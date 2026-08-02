"""Training loop manager for multi-class semantic segmentation."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from src.trainer.base import BaseTrainer, EpochMetrics

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda")


class SegmentationTrainer(BaseTrainer):
    """
    Subclass of BaseTrainer designed for semantic segmentation tasks.

    Handles mixed precision training, computes pixel-wise accuracy, class-wise and
    mean IoU, class-wise and mean Dice coefficients, and logs interactive mask overlays
    to W&B.
    """

    def __init__(
        self,
        use_amp: bool = True,
        class_names: tuple[str, ...] = ("background", "smoke", "fire"),
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.class_names = class_names
        logger.info("SegmentationTrainer initialized (AMP enabled: %s)", use_amp)

    def compute_batch_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute pixel-wise accuracy for the batch."""
        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()
        total = labels.numel()
        return correct / total if total > 0 else 0.0

    def _denormalize(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Denormalize ImageNet-normalized tensor back to RGB numpy array [0, 255]."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
        img_denorm = img_tensor * std + mean
        img_denorm = torch.clamp(img_denorm * 255, 0, 255).permute(1, 2, 0).byte().cpu().numpy()
        return img_denorm

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        epoch: Optional[int] = None,
        collect_predictions: bool = False,
    ) -> EpochMetrics:
        self.model.train(training)
        total_loss = 0.0
        total_pixels_correct = 0.0
        total_pixels = 0
        start_time = time.perf_counter()
        total_batches = len(loader)

        num_classes = len(self.class_names)
        total_intersection = torch.zeros(num_classes, device=DEVICE)
        total_union = torch.zeros(num_classes, device=DEVICE)
        total_target_sum = torch.zeros(num_classes, device=DEVICE)
        total_pred_sum = torch.zeros(num_classes, device=DEVICE)

        # We will collect a few samples for validation overlay logging
        samples_to_log = []

        context = torch.enable_grad() if training else torch.no_grad()

        with context:
            for batch_index, (images, labels) in enumerate(loader, start=1):
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                batch_size = labels.size(0)
                preds = logits.detach().argmax(dim=1)

                correct = (preds == labels).sum().item()
                total = labels.numel()

                total_loss += loss.item() * batch_size
                total_pixels_correct += correct
                total_pixels += total

                # Accumulate class-wise IoU / Dice components
                for c in range(num_classes):
                    pred_c = (preds == c)
                    target_c = (labels == c)
                    total_intersection[c] += (pred_c & target_c).sum().item()
                    total_union[c] += (pred_c | target_c).sum().item()
                    total_pred_sum[c] += pred_c.sum().item()
                    total_target_sum[c] += target_c.sum().item()

                # Collect a small subset of predictions during validation for visual logs
                if not training and collect_predictions and len(samples_to_log) < 4:
                    for i in range(min(batch_size, 4 - len(samples_to_log))):
                        samples_to_log.append((
                            images[i].detach(),
                            preds[i].detach(),
                            labels[i].detach()
                        ))

                if training and batch_index % self.log_every_n_batches == 0:
                    elapsed_sec = time.perf_counter() - start_time
                    batches_per_sec = batch_index / max(elapsed_sec, 1e-6)
                    running_accuracy = total_pixels_correct / max(total_pixels, 1)
                    epoch_prefix = f"Epoch {epoch:03d} " if epoch is not None else ""
                    logger.info(
                        "%s[train] batch %04d/%04d loss=%.4f pixel_acc=%.4f speed=%.2f it/s",
                        epoch_prefix,
                        batch_index,
                        total_batches,
                        loss.item(),
                        running_accuracy,
                        batches_per_sec,
                    )

        duration_sec = time.perf_counter() - start_time

        # Compute final IoU and Dice per class
        eps = 1e-6
        iou_per_class = total_intersection / (total_union + eps)
        dice_per_class = (2.0 * total_intersection) / (total_pred_sum + total_target_sum + eps)

        mean_iou = iou_per_class.mean().item()
        mean_dice = dice_per_class.mean().item()
        pixel_accuracy = total_pixels_correct / max(total_pixels, 1)

        # Hazard-only means exclude class 0 (background). The 3-class mIoU is
        # inflated by background, which is both the easiest class and the
        # overwhelming majority of pixels: 85.22% mIoU = mean(96.18, 83.23,
        # 76.26), so ~11 points of the headline come from correctly labelling
        # empty sky. Reporting both makes the segmentation result comparable to
        # the detector, which is scored only on hazard classes.
        hazard_iou = iou_per_class[1:] if len(iou_per_class) > 1 else iou_per_class
        hazard_dice = dice_per_class[1:] if len(dice_per_class) > 1 else dice_per_class

        extras = {
            "mIoU": mean_iou,
            "mDice": mean_dice,
            "mIoU_hazard_only": hazard_iou.mean().item(),
            "mDice_hazard_only": hazard_dice.mean().item(),
            "pixel_accuracy": pixel_accuracy,
        }
        for c, name in enumerate(self.class_names):
            extras[f"IoU/{name}"] = iou_per_class[c].item()
            extras[f"Dice/{name}"] = dice_per_class[c].item()

        if not training and collect_predictions:
            extras["samples_to_log"] = samples_to_log

        avg_loss = total_loss / max(len(loader.dataset), 1)

        return EpochMetrics(
            loss=avg_loss,
            accuracy=pixel_accuracy,
            duration_sec=duration_sec,
            extras=extras,
        )

    def _log_epoch_metrics(
        self,
        epoch: int,
        phase: str,
        train_metrics: EpochMetrics,
        val_metrics: EpochMetrics,
    ) -> None:
        # Extract samples to log so parent method doesn't try to log them as scalars
        samples_to_log = val_metrics.extras.pop("samples_to_log", None)
        super()._log_epoch_metrics(epoch, phase, train_metrics, val_metrics)

        # Log visual overlays to W&B
        if samples_to_log is not None and self._wandb_run is not None:
            wandb_images = []
            class_labels = {i: name for i, name in enumerate(self.class_names)}

            for i, (image_t, pred_t, target_t) in enumerate(samples_to_log):
                image_np = self._denormalize(image_t)
                pred_np = pred_t.cpu().numpy().astype(np.uint8)
                target_np = target_t.cpu().numpy().astype(np.uint8)

                wandb_images.append(
                    wandb.Image(
                        image_np,
                        masks={
                            "predictions": {
                                "mask_data": pred_np,
                                "class_labels": class_labels,
                            },
                            "ground_truth": {
                                "mask_data": target_np,
                                "class_labels": class_labels,
                            },
                        },
                        caption=f"Sample {i + 1}",
                    )
                )

            prefix = self._phase_prefix(phase)
            wandb.log({f"{prefix}val/predictions": wandb_images}, step=epoch)
