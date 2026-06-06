"""Custom loss functions for semantic segmentation class imbalance."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassDiceLoss(nn.Module):
    """
    Multi-class generalization of Dice Loss.

    Computes Dice coefficient per class and returns the mean loss (1 - Dice).
    Stabilized with a smooth factor to handle cases where classes are absent.
    """

    def __init__(self, smooth: float = 1e-5, ignore_index: Optional[int] = None) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Output tensor of shape (N, C, H, W) before softmax.
            targets: Target tensor of shape (N, H, W) with class indices [0, C-1].
        """
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)

        # Convert targets to one-hot: (N, H, W) -> (N, H, W, C) -> (N, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Reshape to (C, N * H * W) to compute class-wise metrics
        probs = probs.permute(1, 0, 2, 3).reshape(num_classes, -1)
        targets_one_hot = targets_one_hot.permute(1, 0, 2, 3).reshape(num_classes, -1)

        intersection = (probs * targets_one_hot).sum(dim=1)
        denominator = probs.sum(dim=1) + targets_one_hot.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice

        # Filter out classes we want to ignore (e.g. if we have a padding class)
        if self.ignore_index is not None:
            classes_to_keep = [i for i in range(num_classes) if i != self.ignore_index]
            dice_loss = dice_loss[classes_to_keep]

        return dice_loss.mean()


class MulticlassFocalLoss(nn.Module):
    """
    Multi-class Focal Loss to address extreme pixel imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Output tensor of shape (N, C, H, W).
            targets: Target tensor of shape (N, H, W).
        """
        # Compute log softmax probabilities
        log_probs = F.log_softmax(logits, dim=1)

        # Gather the log probabilities of the target classes: (N, 1, H, W)
        log_probs_target = log_probs.gather(dim=1, index=targets.unsqueeze(1))
        probs_target = torch.exp(log_probs_target)

        # Apply the focal scaling term
        focal_weight = (1.0 - probs_target) ** self.gamma
        loss = -self.alpha * focal_weight * log_probs_target

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DiceFocalLoss(nn.Module):
    """
    Linear combination of Multiclass Dice Loss and Multiclass Focal Loss.

    Loss = dice_weight * DiceLoss + focal_weight * FocalLoss
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        smooth: float = 1e-5,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dice_loss = MulticlassDiceLoss(smooth=smooth, ignore_index=ignore_index)
        self.focal_loss = MulticlassFocalLoss(alpha=alpha, gamma=gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        return self.dice_weight * dice + self.focal_weight * focal
