"""Backbone comparison on the 4-class D-Fire task under an identical protocol.

Why a separate script
---------------------
The thesis compares detection *paradigms* -- classification vs detection vs
segmentation. Those differ in task, input resolution, and training budget, so a
cross-paradigm accuracy difference cannot be attributed to architecture. This
script isolates the architecture variable: one task, one resolution, one
augmentation set, one budget, one seed schedule, four trunks.

Combined with ``results/benchmarks.csv``, it answers the question the paradigm
comparison cannot: *within* a paradigm, how much does the architecture choice
move accuracy, and how much does it move inference cost?

Everything except the backbone is held fixed, including the classifier head
(``BackboneClassifier._make_head``), the two-phase freeze/fine-tune protocol,
the optimizer, the class weights, and the data order for a given seed.

Usage::

    python scripts/run_comparison.py --backbone resnet18 --seed 42
    python scripts/run_comparison.py --all --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from src.dataset_multiclass import DFireMulticlassDataModule
from src.model import BACKBONES, BackboneClassifier, FireCNN
from src.results import record_run
from src.trainer.multiclass import MulticlassTrainer
from src.utils import configure_logging, set_seed

logger = logging.getLogger("run_comparison")

#: FireCNN is included as the "no transfer learning" reference point. It has no
#: pretrained trunk, so the two-phase protocol does not apply to it; it is
#: trained end-to-end for the same total number of epochs.
ALL_BACKBONES = ["firecnn", *sorted(BACKBONES)]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(backbone: str, model_cfg: dict, device: str | None) -> torch.nn.Module:
    if backbone == "firecnn":
        # FireCNN's binary head is replaced by a 4-class one so it can compete
        # on the same task. Everything else about the architecture is unchanged.
        model = FireCNN(device=device)
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, model_cfg["num_classes"])
        return model.to(next(model.parameters()).device)

    return BackboneClassifier(
        backbone=backbone,
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.2),
        device=device,
    )


def run_one(backbone: str, seed: int, config: dict, args: argparse.Namespace) -> dict:
    """Train and test one backbone at one seed; return the run summary."""
    set_seed(seed)
    torch.backends.cudnn.benchmark = False

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    wandb_cfg = dict(config.get("wandb", {}))
    wandb_cfg["run_name"] = f"comparison-{backbone}-seed{seed}"

    data_module = DFireMulticlassDataModule(
        root_dir=PROJECT_ROOT / data_cfg["root_dir"],
        image_size=data_cfg["image_size"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        fire_class_id=data_cfg["fire_class_id"],
        smoke_class_id=data_cfg["smoke_class_id"],
        train_split=data_cfg["train_split"],
        val_split=data_cfg["val_split"],
        test_split=data_cfg["test_split"],
        augmentation=data_cfg.get("augmentation", "torchvision"),
        seed=seed,
    )
    data_module.setup()

    class_weights = data_module.class_weights() if data_cfg.get("use_class_weights", True) else None

    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "comparison" / f"{backbone}-seed{seed}"
    model = build_model(backbone, model_cfg, args.device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("=== %s | seed %d | %d parameters ===", backbone, seed, n_params)

    trainer = MulticlassTrainer(
        model=model,
        train_loader=data_module.train_loader,
        val_loader=data_module.val_loader,
        class_names=DFireMulticlassDataModule.CLASS_NAMES,
        learning_rate=train_cfg["head_learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        class_weights=class_weights,
        checkpoint_dir=str(checkpoint_dir),
        wandb_config=wandb_cfg,
        log_every_n_batches=train_cfg.get("log_every_n_batches", 50),
    )

    experiment_config = {
        "experiment": "backbone_comparison",
        "backbone": backbone,
        "seed": seed,
        "n_parameters": n_params,
        **data_cfg,
        **model_cfg,
        **train_cfg,
    }

    test_loader = data_module.test_loader

    if backbone == "firecnn":
        # No pretrained trunk to freeze: train end-to-end for the same total
        # epoch budget the transfer-learning models receive.
        total_epochs = train_cfg["head_epochs"] + train_cfg["finetune_epochs"]
        summary = trainer.fit_with_test(
            epochs=total_epochs,
            experiment_config=experiment_config,
            test_loader=test_loader,
            phase_name="train",
        )
        trainer.finish_wandb()
    else:
        summary = trainer.fit_two_phase(
            head_epochs=train_cfg["head_epochs"],
            finetune_epochs=train_cfg["finetune_epochs"],
            head_learning_rate=train_cfg["head_learning_rate"],
            finetune_learning_rate=train_cfg["finetune_learning_rate"],
            unfreeze_blocks=train_cfg["unfreeze_blocks"],
            experiment_config=experiment_config,
            test_loader=test_loader,
        )

    summary["n_parameters"] = n_params
    record_run(
        "comparison",
        summary,
        seed=seed,
        backbone=backbone,
        split="test",
        config=experiment_config,
        extra={"checkpoint_dir": str(checkpoint_dir)},
    )

    del model, trainer, data_module
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare backbones on the 4-class task.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "comparison.yaml",
        help="Config defining the shared training protocol.",
    )
    parser.add_argument("--backbone", choices=ALL_BACKBONES, help="Train a single backbone.")
    parser.add_argument("--all", action="store_true", help="Train every backbone.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Seeds to run.")
    parser.add_argument("--device", default=None, help="Torch device override.")
    parser.add_argument(
        "--subprocess",
        action="store_true",
        help="Run each (backbone, seed) in a fresh process for full isolation.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    if not args.config.exists():
        logger.error("Config not found: %s", args.config)
        return

    config = load_config(args.config)
    backbones = ALL_BACKBONES if args.all else [args.backbone or "mobilenet_v3_small"]

    if args.subprocess:
        # A fresh interpreter per run guarantees no state leaks between
        # architectures -- cuDNN caches, RNG advancement, or fragmented VRAM.
        for backbone in backbones:
            for seed in args.seeds:
                command = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--config", str(args.config),
                    "--backbone", backbone,
                    "--seeds", str(seed),
                ]
                logger.info("Launching: %s", " ".join(command))
                subprocess.run(command, check=False)
        return

    results: list[tuple[str, int, dict]] = []
    for backbone in backbones:
        for seed in args.seeds:
            try:
                summary = run_one(backbone, seed, config, args)
                results.append((backbone, seed, summary))
            except Exception as exc:  # noqa: BLE001 - one failure must not stop the sweep
                logger.exception("FAILED %s seed %d: %s", backbone, seed, exc)

    if not results:
        return

    print(f"\n{'backbone':<22} {'seed':>5} {'params':>12} {'test acc':>10} {'test F1':>9}")
    print("-" * 62)
    for backbone, seed, summary in results:
        metrics = summary.get("test_metrics", {})
        print(
            f"{backbone:<22} {seed:>5} {summary.get('n_parameters', 0):>12,} "
            f"{summary.get('test_accuracy', 0):>10.4f} {metrics.get('f1_macro', 0):>9.4f}"
        )


if __name__ == "__main__":
    main()
