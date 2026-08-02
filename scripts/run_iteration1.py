"""Entry point for Iteration 1: Binary Fire vs. Normal classification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import DFireDataModule
from src.model import FireCNN
from src.train import Trainer
from src.results import record_run
from src.utils import configure_logging, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Iteration 1 binary fire classifier on D-Fire."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "iteration1.yaml",
        help="Path to YAML experiment configuration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Overrides the config value. Enables reproducible runs.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Suffix for the checkpoint directory, so multi-seed runs do not overwrite each other.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)

    # Seed precedence: CLI > config > unseeded. An unseeded run is still
    # allowed, but it is recorded as such so it can never be mistaken for
    # a reproducible one.
    seed = args.seed if args.seed is not None else config.get('seed')
    if seed is not None:
        set_seed(int(seed))

    # cuDNN autotuning picks algorithms by timing them, which makes kernel
    # selection input- and load-dependent and therefore breaks bit-level
    # reproducibility. Only enable it when the run is explicitly unseeded.
    torch.backends.cudnn.benchmark = seed is None

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    wandb_cfg = config["wandb"]

    data_module = DFireDataModule(
        root_dir=PROJECT_ROOT / data_cfg["root_dir"],
        image_size=data_cfg["image_size"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        fire_class_id=data_cfg["fire_class_id"],
        train_split=data_cfg["train_split"],
        val_split=data_cfg["val_split"],
        test_split=data_cfg["test_split"],
        seed=seed,
    )
    # Keep per-seed checkpoints separate so a multi-seed sweep does not
    # overwrite its own best model.
    suffix = args.tag or (f'seed{seed}' if seed is not None else '')
    checkpoint_dir = PROJECT_ROOT / train_cfg['checkpoint_dir']
    if suffix:
        checkpoint_dir = checkpoint_dir.parent / f'{checkpoint_dir.name}-{suffix}'

    data_module.setup()

    model = FireCNN(
        in_channels=model_cfg["in_channels"],
        num_conv_blocks=model_cfg["num_conv_blocks"],
        base_channels=model_cfg["base_channels"],
    )

    trainer = Trainer(
        model=model,
        train_loader=data_module.train_loader,
        val_loader=data_module.val_loader,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        checkpoint_dir=str(checkpoint_dir),
        wandb_config=wandb_cfg,
        log_every_n_batches=train_cfg.get("log_every_n_batches", 25),
    )

    experiment_config = {
        "iteration": 1,
        "task": "binary_classification",
        **data_cfg,
        **model_cfg,
        **train_cfg,
    }

    test_loader = (
        data_module.test_loader
        if train_cfg.get("run_test_after_training", True)
        else None
    )

    summary = trainer.fit(
        epochs=train_cfg["epochs"],
        experiment_config=experiment_config,
        test_loader=test_loader,
    )

    record_run(
        "iteration1",
        summary,
        seed=seed,
        split="test" if test_loader is not None else "val",
        config=experiment_config,
        extra={"checkpoint_dir": str(checkpoint_dir), "config_path": str(args.config)},
    )


if __name__ == "__main__":
    main()
