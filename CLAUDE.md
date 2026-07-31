# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An iterative computer-vision project that evolves a fire/smoke hazard detector across five stages: binary classification → 4-class transfer learning → robustness/edge quantization → YOLO object detection → semantic segmentation. Each iteration is a self-contained pipeline (dataset → model → trainer → config) sharing common infrastructure in `src/`. See `README.md` for the full per-iteration writeup and results, and `project_architecture.md` for the original design plan.

## Commands

There is no test suite, linter, or build step configured in this repo. Work is run/verified by executing the iteration scripts directly.

```bash
pip install -r requirements.txt      # PyTorch with CUDA must be installed separately first

# Train an iteration (each reads configs/iterationN.yaml by default)
python scripts/run_iteration1.py --config configs/iteration1.yaml   # FireCNN binary classifier
python scripts/run_iteration2.py                                     # MobileNetV3 4-class transfer learning
python scripts/run_iteration3.py                                     # Robust MobileNetV3 + Albumentations + PTQ
python scripts/run_iteration4.py [--skip-export]                     # YOLO26 detector (Ultralytics) + ONNX/TensorRT export
python scripts/run_iteration5.py                                     # Lightweight U-Net segmentation

# Evaluate a trained checkpoint on the test split
python scripts/evaluate_iteration1.py --checkpoint checkpoints/iteration1/best_model.pt
python scripts/evaluate_iteration2.py --checkpoint checkpoints/iteration2/best_model.pt
python scripts/evaluate_iteration4.py --checkpoint checkpoints/iteration4/best_model.pt

# Edge/PTQ simulation and export
python scripts/simulate_edge_iteration3.py --checkpoint checkpoints/iteration3/best_model.pt [--log-wandb]
python scripts/export_iteration4.py --weights <path.pt> --format onnx engine [--log-wandb]

# Dataset sanity checks
python scripts/inspect_dataset.py --data-dir data [--strict]
python scripts/inspect_segmentation_dataset.py

# Inference visualization across all iterations' best checkpoints
python scripts/visualize_predictions.py
```

All scripts insert the project root onto `sys.path` themselves (`PROJECT_ROOT = Path(__file__).resolve().parent.parent`), so run them from anywhere with the repo's Python environment active — no `PYTHONPATH` setup needed.

Every training script takes `--config <path>` pointing at a YAML file under `configs/` (`data`, `model`, `training`, `wandb` sections; iteration 4 also has `export`). To change hyperparameters, edit or copy the relevant YAML rather than passing CLI flags — the scripts have no other tunable arguments.

## Architecture

### Iteration pattern

Every iteration follows the same shape: a `DataModule`/`Dataset` in `src/dataset*.py`, a model in `src/model*.py`, a `Trainer` subclass in `src/trainer/`, and a thin `scripts/run_iterationN.py` entry point that wires config → data module → model → trainer → `trainer.fit_with_test(...)`. When adding a new iteration or modifying an existing one, follow this same wiring rather than introducing a different structure.

### Shared trainer infrastructure (`src/trainer/base.py`)

`BaseTrainer` owns the entire train/val loop, W&B logging, checkpointing, and LR scheduling. Subclasses only need to implement `compute_batch_accuracy` and optionally `compute_epoch_extras`/`build_epoch_extras` for task-specific metrics (F1, confusion matrices, etc.):
- `BinaryTrainer` (`trainer/binary.py`) — iteration 1, also re-exported as `Trainer` from `src/train.py` for backward compatibility.
- `MulticlassTrainer` (`trainer/multiclass.py`) — iteration 2.
- `RobustMulticlassTrainer` (`trainer/robust.py`) — iteration 3, adds two-phase (frozen-backbone → fine-tune) training with cosine/plateau scheduling.
- `SegmentationTrainer` (`trainer/segmentation.py`) — iteration 5, adds AMP (`torch.cuda.amp`) support.

`fit()` saves the best checkpoint by lowest `val_loss` (`best_model.pt` under the config's `checkpoint_dir`); `fit_with_test()` reloads that best checkpoint before running the held-out test split. All metrics are logged to Weights & Biases under project `smoke-fire-detection`.

Iteration 4 (YOLO detection) does **not** use `BaseTrainer` — `src/detection/trainer.py` wraps Ultralytics' own training/validation loop instead, and `src/detection/export.py` handles ONNX/TensorRT export plus edge FPS benchmarking.

### Data labeling (`src/dfire_labels.py`)

All D-Fire-based iterations (1–4) derive labels from the same YOLO-format `.txt` annotation files, using the official D-Fire class mapping (`0 = smoke`, `1 = fire`):
- Iteration 1: `derive_binary_label` — Fire vs. Normal (fire bbox present or not).
- Iterations 2–4: `derive_multiclass_label` — 4-way `Neither / Only_Fire / Only_Smoke / Both` from bbox presence.

Iteration 5 (segmentation) uses a separate COCO polygon-mask dataset (`src/dataset_segmentation.py`), not YOLO label files.

### Device handling

`DEVICE = torch.device("cuda")` is hardcoded in both `src/trainer/base.py` and `src/utils.py` — there is no CPU fallback. Training/evaluation scripts require a CUDA GPU to run.

### Config-driven experiments

YAML configs in `configs/` are the single source of truth per iteration (data paths/splits, model hyperparameters, training schedule, W&B project/entity/tags). Checkpoints are written to `checkpoints/iterationN/` as configured by each YAML's `training.checkpoint_dir`. The `data/` directory expects pre-populated `train/`, `val/`, `test/` splits (D-Fire YOLO format for iterations 1–4; COCO mask format for iteration 5) — it is empty in this repo (only `.gitkeep`) and must be populated separately.
