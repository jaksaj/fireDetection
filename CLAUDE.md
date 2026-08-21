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
# NOTE: iteration 4 takes --weights (not --checkpoint) and needs the Ultralytics path layout
python scripts/evaluate_iteration4.py --weights checkpoints/iteration4/yolo26-dfire/weights/best.pt --split test

# Edge/PTQ simulation and export
python scripts/simulate_edge_iteration3.py --checkpoint checkpoints/iteration3/best_model.pt [--log-wandb]
python scripts/export_iteration4.py --weights <path.pt> --format onnx engine [--log-wandb]

# Dataset sanity checks
python scripts/inspect_dataset.py --data-dir data [--strict]
python scripts/inspect_segmentation_dataset.py

# Inference visualization across all iterations' best checkpoints
python scripts/visualize_predictions.py
```

### Thesis measurement pipeline

These scripts produce the machine-readable evidence under `results/`. They are
the source of every number that should appear in the thesis; nothing is
transcribed by hand from a dashboard.

```bash
# Dataset statistics, split manifest, and cross-split leakage check
python scripts/dataset_stats.py                     # -> results/dataset_stats.{json,csv}, split_manifest.csv
#   (the duplicate scan runs by default; --skip-hash-check opts out and preserves any prior result)

# Inference cost: latency/FLOPs/memory across device x precision x backend.
# Runs unchanged on the workstation and on a Jetson; rows are tagged per host.
python scripts/run_benchmarks.py [--batch-sizes 1 4 8 16] [--quick]   # -> results/benchmarks.csv

# The common-task comparison: every paradigm collapsed to image-level presence
# on the same test images, so accuracy is comparable across methods.
python scripts/evaluate_common.py [--sweep]         # -> results/common_eval.csv
#   --onnx-dir scores quantized ONNX artifacts, but covers the CLASSIFIERS ONLY;
#   it skips iteration 4 and 5 by design. Use evaluate_onnx_detseg.py for those.

# Quantized detector/segmenter accuracy. Always pass the operating point
# explicitly -- the FP32 control and the INT8 model must be scored at the same
# threshold or part of the reported drop is a threshold artifact.
python scripts/evaluate_onnx_detseg.py --onnx <model.onnx> --method iteration4 --conf 0.10
python scripts/evaluate_onnx_detseg.py --onnx <model.onnx> --method iteration5 --mask-area 0.02

# Robustness under a fixed corruption suite (inference only, no retraining)
python scripts/evaluate_robustness.py --seeds 42 43 44 45 46   # -> results/robustness.csv
#   (--seeds shares corruption work across checkpoints: identical results, ~n-times faster)

# Backbone comparison at identical budget/resolution/seed
python scripts/run_comparison.py --all --seeds 42 43 44 [--subprocess]

# Multi-seed reruns of the five methods
python scripts/run_seeds.py --seeds 42 43 44 [--methods iteration1 iteration2]

# Regenerate every thesis table and figure from results/*.csv
python scripts/make_tables.py                       # -> results/figures/, results/tables/
```

All scripts insert the project root onto `sys.path` themselves (`PROJECT_ROOT = Path(__file__).resolve().parent.parent`), so run them from anywhere with the repo's Python environment active — no `PYTHONPATH` setup needed.

Every training script takes `--config <path>` pointing at a YAML file under `configs/` (`data`, `model`, `training`, `wandb` sections; iteration 4 also has `export`). To change hyperparameters, edit or copy the relevant YAML rather than passing CLI flags. The training scripts additionally accept `--seed N` (overrides the config's `seed:`) and `--tag NAME` (suffixes the checkpoint directory so multi-seed sweeps do not overwrite each other).

### Reproducibility and results

- **Seeding.** `src/utils.set_seed` seeds Python/NumPy/torch/CUDA; DataLoaders additionally need `worker_init_fn=seed_worker` and `generator=make_generator(seed)`, which every `DataModule` wires up when given a `seed=`. A seeded run sets `cudnn.deterministic=True` and `cudnn.benchmark=False`; benchmark runs deliberately invert this, because deterministic kernel selection perturbs the quantity being measured.
- **Results.** `src/results.record_run` appends every run's metrics to `results/metrics.csv` (long format) and writes a full JSON record under `results/runs/`, tagged with git SHA, dirty flag, seed, and environment. `src/results.append_rows` handles the wider schemas used by the benchmark and evaluation CSVs.
- **Devices.** `src/utils.resolve_device` replaces the old hardcoded `torch.device("cuda")`. Models take a `device=` argument and no longer force themselves onto CUDA in `__init__`, so they can be instantiated on CPU for edge benchmarking.

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

Every module resolves its device through `src/utils.resolve_device()`, which honours an explicit `device=` argument and otherwise falls back to CPU with a warning when CUDA is unavailable. Models take a `device=` parameter and no longer force themselves onto CUDA in `__init__`.

What actually needs a GPU:

- **Training** (`scripts/run_iteration*.py`, `run_comparison.py`, `run_seeds.py`) — GPU-bound in practice. These will run on CPU but slowly enough to be impractical.

What does **not** need a GPU, and is routinely run without one:

- **Benchmarking** (`run_benchmarks.py`) — CPU and ARM measurements are a core result; the x86-vs-ARM comparison depends on models being instantiable on CPU.
- **Evaluation** (`evaluate_common.py`, `evaluate_robustness.py`, `analyze_failures.py`) via `--device cpu`.
- **Export and quantization** (`export_for_jetson.py`, `quantize_int8.py`) — export runs on CPU by design, and static INT8 calibration is CPU-only.
- **The entire Jetson bundle** (`jetson/`), which has no PyTorch at all on the device.

### Config-driven experiments

YAML configs in `configs/` are the single source of truth per iteration (data paths/splits, model hyperparameters, training schedule, W&B project/entity/tags). Checkpoints are written to `checkpoints/iterationN/` as configured by each YAML's `training.checkpoint_dir`. The `data/` directory expects pre-populated `train/`, `val/`, `test/` splits (D-Fire YOLO format for iterations 1–4; COCO mask format for iteration 5) — it is empty in this repo (only `.gitkeep`) and must be populated separately.
