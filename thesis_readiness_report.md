# Thesis Readiness Report — `fireDetection`

**Audit date:** 2026-07-31
**Scope:** whole repository at commit `ed80d00` (branch `main`), plus the untracked working files (`CLAUDE.md`, `explanation.md`, `verbal_exam_defense_guide.md`, `prez.html`).
**Method:** every `.py`, `.yaml`, and `.md` file in the repo was read. Claims below cite `file:line`. Nothing was executed (no GPU on this machine; training artifacts live on the SSH workstation).

**Context supplied by the author (not derivable from the repo):**
- Thesis scope is **still images only** — no video/temporal requirement.
- All checkpoints and W&B runs are **intact on the workstation**, and retraining is affordable.
- Only the **workstation GPU** is available today; a Raspberry Pi and a Jetson Nano *could* be obtained.
- **How the D-Fire `val/` split was created is unknown** and must be checked on the workstation.

---

## 1. Executive verdict

**Not thesis-ready as-is — but the gap is measurement discipline, not engineering.** The codebase is clean, modular, and does what it says: five working pipelines, five sets of plausible numbers, good W&B instrumentation. What is missing is everything that makes numbers *defensible*. There is not a single random seed anywhere in the repo (`grep` for `seed|manual_seed|deterministic` returns zero matches), so every reported figure is one unrepeated, unreproducible run with no variance — and iterations 2 and 3 differ by ~1 accuracy point, which is inside the noise band of exactly the thing you cannot currently estimate. The thesis title promises a comparison of **architectures on edge devices**, and the repo contains **zero measurements from any edge device**, no FLOPs, no memory, no power; the only cross-device latency numbers that exist anywhere in this project are hardcoded fictions in `presentation/app.js:16-19` (Pi Zero / Pi 4 / Jetson Nano / Orin FPS and latency, invented). The one real GPU latency function does not call `torch.cuda.synchronize()` around its timed loop (`src/edge_simulation.py:78-87`), so the FP32 number it produces — and the speedup ratio derived from it at `src/edge_simulation.py:165` — is not a valid measurement. Worse for the framing: the five iterations solve five *different tasks* at three *different input resolutions* with four *different epoch budgets*, so there is currently no head-to-head architecture comparison in the work at all — only a task-difficulty progression. Two additional claims in `explanation.md` (frozen YOLO backbone, MuSGD optimizer) are contradicted by the code and config. None of this is fatal: the data, the checkpoints, and the GPU time all still exist, and the fixes below are mostly inference-only reruns plus one controlled training sweep. Budget roughly **2–3 focused weeks** of compute and scripting before you start writing, and the work becomes genuinely defensible.

---

## 2. Inventory: what exists, what ran, where results live

### 2.1 What the code can do vs. what was actually run

| # | Experiment | Model | Dataset | Code status | Run status | Where results live |
|---|---|---|---|---|---|---|
| 1 | Binary fire/normal classification | `FireCNN`, 389,153 params (`src/model.py:32-91`) | D-Fire YOLO split | Complete | **Done** — test acc 93.80%, test loss 0.183 | W&B only; quoted in `iteration_results_report.md:25`, `README.md` |
| 2 | 4-class transfer learning | MobileNetV3-Small, 1,075,748 params (`src/model.py:94-178`) | D-Fire, labels derived from bboxes (`src/dfire_labels.py:84-108`) | Complete, two-phase | **Done** — best val acc 89.16%, test F1 macro 85.25%, test F1 `Only_Fire` 68.78% | W&B only; `iteration_results_report.md:92-100`. Report notes **two runs exist**, later one chosen (`iteration_results_report.md:5`) |
| 3 | Robust 4-class + LR schedule | Same MobileNetV3 + Albumentations (`src/augmentations.py`) | D-Fire | Complete | **Done** — test acc 90.25%, test F1 macro 86.72% | W&B only; `iteration_results_report.md:113-118` |
| 3b | PTQ / INT8 edge simulation | `src/edge_simulation.py` | D-Fire test | Complete but **methodologically broken** (§5) | **Unclear** — no INT8 number appears in `iteration_results_report.md`; `README.md` quotes "~1.1 MiB (INT8)" with no matching run write-up | Unknown — needs W&B check |
| 4 | Object detection | YOLO26n, 2,572,280 params (Ultralytics) | D-Fire YOLO boxes | Complete | **Done** — P 75.90, R 69.10, mAP50 75.24, mAP50-95 44.29 | W&B only; `iteration_results_report.md:132-135` |
| 4b | ONNX / TensorRT export + FPS | `src/detection/export.py` | D-Fire test images | Complete, but benchmarks the **`.pt` model, not the exports** (`src/detection/export.py:139-172`) | **Unknown** — no exported-artifact sizes or FPS reported anywhere | Unknown — needs W&B check |
| 5 | Semantic segmentation | `LightweightUNet`, 7,849,667 params (`src/model_segmentation.py`) | Roboflow COCO masks, 7,110 images | Complete, AMP | **Done** — test mIoU 85.22%, Dice 91.81%, pixel acc 96.36%; IoU bg/fire/smoke 96.18/83.23/76.26 | W&B only; `iteration_results_report.md:159-166` |
| — | Dataset inspection (D-Fire) | `src/dataset_inspector.py` | D-Fire | Complete | **Unknown if run** — output is stdout only (`scripts/inspect_dataset.py:53`), never persisted | Nowhere |
| — | Dataset inspection (COCO) | `src/dataset_segmentation_inspector.py` | Roboflow | Complete; saves overlay examples | **Unknown if run** | `data/coco/examples/` (gitignored) |
| — | Qualitative visualization | `scripts/visualize_predictions.py` | Both | Complete but has a **broken YOLO checkpoint path** (§8) | Presumably run (commit `8ed1140 "visualization"`) | `runs/visualizations/` (gitignored) |
| — | Baseline / competing architectures | — | — | **Not implemented** | **Not run** | — |
| — | Robustness evaluation (perturbed test set) | — | — | **Not implemented** | **Not run** | — |
| — | Multi-seed repeats / variance | — | — | **Not implemented** (no seeding anywhere) | **Not run** | — |
| — | On-device (Pi / Jetson) benchmarks | — | — | **Not implemented** | **Not run** | Fabricated placeholders in `presentation/app.js:16-19` |

### 2.2 Artifact reality check

The repository itself contains **no result artifacts of any kind**. `checkpoints/` and `data/` hold only `.gitkeep`; `wandb/`, `logs/`, `runs/`, `*.pt`, `*.onnx`, `*.engine` are all gitignored (`.gitignore:12-24`). `notebooks/` is empty. There is **no code anywhere that writes a CSV or JSON result file** (`grep` for `json.dump|to_csv|csv\.|savetxt`: zero matches). Every number in `README.md`, `iteration_results_report.md`, and `presentation/index.html` was transcribed by hand from the W&B web UI. That is the single biggest practical risk to the *writing* phase: your thesis tables will be hand-copied from a web dashboard, with no local file that can be diffed, re-generated, or checked.

---

## 3. Repo map and pipeline

Entry points are uniform and clean. Each `scripts/run_iterationN.py` inserts the project root on `sys.path` (e.g. `scripts/run_iteration1.py:12-13`), loads one YAML from `configs/`, and wires **config → DataModule → model → Trainer → fit**. Shared training machinery lives in `src/trainer/base.py` (`BaseTrainer`: epoch loop, W&B logging, best-checkpoint-by-val-loss, LR stepping), with four subclasses:

- `BinaryTrainer` (`src/trainer/binary.py`) — iteration 1; overrides `fit()` to reload the best checkpoint and run the test split (`src/trainer/binary.py:66-83`).
- `MulticlassTrainer` (`src/trainer/multiclass.py`) — iteration 2; `fit_two_phase()` = frozen backbone → unfreeze top *N* blocks with differential LR (`src/trainer/multiclass.py:121-199`).
- `RobustMulticlassTrainer` (`src/trainer/robust.py`) — iteration 3; adds cosine/plateau scheduling and the PTQ hook.
- `SegmentationTrainer` (`src/trainer/segmentation.py`) — iteration 5; AMP, per-class IoU/Dice accumulation, W&B mask overlays.

Iteration 4 bypasses `BaseTrainer` entirely and wraps Ultralytics (`src/detection/trainer.py`), with `src/detection/data_config.py` generating `data.yaml` and `src/detection/export.py` handling export + FPS.

**Test-set hygiene is correct in structure:** all four PyTorch iterations select the best checkpoint by validation loss and only then reload it and touch the test split (`src/trainer/base.py:377-406`, `src/trainer/binary.py:66-83`, `src/trainer/multiclass.py:161-193`). No tuning-on-test is visible in the code. That is a genuine strength — say so in the thesis, because it is the first thing a committee probes.

**Architecturally weak points:** `DEVICE = torch.device("cuda")` is module-level in `src/trainer/base.py:19`, `src/utils.py:14`, `src/model.py:13`, `src/model_segmentation.py:11`, and models call `self.to(DEVICE)` inside `__init__` (`src/model.py:66`, `src/model.py:121`, `src/model_segmentation.py:125`). So a model **cannot be instantiated on a CPU-only machine at all**, even though `scripts/visualize_predictions.py:92` offers a `--device cpu` flag. This directly blocks the CPU-side edge measurements the thesis needs.

---

## 4. Data

**What is documented:** the layout expected (`<split>/images/`, `<split>/labels/` — `src/dataset.py:31-42`), the official D-Fire class mapping `0 = smoke, 1 = fire` (`src/dfire_labels.py:11-12`), and the label-derivation rules: binary = "any fire bbox" (`src/dfire_labels.py:76-81`), 4-class = presence cross-product `Neither / Only_Fire / Only_Smoke / Both` (`src/dfire_labels.py:84-108`). Iteration 5 uses a separate Roboflow COCO polygon dataset of 7,110 images (`iteration_results_report.md:153`).

**What is not documented anywhere in the repo:**
- **Dataset sizes.** No image counts for D-Fire train/val/test appear in any file. The 7,110 figure for the Roboflow set is the only dataset size in the entire project.
- **Class balance.** `README.md` blames the low `Only_Fire` F1 (68.78%) on "imbalance", but **no class distribution is recorded anywhere**. That claim is currently unevidenced. `src/dataset_inspector.py` computes exactly these statistics (`src/dataset_inspector.py:101-135`) — but `scripts/inspect_dataset.py:53` only prints them to a terminal.
- **Split provenance.** Every config expects `train/`, `val/`, `test/` (`configs/iteration1.yaml:3-6`), but D-Fire officially ships train/test only. How `val/` was produced is unknown (author-confirmed). **This is blocker #1.**
- **Leakage checks.** None exist. D-Fire contains sequences of frames from the same scenes; if `val/` was carved randomly from a pool that also feeds `test/`, or if near-duplicate frames straddle splits, every number in the thesis is attackable in one question.

**Preprocessing.** All classification paths resize to a fixed square (`transforms.Resize((image_size, image_size))`, `src/dataset.py:138`) — this **distorts aspect ratio**, worth one sentence in Methodology. ImageNet normalization is used consistently, including for `FireCNN` which is trained from scratch (harmless, but note it). Augmentation ladder: iteration 1 = horizontal flip only (`src/dataset.py:136-146`); iteration 2 = flip + ColorJitter (`src/dataset_multiclass.py:167-178`); iteration 3 = Albumentations with `RandomBrightnessContrast`/`HueSaturationValue`/`CLAHE`/`RandomFog` (`src/augmentations.py:12-49`). Eval transforms are deterministic resize+normalize everywhere — correct.

**Segmentation-specific data risks:**
- Iteration 5 reuses the *classification* augmentation pipeline verbatim (`src/dataset_segmentation.py:15,197-198`). Albumentations does propagate geometric ops to the mask, so this is not a correctness bug — but training a **smoke** segmenter with `RandomFog` injected into the images is a defensible-but-questionable choice that a committee will ask about. Have an answer ready, or ablate it.
- Only **polygon** segmentations are rendered (`src/dataset_segmentation.py:133-139`). Any RLE / `iscrowd` annotation in the Roboflow export is silently dropped to background. Verify the export contains no RLE, or those pixels are mislabeled ground truth.
- Overlap resolution draws fire over smoke by sort order (`src/dataset_segmentation.py:125`) — a modeling decision that belongs in Methodology.

---

## 5. Edge-device angle — the weakest chapter, and it is the thesis subject

This section is where the thesis is most exposed, so it is detailed.

**5.1 There are no on-device measurements.** Nothing in this repo has ever run on a Raspberry Pi, a Jetson, or any ARM hardware. Every device claim in `README.md` ("Low-tier ARM CPUs, legacy Raspberry Pi modules", "requires Jetson Orin Nano / Xavier NX platforms") is an *inference from parameter count*, not a measurement. `iteration_results_report.md:63` is admirably honest about this ("These are practical deployment classes rather than exact FPS guarantees") — the README is not.

**5.2 Fabricated numbers are already in the presentation.** `presentation/app.js:16-19` hardcodes a full FPS and latency matrix per model across `pi-zero`, `pi-4`, `jetson-nano`, `jetson-orin`, `desktop-gpu` (e.g. FireCNN 45 FPS / 22 ms on a Pi 4; YOLO26n 3.5 FPS / 285 ms), plus per-device quantization boost factors (`presentation/app.js:30-70`). These are rendered in the presentation as an interactive "edge simulator". **No code in this repository produced these numbers.** If any of them reach the thesis, they are fabricated experimental data — the most serious single item in this audit. Either delete them, replace them with measurements, or label them unmistakably as an illustrative model with a stated source.

**5.3 The GPU latency measurement is invalid.** `benchmark_inference_ms` (`src/edge_simulation.py:53-94`) times a loop of CUDA forward passes with `time.perf_counter()` and **never calls `torch.cuda.synchronize()`** (contrast `src/detection/export.py:107-119`, which does). CUDA launches are asynchronous, so the FP32 figure measures kernel-launch overhead, not execution. Everything derived from it — `edge/latency_speedup_ratio` (`src/edge_simulation.py:165`) — is unusable.

**5.4 The quantization comparison confounds precision with hardware.** `run_edge_simulation` benchmarks **FP32 on GPU** against **INT8 on CPU** (`src/edge_simulation.py:150-162`). There is no FP32-on-CPU baseline, so the effect of INT8 cannot be separated from the effect of moving GPU→CPU. The headline "quantization speedup" is therefore not attributable. Additionally, `quantize_dynamic` is called with `{nn.Linear, nn.Conv2d}` (`src/edge_simulation.py:44-48`); PyTorch's dynamic-quantization default mapping is built around Linear/RNN modules, and Conv coverage varies by version — you must **print the converted module tree and the resulting state-dict size** to know how much of MobileNetV3 actually became INT8 before quoting the "~1.1 MiB" figure in `README.md`. No static (calibrated) PTQ or QAT comparison exists.

**5.5 The YOLO export pipeline never benchmarks the exports.** `run_full_export_pipeline` (`src/detection/export.py:139-172`) calls `benchmark_fps` on `self.model` — the loaded **PyTorch `.pt`** — and *then* exports ONNX/TensorRT, recording only each artifact's **file size** (`src/detection/export.py:162-167`). So the project's central edge-deployment claim ("proves edge-deployment readiness", `project_architecture.md:56`) has zero latency evidence for either export format.

**5.6 The FPS benchmark measures the wrong thing.** `benchmark_fps` (`src/detection/export.py:99-123`) loops `self.model.predict(source=<file path>)`, so **every iteration re-reads and re-decodes the same JPEG from disk** and rebuilds Ultralytics `Results` objects. That is a disk-I/O-inclusive end-to-end pipeline number on one image at batch size 1 — not model latency, and not comparable to any published figure.

**5.7 Missing entirely:** FLOPs/MACs (no `thop`/`fvcore` anywhere), peak memory, power/energy, batch-size sweeps, throughput-vs-latency curves, and warm/cold start behaviour.

**5.8 Fairness across models is not established.** Input resolutions differ (224 / 640 / 256), tasks differ (binary / 4-class / detection / segmentation), and datasets differ for iteration 5. Any sentence of the form "model X is faster than model Y" is currently comparing tasks, not architectures. This must either be reframed (task-cost analysis) or fixed with a controlled comparison (§10).

---

## 6. Evaluation

**Implemented and correct:**
- Multi-class: accuracy, macro-F1, per-class F1, confusion matrix, via scikit-learn with explicit `labels=` and `zero_division=0` (`src/metrics.py:27-65`) — correctly done.
- Segmentation: dataset-level IoU and Dice accumulated over the whole epoch rather than averaged per batch (`src/trainer/segmentation.py:109-115,143-149`) — the right way; per-batch averaging is a common error you avoided.
- Detection: whatever Ultralytics puts in `results_dict` (`src/detection/trainer.py:141-150`).

**Gaps that matter:**
- **Iteration 1 reports accuracy only.** `BinaryTrainer` never overrides `build_epoch_extras`, so the base returns `{}` (`src/trainer/base.py:177-183`). For a *fire detector*, accuracy alone is the wrong headline — recall, false-alarm rate, PR-AUC and an operating-threshold discussion are what a safety-critical framing demands. There is also a latent trap: `src/trainer/base.py:147` computes predictions with `logits.argmax(dim=1)`, which on `(N,1)` binary logits is always class 0; harmless today only because iteration 1 discards those extras.
- **mIoU includes background.** 85.22% = mean(96.18, 76.26, 83.23) — the easiest class inflates the headline. Report hazard-only mIoU (≈79.7%) alongside it, or a committee will compute it for you.
- **No per-class AP for detection.** `_extract_validation_metrics` flattens `results_dict` only; per-class smoke vs fire AP, and small/medium/large AP, are not pulled out — yet "smoke is harder than fire" is a central narrative claim.
- **No per-condition breakdowns.** Nothing splits results by day/night, fog, smoke density, or object size.
- **No failure-case analysis.** `visualize_predictions.py` samples images *randomly* (`scripts/visualize_predictions.py:199`), not by error type. There is no code that surfaces the worst false negatives/positives.
- **No calibration or threshold analysis.** `conf_threshold: 0.25` is declared in `configs/iteration4.yaml:26` and **never passed to `validate()`** (`src/detection/trainer.py:100-109`) — a dead config key.
- **Confusion matrices exist only inside W&B** (`src/trainer/base.py:283-293`); nothing is written locally for thesis figures.

---

## 7. Reproducibility

| Aspect | Status |
|---|---|
| Random seeding | **Absent.** Zero matches for `seed`, `manual_seed`, `deterministic`, `worker_init_fn`, `generator=` across all `.py` files. Every reported number is one unseeded run. |
| Repeats / variance | **Absent.** No multi-seed runs, no mean±std anywhere. |
| Dependency pinning | **Weak.** `requirements.txt` has lower bounds only, no lock file; **torch/torchvision are not listed at all** (prose instruction at `requirements.txt:1`); `matplotlib` is used at `scripts/visualize_predictions.py:380-381` but is not declared. |
| One-command reproduction of a reported number | **No.** You can rerun a *training*, but with no seed it will not reproduce the figure. |
| Config ↔ result linkage | **Weak.** Configs are versioned in git; results are only in W&B. Nothing in the repo maps a commit + config to a run ID. |
| Config drift | `configs/iteration5.yaml` has **no `scheduler` block**, yet `scripts/run_iteration5.py:81` silently defaults to cosine annealing — so the actual LR schedule used for iteration 5 is not recorded in its config. `conf_threshold` (iteration 4) is unused. |
| Experiment tracking | **Good** — thorough W&B integration incl. gradients (`src/trainer/base.py:104`), per-phase prefixes, confusion matrices, mask overlays. This is a real strength; it is just the *only* store. |
| Figure reproducibility | `scripts/visualize_predictions.py` uses unseeded `random.choice` (lines 199, 216, 227) — the same command produces different figures each run. |

---

## 8. Code quality (weighted low, as instructed)

Genuinely good: consistent OOP structure, type hints, docstrings, `from __future__ import annotations`, clean config/CLI separation, a real base-class hierarchy rather than five copy-pasted scripts. This will read well in an engineering chapter.

Defects found:
- **Broken checkpoint path.** `scripts/visualize_predictions.py:688` looks for `checkpoints/iteration4/yolo26-fire/best.pt`, but `configs/iteration4.yaml:24` sets `run_name: yolo26-dfire` and Ultralytics writes to `<dir>/<name>/weights/best.pt` — the correct form is used in `scripts/export_iteration4.py:62-68`. The YOLO panel therefore silently renders "Model N/A" (`scripts/visualize_predictions.py:509`).
- **Dead in-training edge simulation.** `MulticlassTrainer.fit_two_phase` closes the W&B run in its `finally` block (`src/trainer/multiclass.py:195-196`); `RobustMulticlassTrainer` then calls `wandb.log(edge_metrics)` afterwards (`src/trainer/robust.py:100-110`) — after the run is finished. This plausibly explains why no INT8 numbers ever reached `iteration_results_report.md`. The standalone `scripts/simulate_edge_iteration3.py` is the working path.
- **Dead code.** `DFireMulticlassDataModule._build_loader` (`src/dataset_multiclass.py:191-210`) is never called — `setup()` constructs loaders inline (`src/dataset_multiclass.py:212-255`).
- **Missing `evaluate_iteration3.py`** — iteration 3 must be re-evaluated with `evaluate_iteration2.py`, which builds the data module without the `augmentation` argument (`scripts/evaluate_iteration2.py:56-66`). Eval transforms happen to be numerically equivalent, so results are unaffected, but it is fragile.
- **No tests, no linter, no CI** anywhere in the repo.
- `README.md` is UTF-16-encoded with mangled box-drawing/emoji, and `prez.html` is a 0-byte file.
- `explanation.md` states the YOLO backbone was "frozen during the initial epochs" (`explanation.md:161`) — **no `freeze` argument exists** in `src/detection/trainer.py:47-67` — and that the optimizer is "SGD with Momentum (MuSGD)" (`explanation.md:170`) while `configs/iteration4.yaml:19` sets `optimizer: auto`. Both are exactly the kind of statement a committee verifies against the code.

---

## 9. Gap list

### 9.1 BLOCKERS — fix or run before writing

**B1. Unknown split provenance + no leakage check** *(0.5–1 day)*
*What's missing:* nobody knows how `data/val/` was created, and no duplicate/near-duplicate scan across splits has ever been run. D-Fire contains multiple frames from the same scenes.
*Why it matters:* if val was carved from test, model selection contaminated the held-out set; if near-duplicates straddle train/test, every accuracy figure is inflated. This is the single easiest question for a committee to ask and the hardest to recover from mid-defense.
*Steps:* on the workstation, document how `data/` was built; run a perceptual-hash (or file-hash + pHash) scan for cross-split duplicates; if contamination exists, re-split cleanly by source scene and retrain. Persist the split manifest (list of filenames per split) as a versioned file in the repo.
*Files:* `data/` (workstation), new `scripts/check_split_integrity.py`, `src/dataset_inspector.py`.

**B2. Zero seeding → no reproducibility, no variance** *(0.5 day to implement, 2–4 days of GPU time to rerun)*
*What's missing:* seeding of `torch`/`numpy`/`random`/dataloader workers, and ≥3 seeds per reported configuration.
*Why it matters:* iteration 2 (89.16% val) vs iteration 3 (90.25% test) differ by ~1 point. Without variance you cannot claim any of your ordering conclusions, and "did you repeat it?" is a standard question.
*Steps:* add a `set_seed(cfg.seed)` helper (torch, cuda, numpy, python-random, `worker_init_fn`, `generator=` on each DataLoader); add `seed:` to every config; rerun iterations 1, 2, 3, 5 with 3 seeds; report mean±std. Iteration 4 (50 epochs) with 3 seeds if compute allows, otherwise 2 and state it.
*Files:* `src/utils.py`, all `configs/*.yaml`, all `scripts/run_iteration*.py`, `src/dataset*.py`.

**B3. Fabricated edge numbers in the presentation** *(1 hour to remove; see B4 to replace)*
*What's missing:* `presentation/app.js:16-19` and `:30-70` invent per-device FPS/latency/quantization factors.
*Why it matters:* fabricated experimental data in a thesis-adjacent artifact is an integrity issue, not a polish issue.
*Steps:* delete, or replace with measured values, or relabel the whole widget as an analytical model with its formula and assumptions stated.
*Files:* `presentation/app.js`, `presentation/index.html`, `README.md` (hardware table).

**B4. No valid latency/efficiency measurements for any model** *(2–3 days)*
*What's missing:* a single, uniform benchmarking harness. Current code has an unsynchronized CUDA timer (`src/edge_simulation.py:78-87`), a disk-I/O-bound YOLO FPS loop (`src/detection/export.py:99-123`), exports that are never benchmarked (`src/detection/export.py:162-167`), and no FLOPs/memory anywhere.
*Why it matters:* this *is* the thesis topic. Without it, the edge chapter is prose.
*Steps:* write one `src/benchmark.py` applying an identical protocol to every model — batch=1 (plus a batch sweep), ≥50 warmup, ≥200 timed iterations, `torch.cuda.synchronize()` around the timed region, pre-loaded in-memory tensors (no disk I/O), report median + p95 not just mean, plus params, FLOPs (`thop`/`fvcore`), on-disk size, and peak memory (`torch.cuda.max_memory_allocated` / `resource` on CPU). Run each model across: GPU FP32, GPU FP16, CPU FP32, CPU INT8, ONNX Runtime CPU. Write results to CSV.
*Files:* new `src/benchmark.py` + `scripts/run_benchmarks.py`, replacing `src/edge_simulation.py:53-94` and `src/detection/export.py:65-137`.

**B5. Quantization comparison confounds precision with hardware** *(0.5 day, part of B4)*
*What's missing:* an FP32-on-CPU baseline, and verification of what `quantize_dynamic` actually converted.
*Why it matters:* "INT8 gave us a speedup" is currently unattributable, and the "~1.1 MiB INT8" figure in `README.md` is unverified against the code path that would produce it.
*Steps:* measure FP32-CPU, INT8-CPU, FP32-GPU on the same harness; print the post-quantization module tree and state-dict size; state explicitly which layer types were converted.
*Files:* `src/edge_simulation.py:35-50,134-185`, `README.md`.

**B6. Hardcoded CUDA blocks all CPU measurement** *(0.5 day)*
*What's missing:* device injection. `self.to(DEVICE)` inside model constructors (`src/model.py:66,121`, `src/model_segmentation.py:125`) with `DEVICE = torch.device("cuda")` at module scope makes CPU instantiation impossible.
*Why it matters:* B4 and B5 cannot be executed without this.
*Steps:* remove `.to(DEVICE)` from constructors; pass device explicitly from the trainer/benchmark harness.
*Files:* `src/model.py`, `src/model_segmentation.py`, `src/trainer/base.py:19`, `src/utils.py:14`.

**B7. No architecture comparison exists** *(2–4 days incl. training)*
*What's missing:* the thesis promises comparing architectures; the repo delivers a progression across five *different tasks* at three input resolutions with four different epoch budgets (10 / 5+10 / 5+10 / 50 / 40).
*Why it matters:* "which architecture is better for edge fire detection?" is currently unanswerable from your own results, and the budget mismatch confounds any cross-iteration claim.
*Steps:* pick **one** task as the comparison axis — the 4-class D-Fire classification is the cheapest — and train MobileNetV3-Small vs ResNet18 vs EfficientNet-B0 vs your `FireCNN` under **identical** budget, resolution, augmentation, and seeds. Report accuracy/F1 *and* the B4 efficiency metrics for each. That single table is the backbone of the Results chapter.
*Files:* `src/model.py`, new `configs/comparison_*.yaml`, `scripts/run_iteration2.py` (parameterize backbone).

**B8. The "robustness" claim is unmeasured** *(1 day, inference only)*
*What's missing:* `README.md` claims iteration 3 has "drastically higher real-world generalization" — but augmentation was applied to *training only*, and no perturbed test set was ever evaluated. The measured val accuracy actually *dropped* (88.77 vs 89.16).
*Why it matters:* this is an unsupported claim about your own headline contribution, stated in the repo's most public document.
*Steps:* build a fixed corruption suite (fog, motion blur, brightness ±, JPEG compression, Gaussian noise; 3 severities each) applied to the **test** split, and evaluate iteration-2 vs iteration-3 checkpoints on all of it. No retraining needed. This converts a soft claim into a measured, defensible result — the cheapest high-value experiment in this list.
*Files:* new `src/corruptions.py`, new `scripts/evaluate_robustness.py`, reusing `src/augmentations.py`.

### 9.2 IMPORTANT — significantly strengthens the thesis

**I1. No machine-readable results** *(1 day)* — nothing writes CSV/JSON (`grep`: zero matches). Add a results writer that dumps per-run metrics, per-class metrics, confusion matrices, and benchmark rows to `results/*.csv|json`, and a `scripts/make_tables.py` that generates thesis tables/figures from them. Removes hand-transcription errors and makes every table regenerable. *Files:* `src/utils.py`, `src/trainer/base.py`, new `scripts/make_tables.py`.

**I2. Dataset statistics are never persisted** *(0.5 day)* — `src/dataset_inspector.py` already computes counts, class balance, box counts, and integrity checks; `scripts/inspect_dataset.py:53` only prints them. Add `--output json` and commit the result. Your Dataset chapter needs exactly this table, and the "imbalance" explanation for the 68.78% `Only_Fire` F1 is currently unevidenced. *Files:* `scripts/inspect_dataset.py`, `src/dataset_inspector.py`.

**I3. Binary iteration reports accuracy only** *(0.5 day)* — add precision/recall/F1/PR-AUC/ROC-AUC and a threshold sweep for `FireCNN`; a fire detector's operating point is a discussion-worthy result, not a footnote. Also fix the latent `argmax(dim=1)` on binary logits (`src/trainer/base.py:147`). *Files:* `src/trainer/binary.py`, `src/metrics.py`.

**I4. Detection metrics are under-extracted** *(0.5 day)* — pull per-class AP (smoke vs fire) and small/medium/large AP from the Ultralytics results object, and wire the unused `conf_threshold` (`configs/iteration4.yaml:26`) into `validate()` (`src/detection/trainer.py:100-109`). Supports the "smoke is harder" narrative with evidence. *Files:* `src/detection/trainer.py`.

**I5. No failure-case analysis** *(1 day)* — add a script that ranks test images by loss/error and exports the worst *N* per class, plus confusion-pair montages. Discussion chapters live or die on this. *Files:* new `scripts/analyze_failures.py`, `scripts/visualize_predictions.py`.

**I6. Segmentation headline inflated by background** *(1 hour)* — report hazard-only mIoU alongside the 3-class mIoU. *Files:* `src/trainer/segmentation.py:143-159`.

**I7. Code/doc contradictions** *(1 hour)* — `explanation.md:161` (frozen YOLO backbone) and `:170` (MuSGD) contradict `src/detection/trainer.py:47-67` and `configs/iteration4.yaml:19`. Fix the docs or the code before either becomes a thesis sentence. *Files:* `explanation.md`, optionally `src/detection/trainer.py`.

**I8. Environment not pinned** *(1 hour)* — pin torch/torchvision/ultralytics/albumentations versions, add `matplotlib` (used at `scripts/visualize_predictions.py:380`), and freeze a lock file. Record CUDA/driver/GPU model — a Methodology chapter needs the hardware spec, and the workstation is currently undocumented. *Files:* `requirements.txt`.

**I9. Iteration 2 has two runs; one was chosen post hoc** *(1 hour)* — `iteration_results_report.md:5` states the later run was used. Document why (config difference? crash?) or discard the ambiguity by rerunning under B2. *Files:* `iteration_results_report.md`.

**I10. Inconsistent headline metrics** *(1 hour)* — the `README.md` results matrix headlines iteration 2 with **val** accuracy while iterations 1, 3, 5 use **test**. Make every headline test-set. *Files:* `README.md`, `iteration_results_report.md`.

**I11. COCO RLE annotations silently dropped** *(2 hours)* — `src/dataset_segmentation.py:133-139` handles polygon lists only. Verify the export has no RLE/`iscrowd`; if it does, decode it or document the exclusion. *Files:* `src/dataset_segmentation.py`.

**I12. No published-baseline comparison** *(1 day, literature + one table)* — D-Fire has published results; place your numbers next to them. Without this, "is 75.24 mAP50 good?" has no answer.

### 9.3 NICE-TO-HAVE — polish

- **N1** *(0.5 day)* — one physical device run. Even a single Raspberry Pi 4 with ONNX Runtime, or a Jetson Nano with TensorRT, converts the entire edge chapter from projection to measurement. See §10 for the recommendation.
- **N2** *(0.5 day)* — smoke test suite: tiny synthetic dataset, assert each pipeline runs one epoch end-to-end; assert `derive_multiclass_label` against handcrafted label files. There are currently no tests at all.
- **N3** *(1 hour)* — fix `scripts/visualize_predictions.py:688` (`yolo26-fire` → `yolo26-dfire/weights`), seed its `random.choice` calls, and add `--seed`.
- **N4** *(1 hour)* — delete `src/dataset_multiclass.py:191-210` (dead), fix the post-`finish_wandb` logging in `src/trainer/robust.py:100-110`, add the missing `scripts/evaluate_iteration3.py`.
- **N5** *(1 hour)* — re-save `README.md` as UTF-8 (currently UTF-16 with mojibake) and delete the 0-byte `prez.html`.
- **N6** *(0.5 day)* — record model-card-style summaries per iteration (params, FLOPs, size, resolution, train time, energy) auto-generated from the results CSV.
- **N7** *(1 day)* — ablations on iteration 5: with/without `RandomFog` in the segmentation augmentation; Dice-only vs Focal-only vs combined loss. Directly answers likely questions about `src/losses.py`.

---

## 10. Recommendation on physical hardware

You asked whether you actually need a Pi/Jetson. **My recommendation: get one Raspberry Pi 4/5 *or* one Jetson Nano — not both — and spend one afternoon on it.**

Reasoning: your thesis title claims *edge devices*. Everything else in the work can be defended on a workstation, but a committee reading "edge deployment" will ask "on what device, measured how?" Right now the answer is "none, estimated". One board with one benchmark table (params / FLOPs / size / median latency / p95 / throughput / peak RAM, per model, per precision) turns the weakest chapter into the most concrete one, and it costs less than a day of work once the B4 harness exists.

If you decide against it, that is a legitimate scoping choice — but then you must (a) delete the fabricated per-device numbers (B3), (b) retitle the contribution as an *edge-suitability analysis* using proxy metrics measured on the workstation (CPU FP32/INT8 + ONNX Runtime is a reasonable ARM proxy, stated as such), and (c) state the limitation explicitly in both Methodology and Discussion. What is not survivable is claiming device-level results without device-level measurements.

---

## 11. "Ready to write" checklist by chapter

### Methodology
| Have | Missing |
|---|---|
| Full architecture descriptions (`src/model.py`, `src/model_segmentation.py`) and rich rationale in `explanation.md` | Hardware/software environment spec (GPU model, CUDA, torch version) — nowhere in repo (I8) |
| Loss formulations incl. Dice+Focal derivation (`src/losses.py`, `explanation.md:218+`) | Seeding & repeat protocol (B2) |
| Two-phase transfer-learning protocol (`src/trainer/multiclass.py:121-199`) | Benchmarking protocol: warmup, iterations, synchronization, batch size, precision (B4) |
| Augmentation pipelines (`src/augmentations.py`) | Justification for unequal epoch budgets across iterations, or equalized budgets (B7) |
| Config-driven experiment design (`configs/*.yaml`) | Corrected claims: YOLO freezing and optimizer (I7) |
| Correct best-checkpoint-by-val + single test evaluation | Statement of the exact metric definitions used (esp. mIoU incl./excl. background) |

### Dataset
| Have | Missing |
|---|---|
| Source and format description; label-derivation rules (`src/dfire_labels.py`) | **Split provenance and integrity — B1** |
| Inspection tooling that computes everything needed (`src/dataset_inspector.py`) | Actual numbers: split sizes, class balance, box counts (I2) |
| Roboflow set size (7,110) and rationale for the dataset switch (`iteration_results_report.md:145-153`) | Leakage/duplicate analysis (B1) |
| Preprocessing description | Explicit note on aspect-ratio distortion from square resize; RLE handling (I11) |
| | Sample images / class-distribution figures generated from a script |

### Results
| Have | Missing |
|---|---|
| One complete number set per iteration (accuracy, F1, mAP, mIoU, Dice) | **Mean±std across seeds — B2** |
| Per-class F1 (iters 2–3) and per-class IoU (iter 5) | Per-class detection AP (I4); PR/ROC for binary (I3) |
| Parameter counts and FP32 sizes | Measured latency/FLOPs/memory for every model (B4) |
| Confusion matrices (in W&B) | Local, regenerable figures and tables (I1) |
| | The architecture-comparison table (B7) |
| | The robustness-under-corruption table (B8) |
| | Comparison against published D-Fire results (I12) |

### Discussion
| Have | Missing |
|---|---|
| Coherent iteration-to-iteration narrative (`iteration_results_report.md`) | Failure-case analysis with concrete examples (I5) |
| Honest framing of deployment classes (`iteration_results_report.md:63`) | Evidence for the imbalance explanation of low `Only_Fire` F1 (I2) |
| Task-granularity trade-off argument | Evidence for the robustness claim (B8) — currently contradicted by the val numbers |
| Edge-fit reasoning by model size | Accuracy-vs-latency Pareto discussion grounded in measurements (B4) |
| | Explicit limitations section: no on-device tests (or one device, N1), single dataset per task, no video |

---

## 12. Suggested minimal experiment plan

The shortest path from current state to defensible. Ordered by dependency; steps 3–6 are largely parallel on one GPU.

| # | Action | Type | Effort |
|---|---|---|---|
| 1 | Audit `data/` on the workstation: document split creation, run cross-split duplicate scan, commit the split manifest **(B1)** | Analysis | 0.5–1 day |
| 2 | Add seeding + a results→CSV/JSON writer + device injection **(B2, B6, I1)** | Code | 1 day |
| 3 | Rerun iterations 1, 2, 3, 5 with 3 seeds; iteration 4 with 2–3 seeds; report mean±std **(B2)** | Compute | 2–4 days wall-clock |
| 4 | Backbone comparison on the 4-class task at equal budget/resolution/seeds: `FireCNN` / MobileNetV3-S / ResNet18 / EfficientNet-B0 **(B7)** | Compute | 1–2 days |
| 5 | Corruption-suite evaluation of iter-2 vs iter-3 checkpoints on the perturbed test split **(B8)** | Inference only | 1 day |
| 6 | Build the unified benchmark harness and run all models × {GPU FP32, GPU FP16, CPU FP32, CPU INT8, ONNX-RT}; add FLOPs, size, peak memory; benchmark the ONNX/TensorRT exports themselves **(B4, B5)** | Code + measurement | 2–3 days |
| 7 | *(Recommended)* One physical board (Pi 4/5 **or** Jetson Nano): same harness, same table **(N1)** | Measurement | 0.5 day + procurement |
| 8 | Extract per-class detection AP, binary PR/ROC, hazard-only mIoU; persist all confusion matrices locally **(I3, I4, I6)** | Code | 1 day |
| 9 | Failure-case export and Discussion figures **(I5)** | Code | 1 day |
| 10 | Purge fabricated numbers; reconcile README/report/presentation against the new CSVs; fix doc/code contradictions **(B3, I7, I9, I10)** | Cleanup | 0.5 day |

**Total: roughly 12–17 working days**, most of it unattended compute. Steps 1, 2, 3, 5, 6, 10 are the non-negotiable core; 4 and 7 are what elevate the thesis from "adequate" to "strong".

---

## 13. Open questions for you

1. **What exactly is the thesis research question?** "Comparing model architectures on edge devices" and "iterative progression from classification to segmentation" are two different theses. The repo currently supports the second; the first needs B7. Which one are you writing?
2. **How was `data/` populated** — official D-Fire archive as-is, a Roboflow re-export, or a manual re-split? (Blocks B1.)
3. **Did the iteration-3 edge simulation (`scripts/simulate_edge_iteration3.py`) ever run successfully?** If so, what are the INT8 size/latency/accuracy numbers in W&B? The `README.md` "~1.1 MiB (INT8)" figure has no matching write-up in `iteration_results_report.md`.
4. **Did the iteration-4 export pipeline ever complete** — do `.onnx` and `.engine` artifacts exist on the workstation, and were any FPS numbers logged?
5. **What are the workstation specs** (GPU model, VRAM, CPU, CUDA/driver, torch version)? Needed for the Methodology chapter and for every latency table.
6. **Why does iteration 2 appear twice in W&B**, and what distinguished the two runs (`iteration_results_report.md:5`)?
7. **Is `presentation/` a thesis deliverable or just a defense slide deck?** Determines how urgently B3 must be handled and whether the numbers must be replaced rather than removed.
8. **Are you required to compare against published work**, or is an internal comparison sufficient for your department? (Affects I12.)
9. **What is your actual deadline and remaining compute budget?** The plan above assumes ~3 weeks of mostly-unattended GPU time; if that's unavailable, I'd cut steps 4 and 7 and keep 1, 2, 3, 5, 6, 10.
10. **Is the Roboflow segmentation dataset's origin documented** (version, export date, license, whether it overlaps D-Fire imagery)? A committee will ask, and nothing in the repo records it.

---

### Appendix: verification notes

Claims in this report were checked directly against source. Notable verifications: no seeding (`grep -E "seed|manual_seed|deterministic|worker_init_fn|generator="` over all `.py` → 0 matches); no result serialization (`grep -E "json\.dump|to_csv|csv\.|savetxt"` → 0 matches); no FLOPs tooling (`grep -E "thop|flops|profile"` → 0 matches); `matplotlib` imported only at `scripts/visualize_predictions.py:380-381` and absent from `requirements.txt`; repository contains no checkpoints, no data, and no W&B directory (`checkpoints/`, `data/`, `notebooks/` hold only `.gitkeep`; `wandb/`, `runs/`, `logs/` are gitignored at `.gitignore:12-24`).

One initial suspicion was **not** confirmed and is retracted here for the record: `scripts/run_iteration1.py:94` calls `trainer.fit()` rather than `fit_with_test()`, but `BinaryTrainer` overrides `fit()` and does perform the best-checkpoint reload and test evaluation (`src/trainer/binary.py:53-89`). Iteration 1's test-set protocol is sound.
