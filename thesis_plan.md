# Fire Detection — Current State & Plan to Thesis-Grade

> 📎 **HISTORICAL — the plan (2026-07-31).** Substantially executed; the numbers
> quoted are the pre-measurement ones. Current results are in
> [THESIS_STATUS.md](THESIS_STATUS.md).


**Date:** 2026-07-31
**Machine audited:** `DESKTOP-7FJTRSO` (the workstation that actually ran the experiments)
**Supersedes:** the open questions in [thesis_readiness_report.md](thesis_readiness_report.md) — 9 of 10 are answered below from measured evidence.

---

## 0. TL;DR

The previous audit was written on a laptop with no access to data, checkpoints, or W&B, so it had to assume the worst on several points. Having now inspected the real machine, **the situation is better than the audit feared in two important ways and worse in one.**

Better:
- **The test split is clean.** `data/test/` is the official D-Fire test set — contiguous, disjoint filename ID ranges, zero overlap with train or val. The audit's blocker B1 ("val may have been carved from test") is **disproved**. No re-split, no retrain needed for leakage.
- **Compute is a non-issue.** Full measured wall-clock for one seed of all five iterations is **~4.8 hours**. Three seeds of everything is ~14 h — two unattended nights. The plan's bottleneck is writing code, not GPU time.

Worse:
- **Iteration 4's headline numbers (P 75.90 / R 69.10 / mAP50 75.24 / mAP50-95 44.29) are validation numbers, not test numbers.** They are the last row of `checkpoints/iteration4/yolo26-dfire/results.csv`, i.e. epoch-50 val metrics. `runs/detect/val/` is empty and no test evaluation was ever completed. Both `README.md` and `iteration_results_report.md` present them unqualified alongside genuine test numbers from the other iterations. This is the one item that must be fixed before anything is written.

**Verdict:** the engineering is done and mostly sound. What is missing is (a) one honest test-set number for iteration 4, (b) repeats so the numbers have error bars, and (c) the entire hardware-cost half of your stated thesis, which currently exists only as invented constants in `presentation/app.js`. That is **~6 working days of coding plus two nights of unattended compute** — comfortably inside your two weeks.

**On the Jetson: yes, take it.** Reasoning in §5. It is free, you already have access, and it is literally the subject of your thesis.

---

## 1. Your thesis, restated

You said: *"multiple ways of detecting fire, and then for each of those ways a comparison of which hardware it can realistically be run on — a full comparison between each."*

That is a good thesis and it is **not** what the repo currently supports. Written out:

> **RQ:** Across fire-detection paradigms — image classification, object detection, semantic segmentation — what is the accuracy/cost trade-off, and which classes of deployment hardware can realistically run each in real time?

Two axes. The repo has one of them (accuracy, single-run) and none of the other (cost).

### 1.1 The problem nobody has flagged yet, and the fix

Your five iterations report **93.80% binary accuracy**, **90.25% 4-class accuracy**, **75.24 mAP50**, and **85.22% mIoU**. These four numbers are on four different scales measuring four different things. You cannot put them on a shared axis, so "a full comparison between each" is currently impossible *by construction* — not because of missing measurements, but because there is no common yardstick.

**The fix is cheap and it is the single highest-value idea in this document: evaluate every method on one common task.**

Collapse every model's output to an **image-level 4-class label** (`Neither / Only_Fire / Only_Smoke / Both`) on the same 4,306 official test images:

| Method | Native output | Collapse rule |
|---|---|---|
| Iter 1 FireCNN | binary logit | fire present → `Only_Fire ∪ Both`; scored on the binary sub-question only |
| Iter 2/3 MobileNetV3 | 4-class softmax | identity (already the target format) |
| Iter 4 YOLO26n | boxes + scores | any box of class *c* with conf > τ → class *c* present |
| Iter 5 U-Net | per-pixel mask | pixel count of class *c* > τ → class *c* present |

Now all methods produce the same prediction on the same images, scored with the same macro-F1. **That gives you one accuracy axis and one latency axis, and the deliverable of your thesis becomes a single Pareto plot** — accuracy vs. latency, one point per (method × device × precision). Everything else in this plan exists to fill in that plot.

This is inference-only. No retraining. Estimated cost: **one day of scripting, ~20 minutes of GPU.**

### 1.2 What happens to the five iterations

You said you're free to add/remove iterations. My recommendation: **keep all five, but relabel them.** They are not five attempts at one problem, they are four paradigms plus one ablation:

| Current | New role in the thesis |
|---|---|
| Iter 1 — FireCNN binary | Paradigm A: *scene-level binary classification*, trained from scratch. Also your cheapest model — likely the fast end of the Pareto front. |
| Iter 2 — MobileNetV3 4-class | Paradigm B: *scene-level multi-class classification* via transfer learning. |
| Iter 3 — Robust MobileNetV3 | **Not a paradigm — an ablation.** Same architecture, same parameter count, therefore *identical inference cost* to iter 2. Frame it as: "robust training buys accuracy for free at deployment time." That is a genuinely good result and it makes iteration 3 earn its place. Currently the claim is unmeasured (§4, R3). |
| Iter 4 — YOLO26n | Paradigm C: *object detection / localisation*. |
| Iter 5 — LightweightU-Net | Paradigm D: *pixel-level segmentation*. |

Optionally add a **backbone axis** inside paradigm B (§6, item P1-B) — ResNet18 and EfficientNet-B0 alongside MobileNetV3-S at identical budget. ~4 h of GPU. It converts "I compared four paradigms" into "I compared four paradigms *and* showed the architecture choice within a paradigm matters less/more than the paradigm choice." Cheap, and it is the difference between a fine thesis and a strong one.

---

## 2. Answers to the audit's open questions

Everything below is measured on this machine, not inferred.

**Q2 — How was `data/` populated?** *(was blocker B1)*

Resolved, and it is good news. Split sizes and filename ID ranges:

| Split | n | `AoF` range | `WEB` range | `PublicDataset` range |
|---|---|---|---|---|
| train | 14,122 | 0–6722 (interleaved) | 0–9441 (interleaved) | 0–1054 (interleaved) |
| val | 3,099 | 8–6702 (interleaved) | 3–9442 (interleaved) | 4–1052 (interleaved) |
| test | 4,306 | **6723–8383** | **9443–11806** | **1055–1335** |

- `train + val = 17,221` and `test = 4,306` — exactly the official D-Fire train/test sizes.
- Test IDs form **contiguous blocks disjoint from train and val** in all three source prefixes → `data/test/` is the **untouched official D-Fire test split**. Filename-level overlap between all split pairs is **0**.
- `val/` was carved from the official *train* pool by what is evidently a **random ~82/18 shuffle** (val IDs interleave with train IDs from index 3 onward).

**Consequence:** your test numbers are defensible as-is. The random train/val shuffle *does* risk near-duplicate frames straddling train and val (D-Fire contains consecutive frames of the same scenes), but that only affects **model selection**, not the held-out result. Write one honest paragraph in Methodology; do not re-split.

**Q3 — Did the iteration-3 edge simulation ever run?** **No.** There is no W&B run for `simulate_edge_iteration3.py`. The five zero-byte W&B directories from 23:03–23:12 on 2026-06-06 contain only an empty `wandb-summary.json` — crashed launches. **The "~1.1 MiB (INT8)" figure in `README.md` has no run behind it and must be deleted or re-measured.**

**Q4 — Did the iteration-4 export pipeline complete?** **Partially.** `checkpoints/iteration4/yolo26-dfire/weights/best.onnx` exists. **No `.engine` file exists anywhere** and `checkpoints/iteration4/exports/` is empty. No FPS numbers were ever logged for any format. TensorRT export has never succeeded.

**Q5 — Workstation specs?** Ready for Methodology:

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12.9 GB, SM 8.6 (Ampere) |
| CPU | AMD Ryzen, 6 cores (Family 25 Model 33 — Zen 3) |
| OS | Windows 11 Pro 10.0.26200 |
| Python | 3.11.4 |
| torch / torchvision | 2.5.1+cu121 / 0.20.1+cu121, CUDA 12.1 |
| ultralytics | 8.4.60 |
| albumentations | 2.0.8 |

**Q6 — Why does iteration 2 appear twice?** Not a mystery. Run `ymb4db0x` stopped at step 6 with best val acc **81.22%** and **no test metrics** — a truncated run. Run `5elw24hx` ran the full 15 steps (5 head + 10 finetune), best val acc **89.16%**, and did evaluate the test split. The later run is simply the only complete one. One sentence in Methodology closes this.

**Q7 — Is `presentation/` a deliverable?** Your answer: no, the fake per-device table can go, but you want the real equivalent in the thesis. **Plan: delete the invented constants now (they are the only integrity risk in the repo), regenerate the same widget from measured CSV at the end.**

**Q8 — Compare against published D-Fire results?** Your answer: not needed. **Dropped from the plan.**

**Q10 — Roboflow segmentation dataset provenance?** `data/coco/README.dataset.txt` and `README.roboflow.txt` exist on disk and are gitignored. **Copy the licence/version/export-date lines into the thesis** — 10 minutes, and it closes a question a committee will definitely ask.

**Not fully resolved — Q1 (research question):** answered by you in §1 above, now written down.

### 2.1 One audit claim that was wrong

The audit states *"there is not a single random seed anywhere in the repo."* True for iterations 1, 2, 3, 5. **False for iteration 4**: `checkpoints/iteration4/yolo26-dfire/args.yaml` records `seed: 0` and `deterministic: true` (Ultralytics defaults). Iteration 4 is the one reproducible training run you have.

---

## 3. Current state — measured

### 3.1 Dataset statistics (computed now; these were missing everywhere)

Image-level 4-class distribution over the D-Fire splits:

| Split | n | Neither | Only_Fire | Only_Smoke | Both |
|---|---|---|---|---|---|
| train | 14,122 | 6,458 (45.73%) | **770 (5.45%)** | 3,836 (27.16%) | 3,058 (21.65%) |
| val | 3,099 | 1,375 (44.37%) | **174 (5.61%)** | 845 (27.27%) | 705 (22.75%) |
| test | 4,306 | 2,005 (46.56%) | **220 (5.11%)** | 1,186 (27.54%) | 895 (20.78%) |
| **total** | **21,527** | 9,838 (45.70%) | **1,164 (5.41%)** | 5,867 (27.25%) | 4,658 (21.64%) |

Bounding boxes: train 7,794 smoke / 9,638 fire · val 1,756 / 2,176 · test 2,315 / 2,878.
Binary (iteration 1) balance: train 27.11% fire · val 28.36% · test 25.89%.

**This directly evidences the claim your README currently makes without support.** `Only_Fire` is 5.41% of the data — an 8.5× under-representation versus `Neither` — and it is exactly the class with the worst F1 (68.78% in iteration 2). The split ratios are also stable across train/val/test (5.45 / 5.61 / 5.11%), so the shuffle was at least stratified in effect. Put this table in the Dataset chapter.

### 3.2 What ran, what it produced, and how long it took

Runtimes are from the W&B `_runtime` field — real measured wall-clock on the RTX 3060.

| # | Method | Params | Res | Budget | Runtime | Reported result | Split | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | FireCNN binary | 389,153 | 224 | 10 ep | **10.3 min** | acc 93.80%, loss 0.183 | **test** ✅ | complete |
| 2 | MobileNetV3-S 4-class | 1,075,748 | 224 | 5+10 ep | **16.5 min** | val acc 89.16%; test acc **89.46%**, test F1-macro **84.38%** | test ✅ | complete |
| 3 | Robust MobileNetV3-S | 1,075,748 | 224 | 5+10 ep | **24.2 min** | val acc 88.77%; test acc 90.25%, F1-macro 86.72% | test ✅ | complete |
| 4 | YOLO26n | 2,572,280 | 640 | 50 ep | **164.7 min** | P 75.90 / R 69.10 / mAP50 75.24 / mAP50-95 44.29 | ⚠️ **val — no test eval exists** | **incomplete** |
| 5 | LightweightU-Net | 7,849,667 | 256 | 40 ep | **72.5 min** | mIoU 85.22%, Dice 91.81%, pixel acc 96.36% | test ✅ | complete |
| — | Edge/INT8 simulation | — | — | — | — | — | — | **never ran** |
| — | ONNX / TensorRT export | — | — | — | — | ONNX only, no benchmarks | — | **partial** |

**Total for one seed of everything: 288 min ≈ 4.8 h.**

Two numbers worth noting that appear nowhere in your write-ups:
- **Iteration 2's test F1-macro is 84.38%.** `iteration_results_report.md` quotes 85.25%, which is the *validation* finetune F1-macro. Not wrong, but mislabelled against iteration 3's test F1-macro in the same comparison.
- **Iteration 3's `Only_Fire` test F1 is 77.08%** vs iteration 2's 68.78% — an **+8.3 point** gain on your worst class. That is the strongest single result in the project and it is not mentioned in any document.

### 3.3 Confirmed defects (re-verified on this machine)

| Claim | Verified |
|---|---|
| No seeding in iterations 1/2/3/5 | ✅ `grep -rE "seed\|manual_seed\|deterministic" src/ scripts/ configs/` → **0 matches** |
| Nothing writes CSV/JSON results | ✅ `grep -rE "json\.dump\|to_csv\|csv\.writer\|savetxt" src/ scripts/` → **0 matches** |
| CUDA hardcoded, blocks CPU benchmarking | ✅ `DEVICE = torch.device("cuda")` at module scope in 6 files; `self.to(DEVICE)` in 3 constructors |
| GPU timing never synchronizes | ✅ `cuda.synchronize` appears **only** in `src/detection/export.py:108,119` — not in `src/edge_simulation.py` |
| Fabricated per-device numbers | ✅ `presentation/app.js:15-19` — full FPS/latency matrix across 5 devices, plus `quantBoost` multipliers |
| YOLO viz checkpoint path broken | ✅ **already fixed** in your uncommitted working tree (`yolo26-fire` → `yolo26-dfire/weights`) |

---

## 4. What actually blocks the thesis

Reprioritised for *your* stated RQ and your "only what matters" constraint. Renumbered R1–R8; the audit's B/I labels are cross-referenced.

### R1 — Iteration 4 has no test-set result *(audit: new)* — **2 hours**
Your only localisation paradigm reports val metrics next to four genuine test metrics. Run `python scripts/evaluate_iteration4.py --checkpoint checkpoints/iteration4/yolo26-dfire/weights/best.pt --split test`. The code path already exists and defaults to `test`. While you're there, extract **per-class AP (smoke vs fire)** — it directly supports your "smoke is harder" narrative, which is currently asserted without evidence.

### R2 — No common evaluation axis across paradigms *(audit: missed entirely)* — **1 day**
See §1.1. Without it your thesis question is unanswerable. This is the highest-value item in the document.

### R3 — Zero cost measurements; the only ones that exist are invented *(audit: B3+B4+B5)* — **2 days**
`presentation/app.js:15-19` is the only place in this project where per-device latency exists, and no code produced it. Meanwhile the one real GPU timer (`src/edge_simulation.py:78-87`) doesn't synchronize, and the YOLO FPS loop re-decodes a JPEG from disk on every iteration (`src/detection/export.py:99-123`) — that's a disk-I/O benchmark, not a latency benchmark. **Delete the fabricated constants today.** Then build one harness (§6, P0-C).

### R4 — Single unrepeated runs, no variance *(audit: B2)* — **0.5 day code + 2 nights compute**
Iterations 2 and 3 differ by ~1 accuracy point. You cannot currently claim that gap is real. At 4.8 h per full seed this is nearly free — there is no excuse not to do it.

### R5 — Hardcoded CUDA blocks all CPU/ARM measurement *(audit: B6)* — **3 hours**
Strictly a prerequisite for R3. Not worth doing for its own sake.

### R6 — The robustness claim is unmeasured *(audit: B8)* — **1 day, inference only**
README claims iteration 3 has "drastically higher real-world generalization". Augmentation was training-only and no perturbed test set was ever built; val accuracy actually *dropped* (88.77 vs 89.16). Under the new framing this is worth more than before: if robust training gives accuracy at **identical inference cost**, that's a clean deployment finding. If it doesn't, iteration 3 has no reason to exist. Either result is publishable; not knowing is not.

### R7 — Nothing is machine-readable *(audit: I1)* — **0.5 day**
Every number in every document was hand-copied from a web dashboard. With ~5 methods × 4 devices/precisions × 3 seeds coming, hand-transcription will break. You said W&B isn't necessary — good, that simplifies this to a plain CSV/JSON writer.

### R8 — Metric honesty fixes *(audit: I3, I6, I10)* — **0.5 day**
- Iteration 1 reports **accuracy only** — for a fire detector, recall and false-alarm rate are the headline. Add PR/ROC + a threshold sweep.
- Segmentation mIoU 85.22% includes background at 96.18%. Report **hazard-only mIoU ≈ 79.7%** alongside it.
- Latent bug: `src/trainer/base.py:147` does `logits.argmax(dim=1)` on `(N,1)` binary logits → always class 0. Harmless today only because iteration 1 discards those extras; it will bite the moment you add binary metrics.

### Explicitly dropped
| Audit item | Why |
|---|---|
| B1 re-split / leakage retrain | Disproved — test split is clean (§2, Q2). One paragraph, not an experiment. |
| I12 published-baseline comparison | Your call: not needed. |
| N2 test suite, N4 dead code, N5 README encoding | Code quality. Nobody will use this code. Skip. |
| B7 as originally scoped | Reduced to an optional backbone axis (§6, P1-B). |

---

## 5. The Jetson: take it

**Recommendation: yes, borrow the Orin Nano.** It is free, you already have access, and your thesis question is *"which hardware can realistically run each method."* Answering that with zero devices means the entire second axis is projection — and you already know how that ends, because `presentation/app.js` is what projection looks like when nobody checks it.

The strong argument is that **one Orin Nano gives you a whole hardware matrix, not one point:**

| Tier from a single board | How |
|---|---|
| ARM CPU (Cortex-A78AE ×6) | ONNX Runtime CPU execution provider — a genuine low-power ARM tier, no Pi needed |
| Edge GPU FP32 / FP16 | PyTorch or TensorRT on the integrated Ampere GPU |
| Edge GPU INT8 | TensorRT with calibration — also finally lands the iteration-4 export story |
| Power-constrained operating points | `nvpmodel` 7 W / 15 W / 25 W-MAXN — three latency points per model, same silicon |

Add the RTX 3060 as the desktop-GPU reference and the Ryzen as an x86-CPU reference, and you have **~5 hardware configurations × 3 precisions** with one borrowed board and no purchases. That is a real comparison matrix.

**Cost:** ~0.5 day of setup (JetPack, ONNX Runtime, TensorRT) + ~0.5 day of runs, *provided* the harness from P0-C is already portable. Budget one full day for first-time Jetson friction — it is always more annoying than it should be.

**If you decide against it,** that is legitimate, but then you must (a) delete the fabricated constants regardless, (b) retitle the contribution as an *edge-suitability analysis using measured proxies* (x86 CPU FP32/INT8 + ONNX Runtime, explicitly stated as a proxy), and (c) state the limitation in both Methodology and Discussion. What is not survivable is device-level claims without device-level measurements.

---

## 6. The plan

Ordered by dependency. Compute is overnight and unattended throughout — **the critical path is your coding time, roughly 6 working days.**

### P0 — Core (non-negotiable, ~4 days)

**P0-A · Purge fabricated data — 1 hour, do this first**
Delete `modelStats` FPS/latency constants and `quantBoost` multipliers from `presentation/app.js:15-70`; remove the "~1.1 MiB (INT8)" claim and the per-device hardware table from `README.md`. The widget can be restored from real CSV in P2-B. *Files: `presentation/app.js`, `presentation/index.html`, `README.md`.*

**P0-B · Seeding + results persistence + device injection — 1 day** *(R4, R5, R7)*
- `set_seed(seed)` in `src/utils.py`: `torch.manual_seed`, `cuda.manual_seed_all`, `numpy`, `random`, plus `worker_init_fn` and `generator=` on every DataLoader. Add `seed:` to all five configs.
- Remove `self.to(DEVICE)` from `src/model.py:66,121` and `src/model_segmentation.py:125`; pass device from the caller. Make the module-level `DEVICE` a default, not a constant.
- `src/results.py`: append-only writer → `results/metrics.csv` + `results/<run>.json`, keyed by `(method, seed, split, git_sha)`. Every trainer calls it at the end of `fit_with_test`.

**P0-C · The benchmark harness — 1.5 days** *(R3)*
One `src/benchmark.py`, one protocol, every model. This is the technical core of the thesis, so it must be defensible:
- Pre-loaded in-memory tensors — **no disk I/O inside the timed region** (fixes `src/detection/export.py:99-123`).
- ≥50 warmup, ≥200 timed iterations, `torch.cuda.synchronize()` around the timed region (fixes `src/edge_simulation.py:78-87`).
- Report **median and p95**, not mean. Batch size 1 as headline, plus a batch sweep.
- Per model: params, **FLOPs** (`thop` or `fvcore`), on-disk size, peak memory (`torch.cuda.max_memory_allocated` / `psutil` on CPU).
- Backends: PyTorch GPU FP32, GPU FP16, CPU FP32, CPU INT8 dynamic, ONNX Runtime CPU. **Benchmark the exported artifacts themselves**, not the `.pt` (fixes `src/detection/export.py:139-172`).
- For the INT8 claim: print the post-quantization module tree and state-dict size so you know what `quantize_dynamic` actually converted. Measure **FP32-CPU as the baseline** so the speedup is attributable to precision, not to the GPU→CPU move.
- Writes straight to `results/benchmarks.csv`. Must run unmodified on the Jetson.
*Replaces `src/edge_simulation.py:53-94` and `src/detection/export.py:65-137`.*

**P0-D · Common evaluation protocol — 1 day** *(R2, R1)*
`scripts/evaluate_common.py` implementing §1.1: every method → image-level 4-class prediction on the same 4,306 test images → shared macro-F1 + confusion matrix → `results/common_eval.csv`. Includes the iteration-4 test-set run and per-class AP extraction (R1). Sweep the box-confidence and mask-area thresholds τ rather than hardcoding them — the threshold choice is a legitimate Methodology paragraph.

**P0-E · Seeded reruns — 0.5 day setup, 2 unattended nights** *(R4)*
3 seeds × all 5 methods ≈ 14.4 h. Report mean ± std everywhere. Fire and forget.

### P1 — High value, cheap (~2 days)

**P1-A · Corruption robustness suite — 1 day, inference only** *(R6)*
`src/corruptions.py` + `scripts/evaluate_robustness.py`: fog, motion blur, brightness ±, JPEG compression, Gaussian noise × 3 severities, applied to the **test** split. Evaluate iteration-2 vs iteration-3 checkpoints across all 15 conditions. ~12 min of GPU. Converts your softest claim into a measured result, and gives iteration 3 its justification: *accuracy under degradation at zero inference cost.*

**P1-B · Backbone axis — 0.5 day code, ~4 h compute** *(optional but recommended)*
FireCNN / MobileNetV3-S / ResNet18 / EfficientNet-B0 on the 4-class task at identical budget, resolution, augmentation, seeds. 4 backbones × 3 seeds ≈ 3.7 h. Adds a within-paradigm dimension to the cost story. Parameterize the backbone in `src/model.py` and add `configs/comparison_*.yaml`.

**P1-C · Metric honesty — 0.5 day** *(R8)*
Binary PR-AUC/ROC-AUC + threshold sweep for iteration 1; fix the `argmax(dim=1)` latent bug; hazard-only mIoU alongside the 3-class figure; persist all confusion matrices locally.

### P2 — Jetson & write-up support (~2 days)

**P2-A · Jetson Orin Nano measurement — 1 day**
JetPack + ONNX Runtime + TensorRT; export all five models; run the **unmodified** P0-C harness across ARM CPU / GPU FP32 / GPU FP16 / TensorRT INT8 × `nvpmodel` 7 W / 15 W / MAXN. This also finally produces the `.engine` artifacts that iteration 4 never got. Append to `results/benchmarks.csv`.

**P2-B · Figures and tables — 1 day**
`scripts/make_tables.py` generating every thesis table and figure from `results/*.csv`:
1. **The Pareto plot** — common-task macro-F1 vs median latency, one point per (method × device × precision). This is your money figure.
2. Method × device × precision latency matrix (the honest version of the deleted `app.js` widget).
3. Accuracy table with mean ± std over 3 seeds.
4. Robustness degradation curves, iteration 2 vs 3.
5. Dataset distribution table (§3.1 — already computed, just needs formatting).
6. Failure-case montages: worst false negatives per class. Cheap, and the Discussion chapter needs it.

### Not doing
Tests, linting, dead-code removal, README re-encoding, `evaluate_iteration3.py`, published-baseline comparison, RLE-annotation handling, segmentation loss ablations. None affect a thesis number.

### Schedule

| Day | Work | Overnight compute |
|---|---|---|
| 1 | P0-A purge · P0-B seeding/results/device | — |
| 2 | P0-C harness | seeded reruns, batch 1 |
| 3 | P0-C harness finish · P0-D common eval + iter-4 test | seeded reruns, batch 2 |
| 4 | P0-D finish · P1-A corruption suite | backbone comparison (P1-B) |
| 5 | P1-C metrics · P1-B collate | — |
| 6 | P2-A Jetson | — |
| 7 | P2-B figures & tables | — |

**~7 days with a day of slack inside your two weeks.** Days 1–5 are the non-negotiable core; day 6 is the Jetson; day 7 turns everything into thesis figures. If you lose time, cut P1-B first, then P2-A (with the §5 fallback wording).

---

## 7. Decisions still open

1. **Confirm the Jetson.** Everything in P2-A depends on it, and the framing of your entire edge chapter depends on the answer. My recommendation is yes (§5).
2. **Seeds: 3 or 5?** 3 seeds ≈ 14 h, 5 ≈ 24 h. You said you can run 24/7 — I'd do 5 for iterations 1/2/3/5 (cheap) and 3 for iteration 4 (2.7 h each). State the asymmetry.
3. **Keep iteration 5's Roboflow dataset?** It is the only method evaluated on different data, which weakens the common-task comparison in §1.1 — the segmentation model would be scored on images it wasn't trained for. Options: (a) accept and state the limitation, (b) evaluate the U-Net on D-Fire test images for the *presence* task only, which is exactly what §1.1 needs and is legitimate. **I'd do (b)** and note the domain shift explicitly.
4. **Is `RandomFog` in the segmentation augmentation defensible?** Iteration 5 reuses the classification pipeline verbatim (`src/dataset_segmentation.py:15,197-198`), so you are injecting synthetic fog while training a **smoke** segmenter. Either have an answer ready or run the one-line ablation.

---

## Appendix — commands to reproduce this audit's findings

```bash
python -c "import os,re;from collections import defaultdict;d=defaultdict(list);[d[re.match(r'([A-Za-z_]+)(\d+)',f).group(1)].append(int(re.match(r'([A-Za-z_]+)(\d+)',f).group(2))) for f in os.listdir('data/test/images')];[print(k,min(v),max(v),len(v)) for k,v in d.items()]"
```

```bash
python scripts/evaluate_iteration4.py --checkpoint checkpoints/iteration4/yolo26-dfire/weights/best.pt --split test
```
