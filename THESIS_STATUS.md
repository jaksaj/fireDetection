# Thesis Work — Status and Findings

**Updated:** 2026-08-18
**Companion documents:** [thesis_plan.md](thesis_plan.md) (the plan), [thesis_readiness_report.md](thesis_readiness_report.md) (the original audit)

This is the running record of what has been built, what has been measured, and
what those measurements changed. Every number below comes from a file under
`results/` produced by a script in `scripts/` — nothing is transcribed by hand.

---

## 1. Headline: the thesis now has its central result

For the first time, all five methods are scored on **one common task, one test
set, one metric**. Previously they reported 93.80% binary accuracy, 90.25%
4-class accuracy, 74.40 mAP50 and 85.22% mIoU — four incomparable scales, which
made the project's own comparison question unanswerable.

**Common-task ranking** (image-level "fire present", 4,306 official D-Fire test
images, best operating point per method — `results/tables/common_eval_binary.md`):

| Rank | Method | Paradigm | Macro-F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|
| 1 | **YOLO26n** | object detection | **0.9689** | 0.9758 | 0.937 | 0.972 |
| 2 | MobileNetV3-S robust (it. 3) | multiclass classification | 0.9510 | 0.9624 | 0.927 | 0.928 |
| 3 | MobileNetV3-S (it. 2) | multiclass classification | 0.9411 | 0.9538 | 0.884 | 0.945 |
| 4 | FireCNN (it. 1) | binary classification | 0.9169 | 0.9380 | 0.916 | 0.838 |
| 5 | U-Net (it. 5) | semantic segmentation | 0.6999 | 0.7559 | 0.524 | 0.625 |

**Paired with measured cost** (`results/tables/pareto_points.md`, batch 1,
fastest configuration per hardware class):

| Method | Macro-F1 | Desktop GPU<br>(RTX 3060, PyTorch) | Desktop CPU<br>(x86) | **Jetson GPU**<br>(TensorRT FP16) | **Jetson CPU**<br>(ARM, ONNX-RT) |
|---|---|---|---|---|---|
| FireCNN | 0.9169 | 0.72 ms | 6.26 ms | **0.37 ms** | 12.01 ms |
| MobileNetV3-S | 0.9411 | 5.69 ms | 4.90 ms | **0.90 ms** | 4.54 ms |
| MobileNetV3-S robust | 0.9510 | 5.81 ms | 4.85 ms | **0.92 ms** | 4.57 ms |
| YOLO26n | **0.9689** | 16.62 ms | 45.19 ms | **5.55 ms** | 77.76 ms |
| U-Net | 0.6999 | 4.88 ms | 107.78 ms | **3.96 ms** | 230.83 ms |

Jetson figures are at MAXN_SUPER; the 15W and 25W modes are in §2.11. Note the
Jetson-vs-desktop columns compare *different runtimes* (TensorRT vs eager
PyTorch), not hardware alone — see the caveat in §2.11.

**The Pareto front is `FireCNN → MobileNetV3-robust → YOLO26n`.** Iteration 2 is
dominated by iteration 3 (same architecture, same cost, lower accuracy).
Iteration 5 is dominated outright — lower accuracy *and* higher cost than
FireCNN. Figure: `results/figures/pareto_accuracy_vs_latency.png`.

That single figure is the thesis. Everything else supports it.

---

## 2. Findings that change what the thesis can claim

These are measurement results that contradict statements currently in the
repository's documents. Each is now backed by a file.

### 2.1 Fire is harder than smoke — the opposite of the project's narrative

Per-class detection AP had never been extracted. Across **3 seeds** on the test
split (mean ± std):

| Class | mAP50 | mAP50-95 |
|---|---|---|
| smoke | **0.8165 ± 0.0044** | **0.5053 ± 0.0017** |
| fire | **0.6811 ± 0.0014** | **0.3546 ± 0.0015** |

Every write-up in this project asserts that smoke is the harder class. The
measurement says fire is, and the margin is not arguable: the mAP50 gap is
0.135, roughly **30× the seed-to-seed standard deviation**. This is the most
statistically secure finding in the project.

The *reason* is a good Discussion topic: fire regions are small, high-variance
and frequently occluded by the smoke they generate, while smoke plumes are
large and texturally distinctive. It also explains the classification results —
`Only_Fire` is both the rarest class (5.4% of data) and the worst-performing one.

Detector aggregates across the same 3 seeds: mAP50 **0.7488 ± 0.0023**,
mAP50-95 **0.4299 ± 0.0011**.

### 2.2 Iteration 4's published numbers were validation, not test

The reported P/R/mAP were the last row of `results.csv` — epoch-50 **validation**
metrics — presented alongside genuine test numbers from the other iterations. The
real test numbers are now measured:

| | val (previously published) | **test (actual)** |
|---|---|---|
| Precision | 75.90 | 75.08 |
| Recall | 69.10 | 67.94 |
| mAP50 | 75.24 | **74.40** |
| mAP50-95 | 44.29 | **42.68** |

The gap is small, which is itself reassuring — but the label was wrong.

### 2.3 FLOPs do not predict latency

The single most interesting efficiency result, and it is counter-intuitive
enough to carry a Discussion section:

| Model | GFLOPs | GPU latency | CPU latency |
|---|---|---|---|
| FireCNN | **1.50** | **0.72 ms** | 6.26 ms |
| MobileNetV3-S | **0.12** | **5.69 ms** | 4.90 ms |

MobileNetV3-Small does **12× fewer FLOPs** than FireCNN yet runs **8× slower on
GPU**. Depthwise-separable convolutions decompose work into many small,
memory-bound kernels; the GPU is latency-bound on kernel launches, not
compute-bound. FireCNN's four dense conv blocks saturate the hardware far better.

Consequences worth stating explicitly:
- **MobileNetV3 is faster on the CPU (4.90 ms) than on the GPU (5.69 ms).** An
  architecture marketed as "mobile-efficient" is efficient in *FLOPs*, which is
  the wrong currency on a desktop GPU.
- Any thesis claim of the form "model X is more efficient than model Y" must
  name the device and the metric. Parameter count and FLOPs — the only
  efficiency proxies this project used before now — would have ranked these two
  models in exactly the wrong order.

### 2.4 FP16 is slower than FP32 for three of five models

| Model | FP32 | FP16 | Change |
|---|---|---|---|
| FireCNN | 0.72 ms | 0.83 ms | **+15%** |
| MobileNetV3-S | 5.69 ms | 6.48 ms | **+14%** |
| YOLO26n | 16.62 ms | 19.51 ms | **+17%** |
| U-Net | 4.88 ms | **3.90 ms** | −20% |

Only the U-Net — the one genuinely compute-bound model, at 28.2 GFLOPs — benefits
from half precision. For launch-bound models the conversion overhead exceeds the
arithmetic saving. "Use FP16 for edge deployment" is not unconditionally true and
this project can now show why.

### 2.5 The INT8 quantization in this project is essentially a no-op

`torch.quantization.quantize_dynamic(model, {Linear, Conv2d})` was measured
converting:

| Model | Modules converted | Params before → after | Latency change |
|---|---|---|---|
| FireCNN | 2 (+ packed params) | 389,153 → 388,896 | 6.26 → 6.32 ms (none) |
| MobileNetV3-S | 4 (the two head `Linear`s) | 1,075,748 → 927,008 | 4.90 → 5.07 ms (none) |
| U-Net | **0** | 7,849,667 → 7,849,667 | (see caveat below) |

PyTorch dynamic quantization covers `Linear`/RNN layers — **not `Conv2d`**. The
convolutional trunks holding almost all parameters stay in FP32. This
definitively retires the **"~1.1 MiB (INT8)"** figure that was in `README.md`:
no run ever produced it, and the code path referenced could not have. A real
INT8 result requires static (calibrated) PTQ or QAT, which remains undone.

### 2.6 Multi-seed reruns: two published numbers do not survive

Every original figure was a single unrepeated run. With 5 seeds per method
(`results/tables/seed_variance.md`):

| Method | Metric | Originally published | Seeded mean ± std (n=5) |
|---|---|---|---|
| Iteration 1 | test accuracy | 93.80% | **92.80% ± 0.20%** |
| Iteration 2 | test accuracy | 89.46% | 88.86% ± 0.90% |
| Iteration 2 | test macro-F1 | 84.38% | 84.45% ± 1.40% |
| Iteration 3 | test accuracy | 90.25% | 89.89% ± 0.30% |
| Iteration 3 | test macro-F1 | 86.72% | **86.06% ± 0.46%** |
| Iteration 5 | test mIoU | 85.22% | 85.47% ± 1.19% |
| Iteration 5 | hazard-only mIoU | — | **80.05% ± 1.57%** |
| Iteration 4 | test mAP50 | 74.40% | 74.88% ± 0.23% (n=3) |
| Iteration 4 | test mAP50-95 | 42.68% | 42.99% ± 0.11% (n=3) |

Iterations 2, 3, 4 and 5 replicate. **Iteration 1's 93.80% does not** — it sits
~5 std above the seeded mean of 92.80% ± 0.20%, so the original run was an
outlier and 92.80% is the number to report.

### 2.7 Robust training buys stability, not clean accuracy

This is the claim the original audit warned was inside the noise band, and the
audit was right. Iteration 2 vs iteration 3 on the **clean** test split, 5 seeds
each, Welch's t-test:

| Metric | Iteration 2 | Iteration 3 | Difference |
|---|---|---|---|
| Test accuracy | 0.8886 ± 0.0090 | 0.8989 ± 0.0030 | +0.0104 (2.4σ, **p = 0.060**) |
| Test macro-F1 | 0.8445 ± 0.0140 | 0.8606 ± 0.0046 | +0.0161 (2.4σ, **p = 0.060**) |

Two conclusions, both important:

1. **The single-run comparison overstated the gap.** It reported +0.023 macro-F1;
   the seeded estimate is +0.016 at p ≈ 0.06. At α = 0.05 this is *not*
   significant. The thesis must not claim robust training improves clean-data
   accuracy.
2. **Iteration 3's variance is 3.0× smaller on both metrics** (0.0046 vs 0.0140
   on macro-F1). Robust augmentation buys reproducibility even where it barely
   moves the mean — visible only because the runs were repeated.

Combined with §2.8, the defensible claim is: **robust training's value is
stability and degradation resistance, not clean-data accuracy.** That is sharper
and more honest than `README.md`'s original "drastically higher real-world
generalization".

### 2.8 The robustness claim is true — and stronger than claimed

`README.md` asserted iteration 3 had "drastically higher real-world
generalization" with nothing to support it; its *validation* accuracy was
actually lower than iteration 2's (88.77 vs 89.16), which looked like evidence
against. Measured over 8 corruptions × 3 severities on the test split
(`results/robustness.csv`, `results/figures/robustness_curves.png`):

| Condition group | it. 2 mean acc | it. 3 mean acc | it. 2 mean drop | it. 3 mean drop |
|---|---|---|---|---|
| Clean | 0.8946 | 0.8955 | — | — |
| Corruptions **similar** to its training augmentation | 0.8510 | 0.8575 | 0.0436 | 0.0380 |
| Corruptions it **never trained on** | 0.8179 | **0.8574** | 0.0767 | **0.0380** |

Two things make this a good result rather than a trivial one:

1. **Iteration 3's advantage is largest on corruptions it never saw.** Per
   corruption, its edge is +0.05 on defocus blur, +0.04 on Gaussian noise, +0.03
   on JPEG artifacts and +0.03 on motion blur — none of which appear in its
   Albumentations pipeline. On the fog/brightness family it actually trained on,
   the advantage is smaller (+0.01 to +0.02). So this is genuine generalization,
   not memorisation of the specific augmentations.
2. **Its degradation is identical on seen and unseen corruptions (0.0380 both),
   while iteration 2's nearly doubles (0.0436 → 0.0767).** Robust training did
   not just add coverage; it flattened the model's sensitivity to distribution
   shift in general.

And it is free: iterations 2 and 3 are the same architecture with the same
parameter count, measured at 4.90 ms and 4.85 ms CPU respectively — identical
within measurement noise. **Robust augmentation halves accuracy degradation
under unseen corruption at zero inference cost.** That sentence is a thesis
contribution, and until now the project had no evidence for any version of it.

One honest caveat: on `brightness_down` iteration 3 is *worse* (drop 0.0334 vs
0.0231). Worth a sentence rather than a silent omission.

### 2.9 Failure analysis: the dominant error is smoke-vs-background

Ranked by per-image loss (`scripts/analyze_failures.py`, iteration 3, montages in
`results/failures/iteration3/`, per-image records in
`results/iteration3_failures.csv`):

| True | Predicted | Count | Mean confidence |
|---|---|---|---|
| Neither | Only_Smoke | 109 | 0.712 |
| Only_Smoke | Neither | 93 | 0.737 |
| Both | Only_Smoke | 62 | 0.728 |
| Both | Only_Fire | 60 | 0.744 |
| Only_Smoke | Both | 53 | 0.722 |
| Only_Fire | Both | 26 | 0.788 |

Two observations for the Discussion chapter:

1. **The single largest error mode is deciding whether faint smoke is present at
   all** — the symmetric pair Neither↔Only_Smoke accounts for 202 errors, far
   more than any fire-related confusion. The hard problem is not distinguishing
   fire from smoke; it is separating thin smoke from haze, cloud and empty sky.
2. **The model is confidently wrong.** Mean confidence on misclassifications is
   0.69–0.80, so errors are not clustered near the decision boundary. That is a
   calibration problem, and it means a confidence threshold cannot be used to
   suppress these failures — worth stating plainly, since a deployed alerting
   system would naturally reach for exactly that mechanism.

### 2.10 The harness caught a flaw in its own CPU measurements

The U-Net measured 107.8 ms at CPU FP32 and 76.6 ms at CPU "INT8" — a 29%
improvement from a step that converted **zero modules**, i.e. from running the
identical graph twice. That is measurement drift (warm allocator, page cache,
thermal state over a 33-minute sequential sweep), not precision.

Within-run spread was ~10% (p95/median); between-run drift ~30%. **So no CPU
difference below ~30% can be attributed to anything without repeated
measurement.** `scripts/run_benchmarks.py --repeat N` now exists for this, and
any CPU claim in the thesis must cite a repeated run. GPU measurements, being
synchronized, are far tighter and do not show this.

This is worth reporting in Methodology rather than hiding: it is evidence the
measurement protocol is being taken seriously.

---

## 2.11 Real edge hardware: Jetson Orin Nano Super

The edge chapter is no longer a projection. Measured on borrowed hardware:

| | |
|---|---|
| Board | NVIDIA Jetson Orin Nano **Super** Developer Kit |
| Software | JetPack 7.0 / L4T R39.2, Ubuntu 24.04, Python 3.12.3 |
| Compute | 6× Arm Cortex-A78AE, Ampere iGPU, 7 GB shared RAM |
| GPU tier | TensorRT 10.16.2 via `trtexec` |
| ARM CPU tier | ONNX Runtime 1.29.0 |

45 measurements: 5 models × 3 backends × 3 power modes
(`results/tables/jetson_power_modes.md`, `results/figures/jetson_power_modes.png`).

### Latency at 25W (the default mode)

| Model | ARM CPU | TRT FP32 | **TRT FP16** | CPU→FP16 |
|---|---|---|---|---|
| FireCNN | 15.13 ms | 0.92 ms | **0.40 ms** | 38× |
| MobileNetV3-S | 5.62 ms | 1.32 ms | **0.98 ms** | 5.7× |
| MobileNetV3-S robust | 5.62 ms | 1.37 ms | **0.98 ms** | 5.7× |
| YOLO26n | 95.57 ms | 9.25 ms | **6.02 ms** | 16× |
| U-Net | 290.52 ms | 11.70 ms | **4.35 ms** | **67×** |

**Every paradigm is real-time capable on this board.** The slowest configuration
across the entire matrix is YOLO26n at 15W (9.04 ms, 111 FPS). The original
`README.md` guessed the U-Net was "Heavy: requires Jetson Orin Nano / Xavier NX"
and the detector needed "hardware NPU accelerators" — both understated the
hardware substantially.

### FP16 is a win on Jetson and a loss on the desktop

| Model | RTX 3060 FP32→FP16 | Jetson FP32→FP16 |
|---|---|---|
| FireCNN | **+15% slower** | −57% faster |
| MobileNetV3-S | **+14% slower** | −26% faster |
| YOLO26n | **+17% slower** | −35% faster |
| U-Net | −20% faster | −63% faster |

On the RTX 3060 only the compute-bound U-Net benefited (§2.4); on Orin Nano every
model does. "Use FP16 on the edge" turns out to be true *on the edge* and false on
the desktop GPU tested here — the recommendation is hardware-specific, and this
project can now demonstrate that rather than assert it.

### Power modes are not a linear ladder

The GPU improves monotonically with power budget, but the **CPU is slower at 25W
than at 15W**:

| Mode | CPU max clock | MobileNetV3 CPU | YOLO26n GPU FP16 |
|---|---|---|---|
| 15W | 1497.6 MHz | 5.14 ms | 9.04 ms |
| 25W | **1344.0 MHz** | **5.62 ms** | 6.02 ms |
| MAXN_SUPER | 1728.0 MHz | 4.54 ms | 5.55 ms |

25W *reduces* the CPU clock relative to 15W in order to give the GPU more of the
budget. CPU latency tracks the clock almost exactly (15W→25W: clock −11.4%,
latency +9.3%). A deployment doing CPU-side pre/post-processing could therefore
be **slower** at a higher power setting — a genuinely counter-intuitive result
that only multi-mode measurement exposes.

**Diminishing returns above 25W:** 15W→25W buys 33% on YOLO; 25W→MAXN_SUPER buys
a further 8% for a much larger power budget. 25W is the efficiency sweet spot.

### Caveat: the Jetson-vs-desktop comparison confounds runtime with hardware

In `results/figures/pareto_accuracy_vs_latency.png` several Jetson GPU points sit
*left of* the RTX 3060 points — YOLO26n is 6.02 ms on Orin Nano versus 16.62 ms
on the desktop. **This does not mean the edge GPU is faster than the desktop
GPU.** It compares *TensorRT FP16 with a compiled, fused engine* against *eager
PyTorch FP32*, which are different software stacks. The honest statement is:

> An Orin Nano running TensorRT outperforms an RTX 3060 running eager PyTorch on
> this workload.

That is a legitimate and useful deployment comparison — it is what you would
actually ship in each case — but it is not a hardware-only claim. Isolating the
hardware would require running TensorRT on both, which has not been done.

### The FLOPs/latency inversion reproduces on ARM

FireCNN is **slower** than MobileNetV3-S on ARM CPU (15.13 vs 5.62 ms) despite
being 2.8× smaller in parameters, because it does 12× more FLOPs — yet it is the
**fastest** model on the GPU (0.40 ms). The same inversion appears on x86 (§2.3).
Confirming it on a second, architecturally different platform turns a
single-platform curiosity into a robust finding: neither parameter count nor
FLOPs predicts latency, and the ranking flips between CPU and GPU on both.

---

## 2.12 Energy per inference — the metric edge deployment actually budgets for

Measured with `tegrastats` power rails (`jetson/measure_energy.py`), 45 rows in
`results/jetson_energy.csv`. Idle board draw is 4.6 W; energy is reported both as
**total** (mean power under load × latency, what a battery sees) and **marginal**
(load minus idle × latency, the cost of the computation itself).

### Race to idle: the GPU is the power-saving choice

At 25W, GPU (TensorRT FP16) versus ARM CPU:

| Model | GPU energy | CPU energy | GPU advantage | GPU power | CPU power |
|---|---|---|---|---|---|
| FireCNN | **6.44 mJ** | 123.81 mJ | 19× | 16.05 W | 8.18 W |
| MobileNetV3-S | **8.11 mJ** | 46.00 mJ | 5.7× | 8.31 W | 8.18 W |
| MobileNetV3-S robust | **8.15 mJ** | 45.96 mJ | 5.6× | 8.30 W | 8.18 W |
| YOLO26n | **85.76 mJ** | 801.06 mJ | 9.3× | 14.24 W | 8.38 W |
| U-Net | **81.89 mJ** | 2404.43 mJ | **29×** | 18.81 W | 8.28 W |

The GPU draws up to **2.3× more power** yet uses **5.6–29× less energy per
frame**, because it finishes 30–60× sooner. For a battery or solar-powered
sensor node, running inference on the GPU is not a luxury — it is the
energy-efficient option. This is the clearest practical result in the project and
it is invisible to latency-only analysis.

### The fastest power mode is never the most efficient

GPU FP16, all three modes:

| Model | 15W | 25W | MAXN_SUPER | Cheapest |
|---|---|---|---|---|
| FireCNN | **6.19 mJ** | 6.44 | 6.70 | 15W |
| MobileNetV3-S | 9.06 | **8.11 mJ** | 8.17 | 25W |
| MobileNetV3-S robust | 9.17 | **8.15 mJ** | 8.36 | 25W |
| YOLO26n | **82.90 mJ** | 85.76 | 87.82 | 15W |
| U-Net | **75.25 mJ** | 81.90 | 84.81 | 15W |

**MAXN_SUPER is the cheapest mode for no model.** It is always the fastest and
almost always the most wasteful. On the U-Net, going 15W → MAXN_SUPER cuts
latency 39% (6.51 → 3.96 ms) but *raises* energy 13% (75.25 → 84.81 mJ), because
power climbs 85% (11.56 → 21.43 W). This is voltage–frequency scaling: power
grows superlinearly with clock, so the energy integral turns unfavourable even as
latency improves.

The practical rule: **choose the lowest power mode that still meets your frame
deadline.** Since every model is real-time even at 15W (worst case 9.03 ms,
111 FPS), 15W is the correct default for this workload, and the higher modes buy
headroom rather than efficiency.

### Accuracy per millijoule

| Method | Energy (cheapest mode) | Macro-F1 | mJ per F1 point |
|---|---|---|---|
| FireCNN | 6.19 mJ @ 15W | 0.9169 | **6.8** |
| MobileNetV3-S robust | 8.15 mJ @ 25W | 0.9510 | **8.6** |
| MobileNetV3-S | 8.11 mJ @ 25W | 0.9411 | 8.6 |
| YOLO26n | 82.90 mJ @ 15W | 0.9689 | 85.6 |
| U-Net | 75.25 mJ @ 15W | 0.6999 | 107.5 |

**YOLO26n costs 10.2× the energy of MobileNetV3-robust to buy 1.8 points of
macro-F1.** For a mains-powered installation that is free; for a battery or solar
node it is close to indefensible. This is the concrete deployment recommendation
the thesis was set up to produce, and it now rests on measurement rather than
intuition. Figures: `results/figures/accuracy_vs_energy.png`,
`results/figures/jetson_energy_tradeoff.png`.

---

## 3. Data integrity: the original blocker is closed

The audit's top blocker was that nobody knew how `data/val/` was created, with a
risk that model selection had contaminated the held-out set. **Resolved, and the
news is good.**

| Check | Result |
|---|---|
| Split sizes | train 14,122 / val 3,099 / test 4,306 |
| `train + val` | 17,221 = the official D-Fire train pool exactly |
| `test` | 4,306 = the official D-Fire test split exactly |
| Test ID ranges | Contiguous and **disjoint**: AoF 6723–8383, WEB 9443–11806, PublicDataset 1055–1335 |
| Filename overlap between any two splits | **0** |
| **MD5 content duplicates across splits** | **0** |
| MD5 duplicates within a split | **0** |

`data/test/` is the untouched official D-Fire test set. `val/` was carved from
the official *train* pool by a random ~82/18 shuffle (its IDs interleave with
train's). That shuffle risks near-duplicate *frames* of the same scene straddling
train and val, which affects model selection only — never the held-out result.
One honest paragraph in Methodology closes this; no re-split, no retraining.

Persisted to `results/dataset_stats.json`, `results/dataset_stats.csv`, and
`results/split_manifest.csv` (all 21,527 images with split and label, so any
result traces back to the exact data that produced it).

### Class distribution — the evidence the README asserted but never had

| Split | n | Neither | Only_Fire | Only_Smoke | Both |
|---|---|---|---|---|---|
| train | 14,122 | 45.7% | **5.5%** | 27.2% | 21.7% |
| val | 3,099 | 44.4% | **5.6%** | 27.3% | 22.7% |
| test | 4,306 | 46.6% | **5.1%** | 27.5% | 20.8% |
| **all** | **21,527** | 45.7% | **5.4%** | 27.3% | 21.6% |

`Only_Fire` is 5.4% of the data — 8.5× under-represented against `Neither` — and
is exactly the worst-F1 class (68.78% in iteration 2). The imbalance explanation
is now evidenced. Proportions are stable across splits, so the shuffle was
effectively stratified.

### Two data defects found

- **4 test images have out-of-bounds bbox coordinates** (`WEB10769`, `WEB10775`,
  `WEB11598`, `WEB11600`) and are silently dropped by Ultralytics — detection
  metrics are computed on 4,302 of 4,306 images. Must be stated.
- Several test JPEGs were **corrupt and were rewritten in place** by Ultralytics
  during evaluation ("corrupt JPEG restored and saved"). Harmless, but it means
  `data/` is not byte-identical to the original download.

---

## 4. What was built

All new code is documented in `CLAUDE.md`.

| Component | Purpose |
|---|---|
| `src/benchmark.py` | Uniform inference-cost protocol: in-memory inputs, 50 warmup / 200 timed, `cuda.synchronize()` around the timed region, median + p95, params/FLOPs/size/peak-memory. Replaces the unsynchronized timer and the disk-I/O-bound FPS loop. Portable to Jetson unmodified. |
| `src/results.py` | Append-only CSV + JSON result persistence, tagged with git SHA, dirty flag, seed and environment. The project previously wrote **no** result files at all. |
| `src/corruptions.py` | Eight deterministic corruptions × 3 severities, grouped by whether iteration 3 trained on an analogue. |
| `src/model.py` → `BackboneClassifier` | MobileNetV3-S / ResNet18 / EfficientNet-B0 behind one interface with an identical head, so the backbone is the only variable. |
| `scripts/run_benchmarks.py` | Sweeps model × device × precision × batch × backend → `results/benchmarks.csv`. |
| `scripts/evaluate_common.py` | **The common-task protocol.** Collapses every paradigm to image-level presence with threshold sweeps. |
| `scripts/evaluate_robustness.py` | Corruption-suite evaluation, inference only. |
| `scripts/dataset_stats.py` | Dataset statistics, split manifest, cross-split leakage and duplicate scan. |
| `scripts/run_comparison.py` + `configs/comparison.yaml` | Controlled backbone comparison at identical budget/resolution/seed. |
| `scripts/run_seeds.py` | Unattended multi-seed sweep across all methods. |
| `scripts/analyze_failures.py` | Ranks test images by loss; exports worst-case and confusion-pair montages. |
| `scripts/make_tables.py` | Regenerates every thesis table and figure from `results/*.csv`. |
| `scripts/smoke_test.py` | One-epoch validation of every pipeline, to de-risk long unattended sweeps. |

### Correctness fixes

- **Seeding, which did not exist anywhere** for iterations 1/2/3/5: `set_seed()`
  covers Python/NumPy/torch/CUDA, and every DataModule now takes `seed=` and
  wires `worker_init_fn` + `generator` into its DataLoaders. (Iteration 4 was
  already seeded via Ultralytics — the audit's "no seeding anywhere" was wrong
  on that one point.)
- **Hardcoded CUDA removed.** `resolve_device()` replaces module-level
  `torch.device("cuda")`; models take `device=`. CPU benchmarking was impossible
  before this.
- **Latent binary-metrics bug fixed.** `logits.argmax(dim=1)` on `(N,1)` binary
  logits always returns class 0. It was masked only because iteration 1 reported
  no extended metrics; adding any would have produced silent nonsense. Replaced
  with a task-aware `extract_predictions()` hook.
- **Iteration 1 now reports a real metric set** — precision/recall/specificity,
  false-alarm rate, miss rate, PR-AUC, ROC-AUC, and the F1-optimal threshold.
  Accuracy alone is the wrong headline when 74% of the split is non-fire.
- **Hazard-only mIoU** reported alongside the 3-class figure (background at
  96.18% inflates the 85.22% headline by ~11 points).
- **Per-class detection AP** extracted (this is what produced §2.1).
- **Dead `conf_threshold` config key** wired into `validate()`.
- `wandb.mode` honoured, so sweeps don't depend on a network service.
- Fixed the broken YOLO checkpoint path in `visualize_predictions.py`.

### Integrity cleanup

- **Deleted the fabricated per-device FPS/latency matrix** from
  `presentation/app.js` (Pi Zero / Pi 4 / Jetson Nano / Orin / desktop, plus
  invented quantization multipliers). No code in this repository produced those
  numbers. The simulator now reads `results/benchmarks.csv` and renders
  "not measured" for any combination that has not been measured.
- **Removed the "~1.1 MiB (INT8)" claim** and the speculative deployment-target
  column from `README.md`, replaced with measured sizes and an explanation of
  why the INT8 figure was unattainable (§2.5). README re-encoded UTF-8.
- **Corrected `explanation.md`**: it claimed the YOLO backbone was frozen for
  initial epochs (`freeze: null` — it never was) and that the optimizer was
  "SGD with Momentum (MuSGD)" (`optimizer: auto` — never pinned).
- **Corrected the U-Net checkpoint size**: 89.94 MiB on disk, not the ~30 MiB
  claimed (the difference is Adam optimizer state).

---

## 5. Environment (for Methodology)

| | |
|---|---|
| GPU | NVIDIA RTX 3060, 12.9 GB, SM 8.6 (Ampere) |
| CPU | AMD Ryzen, 6 cores (Zen 3) |
| OS | Windows 11 Pro 10.0.26200 |
| Python / torch | 3.11.4 / 2.5.1+cu121, CUDA 12.1 |
| torchvision / ultralytics / albumentations | 0.20.1+cu121 / 8.4.60 / 2.0.8 |

Measured single-seed training wall-clock: iteration 1 **10.3 min**, 2 **16.5 min**,
3 **24.2 min**, 4 **164.7 min**, 5 **72.5 min** — **4.8 h for one seed of
everything**. Repeats are cheap.

---

## 6. Remaining work

All compute-side work is **complete**. 35 training runs are recorded:
iterations 1, 2, 3, 5 at 5 seeds each, iteration 4 at 3 seeds, and the backbone
comparison at 4 architectures × 3 seeds.

| # | Item | Status |
|---|---|---|
| 1 | Robustness under corruption (iter 2 vs 3) | **done** (§2.8) |
| 2 | Backbone comparison, 4 trunks × 3 seeds | **done** |
| 3 | Multi-seed reruns, all methods | **done** (§2.6, §2.7) |
| 4 | Failure-case montages | **done** (§2.9) |
| 5 | Jetson Orin Nano measurement | **done** (§2.11) — 45 rows across 3 power modes |
| 6 | TensorRT FP16/FP32 on device | **done** (§2.11). INT8 calibration still open |
| 7 | Static PTQ / QAT for a real INT8 result | not started (~0.5 day) |

All measurement work is complete. INT8 calibration (item 7) remains optional.
Item 7 is optional but would convert §2.5 from "the quantization we tried does
nothing" into "here is what correctly-applied quantization actually buys".

### Reliability problems hit during the sweeps, and their fixes

Recorded because they cost real GPU time and the fixes are now part of the code:

| Failure | Cause | Fix |
|---|---|---|
| 5 × iteration 3 crashed after training | `wandb.log` called after `fit_two_phase` closed the run | guard on `wandb.run is not None`; logging is best-effort (`src/trainer/robust.py`) |
| 1 × iteration 4 killed mid-training | host-RAM OOM from running two jobs concurrently on a 16 GB machine | queue is strictly sequential and gates on ≥3 GB free (`scripts/recovery_queue.py`) |
| 2 × iteration 4 crashed after training | `bool(numpy_array)` on `ap_class_index` in the per-class AP code | explicit `None` check + `np.atleast_1d` (`src/detection/trainer.py`) |
| iteration 4 left no metrics row despite exit 0 | `run_iteration4.py` never called `record_run` | added (`scripts/run_iteration4.py`) |
| 3 × iteration 4 weights stranded | runs died after training, before recording | `scripts/recover_detection_metrics.py` reloads weights and re-records in ~1 min instead of retraining 2.7 h |

Persistent DataLoader workers were also restricted to the training loader only;
holding them on val and test tripled the resident process count and contributed
to the OOM.

**Item 5 is the only thing blocked on you.** Everything else runs unattended.
Send hostname/IP, SSH user, and whether JetPack is installed, and the benchmark
harness runs there unchanged — it already detects Tegra hardware and tags rows
`jetson-cuda` / `jetson-cpu`.

Note that the workstation results have already made the edge chapter concrete
even without a device: there are real ARM-relevant proxies (ONNX Runtime CPU
measured at 22.8 ms for YOLO26n versus 45.2 ms for PyTorch CPU — a 2× gain purely
from the runtime), and §2.3–2.5 are device-independent architectural findings.
The Jetson would add real ARM CPU, edge GPU, TensorRT INT8, and power-mode
operating points — turning four hardware tiers into a genuine comparison matrix.

---

## 7. Files produced

```
results/
  benchmarks.csv            80 rows: model x device x precision x batch x backend
  common_eval.csv           29 rows: the common-task comparison
  robustness.csv            (in progress)
  metrics.csv               long-format metrics, one row per (run, metric)
  dataset_stats.{json,csv}  split sizes, class balance, integrity checks
  split_manifest.csv        all 21,527 images with split and label
  runs/*.json               full per-run records with git SHA and environment
  figures/                  pareto_accuracy_vs_latency, latency_by_device,
                            throughput_vs_batch, threshold_sensitivity,
                            dataset_distribution  (PNG + PDF)
  tables/                   benchmark_matrix, model_static_cost, pareto_points,
                            common_eval_{binary,multiclass}, dataset_distribution
                            (Markdown + LaTeX)
```

Regenerate everything downstream of the CSVs with:

```bash
python scripts/make_tables.py
```
