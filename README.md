
# Deep Fire & Smoke Detection Pipeline for Edge Devices

An iterative, end-to-end computer vision repository documenting the evolutionary development of an edge-optimized fire and smoke hazard detection system. This project transitions systematically from coarse, scene-level binary classification to real-time spatial object detection and pixel-perfect semantic segmentation.

---

## 🗺️ Project Architecture & Iterative Progression

The system follows a strict design paradigm to address escalating task granularities under constrained deployment environments:

$$\text{Presence} \longrightarrow \text{Multi-Class} \longrightarrow \text{Robustness} \longrightarrow \text{Spatial Location} \longrightarrow \text{Pixel Precision}$$


```

┌────────────────┐     ┌────────────────┐     ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│  Iteration 1   │     │  Iteration 2   │     │  Iteration 3   │     │  Iteration 4   │     │  Iteration 5   │
├────────────────┤     ├────────────────┤     ├────────────────┤     ├────────────────┤     ├────────────────┤
│   FireCNN      │ ──> │ MobileNetV3    │ ──> │ Robust MobileNet│ ──> │ YOLO26n        │ ──> │ Light U-Net    │
│ (Binary Baseline)│   │ (4-Class TL)   │     │ (Augmented/PTQ)│     │ (Detector)     │     │ (Segmentation) │
└────────────────┘     └────────────────┘     └────────────────┘     └────────────────┘     └────────────────┘

```

### 📊 Iteration Results Matrix

| Iteration | Task Type | Architecture | Dataset | Test metric (mean ± std, 5 seeds) | Key finding |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | Binary classification | `FireCNN` (scratch) | D-Fire | **Acc 92.80% ± 0.20** | Confirmed pipeline validity; cannot isolate smoke-only hazards. |
| **2** | 4-class transfer learning | `MobileNetV3-Small` | D-Fire | **Acc 88.86% ± 0.90** <br> **Macro-F1 84.45% ± 1.40** | Isolates mixed cases. `Only_Fire` is the weakest class (5.4% of the data). |
| **3** | Robust 4-class training | `MobileNetV3-Small` + Albumentations | D-Fire | **Acc 89.89% ± 0.30** <br> **Macro-F1 86.06% ± 0.46** | Clean-data gain over iter. 2 is **not** significant (p ≈ 0.06); the real benefit is halved degradation under unseen corruption, at identical inference cost. |
| **4** | Object detection | `YOLO26n` (Ultralytics) | D-Fire | **mAP50 74.88% ± 0.23** <br> **mAP50-95 42.99% ± 0.11** (n=3) | Best method on the common task. Per class, **fire is harder than smoke** (mAP50 0.681 vs 0.817). |
| **5** | Semantic segmentation | Lightweight `U-Net` (Dice+Focal) | Roboflow COCO | **mIoU 85.47% ± 1.19** <br> **hazard-only mIoU 80.05% ± 1.57** | Pixel-level contours. Background inflates the 3-class mIoU by ~5 points. |

> **These supersede the single-run figures previously reported here.** Every value
> is a mean ± standard deviation over repeated seeded runs, regenerated from
> `results/metrics.csv` via `python scripts/make_tables.py`. Two earlier numbers
> did not survive repetition: iteration 1's 93.80% (a ~5σ outlier; the seeded mean
> is 92.80%) and iteration 4's 75.24 mAP50, which was a *validation* figure
> reported alongside test figures from the other iterations.
>
> For the cross-paradigm comparison on a single common task, plus measured
> latency and energy on an RTX 3060 and a Jetson Orin Nano, see
> **[THESIS_STATUS.md](THESIS_STATUS.md)**.

---

## ⚡ Hardware Constraints & Practical Edge Fit

Parameter counts and file sizes below are measured. **Deployment-target
columns have been removed**: they were inferences from parameter count, never
measurements, and nothing in this project had ever run on ARM or edge hardware.

Measured inference cost lives in `results/benchmarks.csv`, produced by
`scripts/run_benchmarks.py` under a single uniform protocol (batch 1, 50 warmup
+ 200 timed iterations, `torch.cuda.synchronize()` around the timed region,
in-memory inputs, median and p95 reported). Regenerate the tables and figures
with `python scripts/make_tables.py`.

| Stage | Model Name | Parameter Count | Checkpoint Size | Input Resolution |
| :---: | :--- | ---: | ---: | :---: |
| **1** | `FireCNN` | 389,153 | 4.48 MiB | $224 \times 224$ |
| **2** | `MobileNetV3-Small` | 1,075,748 | 10.33 MiB | $224 \times 224$ |
| **3** | `MobileNetV3-Small` (robust training) | 1,075,748 | 10.33 MiB | $224 \times 224$ |
| **4** | `YOLO26n` | 2,572,280 | 5.12 MiB (`.pt`), 9.35 MiB (ONNX) | $640 \times 640$ |
| **5** | Lightweight `U-Net` | 7,849,667 | 89.94 MiB | $256 \times 256$ |

> Checkpoint sizes are the actual files on disk, measured. The PyTorch
> checkpoints carry optimizer state as well as weights, which is why the U-Net
> file is 89.94 MiB rather than the ~30 MiB its 7.85 M FP32 parameters would
> occupy alone -- Adam keeps two moment tensors per parameter. An earlier
> version of this table listed ~30.0 MiB, which is the weights-only figure and
> not what exists on disk. For deployment only the weights need shipping, so
> both numbers are meaningful; they are simply not the same number.

### On the previous "~1.1 MiB (INT8)" claim

An earlier version of this table listed a quantized Iteration 3 at ~1.1 MiB.
**No run in this project ever produced that figure** and it has been removed.
The edge-simulation script that would have produced it
(`scripts/simulate_edge_iteration3.py`) has no completed run in `wandb/`.

It is also not achievable by the code path it referred to.
`torch.quantization.quantize_dynamic` was measured converting **4 modules** of
MobileNetV3-Small -- the two `Linear` layers of the classifier head plus their
packed parameters -- and **0 modules** of the U-Net. Dynamic quantization
targets `Linear`/RNN layers; the convolutional trunk holding almost all of the
parameters stays in FP32. A conv-heavy model therefore does not shrink 4x this
way, and the measured INT8 latency difference on CPU is within noise of FP32.
A real INT8 result needs static (calibrated) PTQ or QAT, which this project has
not yet done.

---

## 🛠️ Repository & Codebase Directory Structure

```text
.
├── configs/
│   └── iteration3.yaml               # Robustness training hyperparameters & edge sim parameters
├── data/
│   └── coco/                         # Roboflow COCO masks dataset (7,110 images for Iteration 5)
├── src/
│   ├── detection/
│   │   ├── data_config.py            # Generates Ultralytics specific data.yaml configs
│   │   └── trainer.py                # YOLO26 training script and ONNX/TensorRT export pipelines
│   ├── trainer/
│   │   └── robust.py                 # Cosine learning rate scheduling and Albumentations wrapper
│   ├── dataset.py                    # Iteration 1 binary data pipelines (Collapsing YOLO annotations)
│   ├── dataset_multiclass.py         # Iteration 2 multi-class image-level parsing
│   ├── dataset_segmentation.py       # Iteration 5 mask polygon rendering pipeline
│   ├── model.py                      # Scratch CNN and custom network architectures
│   └── utils.py                      # Shared evaluation matrices & metrics converters
└── README.md                         # Project documentation

```

---

## 🚀 Deep-Dive Iteration Overviews

### 🔹 Iteration 1: Binary Baseline

* **Objective:** Establish a foundational detection mechanism confirming end-to-end data throughput pipelines.
* **Architecture:** $3\text{--}4$ sequential blocks featuring `Conv2d` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d` capped with a memory-efficient Global Average Pooling (GAP) layer and a single dense output neuron utilizing `BCEWithLogitsLoss`.

### 🔹 Iteration 2: 4-Class Transfer Learning

* **Objective:** Expand the semantic resolution to distinct categories: `Neither`, `Only Fire`, `Only Smoke`, and `Both`.
* **Implementation:** Abstracted via an Object-Oriented modular `Trainer` class. Pretrained network parameters are frozen initially to establish classification head updates, followed by low-learning-rate backpropagation across top convolutional levels.

### 🔹 Iteration 3: Robustness and Simulation

* **Objective:** Mitigation of overfit conditions across ambiguous visual scenes (e.g., separating thick mountain fog from structural fire smoke).
* **Techniques:** Integrated `Albumentations` primitives (`RandomFog`, `ColorJitter`, `CLAHE`) alongside a `CosineAnnealingLR` scheduler. Includes Post-Training Quantization (PTQ) converters to systematically evaluate parameter degradation down to `INT8`.

### 🔹 Iteration 4: Object Detection

* **Objective:** Replace holistic scene classification with bounded structural bounding boxes to drive spatial localization.
* **Implementation:** Uses a `YOLO26` design framework to bypass non-maximum suppression latency blocks on hardware layers. Deployed scripts directly wrap compilation out into structured `.onnx` and `.engine` runtimes for live execution targets (targeting a standard 30 FPS boundary).

### 🔹 Iteration 5: Semantic Segmentation

* **Objective:** Pixel-perfect tracking to map complex, irregular, or moving smoke plumes while preventing background noise capture inside square bounding regions.
* **Implementation:** Employs an encoder-decoder `U-Net` pattern driven via Automatic Mixed Precision (`torch.cuda.amp`). Losses combine Dice metrics alongside structural Focal elements to penalize high background pixel imbalances ($96.18\%$ stable background accuracy).

---

## 📈 Engineering Standards & Verification

* **Experiment Monitoring:** Deep telemetry integrated via Weights & Biases (`WandB`) tracking real-time gradient behaviors, F1 macro metrics, Confusion Matrices, and precision-recall boundaries.
* **Loss Formulas:** Integrated mathematical stability via combined objective formulations:

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Dice}} + \alpha \cdot \mathcal{L}_{\text{Focal}}$$
