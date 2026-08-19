# Fire and Smoke Detection: Iteration Results Report

> ⚠️ **SUPERSEDED — do not cite these numbers.**
> This report records the original *single, unrepeated* runs. Several of its
> figures did not survive multi-seed repetition, and one (iteration 4's
> mAP50 75.24 / mAP50-95 44.29) is a **validation** figure presented as if it
> were a test result. Iteration 1's 93.80% is a ~5σ outlier against a seeded
> mean of 92.80% ± 0.20%.
>
> It is kept for provenance — it documents what was believed before the
> measurement infrastructure existed. **For current numbers use
> [THESIS_STATUS.md](THESIS_STATUS.md)** and the CSVs under `results/`.


This report aggregates the five completed training iterations in this workspace and explains how the project progressed from coarse fire presence detection to pixel-level segmentation.

The results below are taken from the saved Weights & Biases summaries under `wandb/`. Iteration 2 appears twice in the run history; the later completed run is used here because it is the most complete recorded experiment for that stage.

## Executive Summary

The project follows a clear refinement path:

1. Iteration 1 established a strong binary baseline for fire vs. normal scenes.
2. Iteration 2 expanded the task to four classes so smoke-only and mixed scenes were no longer collapsed into a binary decision.
3. Iteration 3 kept the same four-class task but improved generalization with stronger augmentation, a scheduler, and edge simulation.
4. Iteration 4 moved from classification to localization with YOLO26 object detection.
5. Iteration 5 moved one step further to semantic segmentation so the model could capture the actual shape of fire and smoke instead of approximate bounding boxes.

The main technical trend is that each iteration increases task granularity:

`presence -> class -> robustness -> location -> pixel precision`

## Results Overview

| Iteration | Task | Main Dataset | Key Result |
| --- | --- | --- | --- |
| 1 | Binary classification | D-Fire YOLO split | Test accuracy: 93.80%, test loss: 0.183 |
| 2 | 4-class transfer learning | D-Fire YOLO split | Best val accuracy: 89.16%, finetune val F1 macro: 85.25% |
| 3 | Robust 4-class training | D-Fire YOLO split | Test accuracy: 90.25%, test F1 macro: 86.72% |
| 4 | Object detection | D-Fire YOLO split | mAP50: 75.24%, mAP50-95: 44.29% |
| 5 | Semantic segmentation | COCO masks dataset | Test mIoU: 85.22%, test Dice: 91.81%, test pixel accuracy: 96.36% |

## Computation Cost And Edge Fit

The most important deployment difference between the five iterations is not just accuracy, but the cost of running them in real time on an edge device.

The model sizes below are based on the actual instantiated networks in this workspace:

| Iteration | Model | Parameters | Approx. FP32 Size | Input Size | Practical Edge Fit |
| --- | --- | --- | --- | --- | --- |
| 1 | FireCNN | 389,153 | ~1.5 MiB | 224 x 224 | Very light; suitable for a Raspberry Pi-class CPU or similar ARM edge board |
| 2 | MobileNetV3-Small classifier | 1,075,748 | ~4.1 MiB | 224 x 224 | Light; suitable for a modern SBC or small NPU/GPU edge board |
| 3 | MobileNetV3-Small classifier | 1,075,748 | ~4.1 MiB | 224 x 224 | Same inference cost as iteration 2; PTQ/INT8 makes it more edge-friendly |
| 4 | YOLO26n detector | 2,572,280 | ~9.8 MiB | 640 x 640 | Moderate to heavy; needs a CUDA GPU or strong NPU for true real-time use |
| 5 | Lightweight U-Net | 7,849,667 | ~30.0 MiB | 256 x 256 | Heaviest in memory and activation cost; best on an entry GPU / Jetson-class device |

### What this means in practice

Iteration 1 is the easiest model to deploy. A small ARM CPU can usually keep up because the network is shallow, the input is only 224 x 224, and the output is a single binary decision. For real-world use, this is the most forgiving option when power budget matters more than fine-grained scene understanding.

Iteration 2 stays in the same low-cost category at inference time. MobileNetV3-Small is still compact, but it is more capable than the scratch CNN. On a modern edge board, this can run comfortably in real time, and on a CPU-only device it remains plausible if the pipeline is optimized. Iteration 3 has the same runtime footprint as iteration 2; the robustness improvements are mostly training-side changes, not inference-side changes. The edge simulation path in the codebase also shows that quantized INT8 CPU inference is the intended deployment direction for this stage.

Iteration 4 is the first stage that becomes clearly compute-bound. YOLO26n is still the smallest detector in the family, but detection at 640 x 640 is much more expensive than classification at 224 x 224. The repository also explicitly targets ONNX/TensorRT export and a 30 FPS edge goal for this stage, which is a strong hint that real-time deployment should use a CUDA-capable board or a similar accelerator. In practice, this means a Jetson Orin Nano / Xavier NX class device, or a desktop NVIDIA GPU. A plain CPU is not a realistic choice for dependable real-time performance here.

Iteration 5 is the most demanding model overall in terms of memory footprint and dense output cost. Even though the input is only 256 x 256, segmentation computes a full class prediction for every pixel, so the decoder has to process and upsample dense feature maps throughout the network. That makes it heavier than the classifiers and more memory-intensive than it first appears from the image size alone. For real-time use, a Jetson-class device or small GPU is the practical minimum; a CPU-only device may work for low frame rates, but it is not the right choice for smooth live monitoring.

One useful way to summarize the deployment story is:

- Iteration 1: CPU-friendly edge inference.
- Iteration 2: Still edge-friendly, with better semantics.
- Iteration 3: Same runtime cost as iteration 2, but more robust and more quantization-ready.
- Iteration 4: Requires accelerator-backed real-time inference.
- Iteration 5: Requires the strongest edge hardware in the project because of dense pixel-wise prediction.

These are practical deployment classes rather than exact FPS guarantees. The actual frame rate depends on batch size, preprocessing, export format, quantization, and the specific accelerator, but the ranking above is stable across common edge devices.

## Iteration 1: Binary Classification Baseline

**Goal.** Establish a simple fire detector that answers a single question: is there fire in the scene or not?

**What was used.** The binary pipeline in [src/dataset.py](src/dataset.py) reads the standard D-Fire image/label folders and collapses the YOLO annotations into a fire-vs-normal label. The model in the first stage is a scratch-built CNN.

**Result.** The baseline was already strong:

- Validation accuracy: 93.68%
- Test accuracy: 93.80%
- Test loss: 0.183

**Interpretation.** This is a good starting point because it confirms the data pipeline, training loop, and evaluation path are all working. The model can reliably detect the presence of fire, but it cannot distinguish smoke-only scenes, combined fire-and-smoke scenes, or the difference between normal and ambiguous situations.

**Why the next iteration was needed.** Binary classification was too coarse for the actual problem. A fire system needs to treat smoke-only scenes differently from fire-only scenes, and it needs to recognize the mixed case where both appear together. That is why the project moved to a four-class formulation in iteration 2.

## Iteration 2: 4-Class Transfer Learning

**Goal.** Replace the binary decision with a richer scene-level classification problem:

- Neither
- Only Fire
- Only Smoke
- Both

**What was used.** The multiclass pipeline in [src/dataset_multiclass.py](src/dataset_multiclass.py) still uses D-Fire YOLO images and labels, but it derives four image-level classes from those annotations. The model switches to a pretrained MobileNetV3-Small backbone, which is much more suitable for edge deployment than a large custom network. Class weights are enabled to help with imbalance.

**Result.** The later completed run for this iteration recorded:

- Best validation accuracy: 89.16%
- Best validation loss: 0.359
- Finetune validation F1 macro: 85.25%
- Test loss: 0.348
- Test F1 for Neither: 94.83%
- Test F1 for Only Fire: 68.78%

**Interpretation.** The model is much better at the easy background-heavy class than at the more subtle classes, especially Only Fire. That gap matters: it shows that the real challenge is not just recognizing that a scene is safe or unsafe, but separating visually similar fire and smoke cases. The pretrained backbone helps, but the class-wise imbalance still limits the minority classes.

**Why the next iteration was needed.** The multiclass classifier still learned from relatively clean training images and standard augmentation. In practice, smoke and fire appear under variable lighting, haze, fog, and motion blur. The project therefore moved to a robustness-focused iteration that adds stronger augmentation, scheduling, and edge simulation.

## Iteration 3: Robustness and Edge Simulation

**Goal.** Improve generalization and make the training setup more realistic for deployment on constrained hardware.

**What was changed.** Iteration 3 keeps the same four-class MobileNetV3-Small architecture, but the training pipeline switches to Albumentations augmentation, adds a cosine learning-rate scheduler, and includes post-training edge simulation. That is reflected in [configs/iteration3.yaml](configs/iteration3.yaml) and the corresponding trainer path in [src/trainer/robust.py](src/trainer/robust.py).

**Result.** The recorded metrics were:

- Best validation accuracy: 88.77%
- Best validation loss: 0.357
- Finetune validation F1 macro: 83.20%
- Test accuracy: 90.25%
- Test F1 macro: 86.72%
- Test loss: 0.318

**Interpretation.** The headline validation accuracy is slightly below iteration 2, but the test-side behavior is healthier and the experiment is more realistic for deployment. This is the expected trade-off when the goal is robustness rather than optimizing one validation metric. The model is being exposed to harder conditions, so the performance profile is more credible for field use.

**Why the next iteration was needed.** Even a robust classifier only answers "what is in the frame?" For alerting and response, the system needs to know where the hazard is. That is why the project moved from scene-level classification to object detection in iteration 4.

## Iteration 4: Fire and Smoke Object Detection

**Goal.** Locate fire and smoke in the image rather than only classifying the image globally.

**What was used.** Iteration 4 builds an Ultralytics YOLO26 detection workflow using the D-Fire YOLO split. The detection data config in [src/detection/data_config.py](src/detection/data_config.py) writes the Ultralytics `data.yaml` file, and the training pipeline in [src/detection/trainer.py](src/detection/trainer.py) is set up for export to ONNX and TensorRT.

**Result.** The saved summary recorded:

- Precision: 75.90%
- Recall: 69.10%
- mAP50: 75.24%
- mAP50-95: 44.29%

**Interpretation.** This is the first iteration that gives the system spatial awareness. The model can now point to hazard regions instead of just saying that a hazard exists. The precision is better than the recall, which means the detector is reasonably conservative but still misses some targets. That is common in the early detection stage when the model is tuned for a balance between false alarms and missed detections.

**Why the next iteration was needed.** Bounding boxes are still an approximation. Smoke plumes are irregular, diffuse, and often much smaller or more spread out than a box suggests. For fire monitoring, a square box can include a lot of irrelevant background. The next step was therefore semantic segmentation.

## Iteration 5: Semantic Segmentation

**Goal.** Classify each pixel instead of drawing a box around the object.

**Why the dataset changed.** This is the biggest dataset shift in the project. Iterations 1 through 4 all use the D-Fire YOLO split, which is built around image-level labels or bounding boxes. Iteration 5 switches to the Roboflow Fire and Smoke Segmentation dataset under [data/coco](data/coco), which contains COCO segmentation annotations and polygon masks.

This change was necessary because the task changed from object detection to semantic segmentation:

- Bounding boxes are good for localization.
- Masks are better for irregular shapes like smoke.
- Pixel labels let the model learn the true contour and spread of the hazard.

The segmentation dataset is also larger and explicitly mask-based: the exported Roboflow dataset contains 7,110 images and COCO segmentation annotations. The custom dataset loader in [src/dataset_segmentation.py](src/dataset_segmentation.py) renders those polygons into pixel masks before training.

**What was used.** Iteration 5 trains a lightweight custom U-Net with mixed precision, and the loss combines Dice and Focal terms so the model can handle the strong background-vs-hazard imbalance.

**Result.** The saved summary recorded:

- Validation accuracy: 95.02%
- Validation mIoU: 81.45%
- Test mIoU: 85.22%
- Test Dice: 91.81%
- Test pixel accuracy: 96.36%
- Test IoU for background: 96.18%
- Test IoU for fire: 83.23%
- Test IoU for smoke: 76.26%

**Interpretation.** This is the most detailed output in the project. Background is easiest, fire is strong, and smoke remains the hardest class, which is exactly what you would expect because smoke is diffuse and occupies fewer pixels. The gap between background and smoke shows why Dice and Focal loss were a good choice: the model needs to pay more attention to the minority hazard pixels rather than learning to predict background everywhere.

## Overall Conclusions

The five iterations show a consistent architectural progression:

- Iteration 1 proved the data and training loop on a binary task.
- Iteration 2 made the label space more realistic by separating smoke, fire, mixed, and normal scenes.
- Iteration 3 improved robustness and deployment realism.
- Iteration 4 added spatial localization.
- Iteration 5 replaced approximate boxes with pixel-accurate masks for the most faithful representation of fire and smoke shape.

The final segmentation stage is the most informative for smoke monitoring, while the detection stage is the most practical for lightweight real-time alerts. Together, they show a clear trade-off between speed, localization, and precision.
