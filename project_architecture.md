# Project Architecture: Iterative Smoke & Fire Detection for Edge Devices

## 1. Project Overview & Methodology
This project documents the iterative development of a computer vision pipeline designed to detect smoke and fire in varying environmental conditions. The ultimate deployment target is an edge computing device. To demonstrate architectural comprehension, the system evolves from a rudimentary binary classifier to an advanced semantic segmentation model.

**Tech Stack & Engineering Standards:**
* **Core Framework:** PyTorch (Leveraging `torch.utils.data`, `torch.nn`, and `torch.optim`).
* **Experiment Tracking:** Weights & Biases (WandB) for logging hyperparameters, loss curves, and artifact versioning.
* **Coding Paradigm:** Object-Oriented Programming (OOP). Codebase will be modularized into discrete Python scripts (`dataset.py`, `model.py`, `train.py`, `utils.py`) to reflect industry best practices.
* **Performance Focus:** Optimization for edge constraints (balancing parameter count, FLOPs, and inference latency).

---

## 2. Iteration 1: The Baseline (Binary Image Classification)
* **Objective:** Establish a baseline model to detect the presence of fire vs. non-fire scenes, proving the custom data pipeline and foundational network mechanics.
* **Dataset:** D-Fire (Filtered into binary classes: "Fire" vs. "Normal").
* **Architecture:** Scratch-built PyTorch Convolutional Neural Network (CNN).
    * 3-4 Convolutional blocks (Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d).
    * Global Average Pooling (to reduce parameter overhead) followed by a Dense output layer.
* **Engineering Best Practices:**
    * Implement a custom OOP `torch.utils.data.Dataset` class to handle disk I/O and preprocessing efficiently.
    * Use PyTorch's `BCEWithLogitsLoss` for numerical stability.
* **Metrics Tracked in WandB:** Binary Accuracy, Training/Validation Loss, Epoch Time.

---

## 3. Iteration 2: Feature Extraction & Transfer Learning
* **Objective:** Expand the classification scope to multi-class (handling the amorphous, semi-transparent nature of smoke) without suffering from exploding parameter counts.
* **Dataset:** D-Fire (4 Classes: Only Fire, Only Smoke, Both, Neither).
* **Architecture:** Transfer Learning utilizing an edge-friendly backbone (e.g., `MobileNetV3-Small` or `EfficientNet-B0` from `torchvision.models`).
* **Engineering Best Practices:**
    * **Modular Trainer Class:** Abstract the training loop into a reusable `Trainer` class that accepts any PyTorch model, optimizer, and dataloader.
    * Freeze the backbone initially, train the custom classification head, then perform fine-tuning by unfreezing the top convolutional blocks with a lower learning rate.
* **Metrics Tracked in WandB:** Categorical CrossEntropy, Class-wise F1-Score (crucial due to potential class imbalances), and dynamically generated Confusion Matrices.

---

## 4. Iteration 3: Robustness & Edge Optimization
* **Objective:** Address overfitting, improve generalization on tricky environmental conditions (fog vs. smoke), and simulate edge deployment constraints.
* **Dataset:** D-Fire (4 Classes).
* **Architecture:** Optimized Iteration 2 Model.
* **Engineering Best Practices:**
    * **Advanced Augmentation Pipeline:** Replace basic transforms with `Albumentations` to apply vectorized, complex augmentations (RandomFog, ColorJitter, CLAHE) to simulate varying lighting and weather.
    * **Learning Rate Scheduling:** Implement `CosineAnnealingLR` or `ReduceLROnPlateau`.
    * **Edge Simulation:** Apply PyTorch Post-Training Quantization (PTQ) converting weights from FP32 to INT8 to measure the trade-off between accuracy and model footprint.
* **Metrics Tracked in WandB:** Learning rate decay curves, Quantized Model Size (MB), Inference Time (ms).

---

## 5. Iteration 4: Real-Time Localization (Object Detection)
* **Objective:** Transition from image-level classification to spatial regression to locate the exact coordinates of the hazard for real-time edge alerting.
* **Dataset:** D-Fire (Utilizing the provided YOLO-format bounding box labels).
* **Architecture:** **YOLO26 (Ultralytics)** - Chosen specifically for its end-to-end NMS-free (Non-Maximum Suppression) design, which eliminates a major latency bottleneck on resource-constrained edge hardware.
* **Engineering Best Practices:**
    * Map the Ultralytics training pipeline to log directly into the established WandB project.
    * Export the final trained YOLO26 `.pt` weights to **ONNX** or **TensorRT** formats to prove edge-deployment readiness.
* **Metrics Tracked in WandB:** mAP@50 (Mean Average Precision), mAP@50-95, Box Loss, Target Edge FPS (Frames Per Second).

---

## 6. Iteration 5: Pixel-Level Semantic Segmentation
* **Objective:** Achieve maximum environmental understanding by classifying hazards at the pixel level. This solves the bounding-box limitation where a square box captures too much empty space around non-linear smoke plumes.
* **Dataset:** Fire and Smoke Segmentation Dataset (Roboflow - Polygonal/Pixel Masks).
* **Architecture:** Custom PyTorch implementation of **U-Net** (or a MobileNet-backed DeepLabV3 for faster edge execution).
* **Engineering Best Practices:**
    * Overhaul the custom `Dataset` class to load and augment image-mask pairs synchronously.
    * **Mixed Precision Training:** Utilize `torch.cuda.amp` (Automatic Mixed Precision) to speed up the heavy encoder-decoder training process and reduce VRAM usage.
    * **Advanced Loss Functions:** Implement a combined **Dice Loss + Focal Loss** to handle extreme pixel imbalance (since smoke typically occupies a minority of total image pixels).
* **Metrics Tracked in WandB:** Mean IoU (Intersection over Union), Dice Coefficient, Pixel-wise Accuracy.