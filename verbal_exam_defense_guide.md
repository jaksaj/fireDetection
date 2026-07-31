# Fire & Smoke Detection Project Defense Guide

This document is a verbal-exam companion for the fire and smoke detection project. It explains the full project story, the technical terms used across the five iterations, and the reasoning behind the design choices. The goal is not just to memorize results, but to understand why each iteration exists and how the project progressed from a simple baseline to a pixel-level segmentation system.

## 1. Big Picture

The project is an iterative computer vision pipeline for detecting fire and smoke in images, with the final goal of deploying usable models on edge hardware. The main theme is progressive task refinement:

presence -> class -> robustness -> location -> pixel precision

In plain words:

- **Presence** means answering whether fire is present or not.
- **Class** means distinguishing different scene types, such as fire-only or smoke-only.
- **Robustness** means making the model more reliable under fog, blur, and lighting changes.
- **Location** means finding where the hazard is in the image.
- **Pixel precision** means identifying the exact shape of fire and smoke.

This progression matters because each step makes the system more useful for real fire monitoring, but also more computationally expensive.

## 2. Project Goals

The project was built around four practical goals:

1. Detect fire and smoke accurately.
2. Improve the level of detail from image-level classification to dense pixel-level segmentation.
3. Keep the models realistic for edge devices such as Raspberry Pi, Jetson, or a desktop GPU.
4. Present the work clearly using a standalone interactive presentation.

If asked why the project is iterative, the answer is simple: each model solves a more realistic version of the problem than the previous one.

## 3. Core Concepts You Must Know

### 3.1 Computer Vision

Computer vision is the field of AI that lets machines interpret images and videos. In this project, the model sees fire and smoke images and learns patterns that distinguish hazardous scenes from safe ones.

### 3.2 Deep Learning

Deep learning uses neural networks with many layers to learn patterns from data. The project uses PyTorch neural networks to learn directly from images instead of hand-writing rules.

### 3.3 Neural Network

A neural network is a function made of layers. Each layer transforms the input into a more useful representation. Early layers often detect edges and textures, while later layers detect more complex shapes like flames, smoke plumes, or object boundaries.

### 3.4 Edge Device

An edge device is a small or local computing device used near the data source, rather than sending data to a cloud server. Examples in this project include Raspberry Pi boards, Jetson boards, and desktop GPUs. Edge devices matter because a fire alert system should work quickly, locally, and often with limited power.

### 3.5 Inference

Inference is the process of using a trained model to make a prediction on new data. Training learns the model; inference uses it.

### 3.6 Generalization

Generalization means how well a model performs on unseen data. A model that overfits the training set but fails on new scenes has poor generalization.

### 3.7 Overfitting

Overfitting happens when a model memorizes training data too closely and fails to handle new cases. This project addresses overfitting with augmentation, scheduling, transfer learning, and robustness training.

### 3.8 Transfer Learning

Transfer learning means starting from a pretrained model instead of training everything from scratch. The project uses MobileNetV3-Small pretrained on ImageNet, then fine-tunes it for fire and smoke tasks.

### 3.9 Fine-Tuning

Fine-tuning means unfreezing some pretrained layers and continuing training with a smaller learning rate. It lets the model adapt to the fire/smoke domain without forgetting the general visual features it already knows.

### 3.10 Quantization

Quantization reduces the numerical precision of model weights and activations, usually from FP32 to INT8. This lowers model size and can improve inference speed on supported hardware.

### 3.11 Post-Training Quantization (PTQ)

PTQ is quantization applied after training is finished. It is useful for testing whether the model can be made smaller and faster without retraining from scratch.

### 3.12 Mixed Precision

Mixed precision uses a combination of FP16 and FP32 during training. It saves memory and can speed up training on GPUs. In this project, it is used in the segmentation iteration.

### 3.13 Metrics

Metrics are numbers that summarize model performance. Different tasks need different metrics:

- Classification uses accuracy and F1-score.
- Detection uses precision, recall, and mAP.
- Segmentation uses IoU, Dice, and pixel accuracy.

## 4. Datasets and Labels

### 4.1 D-Fire Dataset

The D-Fire dataset is used in iterations 1 through 4. It contains fire and smoke scenes, normal scenes, and YOLO-style annotations depending on the task setup.

### 4.2 YOLO Split

The YOLO split refers to the dataset organization used for detection-style labels. Bounding boxes are stored in the YOLO format, which is commonly used for object detection.

### 4.3 Binary Labels

Binary labels mean two classes only, typically fire vs normal. Iteration 1 collapses everything into a yes/no fire decision.

### 4.4 Four-Class Labels

The four-class setup distinguishes:

- Neither
- Only Fire
- Only Smoke
- Both

This is more realistic because smoke-only scenes matter for early warning.

### 4.5 COCO Segmentation Dataset

Iteration 5 switches to a segmentation dataset with COCO-style polygon masks. COCO is a common dataset format for instance and semantic segmentation annotations.

### 4.6 Polygon Masks

Polygon masks are outlines drawn around objects. They are converted into pixel masks so the model can learn the exact shape of fire and smoke.

### 4.7 Pixel Masks

A pixel mask is an image where each pixel stores the class label. For segmentation, this is the ground truth the model tries to predict.

### 4.8 Dataset Shift

Dataset shift means changing from one annotation style or task structure to another. Here the project moves from image-level and box-level labels to pixel-level masks. That is a major conceptual jump and a good oral-exam topic.

## 5. Iteration 1: Binary CNN Baseline

### 5.1 Purpose of Iteration 1

Iteration 1 proves the basic pipeline works. It asks a simple question: is there fire in the image or not?

This stage is important because it validates data loading, preprocessing, model construction, training, and evaluation.

### 5.2 CNN

CNN stands for Convolutional Neural Network. It is the standard architecture for image tasks because it learns local visual patterns such as edges, textures, and shapes.

### 5.3 Convolution

Convolution is a sliding-window operation that extracts features from images. In a CNN, convolution layers learn filters that detect useful patterns.

### 5.4 Conv2d

Conv2d is the 2D convolution layer used for images. It learns kernels that move across the image and generate feature maps.

### 5.5 BatchNorm2d

Batch normalization stabilizes training by normalizing activations within a batch. It often speeds up convergence and helps gradients behave better.

### 5.6 ReLU

ReLU, or Rectified Linear Unit, is an activation function defined as max(0, x). It introduces non-linearity so the network can learn more than simple linear rules.

### 5.7 MaxPool2d

Max pooling reduces the spatial size of feature maps by keeping the strongest response in each region. This helps compress information and reduces computation.

### 5.8 Global Average Pooling

Global Average Pooling averages each feature map into a single value. It replaces large dense layers and keeps the model lightweight.

### 5.9 BCEWithLogitsLoss

BCEWithLogitsLoss is binary cross entropy combined with a sigmoid in a numerically stable form. It is used for binary classification.

### 5.10 Logits

Logits are raw model outputs before sigmoid or softmax is applied. Using logits is often more stable than using probabilities too early.

### 5.11 Why Iteration 1 Matters

This stage proves the project can detect fire at a basic level. It is the foundation for all later work, but it is not enough for real monitoring because it cannot distinguish smoke from fire or explain where the hazard is.

### 5.12 Oral Defense Angle

If asked why a scratch CNN was used first, say that it establishes a clean baseline with low complexity and verifies that the data pipeline and training loop are correct before introducing transfer learning or heavier architectures.

## 6. Iteration 2: 4-Class Transfer Learning

### 6.1 Purpose of Iteration 2

Iteration 2 expands the task from binary detection to multi-class scene understanding. The model now distinguishes fire-only, smoke-only, both, and neither.

### 6.2 Multi-Class Classification

Multi-class classification means selecting one class from more than two options. It is more realistic than binary classification because fire scenes vary.

### 6.3 ImageNet

ImageNet is a large general-purpose image dataset used to pretrain many vision models. A model pretrained on ImageNet already knows useful generic visual patterns.

### 6.4 MobileNetV3-Small

MobileNetV3-Small is a lightweight neural network designed for mobile and edge devices. It balances speed and accuracy better than large backbones.

### 6.5 Backbone

The backbone is the feature extractor part of a model. In transfer learning, the pretrained backbone learns generic visual features, and the custom head converts those features into the project-specific prediction.

### 6.6 Classification Head

The classification head is the final set of layers that map features into class probabilities. In this project, the head is custom because the pretrained backbone needs a new output layer for the four classes.

### 6.7 Freezing Layers

Freezing means keeping some pretrained layers fixed so their weights do not change during early training. This preserves generic features and reduces overfitting.

### 6.8 Unfreezing Layers

Unfreezing means allowing some layers to keep learning. The project unfreezes top blocks during fine-tuning to adapt the backbone to fire and smoke images.

### 6.9 Differential Learning Rate

A differential learning rate means using one learning rate for the pretrained backbone and a different one for the new classifier head. Usually the backbone uses a smaller rate to avoid destroying pretrained knowledge.

### 6.10 Cross-Entropy Loss

Cross-entropy loss measures how far predicted class probabilities are from the true class labels. It is standard for multi-class classification.

### 6.11 Weighted Loss

Weighted loss gives more importance to underrepresented classes. This helps when some classes, such as Only Fire, appear less frequently than the background-heavy class Neither.

### 6.12 F1-Score

F1-score combines precision and recall. It is especially important when classes are imbalanced because accuracy alone can be misleading.

### 6.13 Confusion Matrix

A confusion matrix shows how often each true class is predicted as each other class. It is useful for diagnosing mistakes, especially between visually similar classes like fire and smoke.

### 6.14 Class Imbalance

Class imbalance means some classes have many more examples than others. In this project, the model performs better on the easy majority class than on subtle minority classes.

### 6.15 Oral Defense Angle

If asked why iteration 2 was needed, answer that binary detection is too coarse for real fire monitoring. Smoke-only scenes are important for early alerting, so the task had to be expanded into four classes.

## 7. Iteration 3: Robustness and Edge Simulation

### 7.1 Purpose of Iteration 3

Iteration 3 keeps the same four-class task but improves reliability and deployment realism. It is about generalization, not just training accuracy.

### 7.2 Robustness

Robustness means the model still works under harder conditions such as fog, haze, blur, and lighting changes.

### 7.3 Albumentations

Albumentations is a Python library for image augmentation. It applies fast, diverse augmentations such as RandomFog and ColorJitter.

### 7.4 Data Augmentation

Data augmentation creates modified versions of training images to make the model more tolerant of real-world variation. It helps reduce overfitting and improves generalization.

### 7.5 RandomFog

RandomFog simulates foggy or hazy conditions. This is useful because fog and smoke can look similar.

### 7.6 ColorJitter

ColorJitter randomly changes brightness, contrast, saturation, and hue. It makes the model less sensitive to lighting differences.

### 7.7 CLAHE

CLAHE stands for Contrast Limited Adaptive Histogram Equalization. It improves local contrast and is useful in scenes with uneven lighting.

### 7.8 Learning Rate Scheduler

A scheduler changes the learning rate during training. This can help the model converge more smoothly.

### 7.9 CosineAnnealingLR

CosineAnnealingLR gradually lowers the learning rate following a cosine curve. It helps training settle into a good minimum instead of stopping too abruptly.

### 7.10 Edge Simulation

Edge simulation means testing how the model behaves under hardware constraints similar to deployment conditions. It helps you estimate whether the model is practical on small devices.

### 7.11 INT8

INT8 is an 8-bit integer format. It is smaller and faster than FP32, but may slightly reduce accuracy.

### 7.12 FP32

FP32 is 32-bit floating-point precision. It is the standard training and inference format for many models, but it is larger and slower than INT8.

### 7.13 Model Footprint

Model footprint refers to how much memory a model occupies. It includes weights, activations, and sometimes runtime overhead.

### 7.14 Inference Time

Inference time is how long the model takes to make one prediction. Lower inference time is better for real-time use.

### 7.15 Oral Defense Angle

If asked why iteration 3 exists even though accuracy changes are modest, explain that the goal was not only accuracy. The goal was robustness, better deployment realism, and quantization readiness.

## 8. Iteration 4: Object Detection

### 8.1 Purpose of Iteration 4

Iteration 4 adds spatial localization. Instead of only saying that a fire exists, the model predicts where it is in the image.

### 8.2 Object Detection

Object detection identifies objects and draws bounding boxes around them. It is useful when the system needs location information.

### 8.3 Bounding Box

A bounding box is a rectangle drawn around an object. It provides approximate location but not exact shape.

### 8.4 YOLO

YOLO stands for You Only Look Once. It is a family of real-time object detection models designed for fast inference.

### 8.5 YOLO26n

YOLO26n is the lightweight detector used in this project. The n usually indicates a nano-size or very small model variant designed for speed.

### 8.6 Ultralytics

Ultralytics is the framework used to train and export YOLO models in this project.

### 8.7 Backbone, Neck, and Head

Many detection models split into three parts:

- **Backbone** extracts visual features.
- **Neck** combines features from different scales.
- **Head** predicts boxes and classes.

### 8.8 CSPDarknet

CSPDarknet is a feature-extraction backbone used in some detection systems. It captures multi-scale features efficiently.

### 8.9 FPN

FPN stands for Feature Pyramid Network. It fuses features at different resolutions so small and large objects can both be detected.

### 8.10 PAN / PANet

PAN, or Path Aggregation Network, is a feature fusion design that improves information flow between high-level and low-level features.

### 8.11 Anchor-Free Detection

Anchor-free detection predicts object locations directly rather than comparing them to pre-defined anchor templates. This can simplify training and inference.

### 8.12 NMS

NMS stands for Non-Maximum Suppression. It removes duplicate overlapping boxes by keeping the best ones.

### 8.13 NMS-Free

NMS-free means the detector is designed to avoid the extra post-processing step of NMS. That can reduce latency.

### 8.14 mAP

mAP means mean Average Precision. It is a standard object detection metric that summarizes precision over recall and across classes.

### 8.15 mAP50

mAP50 measures average precision at IoU threshold 0.50. It is easier to satisfy than stricter thresholds.

### 8.16 mAP50-95

mAP50-95 averages mAP across thresholds from 0.50 to 0.95. It is stricter and gives a more realistic picture of detection quality.

### 8.17 Precision and Recall

- **Precision** measures how many predicted boxes are correct.
- **Recall** measures how many true hazards were found.

### 8.18 ONNX

ONNX is an open model exchange format. Exporting to ONNX helps move a model between frameworks and deployment tools.

### 8.19 TensorRT

TensorRT is NVIDIA’s inference optimization platform. It is used to accelerate models on NVIDIA GPUs and Jetson devices.

### 8.20 Oral Defense Angle

If asked why the project moved to detection, say that classification only answers what is present, while emergency response also needs where it is. That is why bounding boxes were introduced.

## 9. Iteration 5: Semantic Segmentation

### 9.1 Purpose of Iteration 5

Iteration 5 gives the most detailed understanding of the scene. It predicts a class for each pixel, which is much more precise than using boxes.

### 9.2 Semantic Segmentation

Semantic segmentation assigns a class label to every pixel in the image. This is important when objects have irregular shapes, like smoke.

### 9.3 U-Net

U-Net is a segmentation architecture with an encoder-decoder shape. It is especially famous for medical and dense pixel tasks.

### 9.4 Encoder

The encoder compresses the image into abstract feature representations. It learns what is in the image.

### 9.5 Decoder

The decoder expands features back into a full-resolution output map. It learns where the classes are located pixel by pixel.

### 9.6 Skip Connections

Skip connections send feature maps from the encoder directly to the decoder. They preserve spatial detail that would otherwise be lost.

### 9.7 Bottleneck

The bottleneck is the narrowest middle part of U-Net. It stores the most compressed, high-level representation.

### 9.8 Upsampling

Upsampling increases spatial resolution. In U-Net, it helps rebuild the segmentation mask from compressed features.

### 9.9 Bilinear Upsampling

Bilinear upsampling is a simple interpolation method for increasing image or feature map size smoothly.

### 9.10 Dice Loss

Dice loss measures overlap between predicted masks and true masks. It is helpful when the foreground class is small compared with the background.

### 9.11 Focal Loss

Focal loss focuses training on hard examples and down-weights easy ones. It helps the model pay attention to minority pixels, like smoke.

### 9.12 Combined Dice + Focal Loss

Using both losses helps the model optimize both overlap quality and hard-pixel learning. That is a strong choice for imbalanced segmentation.

### 9.13 IoU

IoU means Intersection over Union. It compares the overlap between the predicted mask and the ground truth mask.

### 9.14 mIoU

mIoU means mean IoU across classes. It is a common segmentation metric.

### 9.15 Dice Coefficient

The Dice coefficient measures similarity between two masks. It is closely related to Dice loss and is often easier to interpret as a score.

### 9.16 Pixel Accuracy

Pixel accuracy measures the percentage of correctly classified pixels. It can look high even when minority classes are harder, so it should be interpreted together with IoU and Dice.

### 9.17 torch.cuda.amp

torch.cuda.amp is PyTorch’s automatic mixed precision tool. It speeds up training and reduces memory use on GPUs.

### 9.18 Oral Defense Angle

If asked why segmentation was needed after detection, say that fire and smoke are not clean rectangular objects. Smoke especially is diffuse and irregular, so pixel masks are more faithful than boxes.

## 10. Performance Metrics Explained

### 10.1 Accuracy

Accuracy is the percentage of correct predictions. It is useful, but not always sufficient in imbalanced problems.

### 10.2 Loss

Loss is the value the model tries to minimize during training. Lower loss usually means better predictions, but the meaning depends on the task.

### 10.3 Validation Metrics

Validation metrics are measured on held-out data during development. They help choose models and detect overfitting.

### 10.4 Test Metrics

Test metrics are measured after training on completely unseen data. They are the best indicator of final performance.

### 10.5 Precision vs Recall

This is a key oral-exam topic:

- Precision answers: when the model predicts fire, how often is it right?
- Recall answers: how many real fire cases did the model find?

For safety systems, recall is often especially important because missing a hazard can be worse than raising a false alarm.

### 10.6 F1-Score

F1-score is the harmonic mean of precision and recall. It balances the two.

### 10.7 Why Multiple Metrics Matter

Each task requires different evaluation logic:

- Classification is about class correctness.
- Detection is about box correctness.
- Segmentation is about pixel overlap.

No single metric explains everything.

## 11. Edge Deployment and Hardware Terms

### 11.1 Raspberry Pi

Raspberry Pi is a small ARM-based single-board computer. It is useful for low-cost edge inference.

### 11.2 Jetson

Jetson is NVIDIA’s edge AI hardware family. It is more powerful than a Raspberry Pi for GPU-accelerated workloads.

### 11.3 NPU

NPU stands for Neural Processing Unit, a chip designed specifically for AI inference.

### 11.4 GPU

GPU stands for Graphics Processing Unit. GPUs are very good at running neural networks because they can process many operations in parallel.

### 11.5 CPU

CPU stands for Central Processing Unit. It is general-purpose and often slower than a GPU for deep learning inference, but it is common and widely available.

### 11.6 FPS

FPS means frames per second. In inference, it tells you how many images a model can process per second.

### 11.7 Latency

Latency is the delay between input and output. Lower latency is better for real-time detection.

### 11.8 TensorRT and ONNX for Deployment

These export formats show the model can be moved into optimized inference environments, especially on NVIDIA hardware.

### 11.9 Quantization Readiness

A quantization-ready model is one that still performs well after being converted to a lower-precision format.

### 11.10 Oral Defense Angle

If asked why edge deployment is central to the project, answer that fire detection is time-sensitive and may need local inference where internet access is poor or latency matters.

## 12. Training Workflow Terms

### 12.1 Dataset Class

A dataset class is a PyTorch object that defines how data is loaded, transformed, and returned during training.

### 12.2 Dataloader

A dataloader batches data and feeds it to the model efficiently.

### 12.3 Trainer Class

A trainer class wraps the training loop, validation loop, and logging. It makes the code reusable across models.

### 12.4 Epoch

An epoch is one full pass through the training dataset.

### 12.5 Batch

A batch is a small group of samples processed together.

### 12.6 Learning Rate

The learning rate controls how big each parameter update is during training. Too high can destabilize training; too low can make learning slow.

### 12.7 Weights & Biases (W&B)

W&B is an experiment tracking tool. It logs metrics, losses, charts, and training runs so results can be compared later.

### 12.8 Artifact

An artifact is a saved model, dataset version, or other tracked output in W&B.

### 12.9 OOP

Object-Oriented Programming organizes code into classes and objects. In this project it helps separate datasets, models, trainers, and utility functions.

### 12.10 Modular Codebase

A modular codebase splits functionality into smaller files such as dataset, model, train, metrics, and utils. This improves readability and reuse.

## 13. Presentation and Visualization Terms

### 13.1 Glassmorphism

Glassmorphism is a visual style using translucent cards, blur, and layered depth. It gives the presentation a modern dashboard feel.

### 13.2 Dark Mode UI

Dark mode uses dark backgrounds and bright highlights. It is often comfortable for presentations and fits the fire/smoke theme.

### 13.3 SVG

SVG stands for Scalable Vector Graphics. It is ideal for architecture diagrams because it stays sharp at any size.

### 13.4 Chart.js

Chart.js is a JavaScript charting library used to visualize comparison data in the presentation.

### 13.5 Tooltip

A tooltip is a small floating hint box shown when hovering over an element. Here it helps explain architecture blocks.

### 13.6 Standalone HTML App

A standalone HTML app is a web page that runs directly in the browser without needing a build process or server framework.

## 14. Iteration-by-Iteration Defense Summary

### Iteration 1

What it is: a binary CNN baseline.

Why it exists: to prove the pipeline and establish a low-cost starting point.

Main terms: CNN, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Global Average Pooling, BCEWithLogitsLoss.

### Iteration 2

What it is: four-class transfer learning with MobileNetV3-Small.

Why it exists: to distinguish smoke-only, fire-only, mixed, and normal scenes.

Main terms: transfer learning, backbone, classification head, freezing, unfreezing, fine-tuning, cross-entropy, class imbalance, F1-score.

### Iteration 3

What it is: a robustness-focused version of iteration 2.

Why it exists: to make the model more stable in realistic conditions and more suitable for edge deployment.

Main terms: Albumentations, RandomFog, ColorJitter, CLAHE, CosineAnnealingLR, PTQ, INT8, FP32, edge simulation.

### Iteration 4

What it is: object detection with YOLO26n.

Why it exists: to find where fire and smoke are, not just what is in the image.

Main terms: object detection, bounding box, YOLO, anchor-free, NMS-free, mAP50, mAP50-95, precision, recall, ONNX, TensorRT.

### Iteration 5

What it is: semantic segmentation with U-Net.

Why it exists: to capture the exact shape and spread of fire and smoke.

Main terms: semantic segmentation, encoder, decoder, skip connections, bottleneck, upsampling, Dice loss, Focal loss, IoU, mIoU, Dice coefficient, pixel accuracy, mixed precision.

## 15. Common Oral-Exam Questions and Short Answers

### Why did the project start with a binary classifier?

Because it is the simplest correct baseline. It verifies the data pipeline and training code before moving to harder tasks.

### Why use transfer learning in iteration 2?

Because pretrained models already know generic visual features, which makes training faster and more effective than starting from zero.

### Why was iteration 3 needed if accuracy did not improve dramatically?

Because the goal was robustness and deployment realism, not just validation accuracy. Strong augmentation and quantization are important for real-world use.

### Why move from classification to detection?

Because a fire alert system should know where the hazard is, not just whether it exists.

### Why move from detection to segmentation?

Because smoke and fire have irregular shapes, and segmentation captures them more accurately than boxes.

### Why are smoke scenes hard?

Because smoke is diffuse, semi-transparent, and visually similar to fog, clouds, or haze.

### Why is class imbalance important?

Because a model can look accurate while still failing minority classes. Metrics like F1 and class-wise scores reveal that problem.

### Why is edge deployment important?

Because fire detection should work quickly and locally, even with limited power or connectivity.

## 16. Final Defense Narrative

If you need a single story for the verbal exam, use this:

The project started with a scratch CNN to prove the pipeline. It then expanded to a pretrained MobileNetV3 classifier so the system could distinguish fire, smoke, both, and normal scenes. After that, robustness was improved with stronger augmentation, learning-rate scheduling, and quantization to simulate edge constraints. Next, the task moved to YOLO-based object detection so the model could locate hazards spatially. Finally, the system became a segmentation model with U-Net so it could identify the exact pixel-level shape of fire and smoke. The whole project is a progression from simple recognition to precise scene understanding, with edge deployment always kept in mind.

## 17. What to Remember Most

If you only remember a few things for the exam, remember these:

1. The project is iterative because each version solves a more realistic problem.
2. Binary classification is the easiest baseline, but it is too coarse.
3. Transfer learning helps when data is limited and edge deployment matters.
4. Robustness training improves real-world usefulness, not just a single metric.
5. Detection adds location; segmentation adds exact shape.
6. Quantization and mixed precision help with deployment efficiency.
7. Metrics must match the task: accuracy for classification, mAP for detection, IoU and Dice for segmentation.

## 18. Short Glossary

- **Activation function:** a function like ReLU that introduces non-linearity.
- **Anchor-free:** a detector design that avoids anchor templates.
- **Batch normalization:** a normalization layer that stabilizes training.
- **Backbone:** the feature extractor part of a model.
- **Bounding box:** a rectangle around an object.
- **Classification head:** the final layers that output class predictions.
- **Dice loss:** a loss based on mask overlap.
- **Encoder-decoder:** a model structure that compresses and then reconstructs features.
- **Focal loss:** a loss that emphasizes hard examples.
- **Generalization:** performance on unseen data.
- **IoU:** intersection-over-union overlap score.
- **Logits:** raw outputs before probability conversion.
- **mAP:** mean average precision for detection.
- **mIoU:** mean intersection over union for segmentation.
- **NMS:** non-maximum suppression.
- **Pixel mask:** per-pixel class annotation.
- **PTQ:** post-training quantization.
- **Transfer learning:** adapting a pretrained model.
- **U-Net:** a segmentation architecture with skip connections.

## 19. Closing Note

The most important idea in this project is not any single metric or architecture. It is the systematic evolution of the task from coarse to precise understanding while respecting edge-device constraints. If you can explain that trade-off clearly, you can defend the project well.