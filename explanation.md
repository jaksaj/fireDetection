## Iteration 1 – Binary Classification with a Custom CNN

### 1. Architecture Choice
For a simple binary classification task (Fire vs. Normal) on $224 \times 224$ images, starting with a lightweight custom CNN ([FireCNN in src/model.py](file:///c:/git/fireDetection/src/model.py#L32-L91)) is the ideal architectural fit. 
* **Depth & Channel Progression:** The network consists of 4 convolutional blocks. The channel depth progresses from $3 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256$. This sequential expansion allows the network to learn hierarchical spatial features: early blocks with shallow channels capture low-level local patterns (edges, gradients, color transitions), while deeper blocks with wider channels capture high-level semantic abstractions (flame shapes, smoke texture).
* **Conv Blocks:** Each block utilizes $3\times3$ convolutions with a stride of 1 and padding of 1 (`Conv2d(..., kernel_size=3, padding=1, bias=False)`) to preserve spatial boundaries, followed by `BatchNorm2d` to stabilize activation distributions, a non-linear activation (`ReLU`), and $2\times2$ max pooling with a stride of 2 (`MaxPool2d(kernel_size=2, stride=2)`) to downsample spatial dimensions by half.
* **Global Average Pooling (GAP):** Instead of flattening the final feature map ($256 \times 14 \times 14 = 50,176$ values) and feeding it into a massive dense layer, [FireCNN](file:///c:/git/fireDetection/src/model.py#L62) utilizes `nn.AdaptiveAvgPool2d((1, 1))`. GAP averages each of the 256 channel feature maps into a single scalar, yielding a $256 \times 1 \times 1$ tensor. This structural decision reduces the parameters of the classification layer from $\approx 50\text{k}$ to just $256$, drastically lowering the model footprint ($\approx 389\text{K}$ total parameters, $\approx 1.5\text{ MB}$), acting as a strong regularizer that prevents overfitting, and ensuring edge compatibility.

### 2. Activation Functions
* **ReLU (Rectified Linear Unit):** The activation function inside the convolutional blocks is $\text{ReLU}(x) = \max(0, x)$.
  * **Role:** It introduces the non-linearity required to learn complex decision boundaries.
  * **Why ReLU?** ReLU is computationally highly efficient (requiring only a thresholding operation at $0$), and it mitigates the vanishing gradient problem during backpropagation since its derivative is a constant $1$ for all positive activations, preventing the gradient from decaying exponentially through the 4 blocks.
* **Output Activation:** The network outputs a raw real-valued logit ($z$) without an activation function during training. In inference, a **Sigmoid** activation ($\sigma(z) = \frac{1}{1 + e^{-z}}$) is applied to compress the logit into a probability range $[0, 1]$, representing the confidence of fire presence.

### 3. Loss Function
* **Choice:** [BCEWithLogitsLoss](file:///c:/git/fireDetection/src/trainer/binary.py#L29) (Binary Cross-Entropy with Logits).
* **Mathematical Formulation:** 
  $$\mathcal{L} = - [y \cdot \log(\sigma(z)) + (1 - y) \cdot \log(1 - \sigma(z))]$$
  where $y \in \{0, 1\}$ is the ground-truth label, $z$ is the raw model logit, and $\sigma(z)$ is the sigmoid function.
* **Why not a simpler loss (e.g., MSE)?** If we used Mean Squared Error (MSE) combined with a Sigmoid activation, the loss gradient contains the term $\sigma'(z) = \sigma(z)(1-\sigma(z))$. When the model makes a highly confident but incorrect prediction, $z \to \pm\infty$, meaning $\sigma'(z) \to 0$. This saturates the neuron, resulting in vanishing gradients ("flat gradients") and stalling learning. In contrast, the gradient of the BCE loss with respect to the logit is:
  $$\frac{\partial \mathcal{L}}{\partial z} = \sigma(z) - y$$
  This gradient is linear with respect to prediction error, avoiding saturation and accelerating convergence.
* **Numerical Stability:** Combining the Sigmoid and BCE functions into a single class leverages the log-sum-exp trick, preventing underflow or overflow when computing $\log(\sigma(z))$ for highly confident predictions.

### 4. Transfer Learning Strategy
* **Strategy:** Trained **from scratch** (no transfer learning).
* **Reasoning:** As a baseline (Iteration 1), the objective is to validate the data pipeline, preprocessing transformations, and training loop structure without external biases.
* **Initialization:** To ensure stable training from scratch, weights are initialized using **Kaiming (He) Normal Initialization** (`nn.init.kaiming_normal_`) for convolutional layers. This scales the weight variance by $\frac{2}{n_{\text{in}}}$ (where $n_{\text{in}}$ is the number of input connections), compensating for the fact that ReLU zeroes out half of the inputs, thereby maintaining a stable variance of activations across all 4 layers. The linear layer is initialized with **Xavier Uniform** (`nn.init.xavier_uniform_`) to handle the transition from the average pooling layer.

### 5. Output Layer Design
* **Structure:** A single output neuron (`nn.Linear(256, 1)`).
* **Justification:** Since the classification task is binary (Fire vs. Normal), a single output neuron is sufficient. The scalar output represents the log-odds (logit) of the image belonging to the "Fire" class. A single logit output matches the binary cross-entropy formulation, requiring only half the parameters and computation compared to a two-neuron Softmax classification head.

### 6. Key Training Decisions
* **Optimizer:** **Adam** (Adaptive Moment Estimation) with a learning rate of $1 \times 10^{-3}$ and weight decay of $1 \times 10^{-4}$.
  * **Why Adam?** It maintains adaptive learning rates for each parameter based on both the first moment (mean of gradients, $m_t$) and the second moment (uncentered variance of gradients, $v_t$). This allows the optimizer to converge rapidly even if some features are sparse or gradients fluctuate widely on a newly initialized custom network.
  * **Weight Decay:** Acts as $L_2$ regularization, adding a penalty $\frac{1}{2}\lambda w^2$ to the loss. This prevents weights from growing excessively large, smoothing the loss surface and improving generalizability.
* **Batch Size:** 32, balancing GPU parallelization efficiency with the stochastic noise required to escape local minima.

### 7. Why this Network could NOT have been used at a different stage
* **Stage 2 & 3 (4-Class):** The output layer produces a single logit. It is mathematically impossible to represent 4 mutually exclusive classes (Neither, Only Fire, Only Smoke, Both) with a single output without imposing an arbitrary, non-ordinal relationship on the classes.
* **Stage 4 (Object Detection):** Object detection requires predicting both spatial regression coordinates (bounding boxes) and multiple class probabilities per image. A classification CNN with global average pooling discards all spatial coordinates, rendering it incapable of localization.
* **Stage 5 (Semantic Segmentation):** U-Net maps feature vectors back to spatial resolutions. FireCNN contracts spatial dimensions to $1\times1$, discarding the spatial resolution necessary to reconstruct a pixel-level mask.

---

## Iteration 2 – 4-Class Transfer Learning

### 1. Architecture Choice
For the 4-class classification task, we utilize [MobileNetV3-Small](file:///c:/git/fireDetection/src/model.py#L94-L134) (1.07M parameters, ~4.1MB) pretrained on ImageNet.
* **Depthwise Separable Convolutions:** Standard convolution extracts features by filtering and combining channels simultaneously, costing $D_k \cdot D_k \cdot M \cdot N \cdot D_f \cdot D_f$ operations (where $D_k$ is kernel size, $M$/$N$ are input/output channels, and $D_f$ is feature map size). MobileNetV3 splits this into:
  1. *Depthwise Convolution:* A single $3\times3$ filter per channel to extract spatial features ($D_k \cdot D_k \cdot M \cdot D_f \cdot D_f$).
  2. *Pointwise Convolution:* A $1\times1$ convolution to project channels ($M \cdot N \cdot D_f \cdot D_f$).
  This reduces computational costs and parameters by roughly $\frac{1}{N} + \frac{1}{D_k^2} \approx 8\times$ to $9\times$ for $3\times3$ filters, enabling high-speed edge execution.
* **Inverted Residual Blocks (MBConv):** MobileNetV3 uses a "narrow-wide-narrow" structure. It projects low-dimensional features to a high-dimensional space using $1\times1$ expansions, performs depthwise convolution, and projects back to a low-dimensional bottleneck using a linear $1\times1$ convolution. Residual skip connections connect the bottlenecks. This design allows the network to process complex representations in the expanded space while preventing the loss of information through linear bottlenecks.
* **Squeeze-and-Excitation (SE) Blocks:** These apply channel-wise attention by globally pooling feature maps, passing them through a small bottleneck MLP, and scale-multiplying the original channels. This allows the network to prioritize highly relevant feature maps (e.g., flame colors) and suppress irrelevant ones.

### 2. Activation Functions
* **Hardswish:** MobileNetV3 replaces standard Swish ($x \cdot \sigma(x)$) with Hardswish to avoid the exponential function calculation on edge CPUs:
  $$\text{Hardswish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}$$
  where $\text{ReLU6}(x) = \min(\max(0, x), 6)$. 
  * **Role:** Swish/Hardswish are smooth, non-monotonic functions that allow small negative values to pass through. This smooth gradient flow improves generalization compared to ReLU, which creates a sharp gradient discontinuity at $0$.
* **Softmax:** Raw network outputs are projected as 4 logits. The **Softmax** function is applied:
  $$p_c = \frac{e^{-z_c}}{\sum_{j=1}^{4} e^{-z_j}}$$
  converting the logits into a probability distribution over the 4 classes.

### 3. Loss Function
* **Choice:** Weighted [Cross-Entropy Loss](file:///c:/git/fireDetection/src/trainer/multiclass.py#L47).
* **Mathematical Formulation:**
  $$\text{Loss} = - \sum_{c=1}^{4} w_c \cdot y_c \cdot \log(p_c)$$
  where $y$ is the one-hot target, $p_c$ is the softmax probability, and $w_c$ is the class weight.
* **Why weighted?** Forest fire classification datasets are highly imbalanced; "Neither" (background) is highly frequent, while "Both" (fire and smoke) is rare. Without weighting, the network would optimize for the majority class to minimize loss, leading to catastrophic false negatives for minority classes. Weighting by inverse class frequencies ($w_c \propto \frac{1}{N_c}$) scales gradients during backpropagation, penalizing minority class misclassifications more severely and forcing the optimizer to adjust decision boundaries.

### 4. Transfer Learning Strategy
To adapt the ImageNet-pretrained backbone to the fire/smoke domain without destroying features, we implement a **two-phase transfer learning strategy**:
* **Phase 1: Feature Extractor Frozen (5 epochs):** All weights in the feature extraction backbone are frozen (`requires_grad = False`). Only the randomly initialized custom classification head is trained with $lr = 1 \times 10^{-3}$. This "warms up" the classification weights. If the backbone were unfrozen immediately, the massive gradient updates from the random head would propagate back and erase the pretrained ImageNet weights (known as *catastrophic forgetting*).
* **Phase 2: Unfreeze Top Blocks (10 epochs):** The top 3 feature blocks of the backbone are unfrozen (`requires_grad = True`), while the early layers remain frozen. Early layers extract generic visual elements (lines, textures) which generalize perfectly to our task. The top layers represent high-level semantic shapes (objects, complex shapes) which must be fine-tuned to recognize amorphous patterns like smoke and fire. 
* **Differential Learning Rates:** Fine-tuning uses a differential learning rate: the backbone is updated at a lower rate ($1 \times 10^{-4}$), while the classification head is trained at a $10\times$ higher rate ($1 \times 10^{-3}$). This allows the backbone to adjust gently to the new domain without destabilizing.

### 5. Output Layer Design
* **Structure:** A custom classifier head: `nn.Linear(in_features, 256)` $\rightarrow$ `nn.Hardswish` $\rightarrow$ `nn.Dropout(p=0.2)` $\rightarrow$ `nn.Linear(256, 4)`.
* **Justification:** The output maps the 576-channel backbone features to 4 output logits representing the mutually exclusive classes: [Neither, Only Fire, Only Smoke, Both]. The inclusion of `Dropout(0.2)` randomly zeroes $20\%$ of the activations during training, forcing the network to learn redundant feature pathways and preventing the head from overfitting to specific co-occurrences of features.

### 6. Key Training Decisions
* **Optimizer:** **Adam** with a differential learning rate setup.
* **Class Weighting:** Dynamically calculated based on the training dataset distribution, scaling backpropagation updates.
* **Dropout:** Set to $p=0.2$ in the classification head, acting as a structural regularizer during both Phase 1 and Phase 2.

### 7. Why this Network could NOT have been used at a different stage
* **Stage 1 (Binary Classification):** Since Stage 1 is a binary problem, using a 4-class MobileNetV3-Small classifier would introduce unnecessary parameters, positive classes that do not exist in the annotations (like "Smoke" or "Both"), and risk over-parameterization on a simple verification baseline.
* **Stage 4 & 5 (Detection & Segmentation):** The output layer collapses all spatial dimensions using Global Average Pooling before the classification head. The network cannot output bounding boxes or pixel masks.

---

## Iteration 3 – Robustness and Edge Simulation

### 1. Architecture Choice
The model keeps the same [MobileNetV3-Small](file:///c:/git/fireDetection/src/model.py#L94-L134) architecture and 4-class classifier.
* **Quantization-Friendly Architecture:** The architectural choice is highly suitable for **INT8 Post-Training Quantization (PTQ)**. Dynamic PTQ maps FP32 weights ($32\text{-bit}$ floats) to INT8 ($8\text{-bit}$ signed integers), which reduces the model size from $\approx 4.1\text{ MB}$ to $\approx 1.07\text{ MB}$ (a $3.83\times$ compression ratio) and speeds up CPU inference by converting floating-point arithmetic to integer instructions.

### 2. Activation Functions
* **Hardswish and ReLU6:** 
  $$\text{ReLU6}(x) = \min(\max(0, x), 6)$$
  * **Critical role in quantization:** In standard ReLU, the activations are unbounded ($[0, \infty)$). When quantizing activations to 8 bits, we map the dynamic range to 256 integer bins ($[-128, 127]$). If a layer has massive outlier activations, the scaling factor must expand, which squeezes normal activations into fewer bins, causing significant quantization noise. By using `ReLU6`, the activation values are capped at $6$. This tight, bounded range allows for an optimal, low-error mapping of floating-point values to 8-bit integers, preventing accuracy degradation after quantization.

### 3. Loss Function
* **Choice:** Weighted **Cross-Entropy Loss**, matching the formulation of Iteration 2.

### 4. Transfer Learning Strategy
* **Strategy:** Same two-phase differential learning rate fine-tuning strategy as Iteration 2 (freezing features initially, unfreezing top 3 blocks with $10\times$ lower LR on the backbone). This is crucial because the input images now undergo aggressive augmentations, meaning the backbone must adjust its top-level filters to extract robust features despite noise, fog, and contrast distortion.

### 5. Output Layer Design
* **Structure:** Same 4-class classifier head (`nn.Linear(256, 4)`) with dropout.

### 6. Key Training Decisions
* **Albumentations Augmentation Pipeline:**
  1. `RandomFog`: Simulates foggy conditions. Because fog and smoke share low-frequency textures and white/gray features, models trained without fog data frequently false-trigger on weather changes. Exposing the model to simulated fog forces it to learn the distinct visual boundaries and high-frequency textures of smoke.
  2. `ColorJitter`: Randomly shifts brightness, contrast, saturation, and hue to simulate variable lighting conditions (sun glare, twilight, shade).
  3. `CLAHE` (Contrast Limited Adaptive Histogram Equalization): Enhances local contrast. This prevents the model from missing low-contrast smoke plumes against gray skies.
* **Cosine Annealing Learning Rate Scheduler:**
  The learning rate decays following a cosine curve:
  $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)$$
  where $\eta_{\min} = 1 \times 10^{-6}$ and $T_{max}$ is the phase epoch count.
  * **Why Cosine Annealing?** Unlike step decay which drops the learning rate abruptly, the smooth cosine decay allows the optimizer to navigate narrow valleys in the loss landscape, settling into flatter minima. Flatter minima generalize better to augmented test distributions and are more stable under weight quantization.
* **Edge Simulation (Dynamic PTQ):** Applied using PyTorch's `quantize_dynamic` targeting `nn.Linear` and `nn.Conv2d` layers. This simulates edge deployment (e.g., Raspberry Pi) by evaluating the INT8 model on a CPU, benchmarking latency, and measuring the accuracy delta (dynamic range loss).

### 7. Why this Network could NOT have been used at a different stage
* **Stage 2 (Transfer Learning Baseline):** Iteration 2 is designed to establish the transfer learning baseline. Incorporating advanced augmentations, cosine schedules, and PTQ simultaneously would violate ablation study principles, making it impossible to separate the accuracy gains of transfer learning from the generalization gains of scheduling and augmentation.
* **Stage 4 & 5 (Detection & Segmentation):** The architecture remains restricted to image-level classification, unable to provide coordinates or dense pixel-level boundaries.

---

## Iteration 4 – Object Detection with YOLO

### 1. Architecture Choice
For spatial localization, the model switches to a single-stage object detector: **YOLO** (Specifically the nano variant `yolo26n`, 2.57M parameters, ~9.8MB).
* **Single-Stage Architecture:** formulated as a single regression problem. The network processes the image once, predicting bounding boxes and class probabilities directly from the feature maps. This avoids the latency overhead of two-stage detectors (like Faster R-CNN) which require a Region Proposal Network (RPN) followed by region classification.
* **CSPDarknet Backbone:** Cross-Stage Partial networks divide the feature map of a stage into two parts: one passes through a block of convolutions, and the other bypasses it to merge at the end. This halves the gradient copy path, reducing computational complexity and memory usage while preserving feature representation capacity.
* **PANet Neck (Path Aggregation Network):** While a standard Feature Pyramid Network (FPN) passes semantic information down from deep layers to shallow layers, PANet adds a bottom-up path that propagates low-level spatial features (edges, corner points) up to higher-level feature maps. This multi-scale feature aggregation is critical because fire and smoke can appear at vastly different sizes (from small sparks to massive forest fires).
* **Anchor-Free Head:** YOLO predicts bounding boxes directly from feature maps (center offset, width, height) instead of comparing predictions to preset anchor templates. This simplifies the network, avoids hand-crafted anchor shapes, and handles amorphous, variable-shaped objects like smoke.

### 2. Activation Functions
* **SiLU (Sigmoid Linear Unit):** 
  $$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$
  * **Role:** SiLU is used throughout the backbone and neck. It is smooth, differentiable, and non-monotonic. Unlike ReLU, which zeroes out negative gradients entirely, SiLU provides a soft transition near $0$, preventing neurons from dying and allowing the deep detector to backpropagate stable gradients across regression and classification tasks.
* **Sigmoid:** Used in the output heads to bind class probabilities and box coordinate offsets to the range $[0, 1]$.

### 3. Loss Function
YOLO uses a multi-task loss function to optimize classification and localization simultaneously:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}}$$
* **CIoU Loss (Complete Intersection over Union):** Used for bounding box regression. Unlike Mean Squared Error (MSE) on box coordinates (which is highly sensitive to the scale of the object), CIoU is scale-invariant. It measures:
  1. Bounding box overlap area.
  2. Normalized distance between the center points of the predicted and ground-truth boxes.
  3. Aspect ratio consistency ($v$), scaled by a parameter $\alpha$:
     $$v = \frac{4}{\pi^2} \left( \arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h} \right)^2$$
  This forces the model to prioritize box shape alignment, leading to much faster convergence.
* **Distribution Focal Loss (DFL):** Represents box locations as a continuous distribution rather than a single coordinate value. This allows the model to predict precise edges even when boundary edges are blurry, which is common with smoke plumes.
* **BCE Loss:** Used for classifying objects (fire vs. smoke) and determining objectness.

### 4. Transfer Learning Strategy
* **Strategy:** COCO-pretrained weights (`yolo26n.pt`) are loaded. The backbone features are frozen during the initial epochs to retain general feature representations (since COCO contains 80 classes, including fire hydrants and general objects), and then the entire network (backbone, neck, and heads) is fine-tuned to adapt the feature filters to the visual characteristics of fire and smoke.

### 5. Output Layer Design
* **Decoupled Head:** The output head is split into two separate paths:
  1. *Class branch:* Predicts the probability for each class (smoke, fire) at each scale.
  2. *Box branch:* Predicts the 4 bounding box coordinates.
* **Scale Outputs:** The model outputs feature maps at three different strides (usually 8, 16, and 32 times downsampled), matching the target supervision signal of bounding boxes at multiple spatial resolutions.

### 6. Key Training Decisions
* **Optimizer:** **SGD with Momentum** (MuSGD) with a learning rate of $0.01$ and weight decay of $0.0005$. SGD is preferred for long detection training runs (50 epochs) because its uniform gradient steps act as a regularizer, often finding flatter, more generalizable minima than Adam.
* **Real-Time Export (ONNX & TensorRT):** The model is exported to TensorRT. TensorRT optimizes execution on NVIDIA hardware (Jetson/GPUs) by fusing layers (combining convolution, bias, and activation into a single kernel) and using FP16 precision, meeting the real-time target of 30 FPS.

### 7. Why this Network could NOT have been used at a different stage
* **Stage 1, 2 & 3 (Image Classification):** The classification dataset lacks bounding box annotations. YOLO cannot calculate its CIoU or DFL regression losses without bounding box targets, making it impossible to train.
* **Stage 5 (Semantic Segmentation):** YOLO outputs rectangular bounding boxes. Because fire and smoke have highly irregular boundaries, rectangular boxes contain a large percentage of background noise, failing the requirement for pixel-level boundary extraction.

---

## Iteration 5 – Semantic Segmentation with U-Net

### 1. Architecture Choice
For pixel-level segmentation, the model uses a lightweight custom [U-Net](file:///c:/git/fireDetection/src/model_segmentation.py#L80-L160) (7.85M parameters, ~30MB).
* **Symmetric Encoder-Decoder Structure:** 
  * *Encoder (Contracting Path):* Gradually reduces spatial resolution ($256\times256 \rightarrow 16\times16$) using Max Pooling, while increasing channel depth ($3 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256 \rightarrow 512$). It acts as a feature extractor, capturing semantic context ("what is in the image").
  * *Decoder (Expanding Path):* Upsamples the feature maps ($16\times16 \rightarrow 256\times256$) using Bilinear Upsampling, while decreasing channel depth. It projects the compressed representations back to the original resolution, restoring spatial localization ("where are the classes").
* **Skip Connections:** High-resolution spatial feature maps from the encoder are concatenated directly with the upsampled feature maps in the decoder.
  * **Why skip connections are critical:** As the encoder downsamples, high-frequency spatial details (like the fine boundaries of smoke plumes or fire edges) are lost. Skip connections bypass the bottleneck, supplying the decoder with the fine spatial details needed to reconstruct sharp boundaries.
* **Bilinear Upsampling:** [UpBlock](file:///c:/git/fireDetection/src/model_segmentation.py#L46-L78) uses bilinear interpolation followed by a double convolution block (`DoubleConv`). This is computationally lighter than transposed convolutions and avoids checkboard artifacts (unnatural grid patterns in output masks).

### 2. Activation Functions
* **ReLU:** Inside the double convolution blocks (`nn.Conv2d` $\rightarrow$ `nn.BatchNorm2d` $\rightarrow$ `nn.ReLU(inplace=True)`), ReLU is used to introduce non-linearity while keeping backpropagation fast.
* **Softmax:** No activation is applied to the final output projection during training; the model outputs raw logits. During inference, a pixel-wise **Softmax** is applied across the channel dimension ($C=3$: background, smoke, fire) to output a probability distribution for every single pixel.

### 3. Loss Function
* **Choice:** Combined [Dice Loss + Focal Loss](file:///c:/git/fireDetection/src/losses.py#L98-L124) (`DiceFocalLoss`).
* **Mathematical Formulations:**
  * **Multiclass Dice Loss:**
    $$\mathcal{L}_{\text{Dice}} = 1 - \frac{1}{C}\sum_{c=1}^{C} \frac{2\sum_{i} p_{ic} g_{ic} + \epsilon}{\sum_{i} p_{ic} + \sum_{i} g_{ic} + \epsilon}$$
    where $p_{ic}$ is the predicted softmax probability for pixel $i$ and class $c$, $g_{ic}$ is the ground-truth binary label, and $\epsilon$ is a smoothing term ($1 \times 10^{-5}$) to prevent division by zero.
  * **Multiclass Focal Loss:**
    $$\mathcal{L}_{\text{Focal}} = -\frac{1}{N}\sum_{i=1}^{N} \alpha (1 - p_{it})^\gamma \log(p_{it})$$
    where $p_{it}$ is the predicted probability for the target class at pixel $i$, $\gamma=2.0$ is the focusing parameter, and $\alpha=0.25$ is the balance factor.
* **Why this combination?**
  1. *Class Imbalance:* A typical fire scene is dominated by background pixels ($>90\%$), with smoke taking up $\approx 8\%$ and fire only $\approx 2\%$. Standard cross-entropy loss would optimize for the background, ignoring the critical hazard pixels.
  2. *Dice Loss* addresses this by measuring regional overlap (intersection over union), which is scale-invariant and treats small and large regions equally.
  3. *Focal Loss* handles class imbalance by dynamically down-weighting easy-to-classify pixels (e.g., solid background) and forcing the optimizer to focus on hard, misclassified pixels (e.g., diffuse smoke boundaries).
  4. *Linear Combination:* Combining both losses ($1.0 \times \mathcal{L}_{\text{Dice}} + 1.0 \times \mathcal{L}_{\text{Focal}}$) balances global regional overlap with hard-pixel classification.

### 4. Transfer Learning Strategy
* **Strategy:** Trained **from scratch** on the Roboflow Fire and Smoke dataset (7,110 images with COCO-format polygon masks).
* **Reasoning:** Semantic segmentation requires the network to learn dense spatial relationships from the start. Pretrained ImageNet classification features do not map directly to a custom, lightweight decoder path. Training from scratch on a large dataset of 7,110 images allows the 7.85M parameters model to converge effectively without importing classification bias.

### 5. Output Layer Design
* **Structure:** A $1\times1$ convolution projection layer (`nn.Conv2d(32, 3, kernel_size=1)`).
* **Justification:** The final decoder layer output has 32 channels. The $1\times1$ convolution maps these 32 channels to the 3 target classes: [background, smoke, fire]. This yields an output shape of $(N, 3, 256, 256)$, which aligns with the target supervision mask of $(N, 256, 256)$ containing the pixel-wise class indices $\{0, 1, 2\}$.

### 6. Key Training Decisions
* **Automatic Mixed Precision (AMP):** Implemented using `torch.cuda.amp.autocast` and `GradScaler`.
  * **Why AMP?** Autocast runs forward-pass operations (like convolutions) in FP16 precision, which reduces GPU memory usage by $\approx 50\%$ and speeds up execution. It keeps sensitive operations (like losses) in FP32 to maintain numerical stability. The `GradScaler` dynamically scales loss values before backpropagation to prevent small gradients from underflowing (becoming zero in FP16), scaling them back down before the weights update.
* **Optimizer:** **Adam** with a learning rate of $5 \times 10^{-4}$ and weight decay of $1 \times 10^{-4}$, providing stable, adaptive parameter updates across both the contracting and expanding paths of the U-Net.

### 7. Why this Network could NOT have been used at a different stage
* **Stage 1, 2 & 3 (Image Classification):** Classification datasets contain only image-level labels. A U-Net requires dense pixel-level masks to calculate its segmentation loss, making it impossible to train on image-level labels.
* **Stage 4 (Object Detection):** Object detection annotations consist of bounding box coordinates. A U-Net cannot output a set of coordinates, and using it for object detection would require removing the decoder and adding bounding box regression heads, which changes the architecture entirely.

---

## Technical Summary Matrix for Your Defense

| Stage | Task | Model | Params / Size | Key Activation | Loss Function | Optimizer / LR | Output Shape |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Binary Class | Custom CNN | 389K / 1.5MB | ReLU | BCEWithLogitsLoss | Adam ($1\times 10^{-3}$) | $(N, 1)$ |
| **2** | 4-Class | MobileNetV3-Small | 1.07M / 4.1MB | Hardswish / Softmax | Weighted Cross-Entropy | Adam (Differential LR) | $(N, 4)$ |
| **3** | Robust 4-Class | MobileNetV3-Small | 1.07M (FP32) $\rightarrow$ 1.07M (INT8 PTQ) | Hardswish / ReLU6 | Weighted Cross-Entropy | Adam (Cosine LR Schedule) | $(N, 4)$ |
| **4** | Detection | YOLO26n | 2.57M / 9.8MB | SiLU / Sigmoid | CIoU + DFL + BCE | MuSGD ($1\times 10^{-2}$) | Multi-scale Decoupled Heads |
| **5** | Segmentation | Custom U-Net | 7.85M / 30MB | ReLU / Softmax | Dice + Focal Loss | Adam ($5\times 10^{-4}$) with AMP | $(N, 3, 256, 256)$ |