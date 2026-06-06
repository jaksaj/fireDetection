"""Convolutional neural network models for fire and smoke classification."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda")


class ConvBlock(nn.Module):
    """Single convolutional block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FireCNN(nn.Module):
    """
    Scratch-built CNN for binary fire vs. normal classification.

    Architecture:
        - 4 convolutional blocks with channel progression 32 -> 64 -> 128 -> 256
        - Global average pooling to reduce parameter count (edge-friendly)
        - Single logit output for ``BCEWithLogitsLoss``
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_conv_blocks: int = 4,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        if num_conv_blocks < 3 or num_conv_blocks > 4:
            raise ValueError("num_conv_blocks must be 3 or 4 for FireCNN.")

        channels = [base_channels * (2**i) for i in range(num_conv_blocks)]
        blocks: list[nn.Module] = []

        current_channels = in_channels
        for out_channels in channels:
            blocks.append(ConvBlock(current_channels, out_channels))
            current_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels[-1], 1)

        self._initialize_weights()
        self.to(DEVICE)

        param_count = sum(p.numel() for p in self.parameters())
        logger.info(
            "FireCNN initialized (%d conv blocks, %d parameters) on %s",
            num_conv_blocks,
            param_count,
            DEVICE,
        )

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class MobileNetV3FireClassifier(nn.Module):
    """
    Transfer-learning classifier built on MobileNetV3-Small for 4-class D-Fire.

    Supports freezing the ImageNet backbone for head-only training, then
    unfreezing the top feature blocks for fine-tuning at a lower learning rate.
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)

        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        self.to(DEVICE)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "MobileNetV3FireClassifier initialized (%d classes, %d/%d trainable params) on %s",
            num_classes,
            trainable,
            total,
            DEVICE,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze all feature-extraction layers; train classifier head only."""
        for parameter in self.backbone.features.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.classifier.parameters():
            parameter.requires_grad = True
        logger.info("Backbone frozen — classifier head is trainable.")

    def unfreeze_top_layers(self, num_blocks: int = 3) -> None:
        """Unfreeze the top ``num_blocks`` of the feature extractor plus the head."""
        for parameter in self.backbone.features.parameters():
            parameter.requires_grad = False

        feature_blocks = self.backbone.features
        start_index = max(0, len(feature_blocks) - num_blocks)
        for block_index in range(start_index, len(feature_blocks)):
            for parameter in feature_blocks[block_index].parameters():
                parameter.requires_grad = True

        for parameter in self.backbone.classifier.parameters():
            parameter.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "Unfroze top %d feature blocks — %d trainable parameters.",
            num_blocks,
            trainable,
        )

    def trainable_parameter_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """Return (backbone, head) parameter groups for differential learning rates."""
        backbone_params: list[nn.Parameter] = []
        head_params: list[nn.Parameter] = []

        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("backbone.classifier"):
                head_params.append(parameter)
            else:
                backbone_params.append(parameter)

        return backbone_params, head_params
