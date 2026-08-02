"""Convolutional neural network models for fire and smoke classification."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.utils import resolve_device

logger = logging.getLogger(__name__)

# Default only. Every model takes an explicit ``device`` argument so it can be
# instantiated on CPU for edge benchmarking; see src/utils.resolve_device.
DEVICE = resolve_device()


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
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()

        if num_conv_blocks < 3 or num_conv_blocks > 4:
            raise ValueError("num_conv_blocks must be 3 or 4 for FireCNN.")

        target_device = resolve_device(device)

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
        self.to(target_device)

        param_count = sum(p.numel() for p in self.parameters())
        logger.info(
            "FireCNN initialized (%d conv blocks, %d parameters) on %s",
            num_conv_blocks,
            param_count,
            target_device,
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
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()

        target_device = resolve_device(device)

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)

        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        self.to(target_device)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "MobileNetV3FireClassifier initialized (%d classes, %d/%d trainable params) on %s",
            num_classes,
            trainable,
            total,
            target_device,
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


# ----------------------------------------------------------------------------
# Backbone comparison axis
#
# The thesis compares detection *paradigms* (classification / detection /
# segmentation). To separate "the paradigm matters" from "the architecture
# matters", one task -- 4-class D-Fire classification -- is also trained across
# several backbones under an identical budget, resolution, augmentation set and
# seed. This class provides that axis with the same interface the two-phase
# MulticlassTrainer already expects.
# ----------------------------------------------------------------------------

#: Supported backbones -> (torchvision constructor name, weights enum name).
BACKBONES: dict[str, tuple[str, str]] = {
    "mobilenet_v3_small": ("mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
}


class BackboneClassifier(nn.Module):
    """
    Transfer-learning classifier over an interchangeable torchvision backbone.

    Exposes the same ``freeze_backbone`` / ``unfreeze_top_layers`` /
    ``trainable_parameter_groups`` API as :class:`MobileNetV3FireClassifier`, so
    the existing two-phase training protocol applies unchanged across every
    architecture. That equality of protocol is what makes the comparison fair.
    """

    def __init__(
        self,
        backbone: str = "mobilenet_v3_small",
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.2,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()

        if backbone not in BACKBONES:
            raise ValueError(
                f"Unsupported backbone {backbone!r}. Choose from {sorted(BACKBONES)}."
            )

        import torchvision.models as tv_models

        self.backbone_name = backbone
        target_device = resolve_device(device)

        ctor_name, weights_name = BACKBONES[backbone]
        weights = getattr(tv_models, weights_name).DEFAULT if pretrained else None
        self.backbone = getattr(tv_models, ctor_name)(weights=weights)

        # Each family names its classifier differently and nests it differently.
        # Normalise all of them to a single `self.head` attribute so the rest of
        # this class -- and the trainer -- can stay architecture-agnostic.
        if backbone.startswith("mobilenet"):
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = self._make_head(in_features, num_classes, dropout)
            self._head_prefix = "backbone.classifier"
            self._feature_container = self.backbone.features
        elif backbone.startswith("resnet"):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = self._make_head(in_features, num_classes, dropout)
            self._head_prefix = "backbone.fc"
            # ResNet has no `.features` Sequential; its stages are top-level
            # attributes, so build an ordered view of them for freezing.
            self._feature_container = nn.Sequential(
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4,
            )
        elif backbone.startswith("efficientnet"):
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = self._make_head(in_features, num_classes, dropout)
            self._head_prefix = "backbone.classifier"
            self._feature_container = self.backbone.features
        else:  # pragma: no cover - guarded by the BACKBONES check above
            raise ValueError(f"No head-replacement rule for {backbone!r}.")

        self.to(target_device)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "BackboneClassifier[%s] initialized (%d classes, %d/%d trainable params) on %s",
            backbone,
            num_classes,
            trainable,
            total,
            target_device,
        )

    @staticmethod
    def _make_head(in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
        """Identical head across all backbones, so only the trunk varies."""
        return nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def _head_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith(self._head_prefix):
                yield parameter

    def _trunk_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith(self._head_prefix):
                yield parameter

    def freeze_backbone(self) -> None:
        """Freeze all feature-extraction layers; train the classifier head only."""
        for parameter in self._trunk_parameters():
            parameter.requires_grad = False
        for parameter in self._head_parameters():
            parameter.requires_grad = True
        logger.info("[%s] Backbone frozen — head is trainable.", self.backbone_name)

    def unfreeze_top_layers(self, num_blocks: int = 3) -> None:
        """Unfreeze the top ``num_blocks`` feature stages plus the head."""
        for parameter in self._trunk_parameters():
            parameter.requires_grad = False

        blocks = self._feature_container
        start_index = max(0, len(blocks) - num_blocks)
        for block_index in range(start_index, len(blocks)):
            for parameter in blocks[block_index].parameters():
                parameter.requires_grad = True

        for parameter in self._head_parameters():
            parameter.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "[%s] Unfroze top %d stages — %d trainable parameters.",
            self.backbone_name,
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
            if name.startswith(self._head_prefix):
                head_params.append(parameter)
            else:
                backbone_params.append(parameter)

        return backbone_params, head_params
