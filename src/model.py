"""Convolutional neural network for Iteration 1 binary fire detection."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

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
