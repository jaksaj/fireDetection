"""Lightweight Custom U-Net model for multi-class semantic segmentation."""

from __future__ import annotations

import logging
import torch
import torch.nn as nn

from src.utils import resolve_device

logger = logging.getLogger(__name__)

# Default only; pass `device=` to instantiate on CPU for edge benchmarking.
DEVICE = resolve_device()


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm2d -> ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downscaling with MaxPool2d then DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class UpBlock(nn.Module):
    """Upscaling using Bilinear interpolation then DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # Upsample scales spatial resolution, then we adjust channels with DoubleConv
        # In U-Net, we concatenate encoder skip connection, so the input channels
        # to the DoubleConv block will be the upsampled channels + the encoder skip channels.
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Note: in_channels is the concatenated channel size (e.g. upsampled + skip)
        # out_channels is the target channel size after DoubleConv
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Feature map from the lower decoder layer to be upsampled.
            x2: Skip connection feature map from the encoder layer.
        """
        x1 = self.up(x1)
        
        # If the spatial sizes don't match due to rounding, pad x1 to match x2
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_y > 0 or diff_x > 0:
            x1 = nn.functional.pad(
                x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
            )

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class LightweightUNet(nn.Module):
    """
    Lightweight scratch-built U-Net optimized for edge semantic segmentation.

    Uses a standard encoder-decoder architecture with skip connections.
    Channels are scaled 32 -> 64 -> 128 -> 256 -> 512 to balance representational
    capacity and computation cost, resulting in ~7.8M parameters.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        base_channels: int = 32,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels

        target_device = resolve_device(device)

        # Encoder path
        self.inc = DoubleConv(in_channels, base_channels)  # 32
        self.down1 = DownBlock(base_channels, base_channels * 2)  # 64
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)  # 128
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)  # 256
        self.down4 = DownBlock(base_channels * 8, base_channels * 16)  # 512

        # Decoder path
        # up4: inputs are upsampled bottleneck (512) + enc4 (256) = 768 channels.
        # Outputs 256 channels.
        self.up4 = UpBlock(base_channels * 16 + base_channels * 8, base_channels * 8)
        # up3: inputs are upsampled up4 (256) + enc3 (128) = 384 channels.
        # Outputs 128 channels.
        self.up3 = UpBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        # up2: inputs are upsampled up3 (128) + enc2 (64) = 192 channels.
        # Outputs 64 channels.
        self.up2 = UpBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        # up1: inputs are upsampled up2 (64) + enc1 (32) = 96 channels.
        # Outputs 32 channels.
        self.up1 = UpBlock(base_channels * 2 + base_channels, base_channels)

        # Output projection
        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

        self._initialize_weights()
        self.to(target_device)

        param_count = sum(p.numel() for p in self.parameters())
        logger.info(
            "LightweightUNet initialized (base_channels=%d, %d classes, %d parameters) on %s",
            base_channels,
            num_classes,
            param_count,
            target_device,
        )

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder forward with skip connections
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        logits = self.outc(x)
        return logits
