"""Mô hình A2: U-Net tự viết (CNN) và DeepLabV3-ResNet50 pretrained (đối chứng).

Cả hai đều xuất logits (B, 21, H, W) ở độ phân giải đầu vào.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

NUM_CLASSES = 21


def conv_bn(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """U-Net thu gọn: 4 mức encoder, 4 mức decoder, skip connection."""

    def __init__(self, in_ch: int = 3, num_classes: int = NUM_CLASSES, base: int = 32) -> None:
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc1 = conv_bn(in_ch, c[0])
        self.enc2 = conv_bn(c[0], c[1])
        self.enc3 = conv_bn(c[1], c[2])
        self.enc4 = conv_bn(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec3 = conv_bn(c[3], c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2 = conv_bn(c[2], c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = conv_bn(c[1], c[0])
        self.head = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class DeepLabWrap(nn.Module):
    """Bọc DeepLabV3 (torchvision) để forward chỉ trả về logits chính."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
        self.model = models.segmentation.deeplabv3_resnet50(weights=weights, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out["out"] if isinstance(out, dict) else out


def build_model(arch: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    arch = arch.lower()
    if arch == "unet":
        return UNet(num_classes=num_classes)
    if arch == "unet_small":
        return UNet(num_classes=num_classes, base=16)
    if arch == "deeplabv3":
        return DeepLabWrap(num_classes=num_classes, pretrained=True)
    raise ValueError(f"arch không hợp lệ: {arch}")


DEFAULT_LR = {"unet": 1e-3, "unet_small": 1e-3, "deeplabv3": 1e-4}
