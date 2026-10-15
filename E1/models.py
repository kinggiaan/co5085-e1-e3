"""Ba bộ phân loại của E1: softmax tuyến tính, MLP và CNN.

Mọi model nhận đầu vào (B, 1, 28, 28) và trả về logits (B, 10).
- SoftmaxRegression : flatten -> Linear -> softmax (CrossEntropyLoss lo phần softmax).
- MLP               : flatten -> 2 lớp ẩn fully-connected + ReLU.
- CNN               : các khối conv/BN/ReLU/pool + head phân loại.
"""
from __future__ import annotations

import torch
import torch.nn as nn

IN_SHAPE = (1, 28, 28)
IN_DIM = 1 * 28 * 28
NUM_CLASSES = 10


class SoftmaxRegression(nn.Module):
    """Hồi quy softmax: một phép biến đổi tuyến tính trên vector ảnh đã flatten."""

    def __init__(self, in_dim: int = IN_DIM, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class MLP(nn.Module):
    """Perceptron nhiều tầng: 784 -> 256 -> 128 -> 10, ReLU + dropout."""

    def __init__(
        self,
        in_dim: int = IN_DIM,
        hidden: tuple[int, int] = (256, 128),
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def conv_block(cin: int, cout: int, pool: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class CNN(nn.Module):
    """CNN nhỏ cho ảnh 28x28: 3 khối conv (BN + ReLU) + 2 lần max-pool + head FC."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 32, pool=True),   # 28 -> 14
            conv_block(32, 64),
            conv_block(64, 64, pool=True),   # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


MODEL_REGISTRY = {
    "softmax": SoftmaxRegression,
    "mlp": MLP,
    "cnn": CNN,
}


def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"model không hợp lệ: {name}. Chọn trong {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
