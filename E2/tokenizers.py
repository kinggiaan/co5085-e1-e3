"""Các cách biến ảnh 28x28 thành chuỗi token cho Transformer (E2).

Mỗi tokenizer nhận (B, 1, 28, 28) và trả về (B, n_tokens, dim), kèm thuộc tính
`num_tokens` (chưa tính token CLS do model tự thêm).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchTokenizer(nn.Module):
    """Chia ảnh thành lưới patch không chồng lấn, mỗi patch là một token.

    patch_size=4 -> 7x7 = 49 token; patch_size=7 -> 4x4 = 16 token.
    """

    kind = "patch"

    def __init__(self, img_size: int = 28, patch_size: int = 4, in_ch: int = 1, dim: int = 128) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size phải chia hết cho patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_tokens = self.grid * self.grid
        self.patch_dim = patch_size * patch_size * in_ch
        self.proj = nn.Linear(self.patch_dim, dim)
        self.name = f"patch{patch_size}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C*p*p, L) -> (B, L, C*p*p) -> (B, L, dim)
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        tokens = patches.transpose(1, 2)
        return self.proj(tokens)


class RowTokenizer(nn.Module):
    """Mỗi hàng ảnh là một token: 28 token, mỗi token là vector 28 chiều."""

    kind = "row"
    name = "row"

    def __init__(self, img_size: int = 28, in_ch: int = 1, dim: int = 128) -> None:
        super().__init__()
        self.num_tokens = img_size
        self.feat_dim = img_size
        self.proj = nn.Linear(self.feat_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.squeeze(1)  # (B, 28, 28): hàng thứ i là token thứ i
        return self.proj(rows)


class ColumnTokenizer(nn.Module):
    """Mỗi cột ảnh là một token (biến thể của RowTokenizer)."""

    kind = "column"
    name = "col"

    def __init__(self, img_size: int = 28, in_ch: int = 1, dim: int = 128) -> None:
        super().__init__()
        self.num_tokens = img_size
        self.feat_dim = img_size
        self.proj = nn.Linear(self.feat_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cols = x.squeeze(1).transpose(1, 2)  # (B, 28, 28)
        return self.proj(cols)


class CNNStemTokenizer(nn.Module):
    """CNN stem rút feature map 7x7, mỗi vị trí spatial là một token (49 token)."""

    kind = "cnnstem"
    name = "cnnstem"

    def __init__(self, img_size: int = 28, in_ch: int = 1, dim: int = 128) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.num_tokens = (img_size // 4) ** 2
        self.proj = nn.Linear(64, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.stem(x)                     # (B, 64, 7, 7)
        tokens = fmap.flatten(2).transpose(1, 2)  # (B, 49, 64)
        return self.proj(tokens)


TOKENIZER_REGISTRY = {
    "patch": PatchTokenizer,
    "patch7": lambda img_size=28, in_ch=1, dim=128: PatchTokenizer(img_size, patch_size=7, in_ch=in_ch, dim=dim),
    "row": RowTokenizer,
    "col": ColumnTokenizer,
    "cnnstem": CNNStemTokenizer,
}


def build_tokenizer(name: str, dim: int = 128, img_size: int = 28) -> nn.Module:
    if name not in TOKENIZER_REGISTRY:
        raise ValueError(f"tokenizer không hợp lệ: {name}. Chọn trong {list(TOKENIZER_REGISTRY)}")
    return TOKENIZER_REGISTRY[name](dim=dim, img_size=img_size)
