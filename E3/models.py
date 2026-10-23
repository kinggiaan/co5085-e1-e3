"""E3: mô hình hoá ảnh như chuỗi bằng LSTM / GRU.

Ba cách biểu diễn chuỗi (encoder) + một bộ phân loại hồi quy dùng chung:
    rows    : 28 hàng, mỗi hàng là một bước thời gian (feature 28)
    cols    : 28 cột  (biến thể của rows)
    patches : 7x7 patch 4x4, mỗi patch là một bước thời gian (feature 16)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def image_to_rows(x: torch.Tensor) -> torch.Tensor:
    """(B,1,28,28) -> (B,28,28): hàng là bước thời gian."""
    return x.squeeze(1)


def image_to_cols(x: torch.Tensor) -> torch.Tensor:
    """(B,1,28,28) -> (B,28,28): cột là bước thời gian."""
    return x.squeeze(1).transpose(1, 2)


def image_to_patches(x: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
    """(B,1,28,28) -> (B,49,16) theo thứ tự quét từng hàng patch."""
    patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)  # (B, 16, 49)
    return patches.transpose(1, 2)


SEQUENCE_ENCODERS = {
    "rows": (image_to_rows, 28),
    "cols": (image_to_cols, 28),
    "patches": (lambda x: image_to_patches(x, 4), 16),
}


class SequenceClassifier(nn.Module):
    """LSTM/GRU trên chuỗi token của ảnh.

    pooling = "last"  : lấy hidden state cuối (mặc định, phù hợp chuỗi ngắn)
    pooling = "mean"  : trung bình hidden state theo thời gian
    """

    def __init__(
        self,
        input_size: int = 28,
        hidden_size: int = 128,
        num_layers: int = 1,
        cell: str = "lstm",
        bidirectional: bool = False,
        dropout: float = 0.2,
        num_classes: int = 10,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        cell = cell.lower()
        if cell not in {"lstm", "gru"}:
            raise ValueError(f"cell không hợp lệ: {cell}")
        self.cell = cell
        self.bidirectional = bidirectional
        self.pooling = pooling

        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) đã được encode từ ảnh
        out, _ = self.rnn(x)  # (B, T, H*dirs)

        if self.pooling == "mean":
            feat = out.mean(dim=1)
        else:
            if self.bidirectional:
                feat = torch.cat([out[:, -1, : self.rnn.hidden_size], out[:, 0, self.rnn.hidden_size :]], dim=1)
            else:
                feat = out[:, -1]
        return self.head(feat)


class ImageSequenceClassifier(nn.Module):
    """Ghép encoder chuỗi (rows/cols/patches) với LSTM/GRU."""

    def __init__(self, representation: str = "rows", **kwargs) -> None:
        super().__init__()
        if representation not in SEQUENCE_ENCODERS:
            raise ValueError(f"representation không hợp lệ: {representation}")
        self.encoder, input_size = SEQUENCE_ENCODERS[representation]
        self.representation = representation
        self.classifier = SequenceClassifier(input_size=input_size, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))
