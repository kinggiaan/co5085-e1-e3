"""SegFormer-B0 cho A2 — biến thể Transformer đúng theo paper đang phân tích.

Dùng `transformers` với trọng số `nvidia/segformer-b0-finetuned-ade-512-512` (pretrain
ImageNet-1k rồi finetune ADE20K), thay classifier head bằng 21 lớp của VOC.

Lưu ý: logits trả về ở 1/4 độ phân giải đầu vào (đặc trưng của SegFormer), nên vòng lặp
huấn luyện phải interpolate lên kích thước mask — `A2/train.py` đã có sẵn bước này.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

# Cache HuggingFace đặt trong repo (đã gitignore) để tránh lỗi quyền ở %USERPROFILE%\.cache
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".hf"))

NUM_CLASSES = 21
DEFAULT_WEIGHTS = "nvidia/segformer-b0-finetuned-ade-512-512"


class SegFormerWrap(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, weights: str = DEFAULT_WEIGHTS) -> None:
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            weights,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        # head mới cho 21 lớp VOC (ADE có 150 lớp)
        hidden = self.model.config.hidden_sizes[-1]
        self.model.decode_head.classifier = nn.Conv2d(hidden, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        return out.logits


if __name__ == "__main__":
    m = SegFormerWrap()
    n = sum(p.numel() for p in m.parameters())
    y = m(torch.randn(1, 3, 320, 320))
    print(f"SegFormer-B0: {n:,} tham số | logits {tuple(y.shape)}")
