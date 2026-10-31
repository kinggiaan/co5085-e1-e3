"""Mô hình A1: ResNet-18 (CNN) và Swin-T (Transformer) với 4 chế độ huấn luyện.

Chế độ:
    scratch        : khởi tạo ngẫu nhiên, train toàn bộ
    finetune       : pretrained ImageNet-1k, train toàn bộ
    freeze_all     : pretrained, đóng băng backbone, chỉ train head mới
    freeze_partial : pretrained, đóng băng phần lớn, train các block cuối + head

Ghi chú chọn kiến trúc: ViT-B/16 (86M) không vừa 4 GB VRAM ở 224px (đo được 5.048 MB
peak, tràn sang shared memory và chậm gấp ~30 lần), nên Transformer được đại diện bởi
**Swin-T** (28M, phân cấp, cửa sổ dịch) — cũng nằm trong danh sách đề gợi ý và vừa
3,2 GB ở 128px.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 200
MODES = ("scratch", "finetune", "freeze_all", "freeze_partial")
ARCHS = ("resnet18", "swin_t")


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def build_model(arch: str, mode: str, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    arch, mode = arch.lower(), mode.lower()
    if mode not in MODES:
        raise ValueError(f"mode phải là {MODES}")
    use_weights = pretrained and mode != "scratch"

    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if use_weights else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        if mode == "freeze_all":
            _freeze(m)
            for p in m.fc.parameters():
                p.requires_grad = True
        elif mode == "freeze_partial":
            _freeze(m)
            for blk in (m.layer4, m.fc):
                for p in blk.parameters():
                    p.requires_grad = True
        return m

    if arch == "swin_t":
        m = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1 if use_weights else None)
        m.head = nn.Linear(m.head.in_features, num_classes)
        if mode == "freeze_all":
            _freeze(m)
            for p in m.head.parameters():
                p.requires_grad = True
        elif mode == "freeze_partial":
            _freeze(m)
            for blk in (m.features[-1], m.norm, m.head):  # stage cuối + norm + head
                for p in blk.parameters():
                    p.requires_grad = True
        return m

    raise ValueError(f"arch phải là {ARCHS}")


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def param_groups(model: nn.Module, base_lr: float, head_lr: float | None = None) -> list[dict]:
    """Head học nhanh hơn backbone khi finetune/freeze."""
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if ("head" in name or name.startswith("fc.")) else backbone_params).append(p)
    groups = [{"params": backbone_params, "lr": base_lr}]
    if head_params:
        groups.append({"params": head_params, "lr": head_lr or base_lr})
    return [g for g in groups if g["params"]]


DEFAULT_LR = {
    # arch: {mode: (backbone_lr, head_lr)}
    "resnet18": {"scratch": (1e-3, None), "finetune": (1e-4, 1e-3),
                 "freeze_all": (1e-3, None), "freeze_partial": (1e-4, 1e-3)},
    "swin_t": {"scratch": (3e-4, None), "finetune": (2e-5, 1e-3),
               "freeze_all": (1e-3, None), "freeze_partial": (5e-5, 1e-3)},
}
