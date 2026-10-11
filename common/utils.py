"""Tiện ích dùng chung cho E1–E3 (seed, đếm tham số, lưu kết quả)."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Cố định seed cho random / numpy / torch để kết quả tái lập được."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Không bật deterministic của cuDNN: các model rất nhỏ, ưu tiên tốc độ.
    # Seed + cùng phiên bản thư viện là đủ để tái lập trong thực tế.


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


class Timer:
    """Đo thời gian (dùng cho cột 'thời gian huấn luyện' trong báo cáo)."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.last = self.start

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def lap(self) -> float:
        now = time.perf_counter()
        dt = now - self.last
        self.last = now
        return dt


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
