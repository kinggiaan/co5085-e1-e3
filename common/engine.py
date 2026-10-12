"""Vòng lặp huấn luyện viết tay: forward -> loss -> backward -> step.

Không dùng các API cấp cao kiểu `trainer.fit()`; mỗi bước tối ưu được
thực hiện tường minh ở đây và được tái sử dụng cho cả E1, E2, E3.
"""
from __future__ import annotations

import copy
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None,
    device: torch.device,
    collect_predictions: bool = False,
) -> dict:
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        if criterion is not None:
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += y.size(0)
        if collect_predictions:
            all_preds.append(preds.detach().cpu())
            all_labels.append(y.detach().cpu())

    metrics = {
        "loss": total_loss / max(total, 1) if criterion is not None else float("nan"),
        "acc": total_correct / max(total, 1),
        "n": total,
    }
    if collect_predictions:
        metrics["preds"] = torch.cat(all_preds).numpy()
        metrics["labels"] = torch.cat(all_labels).numpy()
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    on_step: Callable[[nn.Module], None] | None = None,
) -> dict:
    model.train()
    running_loss, correct, seen = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = y.size(0)
        running_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == y).sum().item()
        seen += bs
        if on_step is not None:
            on_step(model)

    return {"loss": running_loss / max(seen, 1), "acc": correct / max(seen, 1)}


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | str = "cpu",
    optimizer_name: str = "adam",
    grad_clip: float | None = None,
    verbose: bool = True,
) -> tuple[dict, dict]:
    """Huấn luyện và trả về (history, best_state_dict) theo val accuracy."""
    device = torch.device(device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:  # pragma: no cover
        raise ValueError(f"optimizer không hợp lệ: {optimizer_name}")

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time_s": [],
    }
    best_acc, best_state, best_epoch = -1.0, None, -1

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip=grad_clip)
        va = evaluate(model, val_loader, criterion, device)
        dt = time.perf_counter() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])
        history["epoch_time_s"].append(dt)

        if va["acc"] > best_acc:
            best_acc, best_epoch = va["acc"], epoch
            best_state = copy.deepcopy(model.state_dict())

        if verbose:
            print(
                f"epoch {epoch:02d}/{epochs} | train loss {tr['loss']:.4f} acc {tr['acc']:.4f} "
                f"| val loss {va['loss']:.4f} acc {va['acc']:.4f} | {dt:5.1f}s",
                flush=True,
            )

    history["best_val_acc"] = best_acc
    history["best_epoch"] = best_epoch
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_state or {}


def confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1
    return cm
