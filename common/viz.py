"""Vẽ biểu đồ cho báo cáo: đường cong học, confusion matrix, mẫu lỗi."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.dpi": 140,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def plot_history(history: dict, path: str | Path, title: str = "") -> None:
    """Hai panel: loss và accuracy theo epoch (train vs val)."""
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(epochs, history["train_loss"], "-o", ms=3, color=COLORS[0], label="train")
    axes[0].plot(epochs, history["val_loss"], "-s", ms=3, color=COLORS[1], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, history["train_acc"], "-o", ms=3, color=COLORS[0], label="train")
    axes[1].plot(epochs, history["val_acc"], "-s", ms=3, color=COLORS[1], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend(frameon=False)
    if title:
        fig.suptitle(title)
    fig.savefig(path)
    plt.close(fig)


def plot_many_histories(histories: dict[str, dict], path: str | Path, metric: str = "val_acc") -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, (name, h) in enumerate(histories.items()):
        ax.plot(h["epoch"], h[metric], "-o", ms=3, color=COLORS[i % len(COLORS)], label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, classes: list[str], path: str | Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.grid(False)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if cm[i, j] > thresh else "#222222",
                )
    if title:
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.savefig(path)
    plt.close(fig)


def plot_error_samples(
    dataset,
    preds: np.ndarray,
    labels: np.ndarray,
    path: str | Path,
    n: int = 16,
    title: str = "Mẫu dự đoán sai",
) -> int:
    """Vẽ n ảnh dự đoán sai kèm nhãn thật / dự đoán. Trả về số mẫu đã vẽ."""
    wrong = np.flatnonzero(preds != labels)
    n = min(n, len(wrong))
    if n == 0:
        return 0
    sel = wrong[:n]
    cols = 8 if n > 8 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.45 * cols, 1.7 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, sel):
        img, _ = dataset[int(idx)]
        ax.imshow(img.squeeze(0).numpy(), cmap="gray")
        ax.set_title(f"true {labels[idx]} / pred {preds[idx]}", fontsize=7, color=COLORS[1])
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.savefig(path)
    plt.close(fig)
    return n


def plot_bar_comparison(
    names: list[str],
    values: list[float],
    path: str | Path,
    ylabel: str = "accuracy",
    title: str = "",
    params: list[int] | None = None,
    annotate: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(names)), 3.4))
    bars = ax.bar(names, values, color=[COLORS[i % len(COLORS)] for i in range(len(names))])
    ax.set_ylabel(ylabel)
    vmax, vmin = max(values), min(values)
    if vmax <= 1.0:  # trục accuracy: phóng to khoảng [vmin-0.08, 1.0]
        ax.set_ylim(max(0.0, vmin - 0.08), 1.0)
    else:            # trục thời gian (giây): luôn bắt đầu từ 0
        ax.set_ylim(0, vmax * 1.15)
    ax.tick_params(axis="x", labelrotation=15, labelsize=8)
    if annotate:
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    if params is not None and title == "":
        title = ""
    if title:
        ax.set_title(title, fontsize=10)
    fig.savefig(path)
    plt.close(fig)


def plot_tradeoff(
    names: list[str],
    params: list[int],
    accs: list[float],
    path: str | Path,
    title: str = "Accuracy theo số tham số",
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for i, (n, p, a) in enumerate(zip(names, params, accs)):
        ax.scatter(p, a, s=60, color=COLORS[i % len(COLORS)], label=n, zorder=3)
        ax.annotate(n, (p, a), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("số tham số (tham số học được)")
    ax.set_ylabel("test accuracy")
    ax.set_xscale("log")
    ax.grid(alpha=0.3, which="both")
    fig.savefig(path)
    plt.close(fig)


def plot_attention(attn: torch.Tensor, path: str | Path, title: str = "", head: int | None = None) -> int:
    """Heatmap attention của một mẫu.

    attn: (H, N, N) hoặc (N, N). Nếu là (H, N, N) và `head=None`, tự chọn head có
    phân bố "nhọn" nhất (độ lệch chuẩn lớn nhất) — head 0 sau vài epoch thường
    gần như đều nên nhìn không có thông tin.

    Trả về chỉ số head đã vẽ.
    """
    a = attn.detach().cpu().float()
    if a.dim() == 3:
        h = int(a.std(dim=(1, 2)).argmax()) if head is None else head
    else:
        h = 0
        a = a.unsqueeze(0)
    mat = a[h].numpy()

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(mat, cmap="viridis")
    ax.set_xlabel("key token")
    ax.set_ylabel("query token")
    ax.grid(False)
    ax.set_title(title or f"Attention — head {h}", fontsize=9, pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
    cbar.update_ticks()

    fig.savefig(path)
    plt.close(fig)
    return h

