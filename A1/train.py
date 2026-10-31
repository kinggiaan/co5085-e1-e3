"""Huấn luyện một cấu hình A1 (CNN vs Transformer × chế độ freeze).

Ví dụ:
    python -m A1.train --arch resnet18 --mode scratch --epochs 20
    python -m A1.train --arch swin_t --mode finetune --epochs 8
    python A1/train.py --arch resnet18 --mode freeze_all --limit 2000   # chạy thử
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import viz
from common.engine import evaluate
from common.utils import ensure_dir, get_device, save_json, set_seed
from A1.data import TinyImageNetSplit, build_loaders
from A1.models import DEFAULT_LR, build_model, count_total, count_trainable, param_groups

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="A1: CNN vs Transformer trên Tiny-ImageNet")
    p.add_argument("--arch", choices=["resnet18", "swin_t"], default="resnet18")
    p.add_argument("--mode", choices=["scratch", "finetune", "freeze_all", "freeze_partial"], default="scratch")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--head-lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=None, help="chỉ dùng N ảnh train (để chạy thử)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--data-root", default="data/tiny-imagenet-200")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


DEFAULT_EPOCHS = {"scratch": 20, "finetune": 8, "freeze_all": 6, "freeze_partial": 8}
DEFAULT_BS = {"resnet18": 64, "swin_t": 64}
NUM_CLASSES = 200


def macro_f1(preds: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def per_class_recall(preds: np.ndarray, labels: np.ndarray, n_classes: int) -> list[float]:
    out = []
    for c in range(n_classes):
        mask = labels == c
        out.append(float((preds[mask] == c).mean()) if mask.any() else float("nan"))
    return out


def main(argv=None) -> dict:
    args = parse_args(argv)
    epochs = args.epochs or DEFAULT_EPOCHS[args.mode]
    bs = args.batch_size or DEFAULT_BS[args.arch]
    tag = f"{args.arch}_{args.mode}"
    out_dir = ensure_dir(args.out_dir or RESULTS_DIR / tag)

    set_seed(args.seed)
    device = get_device()
    loaders = build_loaders(args.data_root, batch_size=bs, img_size=args.img_size,
                            num_workers=args.num_workers, limit=args.limit, seed=args.seed)

    model = build_model(args.arch, args.mode)
    n_total, n_train = count_total(model), count_trainable(model)
    base_lr, head_lr = DEFAULT_LR[args.arch][args.mode]
    base_lr = args.lr or base_lr
    head_lr = args.head_lr or head_lr
    print(f"[{tag}] tham số {n_total:,} (trainable {n_train:,}) | bs {bs} | {args.img_size}px | "
          f"epoch {epochs} | lr {base_lr:g}/{head_lr} | device {device}", flush=True)

    model.to(device)
    optimizer = torch.optim.AdamW(param_groups(model, base_lr, head_lr), weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time_s": []}
    best_acc, best_state = -1.0, None
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running, correct, seen = 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            running += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
        scheduler.step()

        tr_loss, tr_acc = running / max(seen, 1), correct / max(seen, 1)
        va = evaluate(model, loaders["val"], criterion, device)
        dt = time.perf_counter() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])
        history["epoch_time_s"].append(dt)

        if va["acc"] > best_acc:
            best_acc = va["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:02d}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va['loss']:.4f} acc {va['acc']:.4f} | {dt:5.1f}s", flush=True)

    train_time = time.perf_counter() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate(model, loaders["val"], criterion, device, collect_predictions=True)
    preds, labels = test["preds"], test["labels"]
    rec = per_class_recall(preds, labels, NUM_CLASSES)

    config = {
        "exercise": "A1", "arch": args.arch, "mode": args.mode, "epochs": epochs,
        "batch_size": bs, "img_size": args.img_size, "lr": base_lr, "head_lr": head_lr,
        "weight_decay": args.weight_decay, "label_smoothing": args.label_smoothing,
        "limit": args.limit, "seed": args.seed,
    }
    metrics = {
        **config, "tag": tag,
        "num_params_total": n_total, "num_params_trainable": n_train,
        "train_time_s": round(train_time, 1),
        "sec_per_epoch": round(train_time / epochs, 1),
        "best_val_acc": best_acc,
        "test_acc": test["acc"],
        "test_loss": test["loss"],
        "macro_f1": macro_f1(preds, labels, NUM_CLASSES),
        "per_class_recall": rec,
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else None,
    }
    save_json(metrics, out_dir / "metrics.json")

    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_s"])
        for row in zip(history["epoch"], history["train_loss"], history["train_acc"],
                       history["val_loss"], history["val_acc"], history["epoch_time_s"]):
            w.writerow(row)

    torch.save({"state_dict": model.state_dict(), "config": config}, out_dir / "model.pt")

    if not args.no_plot:
        viz.plot_history(history, out_dir / "curves.png", title=f"A1 {tag}")
        plot_worst_classes(rec, out_dir / "worst_classes.png", tag)
        save_error_samples(args.data_root, preds, labels, out_dir / "errors.png", n=16)

    print(f"[{tag}] val acc = {test['acc']:.4f} | macro-F1 = {metrics['macro_f1']:.4f} | "
          f"{train_time/60:.1f} phút", flush=True)
    return metrics


def plot_worst_classes(rec: list[float], path: Path, tag: str, k: int = 15) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.array([r if not np.isnan(r) else 0.0 for r in rec])
    idx = np.argsort(arr)[:k][::-1]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh([f"class {i}" for i in idx], arr[idx], color="#d62728")
    ax.set_xlabel("recall")
    ax.set_title(f"A1 {tag} — {k} lớp kém nhất", fontsize=9)
    ax.grid(alpha=0.3, axis="x")
    fig.savefig(path)
    plt.close(fig)


def save_error_samples(root: str, preds: np.ndarray, labels: np.ndarray, path: Path, n: int = 16) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = TinyImageNetSplit(root, "val")
    wrong = np.flatnonzero(preds != labels)[:n]
    if len(wrong) == 0:
        return
    cols = 8
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.9 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, wrong):
        img, _ = ds[int(i)]
        ax.imshow(img)
        ax.set_title(f"T:{labels[i]} P:{preds[i]}", fontsize=6, color="#d62728")
        ax.axis("off")
    fig.suptitle("A1 — mẫu dự đoán sai", fontsize=9)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
