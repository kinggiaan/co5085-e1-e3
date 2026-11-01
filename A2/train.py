"""Huấn luyện một cấu hình segmentation của A2.

    python -m A2.train --arch unet --epochs 30
    python -m A2.train --arch deeplabv3 --epochs 10 --lr 1e-4
    python -m A2.train --arch segformer --epochs 8
    python A2/train.py --arch unet --limit 100 --epochs 1     # chạy thử
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

from common.utils import ensure_dir, get_device, save_json, set_seed
from A2.data import IGNORE_INDEX, NUM_CLASSES, build_loaders
from A2.metrics import SegMetric
from A2.models import DEFAULT_LR, build_model

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="A2: segmentation trên VOC2012")
    p.add_argument("--arch", choices=["unet", "unet_small", "deeplabv3", "segformer"], default="unet")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--img-size", type=int, default=320)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--limit", type=int, default=None, help="chỉ dùng N ảnh train (chạy thử)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--data-root", default="data/VOC2012")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


DEFAULT_EPOCHS = {"unet": 30, "unet_small": 30, "deeplabv3": 10, "segformer": 8}
DEFAULT_BS = {"unet": 8, "unet_small": 16, "deeplabv3": 4, "segformer": 4}


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    metric = SegMetric(NUM_CLASSES)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if logits.shape[-2:] != y.shape[-2:]:
            logits = nn.functional.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
        metric.update(logits.argmax(1), y)
    return metric.compute()


def main(argv=None) -> dict:
    args = parse_args(argv)
    epochs = args.epochs or DEFAULT_EPOCHS[args.arch]
    bs = args.batch_size or DEFAULT_BS[args.arch]
    lr = args.lr or DEFAULT_LR[args.arch]
    tag = args.arch
    out_dir = ensure_dir(args.out_dir or RESULTS_DIR / tag)

    set_seed(args.seed)
    device = get_device()
    loaders = build_loaders(args.data_root, batch_size=bs, img_size=args.img_size,
                            num_workers=args.num_workers, limit=args.limit, seed=args.seed)

    model = build_model(args.arch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{tag}] tham số {n_params:,} (trainable {n_train:,}) | {args.img_size}px | bs {bs} "
          f"| epoch {epochs} | lr {lr:g} | device {device}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs, power=0.9)

    history = {"epoch": [], "train_loss": [], "val_miou": [], "val_dice": [], "val_pixel_acc": [], "epoch_time_s": []}
    best_miou, best_state = -1.0, None
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running, seen = 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            if logits.shape[-2:] != y.shape[-2:]:
                logits = nn.functional.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
        scheduler.step()

        val = evaluate(model, loaders["val"], device)
        dt = time.perf_counter() - t0
        history["epoch"].append(epoch)
        history["train_loss"].append(running / max(seen, 1))
        history["val_miou"].append(val["miou"])
        history["val_dice"].append(val["dice"])
        history["val_pixel_acc"].append(val["pixel_acc"])
        history["epoch_time_s"].append(dt)

        if val["miou"] > best_miou:
            best_miou = val["miou"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:02d}/{epochs} | loss {running/max(seen,1):.4f} | mIoU {val['miou']:.4f} "
              f"| Dice {val['dice']:.4f} | pixel acc {val['pixel_acc']:.4f} | {dt:5.1f}s", flush=True)

    train_time = time.perf_counter() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)
    final = evaluate(model, loaders["val"], device)

    config = {"exercise": "A2", "arch": args.arch, "epochs": epochs, "img_size": args.img_size,
              "batch_size": bs, "lr": lr, "weight_decay": args.weight_decay,
              "limit": args.limit, "seed": args.seed, "dataset": "VOC2012-seg"}
    metrics = {
        **config, "tag": tag,
        "num_params": n_params, "num_params_trainable": n_train,
        "train_time_s": round(train_time, 1),
        "sec_per_epoch": round(train_time / epochs, 1),
        "best_val_miou": best_miou,
        "miou": final["miou"], "dice": final["dice"], "pixel_acc": final["pixel_acc"],
        "per_class_iou": final["per_class_iou"],
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else None,
    }
    save_json(metrics, out_dir / "metrics.json")

    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_miou", "val_dice", "val_pixel_acc", "epoch_time_s"])
        for row in zip(history["epoch"], history["train_loss"], history["val_miou"], history["val_dice"],
                       history["val_pixel_acc"], history["epoch_time_s"]):
            w.writerow(row)

    torch.save({"state_dict": model.state_dict(), "config": config}, out_dir / "model.pt")

    if not args.no_plot:
        plot_curves(history, out_dir / "curves.png", tag)
        plot_pred_samples(model, loaders["val"], device, out_dir / "samples.png", tag)

    print(f"[{tag}] mIoU = {final['miou']:.4f} | Dice = {final['dice']:.4f} | {train_time/60:.1f} phút")
    return metrics


def plot_curves(history: dict, path: Path, tag: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(history["epoch"], history["train_loss"], color="#1f77b4")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("train loss", fontsize=9)
    axes[1].plot(history["epoch"], history["val_miou"], label="mIoU", color="#2ca02c")
    axes[1].plot(history["epoch"], history["val_dice"], label="Dice", color="#d62728")
    axes[1].set_xlabel("epoch")
    axes[1].set_title("validation", fontsize=9)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"A2 {tag}", fontsize=10)
    fig.savefig(path)
    plt.close(fig)


def plot_pred_samples(model, loader, device, path: Path, tag: str, n: int = 4) -> None:
    """Ảnh gốc | ground truth | dự đoán, cho vài mẫu."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    x, y = next(iter(loader))
    x, y = x[:n].to(device), y[:n]
    with torch.no_grad():
        logits = model(x)
        if logits.shape[-2:] != y.shape[-2:]:
            logits = nn.functional.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1).cpu()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.1 * n))
    axes = np.atleast_2d(axes)
    for i in range(n):
        img = (x[i].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("ảnh", fontsize=8)
        axes[i, 1].imshow(y[i].numpy(), vmin=0, vmax=20, cmap="tab20")
        axes[i, 1].set_title("ground truth", fontsize=8)
        axes[i, 2].imshow(pred[i].numpy(), vmin=0, vmax=20, cmap="tab20")
        axes[i, 2].set_title("dự đoán", fontsize=8)
        for ax in axes[i]:
            ax.axis("off")
    fig.suptitle(f"A2 {tag} — so sánh mặt nạ", fontsize=10)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
