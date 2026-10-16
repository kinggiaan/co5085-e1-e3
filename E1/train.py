"""Huấn luyện một model của E1 và lưu kết quả.

Ví dụ:
    python -m E1.train --model cnn --epochs 5
    python E1/train.py --model softmax
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):  # cho phép chạy trực tiếp: python E1/train.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import viz
from common.data import build_loaders, load_datasets
from common.engine import confusion_matrix, evaluate, fit
from common.utils import Timer, count_parameters, ensure_dir, get_device, save_json, set_seed
from E1.models import build_model

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E1: huấn luyện softmax / MLP / CNN trên MNIST")
    p.add_argument("--model", choices=["softmax", "mlp", "cnn"], default="cnn")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


DEFAULTS = {
    "softmax": {"epochs": 8, "lr": 1e-3, "dropout": 0.0},
    "mlp": {"epochs": 8, "lr": 1e-3, "dropout": 0.2},
    "cnn": {"epochs": 5, "lr": 1e-3, "dropout": 0.3},
}


def main(argv=None) -> dict:
    args = parse_args(argv)
    epochs = args.epochs or DEFAULTS[args.model]["epochs"]
    lr = args.lr or DEFAULTS[args.model]["lr"]
    dropout = DEFAULTS[args.model]["dropout"] if args.dropout is None else args.dropout
    out_dir = ensure_dir(args.out_dir or RESULTS_DIR / args.model)

    set_seed(args.seed)
    device = get_device()
    loaders = build_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size, seed=args.seed
    )

    model = build_model(args.model, dropout=dropout) if args.model != "softmax" else build_model("softmax")
    n_params = count_parameters(model)

    print(f"[{args.model}] tham số: {n_params:,} | device: {device} | epochs: {epochs} | lr: {lr}")
    timer = Timer()
    history, _ = fit(
        model,
        loaders["train"],
        loaders["val"],
        epochs=epochs,
        lr=lr,
        weight_decay=args.weight_decay,
        device=device,
    )
    train_time = timer.elapsed()

    test = evaluate(model, loaders["test"], torch.nn.CrossEntropyLoss(), device, collect_predictions=True)
    cm = confusion_matrix(test["preds"], test["labels"])

    config = {
        "exercise": "E1",
        "model": args.model,
        "epochs": epochs,
        "lr": lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "dropout": dropout,
        "seed": args.seed,
    }
    metrics = {
        **config,
        "num_params": n_params,
        "train_time_s": round(train_time, 2),
        "best_val_acc": history["best_val_acc"],
        "best_epoch": history["best_epoch"],
        "test_loss": test["loss"],
        "test_acc": test["acc"],
        "confusion_matrix": cm.tolist(),
    }
    save_json(metrics, out_dir / "metrics.json")

    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_s"])
        for row in zip(
            history["epoch"],
            history["train_loss"],
            history["train_acc"],
            history["val_loss"],
            history["val_acc"],
            history["epoch_time_s"],
        ):
            w.writerow(row)

    torch.save({"state_dict": model.state_dict(), "config": config}, out_dir / "model.pt")

    if not args.no_plot:
        viz.plot_history(history, out_dir / "curves.png", title=f"E1 {args.model}")
        viz.plot_confusion(cm, [str(i) for i in range(10)], out_dir / "confusion.png", title=f"E1 {args.model}")
        _, _, test_clean, test_plain = load_datasets(args.data_dir, seed=args.seed)
        viz.plot_error_samples(
            test_plain,
            test["preds"],
            test["labels"],
            out_dir / "errors.png",
            n=16,
            title=f"E1 {args.model} — 16 mẫu dự đoán sai",
        )

    print(f"[{args.model}] test acc = {test['acc']:.4f} | val acc = {history['best_val_acc']:.4f} | {train_time:.1f}s")
    print(f"Kết quả lưu tại {out_dir}")
    return metrics


if __name__ == "__main__":
    main()
