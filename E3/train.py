"""Huấn luyện LSTM/GRU trên chuỗi ảnh (E3).

Ví dụ:
    python -m E3.train --repr rows --cell lstm --epochs 4
    python E3/train.py --repr patches --cell gru --epochs 4
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import viz
from common.data import build_loaders
from common.engine import confusion_matrix, evaluate, fit
from common.utils import Timer, count_parameters, ensure_dir, get_device, save_json, set_seed
from E3.models import ImageSequenceClassifier

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E3: LSTM/GRU trên chuỗi ảnh MNIST")
    p.add_argument("--repr", dest="representation", choices=["rows", "cols", "patches"], default="rows")
    p.add_argument("--cell", choices=["lstm", "gru"], default="lstm")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--pooling", choices=["last", "mean"], default="last")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    tag = args.tag or f"{args.representation}_{args.cell}"
    out_dir = ensure_dir(args.out_dir or RESULTS_DIR / tag)

    set_seed(args.seed)
    device = get_device()
    loaders = build_loaders(data_dir=args.data_dir, batch_size=args.batch_size, seed=args.seed)

    model = ImageSequenceClassifier(
        representation=args.representation,
        cell=args.cell,
        hidden_size=args.hidden,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        pooling=args.pooling,
        dropout=args.dropout,
    )
    n_params = count_parameters(model)
    print(f"[{tag}] tham số {n_params:,} | device {device}")

    timer = Timer()
    history, _ = fit(
        model,
        loaders["train"],
        loaders["val"],
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        grad_clip=1.0,
    )
    train_time = timer.elapsed()

    test = evaluate(model, loaders["test"], torch.nn.CrossEntropyLoss(), device, collect_predictions=True)
    cm = confusion_matrix(test["preds"], test["labels"])

    config = {
        "exercise": "E3",
        "representation": args.representation,
        "cell": args.cell,
        "hidden": args.hidden,
        "num_layers": args.num_layers,
        "bidirectional": args.bidirectional,
        "pooling": args.pooling,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    metrics = {
        **config,
        "tag": tag,
        "num_params": n_params,
        "train_time_s": round(train_time, 2),
        "sec_per_epoch": round(train_time / max(len(history["epoch"]), 1), 2),
        "best_val_acc": history["best_val_acc"],
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
        viz.plot_history(history, out_dir / "curves.png", title=f"E3 {tag}")
        viz.plot_confusion(cm, [str(i) for i in range(10)], out_dir / "confusion.png", title=f"E3 {tag}")

    print(f"[{tag}] test acc = {test['acc']:.4f} | val acc = {history['best_val_acc']:.4f} | {train_time:.1f}s")
    return metrics


if __name__ == "__main__":
    main()
