"""Huấn luyện Transformer nhỏ của E2 trên MNIST.

Ví dụ:
    python -m E2.train --tokenizer patch --msa manual --epochs 4
    python E2/train.py --tokenizer row --msa torch --epochs 4
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
from common.data import build_loaders, load_datasets
from common.engine import confusion_matrix, evaluate, fit
from common.utils import Timer, count_parameters, ensure_dir, get_device, save_json, set_seed
from E2.models import TinyViT
from E2.tokenizers import build_tokenizer

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E2: MSA tự viết vs PyTorch, nhiều cách tokenize")
    p.add_argument("--tokenizer", choices=["patch", "patch7", "row", "col", "cnnstem"], default="patch")
    p.add_argument("--msa", choices=["manual", "torch"], default="manual")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlp-ratio", type=float, default=2.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--save-attention", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


def run_name(tokenizer: str, msa: str) -> str:
    return f"{tokenizer}_{msa}"


def main(argv=None) -> dict:
    args = parse_args(argv)
    tag = args.tag or run_name(args.tokenizer, args.msa)
    out_dir = ensure_dir(args.out_dir or RESULTS_DIR / tag)

    set_seed(args.seed)
    device = get_device()
    loaders = build_loaders(data_dir=args.data_dir, batch_size=args.batch_size, seed=args.seed)

    tokenizer = build_tokenizer(args.tokenizer, dim=args.dim)
    model = TinyViT(
        tokenizer,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        use_torch_msa=(args.msa == "torch"),
    )
    n_params = count_parameters(model)
    print(
        f"[{tag}] tokens={tokenizer.num_tokens + 1} (kèm CLS) | dim={args.dim} depth={args.depth} "
        f"heads={args.heads} | tham số {n_params:,} | device {device}"
    )

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
        "exercise": "E2",
        "tokenizer": args.tokenizer,
        "msa": args.msa,
        "dim": args.dim,
        "depth": args.depth,
        "num_heads": args.heads,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "num_tokens": tokenizer.num_tokens + 1,
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
        viz.plot_history(history, out_dir / "curves.png", title=f"E2 {tag}")
        viz.plot_confusion(cm, [str(i) for i in range(10)], out_dir / "confusion.png", title=f"E2 {tag}")
        if args.tokenizer == "patch":
            _, _, _, test_plain = load_datasets(args.data_dir, seed=args.seed)
            viz.plot_error_samples(
                test_plain, test["preds"], test["labels"], out_dir / "errors.png", n=8, title=f"E2 {tag} — mẫu sai"
            )

    if args.save_attention:
        x, _ = next(iter(loaders["test"]))
        model.eval()
        with torch.no_grad():
            _, attn = model(x[:1].to(device), return_attn=True)
        if attn is not None:
            viz.plot_attention(attn[0, 0], out_dir / "attention.png", title=f"E2 {tag}: attention head 0")
            torch.save(attn[0].cpu(), out_dir / "attention.pt")

    print(f"[{tag}] test acc = {test['acc']:.4f} | val acc = {history['best_val_acc']:.4f} | {train_time:.1f}s")
    return metrics


if __name__ == "__main__":
    main()
