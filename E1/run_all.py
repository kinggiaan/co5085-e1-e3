"""Chạy cả ba model của E1, so sánh và sinh bảng + biểu đồ tổng hợp.

    python E1/run_all.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import viz
from common.utils import ensure_dir, save_json
from E1.train import RESULTS_DIR, main as train_main

MODELS = ["softmax", "mlp", "cnn"]


def read_history(path: Path) -> dict:
    h = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h["epoch"].append(int(row["epoch"]))
            for k in ("train_loss", "train_acc", "val_loss", "val_acc"):
                h[k].append(float(row[k]))
    return h


def main() -> list[dict]:
    results = []
    for name in MODELS:
        print(f"\n=== E1 / {name} ===")
        results.append(train_main(["--model", name]))

    out = ensure_dir(RESULTS_DIR)
    save_json(results, out / "summary.json")

    # --- bảng markdown ---
    lines = [
        "| Model | Tham số | Val acc (best) | Test acc | Test loss | Thời gian (s) | s/epoch |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['num_params']:,} | {r['best_val_acc']:.4f} | {r['test_acc']:.4f} "
            f"| {r['test_loss']:.4f} | {r['train_time_s']:.1f} | {r['train_time_s'] / r['epochs']:.1f} |"
        )
    table = "\n".join(lines)
    (out / "summary.md").write_text(
        "# E1 — So sánh softmax / MLP / CNN trên MNIST\n\n"
        f"Số epoch: " + ", ".join(f"{r['model']}={r['epochs']}" for r in results) + "\n\n" + table + "\n",
        encoding="utf-8",
    )
    print("\n" + table)

    # --- biểu đồ tổng hợp ---
    histories = {r["model"]: read_history(RESULTS_DIR / r["model"] / "history.csv") for r in results}
    viz.plot_many_histories(histories, out / "val_acc_all.png", metric="val_acc")
    viz.plot_many_histories(histories, out / "val_loss_all.png", metric="val_loss")
    viz.plot_bar_comparison(
        [r["model"] for r in results],
        [r["test_acc"] for r in results],
        out / "test_acc_bar.png",
        ylabel="test accuracy",
        title="E1 — test accuracy",
    )
    viz.plot_tradeoff(
        [r["model"] for r in results],
        [r["num_params"] for r in results],
        [r["test_acc"] for r in results],
        out / "acc_vs_params.png",
    )
    print(f"\nĐã lưu bảng + biểu đồ vào {out}")
    return results


if __name__ == "__main__":
    main()
