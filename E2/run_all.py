"""Chạy toàn bộ thí nghiệm E2: MSA tự viết vs PyTorch, so sánh các cách tokenize.

    python E2/run_all.py

Bước 1: kiểm tra MSA tự viết khớp với nn.MultiheadAttention (E2/test_msa.py).
Bước 2: 3 cách tokenize x 2 bản MSA = 6 lần huấn luyện.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import viz
from common.utils import ensure_dir, save_json
from E2.test_msa import main as equivalence_main
from E2.train import RESULTS_DIR, main as train_main

TOKENIZERS = ["patch", "row", "cnnstem"]
MSA_IMPLS = ["manual", "torch"]


def read_history(path: Path) -> dict:
    h = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h["epoch"].append(int(row["epoch"]))
            for k in ("train_loss", "train_acc", "val_loss", "val_acc"):
                h[k].append(float(row[k]))
    return h


def main() -> list[dict]:
    try:
        equivalence_main()
    except AssertionError as exc:  # pragma: no cover
        print("CẢNH BÁO: kiểm tra tương đương MSA thất bại:", exc)

    results = []
    for tok in TOKENIZERS:
        for msa in MSA_IMPLS:
            print(f"\n=== E2 / {tok} / {msa} ===")
            results.append(train_main(["--tokenizer", tok, "--msa", msa]))

    out = ensure_dir(RESULTS_DIR)
    save_json(results, out / "summary.json")

    lines = [
        "| Tokenize | MSA | Số token | Tham số | Val acc | Test acc | Thời gian (s) | s/epoch |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['tokenizer']} | {r['msa']} | {r['num_tokens']} | {r['num_params']:,} | "
            f"{r['best_val_acc']:.4f} | {r['test_acc']:.4f} | {r['train_time_s']:.1f} | {r['sec_per_epoch']:.1f} |"
        )
    table = "\n".join(lines)
    (out / "summary.md").write_text(
        "# E2 — MSA tự viết vs PyTorch, so sánh tokenization trên MNIST\n\n"
        f"Tất cả: dim=128, depth=2, heads=4, 4 epoch, 50k ảnh train.\n\n{table}\n",
        encoding="utf-8",
    )
    print("\n" + table)

    # So sánh MSA tự viết vs PyTorch theo từng cách tokenize
    manual_acc = [r["test_acc"] for r in results if r["msa"] == "manual"]
    torch_acc = [r["test_acc"] for r in results if r["msa"] == "torch"]
    manual_time = [r["sec_per_epoch"] for r in results if r["msa"] == "manual"]
    torch_time = [r["sec_per_epoch"] for r in results if r["msa"] == "torch"]

    viz.plot_bar_comparison(
        [f"{t}\nmanual" for t in TOKENIZERS] + [f"{t}\ntorch" for t in TOKENIZERS],
        manual_acc + torch_acc,
        out / "test_acc_by_msa.png",
        ylabel="test accuracy",
        title="E2 — MSA tự viết vs nn.MultiheadAttention",
    )
    viz.plot_bar_comparison(
        [f"{t}\nmanual" for t in TOKENIZERS] + [f"{t}\ntorch" for t in TOKENIZERS],
        manual_time + torch_time,
        out / "time_by_msa.png",
        ylabel="giây / epoch",
        title="E2 — thời gian huấn luyện mỗi epoch",
    )

    histories = {
        f"{r['tokenizer']}/{r['msa']}": read_history(RESULTS_DIR / f"{r['tokenizer']}_{r['msa']}" / "history.csv")
        for r in results
    }
    viz.plot_many_histories(histories, out / "val_acc_all.png", metric="val_acc")
    print(f"\nĐã lưu bảng + biểu đồ vào {out}")
    return results


if __name__ == "__main__":
    main()
