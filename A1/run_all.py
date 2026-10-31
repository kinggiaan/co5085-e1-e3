"""Chạy 6 cấu hình của A1 và sinh bảng so sánh.

    python A1/run_all.py                       # đủ 6 cấu hình (vài giờ trên GTX 1650)
    python A1/run_all.py --limit 5000 --epochs 1   # chạy thử nhanh

Mặc định dùng 50.000 ảnh train (subset lấy ngẫu nhiên theo seed) cho cả 6 cấu hình để so
sánh công bằng và vừa thời gian; ghi rõ trong báo cáo.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import viz
from common.utils import ensure_dir, save_json
from A1.train import RESULTS_DIR, main as train_main

CONFIGS = [
    ("resnet18", "scratch"),
    ("swin_t", "scratch"),
    ("resnet18", "finetune"),
    ("swin_t", "finetune"),
    ("resnet18", "freeze_all"),
    ("swin_t", "freeze_partial"),
]


def read_history(path: Path) -> dict:
    h = {"epoch": [], "val_loss": [], "val_acc": []}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h["epoch"].append(int(row["epoch"]))
            h["val_loss"].append(float(row["val_loss"]))
            h["val_acc"].append(float(row["val_acc"]))
    return h


def main() -> list[dict]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50000)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    results = []
    for arch, mode in CONFIGS:
        print(f"\n=== A1 / {arch} / {mode} ===", flush=True)
        extra = ["--limit", str(args.limit)] if args.limit else []
        if args.epochs:
            extra += ["--epochs", str(args.epochs)]
        results.append(train_main(["--arch", arch, "--mode", mode, *extra]))

    out = ensure_dir(RESULTS_DIR)
    save_json(results, out / "summary.json")

    lines = [
        "| Kiến trúc | Chế độ | Tham số trainable / tổng | Val acc | Macro-F1 | s/epoch | VRAM |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['arch']} | {r['mode']} | {r['num_params_trainable']:,} / {r['num_params_total']:,} "
            f"| {r['test_acc']:.2%} | {r['macro_f1']:.2%} | {r['sec_per_epoch']:.0f} "
            f"| {r['peak_vram_mb'] or 0:.0f} MB |"
        )
    table = "\n".join(lines)
    limit_txt = f"{results[0]['limit']:,}" if results[0].get("limit") else "toàn bộ 100k"
    (out / "summary.md").write_text(
        "# A1 — CNN vs Transformer trên Tiny-ImageNet\n\n"
        f"200 lớp, {limit_txt} ảnh train / 10k val, ảnh {results[0]['img_size']}px.\n\n{table}\n",
        encoding="utf-8",
    )
    print("\n" + table)

    viz.plot_bar_comparison(
        [f"{r['arch'].replace('_', '-')}\n{r['mode']}" for r in results],
        [r["test_acc"] for r in results],
        out / "val_acc_bar.png",
        ylabel="val accuracy",
        title="A1 — CNN vs Transformer theo chế độ huấn luyện",
    )
    viz.plot_tradeoff(
        [f"{r['arch']}/{r['mode']}" for r in results],
        [r["num_params_trainable"] for r in results],
        [r["test_acc"] for r in results],
        out / "acc_vs_trainable.png",
    )
    histories = {r["tag"]: read_history(RESULTS_DIR / r["tag"] / "history.csv") for r in results}
    viz.plot_many_histories(histories, out / "val_acc_all.png", metric="val_acc")
    print(f"\nĐã lưu bảng + biểu đồ vào {out}")
    return results


if __name__ == "__main__":
    main()
