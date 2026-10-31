"""Chạy các cấu hình A2 trên VOC2012 và sinh bảng so sánh.

    python A2/run_all.py                        # U-Net + DeepLabV3 (hạn A2.2: 16/11)
    python A2/run_all.py --limit 200 --epochs 1  # chạy thử

Cặp so sánh chính: U-Net (CNN from-scratch) vs DeepLabV3-ResNet50 (CNN pretrained).
Thêm SegFormer-B0 (Transformer) nếu đã cài `transformers` — xem A2/segformer.py.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import ensure_dir, save_json
from A2.train import RESULTS_DIR, main as train_main

CONFIGS = ["unet", "deeplabv3"]


def read_history(path: Path) -> dict:
    h = {"epoch": [], "val_miou": [], "val_dice": []}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h["epoch"].append(int(row["epoch"]))
            h["val_miou"].append(float(row["val_miou"]))
            h["val_dice"].append(float(row["val_dice"]))
    return h


def main() -> list[dict]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    results = []
    for arch in CONFIGS:
        print(f"\n=== A2 / {arch} ===", flush=True)
        extra = []
        if args.limit:
            extra += ["--limit", str(args.limit)]
        if args.epochs:
            extra += ["--epochs", str(args.epochs)]
        results.append(train_main(["--arch", arch, *extra]))

    out = ensure_dir(RESULTS_DIR)
    save_json(results, out / "summary.json")

    lines = [
        "| Mô hình | Pretrain | Tham số | mIoU | Dice | Pixel acc | s/epoch | VRAM |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        pretrain = "có (COCO/VOC)" if r["arch"] == "deeplabv3" else "không"
        lines.append(
            f"| {r['arch']} | {pretrain} | {r['num_params']:,} | {r['miou']:.2%} | {r['dice']:.2%} "
            f"| {r['pixel_acc']:.2%} | {r['sec_per_epoch']:.0f} | {r['peak_vram_mb'] or 0:.0f} MB |"
        )
    table = "\n".join(lines)
    (out / "summary.md").write_text(
        "# A2 — segmentation trên Pascal VOC 2012\n\n"
        f"21 lớp, ảnh {results[0]['img_size']}px, {results[0]['epochs']} epoch.\n\n{table}\n",
        encoding="utf-8",
    )
    print("\n" + table)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for r in results:
        h = read_history(RESULTS_DIR / r["tag"] / "history.csv")
        axes[0].plot(h["epoch"], h["val_miou"], "-o", ms=3, label=r["arch"])
        axes[1].plot(h["epoch"], h["val_dice"], "-o", ms=3, label=r["arch"])
    for ax, name in zip(axes, ("mIoU", "Dice")):
        ax.set_xlabel("epoch"); ax.set_ylabel(name); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3)
    fig.savefig(out / "curves_all.png")
    plt.close(fig)
    print(f"\nĐã lưu bảng + biểu đồ vào {out}")
    return results


if __name__ == "__main__":
    main()
