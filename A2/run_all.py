"""Chạy các cấu hình A2 trên VOC2012 và sinh bảng so sánh.

    python A2/run_all.py                        # U-Net + DeepLabV3 + SegFormer-B0
    python A2/run_all.py --limit 200 --epochs 1  # chạy thử
    python A2/run_all.py --only unet             # chỉ một cấu hình

Ba biến thể tách hai yếu tố thường bị trộn: kiến trúc (CNN vs Transformer) và pretrain.
- unet       : CNN tự viết, train từ đầu
- deeplabv3  : CNN pretrained (COCO/VOC, 21 lớp có sẵn)
- segformer  : Transformer đúng theo paper đang phân tích (ImageNet-1k + ADE20K)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import ensure_dir, save_json
from A2.train import RESULTS_DIR, main as train_main

CONFIGS = ["unet", "deeplabv3", "segformer"]
PRETRAIN = {"unet": "không", "deeplabv3": "COCO/VOC", "segformer": "ImageNet + ADE20K"}
# Ngân sách GPU thật (GTX 1650 4 GB, laptop throttle còn ~1.2 GHz khi nóng):
# ảnh 256px, U-Net 20 epoch (~150 s/epoch), DeepLab 5 (~5 phút/epoch), SegFormer 4.
IMG_SIZE = 256
EPOCHS = {"unet": 20, "deeplabv3": 5, "segformer": 4}


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
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    results = []
    for arch in CONFIGS:
        if args.only and arch != args.only:
            continue
        met = RESULTS_DIR / arch / "metrics.json"
        if met.exists() and not args.force:
            print(f"\n=== A2 / {arch}: đã có kết quả, bỏ qua (--force để chạy lại) ===", flush=True)
            results.append(json.loads(met.read_text(encoding="utf-8")))
            continue

        print(f"\n=== A2 / {arch} ({args.epochs or EPOCHS[arch]} epoch, {IMG_SIZE}px) ===", flush=True)
        extra = ["--img-size", str(IMG_SIZE)]
        if args.limit:
            extra += ["--limit", str(args.limit)]
        extra += ["--epochs", str(args.epochs or EPOCHS[arch])]
        results.append(train_main(["--arch", arch, *extra]))

    out = ensure_dir(RESULTS_DIR)
    # gộp với kết quả đã có để chạy --only từng mô hình mà không làm mất bảng tổng hợp
    merged = {}
    if (out / "summary.json").exists():
        for r in json.loads((out / "summary.json").read_text(encoding="utf-8")):
            merged[r["tag"]] = r
    for r in results:
        merged[r["tag"]] = r
    results = sorted(merged.values(), key=lambda r: CONFIGS.index(r["tag"]) if r["tag"] in CONFIGS else 99)
    save_json(results, out / "summary.json")

    lines = [
        "| Mô hình | Pretrain | Tham số | mIoU | Dice | Pixel acc | s/epoch | VRAM |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['arch']} | {PRETRAIN.get(r['arch'], '?')} | {r['num_params']:,} | {r['miou']:.2%} "
            f"| {r['dice']:.2%} | {r['pixel_acc']:.2%} | {r['sec_per_epoch']:.0f} "
            f"| {r['peak_vram_mb'] or 0:.0f} MB |"
        )
    table = "\n".join(lines)
    img = results[0]["img_size"] if results else 320
    (out / "summary.md").write_text(
        "# A2 — segmentation trên Pascal VOC 2012\n\n"
        f"21 lớp, ảnh {img}px.\n\n{table}\n",
        encoding="utf-8",
    )
    print("\n" + table)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for r in results:
        p = RESULTS_DIR / r["tag"] / "history.csv"
        if not p.exists():
            continue
        h = read_history(p)
        axes[0].plot(h["epoch"], h["val_miou"], "-o", ms=3, label=r["arch"])
        axes[1].plot(h["epoch"], h["val_dice"], "-o", ms=3, label=r["arch"])
    for ax, name in zip(axes, ("mIoU", "Dice")):
        ax.set_xlabel("epoch")
        ax.set_ylabel(name)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)
    fig.savefig(out / "curves_all.png")
    plt.close(fig)
    print(f"\nĐã lưu bảng + biểu đồ vào {out}")
    return results


if __name__ == "__main__":
    main()
