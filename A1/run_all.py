"""Chạy 6 cấu hình của A1 và sinh bảng so sánh.

    python A1/run_all.py                       # chạy các cấu hình còn thiếu
    python A1/run_all.py --force               # chạy lại tất cả
    python A1/run_all.py --only swin_t         # chỉ nhóm Swin
    python A1/run_all.py --limit 5000 --epochs 1   # chạy thử nhanh

Ghi chú về số epoch của Swin-T: đo thực tế **1.663 s/epoch** (27,7 phút) trên GTX 1650 ở 128px,
nên 20 epoch sẽ mất ~9 giờ cho riêng cấu hình scratch. Vì vậy các cấu hình Swin dùng 3 epoch —
đủ để thấy from-scratch Transformer không hội tụ trong ngân sách compute của nhóm, còn bản
pretrained thì có. Điều này được ghi rõ trong báo cáo, không giấu.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import viz
from common.utils import ensure_dir, save_json
from A1.train import RESULTS_DIR, main as train_main

# (arch, mode, epochs)
# Phương án A: bỏ swin_t/scratch (1.663 s/epoch -> 9 h cho 20 epoch, không khả thi trên
# GTX 1650). Đề mục 4.1 yêu cầu có cả CNN và Transformer -> Transformer vẫn có mặt ở
# dạng pretrained (finetune + freeze_partial). Đề mục 4.2 chỉ bắt buộc from-scratch
# cho "ít nhất một họ" -> ResNet-18 đảm nhiệm.
# Bổ sung resnet18/freeze_partial: đề mục 4.3 liệt kê ba kiểu đóng băng như MỘT bộ, nên
# ResNet-18 phải có đủ freeze toàn bộ backbone / freeze một phần / full finetune.
CONFIGS = [
    ("resnet18", "scratch", 20),
    ("resnet18", "finetune", 8),
    ("resnet18", "freeze_all", 6),
    ("resnet18", "freeze_partial", 6),
    ("swin_t", "finetune", 3),
    ("swin_t", "freeze_partial", 3),
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
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=None, help="ép cùng số epoch cho mọi cấu hình")
    ap.add_argument("--force", action="store_true", help="chạy lại cả cấu hình đã có kết quả")
    ap.add_argument("--only", default=None, help="chỉ chạy cấu hình khớp tiền tố, ví dụ swin_t")
    args = ap.parse_args()

    results = []
    for arch, mode, epochs in CONFIGS:
        tag = f"{arch}_{mode}"
        if args.only and not tag.startswith(args.only):
            continue
        met = RESULTS_DIR / tag / "metrics.json"
        if met.exists() and not args.force:
            print(f"\n=== A1 / {tag}: đã có kết quả, bỏ qua (--force để chạy lại) ===", flush=True)
            results.append(json.loads(met.read_text(encoding="utf-8")))
            continue

        print(f"\n=== A1 / {tag} ({args.epochs or epochs} epoch) ===", flush=True)
        extra = ["--limit", str(args.limit)] if args.limit else []
        extra += ["--epochs", str(args.epochs or epochs)]
        results.append(train_main(["--arch", arch, "--mode", mode, *extra]))

    out = ensure_dir(RESULTS_DIR)
    # gộp với kết quả đã có để `--only` không làm mất bảng tổng hợp
    merged = {}
    if (out / "summary.json").exists():
        for r in json.loads((out / "summary.json").read_text(encoding="utf-8")):
            merged[r["tag"]] = r
    for r in results:
        merged[r["tag"]] = r
    order = [f"{a}_{m}" for a, m, _ in CONFIGS]
    results = sorted(merged.values(), key=lambda r: order.index(r["tag"]) if r["tag"] in order else 99)
    save_json(results, out / "summary.json")

    lines = [
        "| Kiến trúc | Chế độ | Epoch | Tham số trainable / tổng | Val acc | Macro-F1 | s/epoch | VRAM |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['arch']} | {r['mode']} | {r['epochs']} | {r['num_params_trainable']:,} / "
            f"{r['num_params_total']:,} | {r['test_acc']:.2%} | {r['macro_f1']:.2%} "
            f"| {r['sec_per_epoch']:.0f} | {r['peak_vram_mb'] or 0:.0f} MB |"
        )
    table = "\n".join(lines)
    limit_txt = f"{results[0]['limit']:,}" if results and results[0].get("limit") else "toàn bộ 100k"
    (out / "summary.md").write_text(
        "# A1 — CNN vs Transformer trên Tiny-ImageNet\n\n"
        f"200 lớp, {limit_txt} ảnh train, val subset, ảnh {results[0]['img_size']}px.\n\n" + table + "\n",
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
    histories = {}
    for r in results:
        p = RESULTS_DIR / r["tag"] / "history.csv"
        if p.exists():
            histories[r["tag"]] = read_history(p)
    viz.plot_many_histories(histories, out / "val_acc_all.png", metric="val_acc")
    print(f"\nĐã lưu bảng + biểu đồ vào {out}")
    return results


if __name__ == "__main__":
    main()
