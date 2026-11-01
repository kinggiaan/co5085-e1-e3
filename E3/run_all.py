"""Chạy toàn bộ E3: LSTM vs GRU trên 3 cách biểu diễn chuỗi + so với baseline E1.

    python E3/run_all.py

Bao gồm một thí nghiệm nhỏ "phá thứ tự": lấy model rows/lstm đã huấn luyện và
đảo ngẫu nhiên thứ tự hàng ở đầu vào — nếu accuracy tụt mạnh thì mô hình thực sự
dựa vào phụ thuộc thời gian chứ không chỉ cộng các hàng lại.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import viz
from common.data import build_loaders
from common.engine import evaluate
from common.utils import ensure_dir, get_device, load_json, save_json, set_seed
from E3.train import RESULTS_DIR, main as train_main

REPRESENTATIONS = ["rows", "cols", "patches"]
CELLS = ["lstm", "gru"]


def read_history(path: Path) -> dict:
    h = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h["epoch"].append(int(row["epoch"]))
            for k in ("train_loss", "train_acc", "val_loss", "val_acc"):
                h[k].append(float(row[k]))
    return h


def evaluate_permuted_rows(run_dir: Path, data_dir: str = "data", seed: int = 123) -> float:
    """Đánh giá lại model rows/* khi thứ tự hàng bị đảo ngẫu nhiên."""
    from E3.models import ImageSequenceClassifier

    ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = ImageSequenceClassifier(
        representation=cfg["representation"],
        cell=cfg["cell"],
        hidden_size=cfg["hidden"],
        num_layers=cfg["num_layers"],
        bidirectional=cfg["bidirectional"],
        pooling=cfg["pooling"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    device = get_device()
    model.to(device)

    loaders = build_loaders(data_dir=data_dir, seed=42)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(28, generator=g)

    total, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in loaders["test"]:
            x = x[:, :, perm, :].to(device)  # đảo thứ tự hàng
            y = y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def load_baselines() -> list[dict]:
    baselines = []
    for name in ("mlp", "cnn"):
        p = Path(__file__).resolve().parents[1] / "E1" / "results" / name / "metrics.json"
        if p.exists():
            m = load_json(p)
            baselines.append(
                {
                    "model": name,
                    "num_params": m["num_params"],
                    "test_acc": m["test_acc"],
                    "best_val_acc": m["best_val_acc"],
                    "train_time_s": m["train_time_s"],
                    "sec_per_epoch": m["train_time_s"] / max(m["epochs"], 1),
                }
            )
        else:
            print(f"(chưa có E1/{name}, bỏ qua baseline này — chạy E1/run_all.py trước)")
    return baselines


def main() -> list[dict]:
    set_seed(42)
    results = []
    for rep in REPRESENTATIONS:
        for cell in CELLS:
            tag = f"{rep}_{cell}"
            met = RESULTS_DIR / tag / "metrics.json"
            if met.exists():
                print(f"\n=== E3 / {tag}: đã có kết quả, bỏ qua (xoá thư mục để chạy lại) ===")
                import json

                results.append(json.loads(met.read_text(encoding="utf-8")))
                continue
            print(f"\n=== E3 / {rep} / {cell} ===")
            results.append(train_main(["--repr", rep, "--cell", cell]))

    # --- hai biến thể mở rộng (không bắt buộc, ăn điểm thảo luận) ---
    extra_tag = RESULTS_DIR / "rows_lstm_bidir"
    if (extra_tag / "metrics.json").exists():
        pass
    else:
        print("\n=== E3 / rows / lstm / bidirectional (hidden 64) ===")
        results.append(train_main(["--repr", "rows", "--cell", "lstm", "--bidirectional",
                                   "--hidden", "64", "--tag", "rows_lstm_bidir"]))
    if not (RESULTS_DIR / "rows_gru_mean" / "metrics.json").exists():
        print("\n=== E3 / rows / gru / mean-pooling ===")
        results.append(train_main(["--repr", "rows", "--cell", "gru", "--pooling", "mean",
                                   "--tag", "rows_gru_mean"]))

    out = ensure_dir(RESULTS_DIR)
    save_json(results, out / "summary.json")

    # --- thí nghiệm phá thứ tự hàng ---
    order_probe = {}
    probe_path = RESULTS_DIR / "rows_lstm"
    if (probe_path / "model.pt").exists():
        acc_shuffled = evaluate_permuted_rows(probe_path)
        acc_original = next(r["test_acc"] for r in results if r["tag"] == "rows_lstm")
        order_probe = {
            "run": "rows_lstm",
            "test_acc_original": acc_original,
            "test_acc_shuffled_rows": acc_shuffled,
            "delta": acc_shuffled - acc_original,
        }
        save_json(order_probe, out / "order_probe.json")
        print(
            f"\nPhá thứ tự hàng (rows_lstm): gốc {acc_original:.4f} -> đảo hàng {acc_shuffled:.4f} "
            f"({order_probe['delta']:+.4f})"
        )

    baselines = load_baselines()

    lines = [
        "| Biểu diễn | Cell | Tham số | Val acc | Test acc | Thời gian (s) | s/epoch |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['representation']} | {r['cell']} | {r['num_params']:,} | {r['best_val_acc']:.4f} "
            f"| {r['test_acc']:.4f} | {r['train_time_s']:.1f} | {r['sec_per_epoch']:.1f} |"
        )
    for b in baselines:
        lines.append(
            f"| baseline {b['model']} | — | {b['num_params']:,} | {b['best_val_acc']:.4f} "
            f"| {b['test_acc']:.4f} | {b['train_time_s']:.1f} | {b['sec_per_epoch']:.1f} |"
        )
    table = "\n".join(lines)
    (out / "summary.md").write_text(
        "# E3 — LSTM/GRU trên chuỗi ảnh so với baseline MLP/CNN (MNIST)\n\n"
        "hidden=128, 1 lớp, 4 epoch. Baseline lấy từ E1 (cùng split).\n\n" + table + "\n",
        encoding="utf-8",
    )
    print("\n" + table)

    names = [r["tag"] for r in results] + [f"baseline_{b['model']}" for b in baselines]
    accs = [r["test_acc"] for r in results] + [b["test_acc"] for b in baselines]
    params = [r["num_params"] for r in results] + [b["num_params"] for b in baselines]
    viz.plot_bar_comparison(names, accs, out / "test_acc_bar.png", ylabel="test accuracy", title="E3 — LSTM/GRU vs baseline")
    viz.plot_tradeoff(names, params, accs, out / "acc_vs_params.png")

    histories = {r["tag"]: read_history(RESULTS_DIR / r["tag"] / "history.csv") for r in results}
    viz.plot_many_histories(histories, out / "val_acc_all.png", metric="val_acc")
    print(f"\nĐã lưu bảng + biểu đồ vào {out}")
    return results


if __name__ == "__main__":
    main()
