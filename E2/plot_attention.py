"""Vẽ bản đồ attention của một lần chạy E2 đã lưu checkpoint.

    python E2/plot_attention.py --run E2/results/patch_manual

Đọc `model.pt` trong thư mục run, lấy một ảnh test, vẽ attention của block đầu
và ghi `attention.png` (chọn head có phân bố nhọn nhất).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import viz
from common.data import build_loaders
from common.utils import get_device
from E2.models import TinyViT
from E2.tokenizers import build_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(Path(__file__).resolve().parent / "results" / "patch_manual"))
    ap.add_argument("--sample", type=int, default=0, help="lấy ảnh test thứ mấy")
    args = ap.parse_args()

    run_dir = Path(args.run)
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    tokenizer = build_tokenizer(cfg["tokenizer"], dim=cfg["dim"])
    model = TinyViT(
        tokenizer,
        dim=cfg["dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        dropout=cfg["dropout"],
        use_torch_msa=(cfg["msa"] == "torch"),
    )
    model.load_state_dict(ckpt["state_dict"])

    device = get_device()
    model.to(device).eval()
    x, y = next(iter(build_loaders(batch_size=1, seed=42)["test"]))
    x = x[args.sample : args.sample + 1] if x.size(0) > 1 else x
    with torch.no_grad():
        logits, attn = model(x.to(device), return_attn=True)

    head = viz.plot_attention(
        attn[0].cpu(),
        run_dir / "attention.png",
        title=f"{cfg['tokenizer']}/{cfg['msa']} — attention, lớp 1",
    )
    print(f"nhãn thật {int(y[0])}, dự đoán {int(logits.argmax(1)[0])}, head được vẽ: {head}")
    print(f"đã ghi {run_dir / 'attention.png'}")


if __name__ == "__main__":
    main()
