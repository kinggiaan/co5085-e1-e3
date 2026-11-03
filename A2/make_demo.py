"""Tạo video demo ngắn cho A2: ảnh | ground truth | dự đoán, chạy qua vài ảnh val.

    python A2/make_demo.py            # dùng DeepLabV3 đã huấn luyện
    python A2/make_demo.py --arch segformer --out docs/a2_demo_segformer.mp4

Kết quả: `docs/a2_demo.mp4` (H.264) để nhúng vào trang GitHub Pages.
Mỗi khung hình giữ 0.8 s, có tiêu đề ghi mIoU của model và tên lớp ở chân ảnh.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from A2.data import CLASS_NAMES, NUM_CLASSES, build_loaders
from A2.models import build_model

ROOT = Path(__file__).resolve().parents[1]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def to_rgb(x: torch.Tensor) -> np.ndarray:
    return (x.cpu() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="deeplabv3")
    ap.add_argument("--n", type=int, default=8, help="số ảnh trong video")
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--out", default="docs/a2_demo.mp4")
    args = ap.parse_args()

    run = ROOT / "A2" / "results" / args.arch
    ckpt = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    model = build_model(args.arch)
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    miou = ckpt["config"].get("miou")
    if miou is None:
        import json

        miou = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["miou"]

    loaders = build_loaders(batch_size=args.n, img_size=ckpt["config"]["img_size"])
    x, y = next(iter(loaders["val"]))
    with torch.no_grad():
        logits = model(x.to(device))
        if logits.shape[-2:] != y.shape[-2:]:
            logits = torch.nn.functional.interpolate(logits, size=y.shape[-2:], mode="bilinear",
                                                     align_corners=False)
        pred = logits.argmax(1).cpu()

    tmp = ROOT / "_demo_frames"
    tmp.mkdir(exist_ok=True)
    frames = []
    cmap = plt.get_cmap("tab20")

    for i in range(min(args.n, x.size(0))):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
        axes[0].imshow(to_rgb(x[i]))
        axes[0].set_title("ảnh", fontsize=9)
        axes[1].imshow(y[i].numpy(), vmin=0, vmax=NUM_CLASSES - 1, cmap=cmap)
        axes[1].set_title("ground truth", fontsize=9)
        axes[2].imshow(pred[i].numpy(), vmin=0, vmax=NUM_CLASSES - 1, cmap=cmap)
        axes[2].set_title("dự đoán", fontsize=9)
        # liệt kê vài lớp có mặt trong dự đoán
        present = [int(c) for c in torch.unique(pred[i]) if 0 <= int(c) < NUM_CLASSES][:4]
        names = ", ".join(CLASS_NAMES[c] for c in present)
        fig.suptitle(f"A2 · {args.arch} · mIoU {miou:.1%} — {names}", fontsize=9)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        path = tmp / f"f{i:02d}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        frames.append(imageio.imread(path))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=args.fps, codec="libx264", quality=8, macro_block_size=1)
    for p in tmp.glob("*.png"):
        p.unlink()
    tmp.rmdir()
    print(f"đã ghi {out} ({out.stat().st_size / 1024:.0f} KB, {len(frames)} khung)")


if __name__ == "__main__":
    main()
