"""Smoke test: kiểm tra môi trường + shape + một vài bước huấn luyện cho E1/E2/E3.

Chạy:
    python scripts/smoke_test.py            # dùng GPU nếu có
    python scripts/smoke_test.py --cpu      # ép chạy CPU

Mục đích: phát hiện sai shape / lỗi import / CUDA không dùng được TRƯỚC khi
chạy cả một loạt thí nghiệm dài.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.data import build_loaders
from common.utils import count_parameters, get_device, set_seed


def bench(model: nn.Module, loader, device, steps: int = 5, warmup: int = 2) -> float:
    """Chạy `steps` batch (train) sau vài bước warm-up, trả về giây/batch."""
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    it = iter(loader)
    for _ in range(warmup):  # bỏ qua chi phí khởi tạo CUDA / cấp phát bộ nhớ
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        crit(model(x), y).backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        n += 1
        if n >= steps:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cpu") if args.cpu else get_device()

    print("=" * 68)
    print("torch        :", torch.__version__)
    print("cuda có sẵn  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU          :", torch.cuda.get_device_name(0))
        print("VRAM         :", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")
    print("device dùng  :", device)
    print("=" * 68)

    print("\n[1/4] Nạp MNIST (lần đầu sẽ tải ~11 MB)...")
    t0 = time.perf_counter()
    loaders = build_loaders(batch_size=64, seed=42)
    print(f"      train={len(loaders['train'].dataset):,} val={len(loaders['val'].dataset):,} "
          f"test={len(loaders['test'].dataset):,}  ({time.perf_counter() - t0:.1f}s)")

    from E1.models import build_model

    print("\n[2/4] E1 — softmax / mlp / cnn")
    for name in ("softmax", "mlp", "cnn"):
        m = build_model(name)
        dt = bench(m, loaders["train"], device, args.steps)
        print(f"      {name:8s} params={count_parameters(m):>8,} | {dt*1000:6.1f} ms/batch "
              f"| ~{dt*len(loaders['train']):5.1f} s/epoch (train)")

    from E2.models import TinyViT
    from E2.msa import ManualMSA, TorchMSA, load_manual_weights_into_torch
    from E2.tokenizers import build_tokenizer

    print("\n[3/4] E2 — TinyViT (patch tokens)")
    for msa in ("manual", "torch"):
        tok = build_tokenizer("patch", dim=128)
        m = TinyViT(tok, dim=128, depth=2, num_heads=4, use_torch_msa=(msa == "torch"))
        dt = bench(m, loaders["train"], device, args.steps)
        print(f"      msa={msa:6s} params={count_parameters(m):>8,} | {dt*1000:6.1f} ms/batch "
              f"| ~{dt*len(loaders['train']):5.1f} s/epoch")

    manual, tmsa = ManualMSA(64, 4).eval(), TorchMSA(64, 4).eval()
    load_manual_weights_into_torch(manual, tmsa)
    with torch.no_grad():
        x = torch.randn(4, 10, 64)
        diff = (manual(x) - tmsa(x)).abs().max().item()
    print(f"      MSA tự viết vs nn.MultiheadAttention: sai khác lớn nhất {diff:.2e} "
          f"-> {'OK' if diff < 1e-4 else 'SAI'}")

    from E3.models import ImageSequenceClassifier

    print("\n[4/4] E3 — LSTM/GRU trên chuỗi ảnh")
    for rep, cell in (("rows", "lstm"), ("patches", "gru")):
        m = ImageSequenceClassifier(representation=rep, cell=cell, hidden_size=128)
        dt = bench(m, loaders["train"], device, args.steps)
        print(f"      {rep:8s}/{cell:4s} params={count_parameters(m):>8,} | {dt*1000:6.1f} ms/batch "
              f"| ~{dt*len(loaders['train']):5.1f} s/epoch")

    print("\nTất cả model chạy được. Môi trường OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
