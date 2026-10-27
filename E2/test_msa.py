"""Kiểm tra MSA tự viết cho kết quả trùng khớp với nn.MultiheadAttention.

Copy nguyên trọng số từ bản PyTorch sang bản tự viết (và ngược lại), chạy trên
cùng input rồi so sánh output + attention weights.

    python E2/test_msa.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from E2.msa import ManualMSA, TorchMSA, load_manual_weights_into_torch, load_torch_weights_into_manual


def check(dim: int, heads: int, n_tokens: int, batch: int, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    manual = ManualMSA(dim, heads).eval()
    torchmsa = TorchMSA(dim, heads).eval()
    load_manual_weights_into_torch(manual, torchmsa)

    x = torch.randn(batch, n_tokens, dim)
    with torch.no_grad():
        y_manual, attn_manual = manual(x, return_attn=True)
        y_torch, attn_torch = torchmsa(x, return_attn=True)

    out_diff = (y_manual - y_torch).abs().max().item()
    attn_diff = (attn_manual - attn_torch).abs().max().item()

    # kiểm tra chiều ngược lại (PyTorch -> tự viết)
    manual2 = ManualMSA(dim, heads).eval()
    load_torch_weights_into_manual(torchmsa, manual2)
    with torch.no_grad():
        y_manual2 = manual2(x)
    out_diff_rev = (y_manual2 - y_torch).abs().max().item()

    # kiểm tra với key_padding_mask
    mask = torch.zeros(batch, n_tokens, dtype=torch.bool)
    mask[:, -3:] = True
    with torch.no_grad():
        y_m_mask = manual(x, key_padding_mask=mask)
        y_t_mask = torchmsa(x, key_padding_mask=mask)
    mask_diff = (y_m_mask - y_t_mask).abs().max().item()

    return {
        "dim": dim,
        "heads": heads,
        "n_tokens": n_tokens,
        "batch": batch,
        "max_abs_diff_forward": out_diff,
        "max_abs_diff_attention": attn_diff,
        "max_abs_diff_reverse_transfer": out_diff_rev,
        "max_abs_diff_with_mask": mask_diff,
    }


def main() -> dict:
    # KHÔNG tắt gradient toàn cục ở đây: run_all.py gọi hàm này rồi huấn luyện tiếp
    # trong cùng process, tắt grad sẽ làm loss.backward() báo "does not require grad".
    configs = [
        dict(dim=128, heads=4, n_tokens=50, batch=8),
        dict(dim=64, heads=2, n_tokens=29, batch=4),
        dict(dim=96, heads=3, n_tokens=16, batch=2),
    ]
    results = [check(**c) for c in configs]
    worst = max(max(r["max_abs_diff_forward"], r["max_abs_diff_attention"]) for r in results)
    ok = worst < 1e-4

    out = {"all_close": ok, "worst_abs_diff": worst, "cases": results}
    path = Path(__file__).resolve().parent / "results" / "msa_equivalence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for r in results:
        print(
            f"dim={r['dim']:3d} heads={r['heads']} N={r['n_tokens']:2d} | "
            f"out diff {r['max_abs_diff_forward']:.2e} | attn diff {r['max_abs_diff_attention']:.2e} | "
            f"reverse {r['max_abs_diff_reverse_transfer']:.2e} | mask {r['max_abs_diff_with_mask']:.2e}"
        )
    print("=> KẾT LUẬN:", "PASS" if ok else "FAIL", f"(sai khác lớn nhất {worst:.2e})")
    assert ok, "ManualMSA không khớp với nn.MultiheadAttention"
    return out


if __name__ == "__main__":
    main()
