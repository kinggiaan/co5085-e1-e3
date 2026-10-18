"""Multi-Head Self-Attention (MSA) — bản tự viết và bản PyTorch đối chiếu.

`ManualMSA` chỉ dùng các phép toán cơ bản: Linear (chiếu Q/K/V), nhân ma trận
(`@` / `einsum`), softmax, và ghép các head bằng reshape/transpose.
Không dùng `nn.MultiheadAttention` hay `nn.TransformerEncoder*`.

`TorchMSA` bọc `nn.MultiheadAttention` để làm bản đối chiếu.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ManualMSA(nn.Module):
    """Multi-head self-attention viết tay.

    Tham số:
        dim        : số chiều của token (model dim)
        num_heads  : số head; yêu cầu dim % num_heads == 0
        attn_dropout / proj_dropout : dropout trên attention weights / trên output

    Forward:
        x            : (B, N, C)
        key_padding_mask : (B, N) bool, True = bị mask (không attend tới)
    Trả về:
        out          : (B, N, C)
        attn (nếu return_attn=True) : (B, H, N, N) sau softmax
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} phải chia hết cho num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None, return_attn: bool = False):
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        # 1) chiếu tuyến tính rồi tách thành H head: (B, H, N, Dh)
        q = self.q_proj(x).reshape(B, N, H, D).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, H, D).transpose(1, 2)

        # 2) điểm tương đồng đã scale: (B, H, N, N)
        attn = torch.einsum("bhnd,bhmd->bhnm", q, k) * self.scale

        # 3) mask (tuỳ chọn) + softmax theo trục key
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].bool()  # (B,1,1,N)
            attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 4) tổng có trọng số của value và ghép head: (B, N, C)
        out = torch.einsum("bhnm,bhmd->bhnd", attn, v)
        out = out.transpose(1, 2).reshape(B, N, C)

        # 5) chiếu ra output
        out = self.proj_drop(self.out_proj(out))
        if return_attn:
            return out, attn
        return out


class TorchMSA(nn.Module):
    """Bản đối chiếu: dùng nn.MultiheadAttention có sẵn của PyTorch."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None, return_attn: bool = False):
        out, weights = self.mha(
            x, x, x, key_padding_mask=key_padding_mask, need_weights=return_attn, average_attn_weights=False
        )
        if return_attn:
            return out, weights
        return out


def load_manual_weights_into_torch(manual: ManualMSA, torch_msa: TorchMSA) -> None:
    """Copy trọng số ManualMSA -> TorchMSA để kiểm tra hai bản cho cùng kết quả."""
    with torch.no_grad():
        w = torch.cat([manual.q_proj.weight, manual.k_proj.weight, manual.v_proj.weight], dim=0)
        b = torch.cat([manual.q_proj.bias, manual.k_proj.bias, manual.v_proj.bias], dim=0)
        torch_msa.mha.in_proj_weight.copy_(w)
        torch_msa.mha.in_proj_bias.copy_(b)
        torch_msa.mha.out_proj.weight.copy_(manual.out_proj.weight)
        torch_msa.mha.out_proj.bias.copy_(manual.out_proj.bias)


def load_torch_weights_into_manual(torch_msa: TorchMSA, manual: ManualMSA) -> None:
    with torch.no_grad():
        w, b = torch_msa.mha.in_proj_weight, torch_msa.mha.in_proj_bias
        d = manual.dim
        manual.q_proj.weight.copy_(w[0:d])
        manual.k_proj.weight.copy_(w[d : 2 * d])
        manual.v_proj.weight.copy_(w[2 * d :])
        manual.q_proj.bias.copy_(b[0:d])
        manual.k_proj.bias.copy_(b[d : 2 * d])
        manual.v_proj.bias.copy_(b[2 * d :])
        manual.out_proj.weight.copy_(torch_msa.mha.out_proj.weight)
        manual.out_proj.bias.copy_(torch_msa.mha.out_proj.bias)
