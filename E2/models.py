"""Transformer nhỏ (ViT-lite) cho ảnh 28x28: tokenizer -> [CLS] + pos-emb -> block -> head."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from E2.msa import ManualMSA, TorchMSA


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    """Pre-LN Transformer block: x -> x + MSA(LN(x)) -> x + MLP(LN(x))."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.1, use_torch_msa: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = (
            TorchMSA(dim, num_heads, dropout=dropout)
            if use_torch_msa
            else ManualMSA(dim, num_heads, attn_dropout=dropout, proj_dropout=dropout)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        h = self.norm1(x)
        if return_attn:
            attn_out, attn = self.attn(h, return_attn=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn
        x = x + self.attn(h)
        x = x + self.mlp(self.norm2(x))
        return x, None


class TinyViT(nn.Module):
    """Bộ phân loại Transformer nông cho ảnh nhỏ.

    tokenizer    : module trả về (B, n_tokens, dim), có thuộc tính num_tokens
    depth        : số block (2 cho "encoder nông")
    use_torch_msa: True -> dùng nn.MultiheadAttention, False -> MSA tự viết
    """

    def __init__(
        self,
        tokenizer: nn.Module,
        dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        num_classes: int = 10,
        dropout: float = 0.1,
        use_torch_msa: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.dim = dim
        self.num_tokens = tokenizer.num_tokens

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens + 1, dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                Block(dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout, use_torch_msa=use_torch_msa)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        tokens = self.tokenizer(x)                       # (B, N, dim)
        B = tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        tokens = self.pos_drop(tokens)

        attn0 = None
        for i, blk in enumerate(self.blocks):
            tokens, attn = blk(tokens, return_attn=return_attn and i == 0)
            if attn is not None:
                attn0 = attn

        logits = self.head(self.norm(tokens[:, 0]))
        if return_attn:
            return logits, attn0
        return logits
