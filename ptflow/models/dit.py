"""LightningDiT backbone, identical in structure and parameter names to W-Flow.

Parameter names must match the released W-Flow checkpoints exactly, so the
module layout below mirrors ``models/generator.py`` of github.com/hanjq17/W-Flow.
The only functional additions are ``forward_hidden`` (the trunk without the
readout, so several heads can share it) and ``unpatchify``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


def _sincos_1d(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def sincos_pos_embed_2d(embed_dim: int, grid_size: int) -> np.ndarray:
    grid = np.meshgrid(np.arange(grid_size, dtype=np.float32), np.arange(grid_size, dtype=np.float32))
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    return np.concatenate([_sincos_1d(embed_dim // 2, grid[0]), _sincos_1d(embed_dim // 2, grid[1])], axis=1)


class TorchLinear(nn.Module):
    """nn.Linear stored under ``.linear`` (W-Flow key layout)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, zero_init: bool = False, std: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if zero_init:
            nn.init.zeros_(self.linear.weight)
        elif std > 0:
            nn.init.normal_(self.linear.weight, std=std)
        else:
            nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(var + self.eps) * self.weight).to(dtype=x.dtype)


def modulate(x, shift, scale):
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden: int, inner: int):
        super().__init__()
        self.w1 = TorchLinear(hidden, inner)
        self.w3 = TorchLinear(hidden, inner)
        self.w2 = TorchLinear(inner, hidden)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class StandardMLP(nn.Module):
    def __init__(self, hidden: int, inner: int):
        super().__init__()
        self.fc1 = TorchLinear(hidden, inner)
        self.fc2 = TorchLinear(inner, hidden)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm, use_rmsnorm, use_rope, attn_fp32, max_seq_len):
        super().__init__()
        self.dim, self.num_heads = int(dim), int(num_heads)
        self.use_rope, self.attn_fp32 = bool(use_rope), bool(attn_fp32)
        head_dim = self.dim // self.num_heads
        self.qkv = TorchLinear(self.dim, 3 * self.dim)
        if qk_norm:
            norm = (lambda: RMSNorm(head_dim)) if use_rmsnorm else (lambda: nn.LayerNorm(head_dim, eps=1e-6))
            self.q_norm, self.k_norm = norm(), norm()
        else:
            self.q_norm = self.k_norm = None
        self.proj = TorchLinear(self.dim, self.dim)
        if self.use_rope:
            half = head_dim // 2
            freqs = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
            emb = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), freqs)
            emb = torch.cat([emb, emb], dim=-1)
            self.register_buffer("_rope_cos", torch.cos(emb)[None, :, None, :], persistent=False)
            self.register_buffer("_rope_sin", torch.sin(emb)[None, :, None, :], persistent=False)

    def forward(self, x):
        B, N, C = x.shape
        hd = C // self.num_heads
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, hd).unbind(2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.use_rope:
            dt = torch.float32 if self.attn_fp32 else q.dtype
            cos, sin = self._rope_cos[:, :N].to(dt), self._rope_sin[:, :N].to(dt)
            h = hd // 2
            q = q * cos + torch.cat([-q[..., h:], q[..., :h]], dim=-1) * sin
            k = k * cos + torch.cat([-k[..., h:], k[..., :h]], dim=-1) * sin
        if self.attn_fp32:
            q, k, v = q.float(), k.float(), v.float()
        q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))
        if self.attn_fp32:
            with torch.amp.autocast(device_type=q.device.type, enabled=False):
                out = F.scaled_dot_product_attention(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C).to(dtype=x.dtype)
        return self.proj(out)


class LightningDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio, use_qknorm, use_swiglu, use_rmsnorm, use_rope,
                 attn_fp32, max_seq_len):
        super().__init__()
        h = int(hidden_size)
        norm = (lambda: RMSNorm(h)) if use_rmsnorm else (lambda: nn.LayerNorm(h, eps=1e-6, elementwise_affine=False))
        self.norm1, self.norm2 = norm(), norm()
        self.attn = Attention(h, num_heads, use_qknorm, use_rmsnorm, use_rope, attn_fp32, max_seq_len)
        inner = int(h * float(mlp_ratio))
        if use_swiglu:
            inner = (int(2 / 3 * inner) + 31) // 32 * 32
            self.mlp = SwiGLUFFN(h, inner)
        else:
            self.mlp = StandardMLP(h, inner)
        self.adaLN_mod = nn.Sequential(nn.SiLU(), TorchLinear(h, 6 * h, zero_init=True))

    def forward(self, x, c):
        sm, cm, gm, sf, cf, gf = self.adaLN_mod(c.float()).to(dtype=x.dtype).chunk(6, dim=1)
        x = x + gm.unsqueeze(1) * self.attn(modulate(self.norm1(x), sm, cm))
        return x + gf.unsqueeze(1) * self.mlp(modulate(self.norm2(x), sf, cf))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm):
        super().__init__()
        h = int(hidden_size)
        self.norm_final = RMSNorm(h) if use_rmsnorm else nn.LayerNorm(h, eps=1e-6, elementwise_affine=False)
        self.adaLN_mod = nn.Sequential(nn.SiLU(), TorchLinear(h, 2 * h, zero_init=True))
        self.linear = TorchLinear(h, patch_size * patch_size * out_channels, zero_init=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_mod(c.float()).to(dtype=x.dtype).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class LightningDiT(nn.Module):
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_channels: int = 4,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        cond_dim: Optional[int] = None,
        n_cls_tokens: int = 0,
        attn_fp32: bool = True,
        use_remat: bool = False,
    ):
        super().__init__()
        self.input_size, self.patch_size = int(input_size), int(patch_size)
        self.in_channels, self.out_channels = int(in_channels), int(out_channels)
        self.hidden_size, self.n_cls_tokens = int(hidden_size), int(n_cls_tokens)
        self.use_remat = bool(use_remat)
        grid = self.input_size // self.patch_size
        max_seq_len = grid * grid + self.n_cls_tokens

        self.patch_embed = TorchLinear(self.patch_size**2 * self.in_channels, self.hidden_size)
        self.blocks = nn.ModuleList([
            LightningDiTBlock(hidden_size, num_heads, mlp_ratio, use_qknorm, use_swiglu, use_rmsnorm, use_rope,
                              attn_fp32, max_seq_len)
            for _ in range(int(depth))
        ])
        self.final_layer = FinalLayer(hidden_size, self.patch_size, self.out_channels, use_rmsnorm)
        pe = torch.as_tensor(sincos_pos_embed_2d(self.hidden_size, grid), dtype=torch.float32)
        self.pos_embed = nn.Parameter(pe.unsqueeze(0))
        if self.n_cls_tokens > 0:
            if cond_dim is None:
                raise ValueError("cond_dim must be set when n_cls_tokens > 0")
            self.cls_embed = nn.Parameter(torch.randn(1, self.n_cls_tokens, self.hidden_size) * 0.02)
            self.c_token_proj = TorchLinear(int(cond_dim), self.hidden_size)
        else:
            self.register_parameter("cls_embed", None)
            self.c_token_proj = None

    def forward_hidden(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: [B, H, W, C] -> tokens [B, n_cls + N, hidden] (before the readout)."""
        B, H, W, C = x.shape
        g, p = self.input_size // self.patch_size, H // (self.input_size // self.patch_size)
        x = x.reshape(B, g, p, g, p, C).permute(0, 1, 3, 2, 4, 5).reshape(B, g * g, p * p * C)
        x = self.patch_embed(x)
        x = x + self.pos_embed.to(dtype=x.dtype)
        if self.n_cls_tokens > 0:
            tok = self.c_token_proj(c).unsqueeze(1).repeat(1, self.n_cls_tokens, 1)
            x = torch.cat([tok + self.cls_embed.to(dtype=x.dtype), x], dim=1)
        for blk in self.blocks:
            if self.use_remat and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(blk, x, c, use_reentrant=False)
            else:
                x = blk(x, c)
        return x

    def unpatchify(self, tokens: torch.Tensor, channels: int) -> torch.Tensor:
        """Readout tokens [B, n_cls + N, p*p*channels] -> [B, H, W, channels]."""
        tokens = tokens[:, self.n_cls_tokens:, :]
        B, g, p = tokens.shape[0], self.input_size // self.patch_size, self.patch_size
        x = tokens.reshape(B, g, g, p, p, channels).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(B, self.input_size, self.input_size, channels)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.forward_hidden(x, c)
        return self.unpatchify(self.final_layer(h, c), self.out_channels)


class TimestepEmbedder(nn.Module):
    """Embeds the CFG scale (W-Flow conditions its generator on cfg = 1 + w)."""

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = int(freq_dim)
        self.fc1 = TorchLinear(self.freq_dim, hidden_size, std=0.02)
        self.fc2 = TorchLinear(hidden_size, hidden_size, std=0.02)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.fc2(F.silu(self.fc1(emb)))
