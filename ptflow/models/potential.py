"""The data-side potential phi_theta(x, c) and its derivatives.

Parameterization: the network outputs phi_theta in physical (transport) units
and every equation uses u_theta = phi_theta / (2 eps) (paper, Sec. 2 and App. B).
The two are the same model; phi is chosen because it has a finite eps -> 0 limit
(the Brenier potential), so the network's target does not rescale as eps anneals,
and the prox condition  grad phi(y*) + y* - x0 = 0  is eps-free.

Precision: the importance weights depend on differences of u = phi/(2 eps) across
nearby proposal points, so the potential always runs in fp32 with TF32 disabled.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Tuple

import torch
import torch.nn as nn

from ptflow.models.dit import LightningDiT


@contextmanager
def fp32_matmul():
    """Disable TF32 inside the block (its 10-bit mantissa would swamp phi/(2 eps) differences)."""
    old_mm = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_mm
        torch.backends.cudnn.allow_tf32 = old_cudnn


@contextmanager
def frozen(module: nn.Module):
    """Build graphs through ``module`` without giving its parameters gradients."""
    flags = [(p, p.requires_grad) for p in module.parameters()]
    try:
        for p, _ in flags:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)


def _math_sdpa(enabled: bool):
    """Fused attention kernels lack double backward; HVPs need the math kernel."""
    if not enabled:
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel(SDPBackend.MATH)


class Potential(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        input_size: int = 32,
        in_channels: int = 4,
        patch_size: int = 2,
        hidden_size: int = 768,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        n_cls_tokens: int = 0,
        **unused,
    ):
        super().__init__()
        for k in ("init_from_generator", "use_bf16", "attn_fp32", "cond_dim", "use_remat"):
            unused.pop(k, None)
        if unused:
            raise ValueError(f"Unknown potential options: {sorted(unused)}")
        self.num_classes = int(num_classes)
        self.null_class = self.num_classes            # label used for the unconditional potential
        self.dim = int(input_size) ** 2 * int(in_channels)
        self.class_embed = nn.Embedding(self.num_classes + 1, int(hidden_size))
        nn.init.normal_(self.class_embed.weight, std=0.02)
        self.register_buffer("cond_bias", torch.zeros(int(hidden_size)))
        # Scalar field per latent position; the zero-init readout makes phi == 0,
        # so prox_phi = identity at initialization.
        self.trunk = LightningDiT(
            input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, out_channels=1, use_qknorm=use_qknorm,
            use_swiglu=use_swiglu, use_rope=use_rope, use_rmsnorm=use_rmsnorm, cond_dim=hidden_size,
            n_cls_tokens=n_cls_tokens, attn_fp32=True, use_remat=False,
        )

    def null_labels(self, c: torch.Tensor) -> torch.Tensor:
        return torch.full_like(c, self.null_class)

    def drop_labels(self, c: torch.Tensor, p: float, generator=None) -> torch.Tensor:
        if p <= 0:
            return c
        r = torch.rand(c.shape, generator=generator, device=c.device)
        return torch.where(r < p, self.null_labels(c), c)

    def phi(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """phi_theta(x, c): [B, H, W, C] -> [B], fp32."""
        with torch.autocast(device_type=x.device.type, enabled=False), fp32_matmul():
            cond = self.class_embed(c) + self.cond_bias
            field = self.trunk(x.float(), cond)
        return field.sum(dim=(1, 2, 3))

    def forward(self, x, c):
        return self.phi(x, c)


@torch.no_grad()
def init_potential_from_generator(pot: Potential, gen) -> dict:
    """Copy W-Flow's trunk (first ``depth`` blocks) into the potential; readout stays zero.

    The potential's conditioning bias reproduces the generator's conditioning at
    w = 0 (folded noise code + cfg embedding), so the copied blocks see inputs of
    the kind they were trained on.
    """
    src, dst = gen.model, pot.trunk
    if src.hidden_size != dst.hidden_size or src.n_cls_tokens != dst.n_cls_tokens:
        raise ValueError("init_from_generator needs the potential's hidden_size and n_cls_tokens to match the generator")
    if len(dst.blocks) > len(src.blocks):
        raise ValueError("Potential is deeper than the generator; cannot copy its trunk")
    dst.patch_embed.load_state_dict(src.patch_embed.state_dict())
    dst.pos_embed.copy_(src.pos_embed)
    if dst.n_cls_tokens > 0:
        dst.cls_embed.copy_(src.cls_embed)
        dst.c_token_proj.load_state_dict(src.c_token_proj.state_dict())
    for b_dst, b_src in zip(dst.blocks, src.blocks):
        b_dst.load_state_dict(b_src.state_dict())
    pot.class_embed.weight[: gen.num_classes].copy_(gen.class_embed.weight)
    pot.class_embed.weight[gen.num_classes].copy_(gen.class_embed.weight.mean(dim=0))
    c0 = torch.zeros(1, dtype=torch.long, device=gen.class_embed.weight.device)
    bias = gen.condition(c0, 0.0)[0] - gen.class_embed(c0)[0]
    pot.cond_bias.copy_(bias.to(pot.cond_bias))
    return {"copied_blocks": len(dst.blocks)}


# ---------------------------------------------------------------------------
# Derivatives.  All returned in phi-units (physical transport units).
# ---------------------------------------------------------------------------

def grad_phi(pot: Potential, x: torch.Tensor, c: torch.Tensor, *, create_graph: bool = False
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(grad_x phi(x, c), phi(x, c)).  Keeps x's upstream graph when create_graph=True."""
    xin = x.float()
    if not (create_graph and xin.requires_grad):
        xin = xin.detach().requires_grad_(True)
    with torch.enable_grad(), _math_sdpa(create_graph):
        phi = pot.phi(xin, c)
        (g,) = torch.autograd.grad(phi.sum(), xin, create_graph=create_graph)
    return g, (phi if create_graph else phi.detach())


def guided_grad(pot: Potential, x: torch.Tensor, c: torch.Tensor, w, *, create_graph: bool = False) -> torch.Tensor:
    """grad phi^w = (1 + w) grad phi(., c) - w grad phi(., null)   (eq. 17)."""
    w = torch.as_tensor(w, device=x.device, dtype=torch.float32)
    if w.ndim == 0:
        w = w.expand(x.shape[0])
    if not bool((w != 0).any()):
        return grad_phi(pot, x, c, create_graph=create_graph)[0]
    x = x.float()
    g2, _ = grad_phi(pot, torch.cat([x, x]), torch.cat([c, pot.null_labels(c)]), create_graph=create_graph)
    gc, gu = g2.chunk(2)
    wv = w.view(-1, *([1] * (x.ndim - 1)))
    return (1.0 + wv) * gc - wv * gu


def guided_hvp(pot: Potential, x: torch.Tensor, c: torch.Tensor, w, v: torch.Tensor) -> torch.Tensor:
    """Hessian-vector product  grad^2 phi^w(x) v  at a detached x (no parameter gradients)."""
    xin = x.detach().float().requires_grad_(True)
    with torch.enable_grad(), frozen(pot):
        g = guided_grad(pot, xin, c, w, create_graph=True)
        with _math_sdpa(True):
            (hv,) = torch.autograd.grad((g * v).sum(), xin)
    return hv.detach()


def prox_residual(pot: Potential, m: torch.Tensor, x0: torch.Tensor, c: torch.Tensor, w=0.0, *,
                  create_graph: bool = False) -> torch.Tensor:
    """r = grad phi^w(m) + m - x0: the first-order prox condition (eq. 16), in phi-units."""
    return guided_grad(pot, m, c, w, create_graph=create_graph) + m.float() - x0.float()


def prox_energy(pot: Potential, y: torch.Tensor, x0: torch.Tensor, c: torch.Tensor, w=0.0) -> torch.Tensor:
    """F_x0(y) = phi^w(y) + |y - x0|^2 / 2, whose minimizer is prox_{phi^w}(x0)."""
    w = torch.as_tensor(w, device=y.device, dtype=torch.float32)
    if w.ndim == 0:
        w = w.expand(y.shape[0])
    phi = pot.phi(y, c).double()
    if bool((w != 0).any()):
        phi = (1.0 + w.double()) * phi - w.double() * pot.phi(y, pot.null_labels(c)).double()
    return phi + 0.5 * (y.double() - x0.double()).flatten(1).square().sum(1)
