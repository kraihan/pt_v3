"""The PT-Flow generator / proposal pair (m_eta, s_eta), initialized from W-Flow.

Paper, Sec. 3.6-3.7: one network gives the proposal mean m_eta(x0, c, w), which
is also the one-step generator (T = prox = m), and a diagonal log-scale head
s_eta(x0, c, w) implementing S in eq. 10.  Here both heads share the W-Flow
LightningDiT trunk.  The mean head is W-Flow's own readout, so at step 0
m_eta is exactly the pretrained W-Flow sampler; the scale head is zero-initialized
(S = I).

Deterministic generator.  W-Flow also feeds 32 random discrete "noise codes"
(64 classes each) into its conditioning, so one x0 maps to many outputs.  The
theory needs a single map x0 -> prox(x0), so the codes are folded into one
fixed conditioning bias (``code_bias``) when loading W-Flow weights.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ptflow.models.dit import FinalLayer, LightningDiT, RMSNorm, TimestepEmbedder


class Generator(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        num_classes: int = 1000,
        input_size: int = 32,
        in_channels: int = 4,
        out_channels: int = 4,
        patch_size: int = 2,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        n_cls_tokens: int = 0,
        use_bf16: bool = True,
        attn_fp32: bool = False,
        use_remat: bool = False,
        scale_max: float = 3.0,
        **unused,
    ):
        super().__init__()
        unused.pop("noise_classes", None)
        unused.pop("noise_coords", None)
        if unused:
            raise ValueError(f"Unknown generator options: {sorted(unused)}")
        if out_channels != in_channels:
            raise ValueError("The prox map acts on the latent space: out_channels must equal in_channels")
        self.num_classes, self.cond_dim = int(num_classes), int(cond_dim)
        self.input_size, self.channels = int(input_size), int(in_channels)
        self.use_bf16, self.scale_max = bool(use_bf16), float(scale_max)

        self.class_embed = nn.Embedding(self.num_classes, self.cond_dim)
        nn.init.normal_(self.class_embed.weight, std=0.02)
        self.cfg_embedder = TimestepEmbedder(self.cond_dim)
        self.cfg_norm = RMSNorm(self.cond_dim)
        self.register_buffer("code_bias", torch.zeros(self.cond_dim))
        self.model = LightningDiT(
            input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size,
            depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, out_channels=out_channels,
            use_qknorm=use_qknorm, use_swiglu=use_swiglu, use_rope=use_rope, use_rmsnorm=use_rmsnorm,
            cond_dim=cond_dim, n_cls_tokens=n_cls_tokens, attn_fp32=attn_fp32, use_remat=use_remat,
        )
        # s_eta: second readout on the shared trunk, zero-init => S = I at step 0.
        self.scale_layer = FinalLayer(hidden_size, patch_size, in_channels, use_rmsnorm)

    @property
    def dim(self) -> int:
        return self.input_size * self.input_size * self.channels

    def _autocast(self, device: torch.device):
        if self.use_bf16 and device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def condition(self, c: torch.Tensor, w, code_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Conditioning vector; W-Flow's cfg input is the guidance weight plus one."""
        w = torch.as_tensor(w, device=c.device, dtype=torch.float32)
        if w.ndim == 0:
            w = w.expand(c.shape[0])
        e = self.cfg_norm(self.cfg_embedder(1.0 + w).float())
        bias = self.code_bias if code_bias is None else code_bias
        return self.class_embed(c) + bias + 0.02 * e.to(self.class_embed.weight.dtype)

    def forward(self, x0: torch.Tensor, c: torch.Tensor, w=0.0, *, with_scale: bool = True,
                code_bias: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """x0: [B, H, W, C] noise.  Returns (m, s), both fp32 [B, H, W, C]; s is None if not requested."""
        with self._autocast(x0.device):
            cond = self.condition(c, w, code_bias)
            h = self.model.forward_hidden(x0, cond)
            m = self.model.unpatchify(self.model.final_layer(h, cond), self.channels).float()
            s = None
            if with_scale:
                raw = self.model.unpatchify(self.scale_layer(h, cond), self.channels).float()
                s = self.scale_max * torch.tanh(raw / self.scale_max)
        return m, s


# ---------------------------------------------------------------------------
# W-Flow checkpoint loading
# ---------------------------------------------------------------------------

def _canonical(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        for prefix in ("module.", "_orig_mod."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        out[k.replace("._orig_mod.", ".")] = v
    return out


def read_wflow_state(path: str, weights: str = "ema") -> Tuple[Dict[str, torch.Tensor], int]:
    """Read generator weights from a W-Flow ``state_XXXXXXXX.pt`` (EMA by default)."""
    path = str(Path(path).expanduser())
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    key = {"ema": "ema_model", "raw": "model"}[weights]
    state = payload.get(key) if isinstance(payload, dict) else None
    if state is None:
        if isinstance(payload, dict) and "class_embed.weight" in payload:
            state = payload  # bare state_dict
        else:
            raise KeyError(f"{path} has no '{key}' weights")
    step = int(payload.get("step", -1)) if isinstance(payload, dict) else -1
    return _canonical(state), step


def noise_code_table(state: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    """W-Flow noise-code embeddings stacked as [coords, classes, cond_dim], or None."""
    keys = sorted((k for k in state if k.startswith("noise_embeds.")), key=lambda k: int(k.split(".")[1]))
    if not keys:
        return None
    return torch.stack([state[k].float() for k in keys], dim=0)


def fold_noise_code(table: Optional[torch.Tensor], mode: str, cond_dim: int) -> torch.Tensor:
    """Collapse W-Flow's random codes into one conditioning bias.

    mode="fixed:<i>" uses code i on every coordinate (a configuration W-Flow saw
    in training); mode="mean" uses the average embedding of every coordinate.
    """
    if table is None:
        return torch.zeros(cond_dim)
    if mode == "mean":
        return table.mean(dim=1).sum(dim=0)
    if mode.startswith("fixed"):
        idx = int(mode.split(":")[1]) if ":" in mode else 0
        return table[:, idx, :].sum(dim=0)
    raise ValueError(f"noise_code must be 'fixed:<i>' or 'mean', got {mode!r}")


def load_wflow_generator(gen: Generator, path: str, *, weights: str = "ema", noise_code: str = "fixed:0") -> Dict:
    """Initialize the generator from W-Flow weights.  Every trunk/readout tensor must match."""
    state, step = read_wflow_state(path, weights)
    table = noise_code_table(state)
    target = gen.state_dict()
    new_keys = {k for k in target if k.startswith("scale_layer.")} | {"code_bias"}
    missing = sorted(k for k in target if k not in state and k not in new_keys)
    unexpected = sorted(k for k in state if k not in target and not k.startswith("noise_embeds."))
    if missing or unexpected:
        raise ValueError(f"W-Flow checkpoint does not match the configured architecture: "
                         f"missing={missing[:6]} unexpected={unexpected[:6]}")
    for k in target:
        if k in state and state[k].shape != target[k].shape:
            raise ValueError(f"Shape mismatch for {k}: checkpoint {tuple(state[k].shape)} vs model {tuple(target[k].shape)}")
    loaded = {k: state[k].to(target[k].dtype) for k in target if k in state}
    gen.load_state_dict(loaded, strict=False)
    with torch.no_grad():
        gen.code_bias.copy_(fold_noise_code(table, noise_code, gen.cond_dim).to(gen.code_bias))
    return {"source": str(path), "source_step": step, "weights": weights, "noise_code": noise_code,
            "noise_code_table": table}
