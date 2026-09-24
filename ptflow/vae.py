"""SD-VAE (stabilityai/sd-vae-ft-mse) decoding of NHWC latents to images in [0, 1]."""

from __future__ import annotations

import torch

from ptflow.config import env_path

_VAE = {}


def load_vae(device: torch.device):
    key = str(device)
    if key not in _VAE:
        from diffusers.models import AutoencoderKL
        _VAE[key] = AutoencoderKL.from_pretrained(env_path("VAE_PATH")).to(device).eval().requires_grad_(False)
    return _VAE[key]


@torch.no_grad()
def decode(latents: torch.Tensor) -> torch.Tensor:
    """[B, 32, 32, 4] latents (x 0.18215 scaling) -> [B, 3, 256, 256] in [0, 1]."""
    if not torch.isfinite(latents).all():
        raise FloatingPointError("Nonfinite latents reached the decoder")
    vae = load_vae(latents.device)
    img = vae.decode(latents.permute(0, 3, 1, 2).float() / 0.18215).sample
    return ((img + 1.0) / 2.0).clamp(0.0, 1.0)


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    return (images * 255.0).round().clamp(0, 255).to(torch.uint8)
