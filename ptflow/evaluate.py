"""Class-balanced FID evaluation for sampling modes A / B / C (multi-GPU, streaming)."""

from __future__ import annotations

import math
import time
from typing import Dict, Optional

import torch

from ptflow import dist
from ptflow.fid import InceptionStream, fid_from_features, inception_score
from ptflow.sampling import sample
from ptflow.vae import decode, to_uint8


def random_code_bias(table: torch.Tensor, n: int, generator) -> torch.Tensor:
    """W-Flow's original random noise codes: sum_i E_i[l_i] with l_i uniform (baseline only)."""
    coords, classes, _ = table.shape
    idx = torch.randint(0, classes, (n, coords), generator=generator, device=table.device)
    return table[torch.arange(coords, device=table.device), idx].sum(dim=1)


def evaluate_fid(gen, pot, *, num_samples: int, cfg_scale: float, mode: str = "A", eps: float = 0.1,
                 alpha: float = 0.1, K: int = 8, refine_steps: int = 3, seed: int = 0, batch: int = 64,
                 ref_path: str, device: torch.device, code_table: Optional[torch.Tensor] = None,
                 chunk: int = 0, log_every: int = 50) -> Dict:
    """FID-N against ``ref_path``.  cfg_scale follows W-Flow's convention (cfg = 1 + w)."""
    w = float(cfg_scale) - 1.0
    num_classes = gen.num_classes
    per_rank = math.ceil(num_samples / dist.world())
    start = dist.rank() * per_rank
    stream = InceptionStream(device)
    t0 = time.time()
    shape = (gen.input_size, gen.input_size, gen.channels)
    for j, b0 in enumerate(range(0, per_rank, batch)):
        idx = torch.arange(start + b0, start + min(b0 + batch, per_rank), device=device)
        g = torch.Generator(device=device).manual_seed(int(seed) * 1_000_003 + int(idx[0]))
        c = idx % num_classes
        x0 = torch.randn((len(idx), *shape), generator=g, device=device)
        if code_table is not None:
            with torch.no_grad():
                x1 = gen(x0, c, w, with_scale=False, code_bias=random_code_bias(code_table, len(idx), g))[0]
        else:
            x1 = sample(mode, gen, pot, x0, c, w, eps=eps, refine_steps=refine_steps, K=K, alpha=alpha,
                        generator=g, chunk=chunk)
        stream.add(to_uint8(decode(x1)))
        if dist.is_main() and log_every and j % log_every == 0:
            print(f"[fid] {b0 + len(idx)}/{per_rank} per rank, {time.time() - t0:.0f}s", flush=True)
    feats, logits = stream.gather(num_samples)
    out = {"num_samples": int(num_samples), "cfg_scale": float(cfg_scale), "w": w, "mode": mode, "seed": int(seed),
           "seconds": time.time() - t0}
    if dist.is_main():
        out["fid"] = fid_from_features(feats, ref_path)
        out.update(inception_score(logits))
    return out
