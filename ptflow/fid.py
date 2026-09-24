"""Streaming FID / Inception Score with torch-fidelity's inception-v3-compat.

Images are converted to features batch by batch, so a 50k evaluation never
holds the images in memory.  The extractor and the reference statistics
(jit_in256_stats.npz) are the ones W-Flow reports against.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import torch

from ptflow import dist


class InceptionStream:
    def __init__(self, device: torch.device):
        hub = os.environ.get("TORCH_HUB_DIR")
        if hub:
            torch.hub.set_dir(hub)
        from torch_fidelity.utils import create_feature_extractor
        self.fe = create_feature_extractor("inception-v3-compat", ["2048", "logits_unbiased"],
                                           cuda=device.type == "cuda").eval()
        self.device = device
        self.feats: List[torch.Tensor] = []
        self.logits: List[torch.Tensor] = []

    @torch.no_grad()
    def add(self, images_uint8: torch.Tensor) -> None:
        """images_uint8: [B, 3, H, W] uint8."""
        f, l = self.fe(images_uint8.to(self.device))
        self.feats.append(f.double().cpu())
        self.logits.append(l.double().cpu())

    def gather(self, n: int):
        """All ranks' features, truncated to n (ranks must hold equal counts)."""
        f = torch.cat(self.feats)
        l = torch.cat(self.logits)
        f = dist.all_gather_cat(f.float().to(self.device)).double().cpu()[:n]
        l = dist.all_gather_cat(l.float().to(self.device)).double().cpu()[:n]
        return f.numpy(), l.numpy()


def load_reference(path: str):
    data = np.load(path)
    if "mu" in data:
        return data["mu"], data["sigma"]
    return data["ref_mu"], data["ref_sigma"]


def frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    from scipy import linalg
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + off).dot(sigma2 + off))
    if np.iscomplexobj(covmean):
        if np.abs(covmean.imag).max() > 1e-3:
            raise ValueError(f"Significant imaginary component in covariance sqrt: {np.abs(covmean.imag).max():.3g}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


def inception_score(logits: np.ndarray, splits: int = 10) -> Dict[str, float]:
    logits = logits[np.random.RandomState(2020).permutation(len(logits))]
    p = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    n = len(p) // splits
    scores = []
    for i in range(splits):
        part = p[i * n:(i + 1) * n]
        py = part.mean(0, keepdims=True)
        scores.append(np.exp((part * (np.log(part + 1e-10) - np.log(py + 1e-10))).sum(1).mean()))
    return {"is_mean": float(np.mean(scores)), "is_std": float(np.std(scores))}


def fid_from_features(feats: np.ndarray, ref_path: str) -> float:
    mu_r, sig_r = load_reference(ref_path)
    return frechet_distance(feats.mean(0), np.cov(feats, rowvar=False), mu_r, sig_r)
