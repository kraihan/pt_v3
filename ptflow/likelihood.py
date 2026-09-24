"""Latent-space likelihood of the model marginal (Thm 3.5, Appendix J).

    log rho_hat_1(x) = log psi_hat_1(x) - u(x),
    psi_hat_1(x)     = E_{z ~ N(0, I)}[ rho_0(y) / psi_0(y) ],   y = x + sqrt(2 eps) z,

with every inner psi_0(y) estimated by the prox-tilted estimator centred at
m_eta(y).  The density is normalized analytically, but this estimator is nested
Monte Carlo (an inner estimate inside a reciprocal, an outer average and a log):
it is neither unbiased nor a certified bound, so it must be reported over a grid
of (outer, inner) budgets and seeds.  It is a density of the SD-VAE latent, not
of pixels.
"""

from __future__ import annotations

import math
from typing import Dict

import torch

from ptflow.estimator import draw_noise, tilted_estimate, weight_stats


@torch.no_grad()
def nested_log_likelihood(gen, pot, x1, c, eps: float, *, k_outer: int = 16, k_inner: int = 16, alpha: float = 0.1,
                          generator=None, chunk: int = 0) -> Dict[str, torch.Tensor]:
    """Per-example log rho_hat_1(x1 | c) in nats, plus estimator diagnostics."""
    B, tail = x1.shape[0], x1.shape[1:]
    d = x1[0].numel()
    x1 = x1.float()
    z, _ = draw_noise(x1, k_outer, 0.0, generator=generator, antithetic=k_outer % 2 == 0)
    y = (x1.unsqueeze(1) + math.sqrt(2.0 * eps) * z).reshape(B * k_outer, *tail)
    c_rep = c.repeat_interleave(k_outer, dim=0)
    step = int(chunk) if chunk and chunk > 0 else y.shape[0]
    log_psi0, inner_ess = [], []
    for i in range(0, y.shape[0], step):
        yi, ci = y[i:i + step], c_rep[i:i + step]
        m, s = gen(yi, ci, 0.0)
        est = tilted_estimate(pot.phi, yi, ci, m, s, eps, K=k_inner, alpha=alpha, generator=generator, chunk=chunk)
        log_psi0.append(est.log_psi0)
        inner_ess.append(weight_stats(est.log_w)["control_ess"])
    log_psi0 = torch.cat(log_psi0).view(B, k_outer)
    y64 = y.double().view(B, k_outer, -1)
    log_rho0 = -0.5 * y64.square().sum(-1) - 0.5 * d * math.log(2 * math.pi)
    outer = log_rho0 - log_psi0
    log_psi1 = torch.logsumexp(outer, dim=1) - math.log(k_outer)
    u_x = pot.phi(x1, c).double() / (2.0 * eps)
    log_rho1 = log_psi1 - u_x
    return {
        "log_likelihood": log_rho1,
        "nll_nats_per_dim": -log_rho1 / d,
        "nll_bits_per_dim": -log_rho1 / (d * math.log(2)),
        "outer_control_ess": weight_stats(outer)["control_ess"],
        "inner_control_ess": torch.cat(inner_ess).view(B, k_outer).mean(1),
    }
