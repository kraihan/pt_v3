"""PT-Flow objectives, all in phi-units and normalized per latent coordinate.

potential_loss     eq. 14:  L_theta = E[log psi0_tilt(x0)] + E[u_theta(x1)]
                   Multiplied by 2 eps / d (a positive constant for fixed eps):
                       (E[phi(x1)] - E[phi0_hat(x0)]) / d.
generator_loss     eq. 16 (Alg. 1 lines 7-9):  |2 eps g_w(m) + m - x0|^2 plus the
                   Hutchinson scale match |e^{-s} - diag(I + grad^2 phi^w(m))|^2.
calibration_loss   the same prox residual minimized over theta at frozen eta.  Used
                   once, before Algorithm 1, to make the W-Flow generator the prox of
                   the potential (T = prox = m at step 0).
curvature_hinge    soft (A2) enforcement: penalize directional curvature below -allow.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ptflow.estimator import TiltedEstimate, tilted_estimate, weight_stats
from ptflow.models.potential import frozen, grad_phi, guided_grad, guided_hvp, prox_residual


def _sq_per_dim(r: torch.Tensor) -> torch.Tensor:
    return r.float().flatten(1).square().sum(1) / r[0].numel()


def potential_loss(pot, x0, c0, m0, s0, x1, c1, eps: float, *, K: int, alpha: float, lambda_gauge: float = 0.0,
                   generator=None, chunk: int = 0) -> Tuple[torch.Tensor, TiltedEstimate, Dict[str, torch.Tensor]]:
    """Potential step (B) at frozen eta and guidance weight zero."""
    d = float(x0[0].numel())
    est = tilted_estimate(pot.phi, x0, c0, m0, s0, eps, K=K, alpha=alpha, generator=generator, chunk=chunk)
    phi_data = pot.phi(x1, c1) / d
    phi0 = est.phi0(eps).float() / d
    loss = phi_data.mean() - phi0.mean()
    # The objective is invariant to phi -> phi + const (its only flat direction,
    # Thm 3.6(ii)); pinning that coordinate does not move the optimum.
    gauge = 0.5 * (phi_data.mean() + phi0.mean())
    if lambda_gauge > 0:
        loss = loss + lambda_gauge * gauge.square()
    st = weight_stats(est.log_w)
    metrics = {
        "pot/loss": loss.detach(), "pot/phi_data": phi_data.mean().detach(), "pot/phi0_noise": phi0.mean().detach(),
        "pot/gauge": gauge.detach(), "est/ess_frac": st["ess_frac"].mean(), "est/control_ess": st["control_ess"].mean(),
        "est/var_logw": st["var_logw"].mean(), "est/chi2": st["chi2"].mean(), "est/max_w": st["max_w"].mean(),
    }
    return loss, est, metrics


def hutchinson_diag(pot, m, c, w, probes: int = 1, generator=None) -> torch.Tensor:
    """Unbiased estimate of diag(I + grad^2 phi^w(m)) from Rademacher probes."""
    acc = torch.zeros_like(m, dtype=torch.float32)
    for _ in range(int(probes)):
        v = torch.randint(0, 2, m.shape, generator=generator, device=m.device).float().mul_(2).sub_(1)
        acc += v * guided_hvp(pot, m, c, w, v)
    return 1.0 + acc / int(probes)


def generator_loss(pot, gen, x0, c, w, *, mode: str = "full", lambda_scale: float = 0.1, probes: int = 1,
                   generator=None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Generator step (A) at frozen theta.

    mode="full" differentiates the squared residual through grad phi(m) (the
    paper's eq. 16, one Hessian-vector product).  mode="detach" regresses m onto
    the fixed-point target x0 - grad phi^w(m) (cheaper; same zero set).
    """
    m, s = gen(x0, c, w, with_scale=lambda_scale > 0)
    with frozen(pot):
        if mode == "full":
            r = prox_residual(pot, m, x0, c, w, create_graph=True)
        elif mode == "detach":
            g = guided_grad(pot, m.detach(), c, w)
            r = m - (x0.float() - g).detach()
        else:
            raise ValueError(f"generator mode must be 'full' or 'detach', got {mode!r}")
    loss_loc = _sq_per_dim(r).mean()
    loss = loss_loc
    with torch.no_grad():
        rn = r.detach().flatten(1).norm(dim=1)
        disp = (m.detach() - x0).flatten(1).norm(dim=1)
    metrics = {"gen/loss_prox": loss_loc.detach(), "gen/resid": rn.mean(),
               "gen/resid_rel": (rn / disp.clamp_min(1e-6)).mean(), "gen/displacement_rms": (disp / x0[0].numel() ** 0.5).mean()}
    if lambda_scale > 0:
        target = hutchinson_diag(pot, m.detach(), c, w, probes=probes, generator=generator)
        loss_scale = (torch.exp(-s) - target).square().mean()
        loss = loss + lambda_scale * loss_scale
        metrics.update({"gen/loss_scale": loss_scale.detach(), "gen/log_scale_mean": s.detach().mean(),
                        "gen/log_scale_std": s.detach().std(), "gen/diag_h_mean": target.mean()})
    return loss, metrics


def calibration_loss(pot, m, x0, c, w) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """|grad phi^w(m) + m - x0|^2 / d minimized over theta (generator frozen)."""
    r = prox_residual(pot, m.detach(), x0, c, w, create_graph=True)
    loss = _sq_per_dim(r).mean()
    with torch.no_grad():
        rn = r.detach().flatten(1).norm(dim=1)
        disp = (m.detach() - x0).flatten(1).norm(dim=1)
    return loss, {"calib/loss": loss.detach(), "calib/resid_rel": (rn / disp.clamp_min(1e-6)).mean()}


def curvature_hinge(pot, y, c, *, allow: float = 0.5, h: float = 1e-2, generator=None
                    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Directional curvature v^T grad^2 phi v / |v|^2 by a gradient difference; hinge below -allow.

    (A2) asks grad^2 phi >= -lambda I with lambda < 1 so that F_x is strongly convex
    and the prox is unique.
    """
    y = y.detach().float()
    v = torch.randn(y.shape, generator=generator, device=y.device)
    g0, _ = grad_phi(pot, y, c, create_graph=True)
    g1, _ = grad_phi(pot, y + h * v, c, create_graph=True)
    curv = ((g1 - g0) * v).flatten(1).sum(1) / (h * v.flatten(1).square().sum(1))
    pen = torch.relu(-curv - allow).mean()
    return pen, {"curv/mean": curv.detach().mean(), "curv/min": curv.detach().min(),
                 "curv/violation_frac": (curv.detach() < -allow).float().mean(), "curv/penalty": pen.detach()}
