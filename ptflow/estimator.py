"""Prox-tilted importance estimator of psi_0 (paper Sec. 3.4-3.5, eqs. 3, 9, 10, 15).

    psi_0(x0) = E_{y ~ q}[ G_{2 eps}(y - x0) exp(-u(y)) / q(y) ],      u = phi / (2 eps)

with the defensive Laplace proposal

    q_def(. | x0) = (1 - alpha) N(m, 2 eps diag(e^s)) + alpha N(x0, 2 eps I),

m = m_eta(x0) ~ prox_phi(x0) and e^{-s} ~ diag(I + grad^2 phi(m)).  Everything is
in the log domain.  N(y; x0, 2 eps I) *is* G_{2 eps}(y - x0), so the Gaussian
normalizers cancel and only squared norms remain (computed in float64).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

PhiFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def draw_noise(x0: torch.Tensor, K: int, alpha: float, *, generator=None, antithetic: bool = True
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard-normal draws z [B, K, ...] and defensive-component flags [B, K].

    Antithetic +/- pairs share a mixture component so each pair stays antithetic.
    """
    B, tail = x0.shape[0], x0.shape[1:]
    if antithetic:
        if K % 2:
            raise ValueError("Antithetic sampling needs an even K")
        half = torch.randn((B, K // 2, *tail), generator=generator, device=x0.device)
        z = torch.cat([half, -half], dim=1)
        u = torch.rand((B, K // 2), generator=generator, device=x0.device)
        from_def = torch.cat([u, u], dim=1) < float(alpha)
    else:
        z = torch.randn((B, K, *tail), generator=generator, device=x0.device)
        from_def = torch.rand((B, K), generator=generator, device=x0.device) < float(alpha)
    return z, from_def


def proposal_points(x0, m, s, eps: float, z, from_def) -> torch.Tensor:
    """Map noise draws to proposal points y [B, K, ...]."""
    sd = math.sqrt(2.0 * eps)
    scale = torch.ones_like(m) if s is None else torch.exp(0.5 * s)
    y_tilt = m.unsqueeze(1) + sd * scale.unsqueeze(1) * z
    y_def = x0.unsqueeze(1) + sd * z
    mask = from_def.view(*from_def.shape, *([1] * (x0.ndim - 1)))
    return torch.where(mask, y_def, y_tilt)


def log_kernel_over_proposal(y, x0, m, s, eps: float, alpha: float) -> torch.Tensor:
    """log G_{2 eps}(y - x0) - log q_def(y | x0), float64, [B, K]."""
    B, K = y.shape[:2]
    yf = y.reshape(B, K, -1).double()
    x0f, mf = x0.reshape(B, 1, -1).double(), m.reshape(B, 1, -1).double()
    four_eps = 4.0 * eps
    to_x0 = (yf - x0f).square().sum(-1) / four_eps
    if s is None:
        to_m = (yf - mf).square().sum(-1) / four_eps
        half_logdet = 0.0
    else:
        sf = s.reshape(B, 1, -1).double()
        to_m = ((yf - mf) * torch.exp(-0.5 * sf)).square().sum(-1) / four_eps
        half_logdet = 0.5 * sf.sum(-1)
    r = to_x0 - to_m - half_logdet                         # log q_tilt - log G
    if alpha <= 0.0:
        return -r
    if alpha >= 1.0:
        return torch.zeros_like(r)
    return -torch.logaddexp(r + math.log1p(-alpha), torch.full_like(r, math.log(alpha)))


@dataclass
class TiltedEstimate:
    log_psi0: torch.Tensor   # [B] float64, differentiable w.r.t. the potential
    log_w: torch.Tensor      # [B, K] float64 unnormalized log-weights (differentiable)
    y: torch.Tensor          # [B, K, ...] proposal points (detached)
    phi_y: torch.Tensor      # [B, K] potential at the proposal points

    def phi0(self, eps: float) -> torch.Tensor:
        """Estimated smoothed potential phi_0 = -2 eps log psi_0 (phi-units)."""
        return -2.0 * eps * self.log_psi0

    def normalized_weights(self) -> torch.Tensor:
        return torch.softmax(self.log_w.detach(), dim=1)

    def posterior_mean(self) -> torch.Tensor:
        """SNIS estimate of the bridge conditional mean T_eps(x0) = E[X1 | x0]."""
        w = self.normalized_weights().to(self.y.dtype)
        return (w.view(*w.shape, *([1] * (self.y.ndim - 2))) * self.y).sum(dim=1)


def tilted_estimate(phi_fn: PhiFn, x0, c, m, s, eps: float, *, K: int = 8, alpha: float = 0.1,
                    generator=None, antithetic: bool = True, draws=None, chunk: int = 0) -> TiltedEstimate:
    """Eq. 15: log psi0_hat = logsumexp_k[log G - u(y_k) - log q_def] - log K.

    ``phi_fn(y, c) -> [N]`` is the (possibly guided) potential in phi-units.
    ``m`` and ``s`` must be detached: the potential step runs at frozen eta.
    ``draws=(z, from_def)`` reuses common random numbers.
    """
    if eps <= 0 or not 0.0 <= alpha <= 1.0:
        raise ValueError("Require eps > 0 and alpha in [0, 1]")
    m, s = m.detach().float(), (None if s is None else s.detach().float())
    x0 = x0.detach().float()
    with torch.no_grad():
        z, from_def = draws if draws is not None else draw_noise(x0, K, alpha, generator=generator, antithetic=antithetic)
        K = z.shape[1]
        y = proposal_points(x0, m, s, eps, z, from_def)
        log_ratio = log_kernel_over_proposal(y, x0, m, s, eps, alpha)
    B, tail = x0.shape[0], x0.shape[1:]
    y_flat, c_rep = y.reshape(B * K, *tail), c.repeat_interleave(K, dim=0)
    step = int(chunk) if chunk and chunk > 0 else B * K
    phi_y = torch.cat([phi_fn(y_flat[i:i + step], c_rep[i:i + step]) for i in range(0, B * K, step)]).view(B, K)
    # Exact centring: subtracting a k-independent constant keeps the log-weights O(1).
    ref = phi_y.detach().double().mean(dim=1, keepdim=True)
    log_w = log_ratio - (phi_y.double() - ref) / (2.0 * eps)
    log_psi0 = torch.logsumexp(log_w, dim=1) - math.log(K) - ref.squeeze(1) / (2.0 * eps)
    return TiltedEstimate(log_psi0=log_psi0, log_w=log_w, y=y, phi_y=phi_y)


@torch.no_grad()
def weight_stats(log_w: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per-row diagnostics of importance weights [B, K].

    ess            (sum w)^2 / sum w^2, in draws (1 .. K)
    ess_frac       ess / K
    control_ess    (ess - 1) / (K - 1): 0 when one draw carries everything, 1 when uniform
    var_logw       Var_k(log w)   -- s_res^2 of Theorem 3.9
    chi2           mean(w^2) / mean(w)^2 - 1, the chi-square divergence estimate
    max_w          largest normalized weight
    """
    lw = log_w.detach().double()
    K = lw.shape[1]
    lse1, lse2 = torch.logsumexp(lw, 1), torch.logsumexp(2 * lw, 1)
    ess = torch.exp(2 * lse1 - lse2).clamp(1.0, float(K))
    chi2 = torch.exp(lse2 - math.log(K) - 2 * (lse1 - math.log(K))) - 1.0
    return {
        "ess": ess,
        "ess_frac": ess / K,
        "control_ess": ((ess - 1.0) / max(K - 1, 1)).clamp(0.0, 1.0),
        "var_logw": lw.var(dim=1, correction=0),
        "chi2": chi2,
        "max_w": torch.softmax(lw, dim=1).max(dim=1).values,
    }


@torch.no_grad()
def snis_resample(est: TiltedEstimate, generator=None) -> torch.Tensor:
    """Mode C: draw one proposal point per row with probability proportional to its weight."""
    idx = torch.multinomial(est.normalized_weights().float(), 1, generator=generator).squeeze(1)
    B = est.y.shape[0]
    return est.y[torch.arange(B, device=est.y.device), idx]


def potential_phi_fn(pot, w=0.0) -> PhiFn:
    """phi^w(y, c) = (1 + w) phi(y, c) - w phi(y, null)  (eq. 17); w is a scalar here."""
    w = float(w)
    if w == 0.0:
        return pot.phi

    def fn(y, c):
        return (1.0 + w) * pot.phi(y, c) - w * pot.phi(y, pot.null_labels(c))
    return fn
