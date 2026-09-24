"""Algorithm 2: PT-Flow sampling.

Mode A  one generator call: x1 = m_eta(x0, c, w)                     (1 NFE)
Mode B  Mode A followed by n prox-refinement steps on F_x0(y) = phi^w(y) + |y - x0|^2/2,
        y <- y - gamma (grad phi^w(y) + y - x0), with backtracking on F
Mode C  self-normalized importance resampling from the defensive proposal (K draws);
        consistent for the bridge kernel p(x1 | x0) as K grows, under coverage
"""

from __future__ import annotations

import torch

from ptflow.estimator import potential_phi_fn, snis_resample, tilted_estimate
from ptflow.models.potential import prox_energy, prox_residual


@torch.no_grad()
def mode_a(gen, x0, c, w=0.0) -> torch.Tensor:
    return gen(x0, c, w, with_scale=False)[0]


def refine(pot, y, x0, c, w=0.0, *, steps: int = 3, gamma: float = 0.5, backtrack: int = 6):
    """Prox refinement with per-sample backtracking (a step is kept only if F decreases)."""
    y = y.detach().float()
    energy = prox_energy(pot, y, x0, c, w).detach()
    for _ in range(int(steps)):
        r = prox_residual(pot, y, x0, c, w).detach()
        rate = torch.full((y.shape[0],), float(gamma), device=y.device)
        done = torch.zeros(y.shape[0], dtype=torch.bool, device=y.device)
        for _ in range(int(backtrack)):
            with torch.no_grad():
                trial = y - rate.view(-1, *([1] * (y.ndim - 1))) * r
                e_trial = prox_energy(pot, trial, x0, c, w)
                ok = (~done) & torch.isfinite(e_trial) & (e_trial < energy)
                y = torch.where(ok.view(-1, *([1] * (y.ndim - 1))), trial, y)
                energy = torch.where(ok, e_trial, energy)
                done |= ok
                rate = torch.where(done, rate, 0.5 * rate)
            if bool(done.all()):
                break
    return y, energy


def mode_b(gen, pot, x0, c, w=0.0, *, steps: int = 3, gamma: float = 0.5) -> torch.Tensor:
    return refine(pot, mode_a(gen, x0, c, w), x0, c, w, steps=steps, gamma=gamma)[0]


@torch.no_grad()
def mode_c(gen, pot, x0, c, w=0.0, *, eps: float, K: int = 8, alpha: float = 0.1, generator=None, chunk: int = 0):
    m, s = gen(x0, c, w)
    est = tilted_estimate(potential_phi_fn(pot, w), x0, c, m, s, eps, K=K, alpha=alpha, generator=generator, chunk=chunk)
    return snis_resample(est, generator=generator)


def sample(mode: str, gen, pot, x0, c, w=0.0, *, eps: float = 0.1, refine_steps: int = 3, K: int = 8,
           alpha: float = 0.1, generator=None, chunk: int = 0) -> torch.Tensor:
    mode = mode.upper()
    if mode == "A":
        return mode_a(gen, x0, c, w)
    if pot is None:
        raise ValueError(f"Mode {mode} needs the potential")
    if mode == "B":
        return mode_b(gen, pot, x0, c, w, steps=refine_steps)
    if mode == "C":
        return mode_c(gen, pot, x0, c, w, eps=eps, K=K, alpha=alpha, generator=generator, chunk=chunk)
    raise ValueError(f"Unknown sampling mode {mode!r}")
