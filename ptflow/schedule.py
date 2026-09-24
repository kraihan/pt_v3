"""Training phases, the eps anneal and the ESS controller (Algorithm 1, step C).

Phase "calibrate" (runs once, before Algorithm 1):
    The W-Flow generator is frozen and the potential is fitted so that m_eta is its
    prox (calibration_loss).  It ends when the relative prox residual is small AND
    the generator-centred proposal is healthy, or after ``calib_max_steps``.
    Starting Algorithm 1 from an inconsistent pair (m far from prox_phi) leaves the
    proposal many standard deviations from the target mode and the weights degenerate.

Phase "pt" (Algorithm 1):
    healthy    control ESS >= ess_ok   anneal eps toward eps_min; base alpha; base n_eta
    degrading  ess_bad .. ess_ok       hold eps; raise alpha; double n_eta
    broken     control ESS < ess_bad   freeze theta; retrain eta (n_eta_broken) until recovered

The health signal is the *control ESS* (ESS - 1) / (K - 1), an affine rescaling
of the paper's ESS/K that removes its 1/K floor (with ESS/K the "broken" threshold
0.05 is unreachable for K < 20).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

HEALTHY, DEGRADING, BROKEN = "healthy", "degrading", "broken"


@dataclass
class PTSchedule:
    eps_max: float = 0.2
    eps_min: float = 0.1
    eps_schedule: str = "cosine"          # cosine | linear | exp | constant
    anneal_steps: int = 50000             # counted in healthy Algorithm-1 steps
    anneal_on: str = "healthy"            # "healthy" (Alg. 1) | "always" (open loop, e.g. an eps-sweep run)
    alpha_def: float = 0.1
    alpha_degraded: float = 0.25
    n_eta: int = 1
    n_eta_degraded: int = 2
    n_eta_broken: int = 4
    ess_ok: float = 0.3
    ess_bad: float = 0.05
    ess_ema: float = 0.9
    calib_min_steps: int = 500
    calib_max_steps: int = 20000
    calib_tol: float = 0.1                # relative prox residual |r| / |m - x0|
    calib_check_every: int = 50

    # mutable state (checkpointed)
    phase: str = "calibrate"
    calib_step: int = 0
    pt_step: int = 0
    anneal_progress: int = 0
    ess_value: float = -1.0
    resid_value: float = -1.0
    state: str = HEALTHY
    calib_exit: str = ""

    # -- schedules -----------------------------------------------------------
    def eps(self) -> float:
        mode = self.eps_schedule
        if mode == "constant":
            return float(self.eps_min)
        t = min(self.anneal_progress / max(1, self.anneal_steps), 1.0)
        if mode == "cosine":
            return float(self.eps_min + (self.eps_max - self.eps_min) * 0.5 * (1 + math.cos(math.pi * t)))
        if mode == "linear":
            return float(self.eps_max + (self.eps_min - self.eps_max) * t)
        if mode == "exp":
            return float(math.exp(math.log(self.eps_max) + (math.log(self.eps_min) - math.log(self.eps_max)) * t))
        raise ValueError(f"Unknown eps_schedule {mode!r}")

    def alpha(self) -> float:
        return float(self.alpha_def if self.state == HEALTHY else self.alpha_degraded)

    def eta_steps(self) -> int:
        return {HEALTHY: self.n_eta, DEGRADING: self.n_eta_degraded, BROKEN: self.n_eta_broken}[self.state]

    def update_theta(self) -> bool:
        return self.state != BROKEN

    # -- observations ----------------------------------------------------------
    def _ema(self, old: float, new: float) -> float:
        return float(new) if old < 0 else self.ess_ema * old + (1 - self.ess_ema) * float(new)

    def observe_calibration(self, resid_rel: float | None = None, control_ess: float | None = None) -> None:
        """One calibration step; the two measurements are supplied every ``calib_check_every`` steps."""
        self.calib_step += 1
        if resid_rel is not None:
            self.resid_value = self._ema(self.resid_value, resid_rel)
        if control_ess is not None:
            self.ess_value = self._ema(self.ess_value, control_ess)
        ready = (self.calib_step >= self.calib_min_steps and 0 <= self.resid_value <= self.calib_tol
                 and self.ess_value >= self.ess_ok)
        if ready:
            self.calib_exit = f"converged at calibration step {self.calib_step}"
        elif self.calib_step >= self.calib_max_steps:
            self.calib_exit = (f"max steps reached (resid_rel={self.resid_value:.3g}, "
                               f"control_ess={self.ess_value:.3g}); Algorithm 1 starts from an imperfect pair")
        if self.calib_exit:
            self.phase = "pt"
            self.state = HEALTHY if self.ess_value >= self.ess_ok else (DEGRADING if self.ess_value >= self.ess_bad else BROKEN)

    def observe(self, control_ess: float) -> str:
        """Algorithm 1 step (C): feed this step's control ESS and update the controller."""
        if not math.isfinite(control_ess):
            control_ess = 0.0
        self.ess_value = self._ema(self.ess_value, control_ess)
        if self.ess_value >= self.ess_ok:
            self.state = HEALTHY
        elif self.ess_value >= self.ess_bad:
            self.state = DEGRADING
        else:
            self.state = BROKEN
        if self.state == HEALTHY or self.anneal_on == "always":
            self.anneal_progress += 1
        self.pt_step += 1
        return self.state

    def metrics(self) -> dict:
        return {"sched/eps": self.eps(), "sched/alpha": self.alpha(), "sched/n_eta": float(self.eta_steps()),
                "sched/control_ess_ema": self.ess_value, "sched/resid_rel_ema": self.resid_value,
                "sched/health": {HEALTHY: 2.0, DEGRADING: 1.0, BROKEN: 0.0}[self.state],
                "sched/calibrating": float(self.phase == "calibrate"), "sched/anneal_progress": float(self.anneal_progress)}

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, d: dict) -> None:
        for k in ("phase", "calib_step", "pt_step", "anneal_progress", "ess_value", "resid_value", "state", "calib_exit"):
            if k in d:
                setattr(self, k, type(getattr(self, k))(d[k]))


def build_schedule(cfg: dict) -> PTSchedule:
    known = set(PTSchedule.__dataclass_fields__)
    unknown = set(cfg) - known
    if unknown:
        raise ValueError(f"Unknown schedule keys: {sorted(unknown)}")
    sched = PTSchedule(**cfg)
    if sched.anneal_on not in ("healthy", "always"):
        raise ValueError(f"anneal_on must be 'healthy' or 'always', got {sched.anneal_on!r}")
    return sched
