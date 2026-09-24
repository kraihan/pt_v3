"""Mechanism experiments on a fixed PT-Flow checkpoint (the real model; no training).

Each command writes <out>/<name>.csv, <name>.json (means with bootstrap 95% CIs) and
<name>_raw.npz (per-example values used by scripts/make_figures.py).

    python experiments.py proposals  --ckpt CKPT --out R   # Fig 1: naive / recentred / curvature-scaled / full
    python experiments.py eps-sweep  --ckpt CKPT --out R   # Fig 2: ESS, Var(log w), chi2, T_eps - m vs eps
    python experiments.py prox       --ckpt CKPT --out R   # Fig 3: prox residual, Mode-B refinement
    python experiments.py mismatch   --ckpt CKPT --out R   # Fig 4: centre shift / variance scaling, vs Gaussian theory
    python experiments.py nll        --ckpt CKPT --out R   # Fig 5: nested-MC latent NLL over budgets x seeds
    python experiments.py jacobian   --config configs/B.yaml --out R   # W-Flow map: gradient-map audit
    python experiments.py toy        --out R               # exact T_eps vs prox (quadrature)

Proposals, with common random numbers (same x0, labels, z and mixture draws) across all of them:
    naive       N(x0, 2 eps I)                                        eq. 8
    recentered  N(m, 2 eps I)                                         centre only
    curvature   N(m, 2 eps diag e^s)                                  centre + diagonal curvature scale
    full        (1 - alpha) N(m, 2 eps diag e^s) + alpha N(x0, 2 eps I)   eqs. 9-10
s is the generator's learned scale head (--scale learned) or the Hutchinson estimate of
-log diag(I + grad^2 phi(m)) from the potential itself (--scale hutchinson), which is what the
scale head is trained to match (use it at a calibrated checkpoint, where the head is still zero).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from ptflow import dist
from ptflow.estimator import log_kernel_over_proposal, potential_phi_fn, proposal_points, weight_stats
from ptflow.losses import hutchinson_diag
from ptflow.models.potential import guided_hvp, prox_energy, prox_residual

PROPOSALS = ("naive", "recentered", "curvature", "full")
METRICS = ("control_ess", "ess_frac", "var_logw", "log10_1pchi2", "max_w")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def write(out: Path, name: str, rows: List[Dict], summary: Dict, raw: Optional[Dict[str, np.ndarray]] = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
    if raw:
        np.savez_compressed(out / f"{name}_raw.npz", **raw)
    (out / f"{name}.json").write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, default=float))


def inputs(gen, n: int, seed: int, device, w: float = 0.0):
    g = torch.Generator(device=device).manual_seed(seed)
    c = torch.randint(0, gen.num_classes, (n,), generator=g, device=device)
    x0 = torch.randn((n, gen.input_size, gen.input_size, gen.channels), generator=g, device=device)
    with torch.no_grad():
        m, s = gen(x0, c, w)
    return x0, c, m, s, g


def crn_draws(x0, K: int, g):
    """Interleaved antithetic pairs (z1, -z1, z2, -z2, ...) so every even prefix stays antithetic,
    plus one uniform per pair for the defensive-component choice."""
    B, tail = x0.shape[0], x0.shape[1:]
    half = torch.randn((B, K // 2, *tail), generator=g, device=x0.device)
    z = torch.stack([half, -half], dim=2).reshape(B, K, *tail)
    u = torch.rand((B, K // 2), generator=g, device=x0.device).repeat_interleave(2, dim=1)
    return z, u


def curvature_scale(pot, m, c, probes: int, g, scale_max: float = 3.0) -> torch.Tensor:
    """s = -log diag(I + grad^2 phi(m)) (eq. 10, S^-1 ~ D), Hutchinson with ``probes`` Rademacher probes."""
    diag = hutchinson_diag(pot, m, c, 0.0, probes=probes, generator=g)
    return -torch.log(diag.clamp(math.exp(-scale_max), math.exp(scale_max)))


def proposal_scale(a, pot, m, c, s_learned, g) -> torch.Tensor:
    if a.scale == "learned":
        return s_learned
    return torch.cat([curvature_scale(pot, m[i:i + a.batch], c[i:i + a.batch], a.hutch_probes, g)
                      for i in range(0, len(m), a.batch)])


def proposal_spec(name: str, x0, m, s, alpha: float):
    """(centre, log-scale, defensive alpha) of each proposal."""
    return {"naive": (x0, None, 1.0), "recentered": (m, None, 0.0),
            "curvature": (m, s, 0.0), "full": (m, s, alpha)}[name]


@torch.no_grad()
def log_weights(pot, x0, c, center, s, alpha: float, eps: float, z, u, chunk: int, w: float = 0.0,
                return_y: bool = False):
    """Unnormalized log-weights [B, K] of the tilted estimator for a given proposal and fixed draws."""
    y = proposal_points(x0, center, s, eps, z, u < alpha)
    lr = log_kernel_over_proposal(y, x0, center, s, eps, alpha)
    B, K = y.shape[:2]
    yf, cr = y.reshape(B * K, *y.shape[2:]), c.repeat_interleave(K)
    fn = potential_phi_fn(pot, w)
    step = chunk or B * K
    phi = torch.cat([fn(yf[i:i + step], cr[i:i + step]) for i in range(0, B * K, step)]).view(B, K).double()
    lw = lr - (phi - phi.mean(1, keepdim=True)) / (2 * eps)
    return (lw, y) if return_y else lw


def batched_log_weights(a, pot, x0, c, center, s, alpha, eps, z, u, m=None):
    """Log-weights for all examples; with ``m`` also the per-example gap |T_eps_hat - m| / sqrt(d)."""
    lws, gaps = [], []
    for i in range(0, len(x0), a.batch):
        sl = slice(i, i + a.batch)
        lw, y = log_weights(pot, x0[sl], c[sl], center[sl], None if s is None else s[sl], alpha, eps, z[sl], u[sl],
                            a.chunk, return_y=True)
        lws.append(lw)
        if m is not None:
            wn = torch.softmax(lw, dim=1).to(y.dtype).view(*lw.shape, *([1] * (y.ndim - 2)))
            t_hat = (wn * y).sum(1)                     # SNIS bridge conditional mean E[X1 | x0]
            gaps.append((t_hat - m[sl]).flatten(1).norm(dim=1) / m[0].numel() ** 0.5)
    return torch.cat(lws), (torch.cat(gaps) if gaps else None)


def per_example(lw: torch.Tensor) -> Dict[str, np.ndarray]:
    """Per-example weight diagnostics; chi2 is carried as log10(1 + chi2) = log10(E w^2 / (E w)^2)."""
    st = weight_stats(lw)
    lw = lw.double()
    K = lw.shape[1]
    l1, l2 = torch.logsumexp(lw, 1), torch.logsumexp(2 * lw, 1)
    log_1pchi2 = (l2 - math.log(K)) - 2 * (l1 - math.log(K))
    out = {k: st[k].cpu().numpy() for k in ("control_ess", "ess_frac", "var_logw", "max_w")}
    out["log10_1pchi2"] = (log_1pchi2 / math.log(10)).cpu().numpy()
    out["nonfinite"] = (~torch.isfinite(lw).all(1)).double().cpu().numpy()
    return out


def boot_ci(x: np.ndarray, rng, n_boot: int = 1000):
    x = np.asarray(x, dtype=np.float64)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(pe: Dict[str, np.ndarray], K: int, rng) -> Dict[str, float]:
    out = {}
    for k in METRICS:
        out[k], out[f"{k}_lo"], out[f"{k}_hi"] = boot_ci(pe[k], rng)
    out["control_ess_median"] = float(np.median(pe["control_ess"]))
    out["nonfinite_rate"] = float(pe["nonfinite"].mean())
    med = float(np.median(pe["log10_1pchi2"]))            # budget K* ~ chi2 / delta^2 (Sec. 3.5)
    log10_chi2 = med if med > 1 else math.log10(max(10**med - 1, 1e-12))   # chi2 ~ 1 + chi2 once it is large
    for delta in (0.1, 0.05):
        out[f"log10_budget_delta{delta}"] = max(0.0, log10_chi2 - 2 * math.log10(delta))
    return out


def load(a, device):
    from ptflow.build import load_checkpoint_models
    gen, pot, sched, cfg, step = load_checkpoint_models(a.ckpt, device, ema=not a.raw)
    return gen, pot, sched, step


def meta(a, step, eps=None, **kw):
    return {"checkpoint": a.ckpt, "step": step, "n": a.n, "scale": a.scale, "alpha": a.alpha,
            **({"eps": eps} if eps is not None else {}), **kw}


# ---------------------------------------------------------------------------
# Fig 1. full proposal vs its components
# ---------------------------------------------------------------------------

def exp_proposals(a, device):
    gen, pot, sched, step = load(a, device)
    eps = a.eps or sched.eps()
    ks = [int(k) for k in a.ks.split(",")]
    x0, c, m, s, g = inputs(gen, a.n, a.seed, device)
    s = proposal_scale(a, pot, m, c, s, g)
    z, u = crn_draws(x0, max(ks), g)
    rng = np.random.default_rng(a.seed)
    rows, raw = [], {}
    for name in PROPOSALS:
        center, scale, alpha = proposal_spec(name, x0, m, s, a.alpha)
        t0 = time.time()
        lw, _ = batched_log_weights(a, pot, x0, c, center, scale, alpha, eps, z, u)
        seconds = (time.time() - t0) / a.n
        for k in ks:
            pe = per_example(lw[:, :k])
            rows.append({"proposal": name, "K": k, "eps": eps, "alpha": alpha, "sec_per_example": seconds,
                         **summarize(pe, k, rng)})
            raw.update({f"{name}|K{k}|{m_}": v for m_, v in pe.items()})
    write(Path(a.out), "proposals", rows, meta(a, step, eps, ks=ks, rows=rows), raw)


# ---------------------------------------------------------------------------
# Fig 2. ESS / variance vs eps, with the Laplace-mismatch floor and the T_eps - m gap
# ---------------------------------------------------------------------------

def laplace_mismatch(pot, m, c, s, probes: int, g) -> torch.Tensor:
    """Hutchinson estimate of 0.5 ||S^1/2 (I + grad^2 phi) S^1/2 - I||_F^2 per example.

    The eps-independent part of Var(log w) for a Gaussian proposal centred at the mode
    (Thm 3.9, the 0.5 z^T B z term); the O(eps) reversal needs it small.
    """
    root = torch.ones_like(m) if s is None else torch.exp(0.5 * s)
    acc = torch.zeros(m.shape[0], device=m.device, dtype=torch.float64)
    for _ in range(probes):
        v = torch.randn(m.shape, generator=g, device=m.device)
        vs = root * v
        mv = root * (vs + guided_hvp(pot, m, c, 0.0, vs))
        acc += (mv - v).flatten(1).double().square().sum(1)
    return 0.5 * acc / probes


def exp_eps_sweep(a, device):
    gen, pot, sched, step = load(a, device)
    eps_list = [float(e) for e in a.eps_list.split(",")]
    x0, c, m, s, g = inputs(gen, a.n, a.seed, device)
    s = proposal_scale(a, pot, m, c, s, g)
    z, u = crn_draws(x0, a.K, g)
    rng = np.random.default_rng(a.seed)
    rows, raw = [], {}
    for eps in eps_list:
        for name in PROPOSALS:
            center, scale, alpha = proposal_spec(name, x0, m, s, a.alpha)
            lw, gap = batched_log_weights(a, pot, x0, c, center, scale, alpha, eps, z, u, m=m)
            pe = per_example(lw)
            pe["tgap_rms"] = gap.cpu().numpy()
            row = {"eps": eps, "eps_times_d": eps * m[0].numel(), "proposal": name, "K": a.K, **summarize(pe, a.K, rng)}
            row["tgap_rms"], row["tgap_rms_lo"], row["tgap_rms_hi"] = boot_ci(pe["tgap_rms"], rng)
            rows.append(row)
            raw.update({f"{name}|eps{eps}|{k}": v for k, v in pe.items()})
    mism = {}
    for label, sc in (("proposal_scale", s), ("identity_scale", None)):
        vals = torch.cat([laplace_mismatch(pot, m[i:i + a.batch], c[i:i + a.batch],
                                           None if sc is None else sc[i:i + a.batch], a.probes, g)
                          for i in range(0, a.n, a.batch)]).cpu().numpy()
        mean, lo, hi = boot_ci(vals, rng)
        mism[label] = {"mean": mean, "lo": lo, "hi": hi, "median": float(np.median(vals))}
        raw[f"mismatch|{label}"] = vals
    slopes = {}
    le = np.log(np.array(eps_list))
    for name in PROPOSALS:
        per = np.stack([raw[f"{name}|eps{e}|var_logw"] for e in eps_list])       # [E, n]

        def slope(idx):
            return float(np.polyfit(le, np.log(np.clip(per[:, idx].mean(1), 1e-12, None)), 1)[0])
        boot = [slope(rng.integers(0, per.shape[1], per.shape[1])) for _ in range(500)]
        slopes[name] = {"slope": slope(np.arange(per.shape[1])),
                        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    write(Path(a.out), "eps_sweep", rows,
          meta(a, step, K=a.K, eps_list=eps_list, laplace_mismatch_half_frobenius_sq=mism, slopes=slopes,
               theory="naive: Var ~ |grad phi|^2 / (2 eps) (slope -1); tilted: Var <= eps (...) (slope +1) "
                      "only if the mismatch term is O(eps)", rows=rows), raw)


# ---------------------------------------------------------------------------
# Fig 3. prox residual / approximation quality
# ---------------------------------------------------------------------------

def exp_prox(a, device):
    from ptflow.sampling import refine
    gen, pot, sched, step = load(a, device)
    rows, raw = [], {}
    q = lambda t: {f"q{p}": float(np.percentile(t, p)) for p in (10, 25, 50, 75, 90, 99)}
    for w in [float(x) for x in a.ws.split(",")]:
        x0, c, m, s, g = inputs(gen, a.n, a.seed, device, w)
        d = m[0].numel()
        r = torch.cat([prox_residual(pot, m[i:i + a.batch], x0[i:i + a.batch], c[i:i + a.batch], w).detach()
                       for i in range(0, a.n, a.batch)])
        rn = r.flatten(1).norm(dim=1)
        disp = (m - x0).flatten(1).norm(dim=1).clamp_min(1e-6)
        stats = {"resid_rel": (rn / disp).cpu().numpy(), "resid_rms": (rn / d**0.5).cpu().numpy(),
                 "displacement_rms": (disp / d**0.5).cpu().numpy()}
        e0 = torch.cat([prox_energy(pot, m[i:i + a.batch], x0[i:i + a.batch], c[i:i + a.batch], w)
                        for i in range(0, a.n, a.batch)])
        for n_ref in [int(x) for x in a.refine.split(",")]:
            ys, es = zip(*[refine(pot, m[i:i + a.batch], x0[i:i + a.batch], c[i:i + a.batch], w, steps=n_ref)
                           for i in range(0, a.n, a.batch)])
            y, e = torch.cat(ys), torch.cat(es)
            rr = torch.cat([prox_residual(pot, y[i:i + a.batch], x0[i:i + a.batch], c[i:i + a.batch], w).detach()
                            for i in range(0, a.n, a.batch)]).flatten(1).norm(dim=1)
            stats[f"refine{n_ref}_resid_rel"] = (rr / disp).cpu().numpy()
            stats[f"refine{n_ref}_energy_drop_per_dim"] = ((e0 - e) / d).cpu().numpy()
            stats[f"refine{n_ref}_move_rms"] = ((y - m).flatten(1).norm(dim=1) / d**0.5).cpu().numpy()
        for k, v in stats.items():
            rows.append({"w": w, "stat": k, "mean": float(v.mean()), **q(v)})
            raw[f"w{w}|{k}"] = v
    write(Path(a.out), "prox", rows, meta(a, step, eps=sched.eps(), rows=rows), raw)


# ---------------------------------------------------------------------------
# Fig 4. proposal mismatch, against the Gaussian prediction
# ---------------------------------------------------------------------------

def exp_mismatch(a, device):
    """Perturb the proposal around the checkpoint's own (full) proposal and measure the damage.

    Predictions for a Gaussian target matched by the unperturbed proposal:
      centre shift delta (in proposal std per coordinate):  extra Var(log w) = |delta * dir / e^{s/2}|^2
      variance multiplier kappa:                          extra Var(log w) = (d / 2) (kappa - 1)^2
    """
    gen, pot, sched, step = load(a, device)
    eps = a.eps or sched.eps()
    x0, c, m, s, g = inputs(gen, a.n, a.seed, device)
    s = proposal_scale(a, pot, m, c, s, g)
    if s is None:
        s = torch.zeros_like(m)
    z, u = crn_draws(x0, a.K, g)
    direction = torch.randn(m.shape, generator=g, device=device)
    d = m[0].numel()
    rng = np.random.default_rng(a.seed)
    rows, raw = [], {}
    grid = [("shift", float(x), 1.0) for x in a.shifts.split(",")] + [("scale", 0.0, float(x)) for x in a.scales.split(",")]
    for alpha in [float(x) for x in a.alphas.split(",")]:
        for kind, delta, kappa in grid:
            center = m + delta * math.sqrt(2 * eps) * direction
            scale = s + math.log(kappa)
            lw, _ = batched_log_weights(a, pot, x0, c, center, scale, alpha, eps, z, u)
            pe = per_example(lw)
            pred_shift = float((delta * direction * torch.exp(-0.5 * s)).flatten(1).square().sum(1).mean())
            rows.append({"alpha": alpha, "kind": kind, "center_shift_std": delta, "variance_multiplier": kappa,
                         "pred_extra_var_logw": pred_shift if kind == "shift" else 0.5 * d * (kappa - 1) ** 2,
                         **summarize(pe, a.K, rng)})
            raw.update({f"a{alpha}|{kind}|{delta}|{kappa}|{k}": v for k, v in pe.items()})
    write(Path(a.out), "mismatch", rows, meta(a, step, eps, K=a.K, rows=rows), raw)


# ---------------------------------------------------------------------------
# Fig 5. likelihood / NLL on held-out ImageNet latents
# ---------------------------------------------------------------------------

def exp_nll(a, device):
    from ptflow.data import build_dataset
    from ptflow.likelihood import nested_log_likelihood
    gen, pot, sched, step = load(a, device)
    eps = a.eps or sched.eps()
    cfg = torch.load(a.ckpt, map_location="cpu", weights_only=False)["config"]
    ds = build_dataset(cfg["data"], "val")
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(a.seed))[: a.n]
    x1 = torch.stack([ds[int(i)][0] for i in idx]).to(device)
    c = torch.tensor([ds[int(i)][1] for i in idx], device=device)
    rng = np.random.default_rng(a.seed)
    rows, raw = [], {}
    summary = meta(a, step, eps, label="latent-space nested-MC NLL estimate (not a bound; not pixel bits/dim)", grid=[])
    for ko in [int(x) for x in a.k_outer.split(",")]:
        for ki in [int(x) for x in a.k_inner.split(",")]:
            per_seed, outer_ess, inner_ess = [], [], []
            for seed in range(a.seeds):
                g = torch.Generator(device=device).manual_seed(1000 + seed)
                out = [nested_log_likelihood(gen, pot, x1[i:i + a.batch], c[i:i + a.batch], eps, k_outer=ko,
                                             k_inner=ki, alpha=a.alpha, generator=g, chunk=a.chunk)
                       for i in range(0, a.n, a.batch)]
                res = {k: torch.cat([o[k] for o in out]).cpu().numpy() for k in out[0]}
                per_seed.append(res["nll_bits_per_dim"])
                outer_ess.append(res["outer_control_ess"])
                inner_ess.append(res["inner_control_ess"])
                for j in range(a.n):
                    rows.append({"k_outer": ko, "k_inner": ki, "seed": seed, "example": int(idx[j]),
                                 "nll_bits_per_dim": float(res["nll_bits_per_dim"][j]),
                                 "outer_control_ess": float(res["outer_control_ess"][j]),
                                 "inner_control_ess": float(res["inner_control_ess"][j])})
            bpd = np.stack(per_seed)                                   # [seeds, n]
            mean, lo, hi = boot_ci(bpd.mean(0), rng)
            summary["grid"].append({"k_outer": ko, "k_inner": ki, "bits_per_dim": mean, "lo": lo, "hi": hi,
                                    "seed_std": float(bpd.std(0).mean()) if a.seeds > 1 else None,
                                    "outer_control_ess": float(np.mean(outer_ess)),
                                    "inner_control_ess": float(np.mean(inner_ess))})
            raw[f"ko{ko}|ki{ki}|bpd"] = bpd
    write(Path(a.out), "nll", rows, summary, raw)


# ---------------------------------------------------------------------------
# Jacobian audit of a generator: can it be the prox of a potential?
# ---------------------------------------------------------------------------

def generator_jacobian(gen, x0, c, chunk: int) -> torch.Tensor:
    """Full d x d Jacobian dm/dx0 at one point via batched reverse-mode rows (fp32 forward)."""
    d = x0.numel()
    rows = []
    eye = torch.eye(d, device=x0.device)
    for i in range(0, d, chunk):
        n = min(chunk, d - i)
        xb = x0.detach().expand(n, *x0.shape).clone().requires_grad_(True)
        with torch.enable_grad():
            use = gen.use_bf16
            gen.use_bf16 = False
            m, _ = gen(xb, c.expand(n), 0.0, with_scale=False)
            gen.use_bf16 = use
            (gx,) = torch.autograd.grad(m.reshape(n, d), xb, grad_outputs=eye[i:i + n])
        rows.append(gx.reshape(n, d).double())
    return torch.cat(rows)                                    # J[i, j] = d m_i / d x0_j


def exp_jacobian(a, device):
    """If m = prox_phi then dm/dx0 = (I + grad^2 phi)^-1: symmetric positive definite.  Its antisymmetric
    part bounds how well calibration can succeed; H = J_sym^-1 gives the ESS ceiling of a diagonal
    proposal even for a potential perfectly matched to this generator."""
    from ptflow.build import load_checkpoint_models, load_wflow_only
    from ptflow.config import load_config
    if a.ckpt:
        gen = load_checkpoint_models(a.ckpt, device)[0]
    else:
        cfg = load_config(a.config)
        gen, _ = load_wflow_only(cfg["generator"], a.wflow_ckpt or cfg["init"]["wflow_ckpt"], device,
                                 noise_code=cfg["init"].get("noise_code", "fixed:0"))
    x0, c, m, _, g = inputs(gen, a.n, a.seed, device)
    rows = []
    for i in range(a.n):
        J = generator_jacobian(gen, x0[i], c[i:i + 1], a.chunk)
        sym, asym = 0.5 * (J + J.T), 0.5 * (J - J.T)
        ev, V = torch.linalg.eigh(sym)
        row = {"example": i, "asymmetry_rel": float(asym.norm() / J.norm()), "eig_min": float(ev.min()),
               "eig_max": float(ev.max()), "negative_eig_frac": float((ev <= 0).double().mean())}
        if ev.min() > 0:
            H = (V / ev) @ V.T                                  # (I + grad^2 phi) implied by m = prox
            D = torch.diagonal(H)
            B = (H - torch.diag(D)) / torch.sqrt(D[:, None] * D[None, :])
            row["half_offdiag_frobenius_sq"] = float(0.5 * B.square().sum())
            # ESS of the best diagonal proposal on the Laplace target (Gaussian, eps-independent)
            L = torch.linalg.cholesky(torch.eye(len(D), device=device, dtype=torch.float64) + B)
            zz = torch.randn((a.K * 64, len(D)), generator=g, device=device, dtype=torch.float64)
            lw = (-0.5 * (zz @ L).square().sum(1) + 0.5 * zz.square().sum(1)).view(64, a.K)
            row["diag_proposal_control_ess"] = float(weight_stats(lw)["control_ess"].mean())
        rows.append(row)
        print(json.dumps(row), flush=True)
    keys = [k for k in rows[0] if k != "example"]
    summary = {k: float(np.nanmean([r.get(k, np.nan) for r in rows])) for k in keys}
    summary.update(n=a.n, K=a.K, rows=rows)
    write(Path(a.out), "jacobian", rows, summary)


# ---------------------------------------------------------------------------
# Toy: exact bridge conditional mean vs prox, by quadrature (separable, any d)
# ---------------------------------------------------------------------------

def exp_toy(a, device):
    """phi(y) = sum_i [a y_i^2 / 2 + b log cosh(y_i - mu_i)] (separable, convex).

    Per coordinate:  T_eps(x) = E[y] under exp(-(phi(y) + (y - x)^2 / 2) / (2 eps))  (quadrature),
                     y* = prox_phi(x)                                               (Newton).
    Reports |T_eps - y*| vs eps (Sec. 3.2: exact up to O(eps)) and the Prop. 3.4 kernel spread.
    """
    rng = np.random.default_rng(a.seed)
    d, A, Bc = a.toy_dim, 0.5, 2.0
    mu, x = rng.normal(size=d) * 2.0, rng.normal(size=d)
    phi = lambda y, j: 0.5 * A * y**2 + Bc * np.log(np.cosh(y - mu[j]))
    dphi = lambda y, j: A * y + Bc * np.tanh(y - mu[j])
    ddphi = lambda y, j: A + Bc / np.cosh(y - mu[j]) ** 2
    ystar = x.copy()
    for _ in range(100):
        ystar -= (np.array([dphi(ystar[j], j) for j in range(d)]) + ystar - x) / (1 + np.array([ddphi(ystar[j], j) for j in range(d)]))
    grid = np.linspace(-15, 15, 200001)
    rows = []
    for eps in [float(e) for e in a.eps_list.split(",")]:
        T, var = np.zeros(d), np.zeros(d)
        for j in range(d):
            e = -(phi(grid, j) + 0.5 * (grid - x[j]) ** 2) / (2 * eps)
            p = np.exp(e - e.max())
            p /= p.sum()
            T[j] = (p * grid).sum()
            var[j] = (p * (grid - T[j]) ** 2).sum()
        rows.append({"eps": eps, "T_minus_prox_norm": float(np.linalg.norm(T - ystar)),
                     "T_minus_prox_rms": float(np.sqrt(np.mean((T - ystar) ** 2))),
                     "kernel_spread_sqrt_tr_cov": float(np.sqrt(var.sum())),
                     "laplace_spread": float(np.sqrt(sum(2 * eps / (1 + ddphi(ystar[j], j)) for j in range(d))))})
    le = np.log([r["eps"] for r in rows])
    slope = float(np.polyfit(le, np.log([r["T_minus_prox_norm"] for r in rows]), 1)[0])
    write(Path(a.out), "toy", rows, {"dim": d, "slope_log_gap_vs_log_eps": slope,
                                     "theory": "gap = O(eps) -> slope ~ 1", "rows": rows})


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["proposals", "eps-sweep", "prox", "mismatch", "nll", "jacobian", "toy"])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--wflow-ckpt", default="")
    ap.add_argument("--out", default="results")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.0, help="default: the checkpoint's eps")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--ks", default="8,16,64")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--probes", type=int, default=4)
    ap.add_argument("--eps-list", default="0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001")
    ap.add_argument("--scale", default="learned", choices=["learned", "hutchinson"],
                    help="diagonal proposal scale: the generator's head, or Hutchinson on the potential")
    ap.add_argument("--hutch-probes", type=int, default=16)
    ap.add_argument("--ws", default="0,0.2,1.0")
    ap.add_argument("--refine", default="1,3,10")
    ap.add_argument("--shifts", default="0,0.002,0.005,0.01,0.02,0.05,0.1")
    ap.add_argument("--scales", default="0.8,0.9,0.95,1,1.05,1.1,1.25")
    ap.add_argument("--alphas", default="0,0.1,0.5")
    ap.add_argument("--k-outer", default="4,16,64")
    ap.add_argument("--k-inner", default="4,16,64")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--toy-dim", type=int, default=2)
    a = ap.parse_args()
    device = dist.init()
    fn = {"proposals": exp_proposals, "eps-sweep": exp_eps_sweep, "prox": exp_prox, "mismatch": exp_mismatch,
          "nll": exp_nll, "jacobian": exp_jacobian, "toy": exp_toy}[a.experiment]
    if a.experiment not in ("toy", "jacobian") and not a.ckpt:
        raise SystemExit("--ckpt is required")
    fn(a, device)


if __name__ == "__main__":
    main()
