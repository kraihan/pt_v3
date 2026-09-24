"""Mechanism experiments at a fixed checkpoint (no training).  Each writes <out>/<name>.csv and .json.

    python experiments.py proposals  --ckpt CKPT --out results/   # full proposal vs its components
    python experiments.py eps-sweep  --ckpt CKPT --out results/   # ESS / Var(log w) / chi2 vs eps
    python experiments.py prox       --ckpt CKPT --out results/   # prox residual, refinement, T = prox = m gap
    python experiments.py mismatch   --ckpt CKPT --out results/   # controlled proposal mismatch
    python experiments.py nll        --ckpt CKPT --out results/   # nested-MC latent NLL, budget grid x seeds
    python experiments.py jacobian   --config configs/B.yaml --out results/   # W-Flow map: gradient-map audit
    python experiments.py toy        --out results/               # exact T_eps vs prox (quadrature, no network)

Proposals (common random numbers across all of them):
    naive       N(x0, 2 eps I)                               the estimator of eq. 8
    recentered  N(m, 2 eps I)                                centre only
    diagonal    N(m, 2 eps diag e^s)                         centre + learned diagonal scale
    full        (1-alpha) N(m, 2 eps diag e^s) + alpha N(x0, 2 eps I)   the full proposal (eqs. 9-10)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from ptflow import dist
from ptflow.estimator import log_kernel_over_proposal, proposal_points, weight_stats
from ptflow.models.potential import guided_hvp, prox_energy, prox_residual

PROPOSALS = ("naive", "recentered", "diagonal", "full")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def write(out: Path, name: str, rows: List[Dict], summary: Dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
    (out / f"{name}.json").write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))


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


def proposal_spec(name: str, x0, m, s, alpha: float):
    """(centre, log-scale, defensive alpha) of each proposal."""
    if name == "naive":
        return x0, None, 1.0
    if name == "recentered":
        return m, None, 0.0
    if name == "diagonal":
        return m, s, 0.0
    if name == "full":
        return m, s, alpha
    raise ValueError(name)


@torch.no_grad()
def log_weights(pot, x0, c, center, s, alpha: float, eps: float, z, u, chunk: int, w: float = 0.0) -> torch.Tensor:
    """Unnormalized log-weights [B, K] of the tilted estimator for a given proposal and fixed draws."""
    from ptflow.estimator import potential_phi_fn
    from_def = u < alpha
    y = proposal_points(x0, center, s, eps, z, from_def)
    lr = log_kernel_over_proposal(y, x0, center, s, eps, alpha)
    B, K = y.shape[:2]
    yf, cr = y.reshape(B * K, *y.shape[2:]), c.repeat_interleave(K)
    fn = potential_phi_fn(pot, w)
    step = chunk or B * K
    phi = torch.cat([fn(yf[i:i + step], cr[i:i + step]) for i in range(0, B * K, step)]).view(B, K).double()
    return lr - (phi - phi.mean(1, keepdim=True)) / (2 * eps)


def summarize_weights(lw: torch.Tensor, K: int) -> Dict[str, float]:
    st = weight_stats(lw)
    finite = torch.isfinite(lw).all(dim=1).float()
    chi2 = st["chi2"]
    out = {k: float(v.mean()) for k, v in st.items()}
    out.update(control_ess_median=float(st["control_ess"].median()), chi2_median=float(chi2.median()),
               rel_var=float((chi2 / K).mean()), nonfinite_rate=float(1 - finite.mean()))
    for delta in (0.1, 0.05):
        out[f"budget_delta{delta}"] = float(torch.ceil(chi2.median() / delta**2).clamp_min(1))
    return out


def load(a, device):
    from ptflow.build import load_checkpoint_models
    gen, pot, sched, cfg, step = load_checkpoint_models(a.ckpt, device, ema=not a.raw)
    return gen, pot, sched, step


# ---------------------------------------------------------------------------
# A. full proposal vs components
# ---------------------------------------------------------------------------

def exp_proposals(a, device):
    gen, pot, sched, step = load(a, device)
    eps = a.eps or sched.eps()
    ks = [int(k) for k in a.ks.split(",")]
    x0, c, m, s, g = inputs(gen, a.n, a.seed, device)
    z, u = crn_draws(x0, max(ks), g)
    rows = []
    for name in PROPOSALS:
        center, scale, alpha = proposal_spec(name, x0, m, s, a.alpha)
        rows_name = []
        for i in range(0, a.n, a.batch):
            sl = slice(i, i + a.batch)
            t0 = time.time()
            lw = log_weights(pot, x0[sl], c[sl], center[sl], None if scale is None else scale[sl], alpha, eps,
                             z[sl], u[sl], a.chunk)
            rows_name.append((lw, time.time() - t0))
        lw = torch.cat([r[0] for r in rows_name])
        seconds = sum(r[1] for r in rows_name) / a.n
        for k in ks:
            rows.append({"proposal": name, "K": k, "eps": eps, "alpha": alpha, "sec_per_example": seconds,
                         **summarize_weights(lw[:, :k], k)})
    write(Path(a.out), "proposals", rows, {"checkpoint": a.ckpt, "step": step, "eps": eps, "n": a.n, "rows": rows})


# ---------------------------------------------------------------------------
# B. variance vs eps, with the Laplace mismatch floor
# ---------------------------------------------------------------------------

def laplace_mismatch(pot, m, c, s, probes: int, g) -> torch.Tensor:
    """Hutchinson estimate of 0.5 ||S^1/2 (I + grad^2 phi) S^1/2 - I||_F^2 per example.

    This is the eps-independent part of Var(log w) for a Gaussian proposal centred
    at the mode (Thm 3.9: the 0.5 z^T B z term); the O(eps) reversal needs it small.
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
    z, u = crn_draws(x0, a.K, g)
    rows = []
    for eps in eps_list:
        for name in PROPOSALS:
            center, scale, alpha = proposal_spec(name, x0, m, s, a.alpha)
            lw = torch.cat([log_weights(pot, x0[i:i + a.batch], c[i:i + a.batch], center[i:i + a.batch],
                                        None if scale is None else scale[i:i + a.batch], alpha, eps,
                                        z[i:i + a.batch], u[i:i + a.batch], a.chunk) for i in range(0, a.n, a.batch)])
            rows.append({"eps": eps, "proposal": name, "K": a.K, **summarize_weights(lw, a.K),
                         "_var_logw_per_example": lw.var(dim=1, correction=0).cpu().numpy()})
    mism = {}
    for label, sc in (("learned_scale", s), ("identity_scale", None)):
        vals = torch.cat([laplace_mismatch(pot, m[i:i + a.batch], c[i:i + a.batch],
                                           None if sc is None else sc[i:i + a.batch], a.probes, g)
                          for i in range(0, a.n, a.batch)])
        mism[label] = {"mean": float(vals.mean()), "median": float(vals.median())}
    slopes = {}
    rng = np.random.default_rng(a.seed)
    for name in PROPOSALS:
        per = np.stack([r["_var_logw_per_example"] for r in rows if r["proposal"] == name])   # [E, n]
        le = np.log(np.array(eps_list))

        def slope(idx):
            v = np.log(np.clip(per[:, idx].mean(1), 1e-12, None))
            return float(np.polyfit(le, v, 1)[0])
        boot = [slope(rng.integers(0, per.shape[1], per.shape[1])) for _ in range(200)]
        slopes[name] = {"slope_log_var_vs_log_eps": slope(np.arange(per.shape[1])),
                        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    for r in rows:
        r.pop("_var_logw_per_example")
    write(Path(a.out), "eps_sweep", rows, {"checkpoint": a.ckpt, "step": step, "n": a.n, "K": a.K,
                                           "laplace_mismatch_half_frobenius_sq": mism,
                                           "theory": "Thm 3.9 predicts slope ~1 only if the mismatch term is O(eps)",
                                           "slopes": slopes, "rows": rows})


# ---------------------------------------------------------------------------
# C. prox residual / approximation quality:  T_eps  vs  prox  vs  m
# ---------------------------------------------------------------------------

def exp_prox(a, device):
    from ptflow.estimator import tilted_estimate
    from ptflow.sampling import refine
    gen, pot, sched, step = load(a, device)
    eps = a.eps or sched.eps()
    rows, summary = [], {"checkpoint": a.ckpt, "step": step, "eps": eps, "n": a.n}
    q = lambda t: {f"q{p}": float(torch.quantile(t.float(), p / 100)) for p in (10, 50, 90, 99)}
    for w in [float(x) for x in a.ws.split(",")]:
        x0, c, m, s, g = inputs(gen, a.n, a.seed, device, w)
        d = m[0].numel()
        r = torch.cat([prox_residual(pot, m[i:i + a.batch], x0[i:i + a.batch], c[i:i + a.batch], w).detach()
                       for i in range(0, a.n, a.batch)])
        rn = r.flatten(1).norm(dim=1)
        disp = (m - x0).flatten(1).norm(dim=1)
        rows.append({"w": w, "stat": "resid_abs", **q(rn)})
        rows.append({"w": w, "stat": "resid_rel", **q(rn / disp.clamp_min(1e-6))})
        e0 = prox_energy(pot, m, x0, c, w)
        for n_ref in [int(x) for x in a.refine.split(",")]:
            y, e = refine(pot, m, x0, c, w, steps=n_ref)
            rr = prox_residual(pot, y, x0, c, w).detach().flatten(1).norm(dim=1)
            rows.append({"w": w, "stat": f"refine{n_ref}_energy_drop_per_dim", **q((e0 - e) / d)})
            rows.append({"w": w, "stat": f"refine{n_ref}_distance_from_A_rms", **q((y - m).flatten(1).norm(dim=1) / d**0.5)})
            rows.append({"w": w, "stat": f"refine{n_ref}_resid_rel", **q(rr / disp.clamp_min(1e-6))})
        if w == 0.0:
            # Direct test of T = prox = m: the SNIS posterior mean estimates the bridge conditional mean T_eps(x0).
            gaps = []
            for i in range(0, a.n, a.batch):
                sl = slice(i, i + a.batch)
                est = tilted_estimate(pot.phi, x0[sl], c[sl], m[sl], s[sl], eps, K=a.K, alpha=a.alpha, generator=g,
                                      chunk=a.chunk)
                gaps.append(((est.posterior_mean() - m[sl]).flatten(1).norm(dim=1) / d**0.5,
                             weight_stats(est.log_w)["control_ess"]))
            gap, ess = torch.cat([x[0] for x in gaps]), torch.cat([x[1] for x in gaps])
            rows.append({"w": w, "stat": "T_eps_minus_m_rms", **q(gap)})
            summary["T_eps_minus_m"] = {"median_rms": float(gap.median()), "control_ess_mean": float(ess.mean()),
                                        "note": "reliable only where the control ESS is not near zero"}
    summary["rows"] = rows
    write(Path(a.out), "prox", rows, summary)


# ---------------------------------------------------------------------------
# D. proposal mismatch stress test
# ---------------------------------------------------------------------------

def exp_mismatch(a, device):
    gen, pot, sched, step = load(a, device)
    eps = a.eps or sched.eps()
    x0, c, m, s, g = inputs(gen, a.n, a.seed, device)
    z, u = crn_draws(x0, a.K, g)
    direction = torch.randn(m.shape, generator=g, device=device)
    rows = []
    grid = [("shift", float(x), 1.0) for x in a.shifts.split(",")] + [("scale", 0.0, float(x)) for x in a.scales.split(",")]
    for alpha in [float(x) for x in a.alphas.split(",")]:
        for kind, delta, kappa in grid:
            center = m + delta * math.sqrt(2 * eps) * direction      # delta proposal std per coordinate
            scale = s + math.log(kappa)                               # variance multiplier kappa
            lw = torch.cat([log_weights(pot, x0[i:i + a.batch], c[i:i + a.batch], center[i:i + a.batch],
                                        scale[i:i + a.batch], alpha, eps, z[i:i + a.batch], u[i:i + a.batch], a.chunk)
                            for i in range(0, a.n, a.batch)])
            rows.append({"alpha": alpha, "kind": kind, "center_shift_std": delta, "variance_multiplier": kappa,
                         **summarize_weights(lw, a.K)})
    write(Path(a.out), "mismatch", rows, {"checkpoint": a.ckpt, "step": step, "eps": eps, "K": a.K, "rows": rows})


# ---------------------------------------------------------------------------
# E. likelihood / NLL
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
    rows, summary = [], {"checkpoint": a.ckpt, "step": step, "eps": eps, "n": a.n,
                         "label": "latent-space nested-MC NLL estimate (not a bound; not pixel bits/dim)", "grid": []}
    for ko in [int(x) for x in a.k_outer.split(",")]:
        for ki in [int(x) for x in a.k_inner.split(",")]:
            per_seed = []
            for seed in range(a.seeds):
                g = torch.Generator(device=device).manual_seed(1000 + seed)
                out = [nested_log_likelihood(gen, pot, x1[i:i + a.batch], c[i:i + a.batch], eps, k_outer=ko,
                                             k_inner=ki, alpha=a.alpha, generator=g, chunk=a.chunk)
                       for i in range(0, a.n, a.batch)]
                res = {k: torch.cat([o[k] for o in out]) for k in out[0]}
                per_seed.append(res["nll_bits_per_dim"])
                for j in range(a.n):
                    rows.append({"k_outer": ko, "k_inner": ki, "seed": seed, "example": int(idx[j]),
                                 "nll_bits_per_dim": float(res["nll_bits_per_dim"][j]),
                                 "nll_nats_per_dim": float(res["nll_nats_per_dim"][j]),
                                 "outer_control_ess": float(res["outer_control_ess"][j]),
                                 "inner_control_ess": float(res["inner_control_ess"][j])})
            bpd = torch.stack(per_seed)                        # [seeds, n]
            ex_mean = bpd.mean(0)
            summary["grid"].append({
                "k_outer": ko, "k_inner": ki, "bits_per_dim_mean": float(ex_mean.mean()),
                "ci95_over_examples": 1.96 * float(ex_mean.std() / math.sqrt(a.n)),
                "seed_std_per_example_mean": float(bpd.std(0).mean()) if a.seeds > 1 else None})
    write(Path(a.out), "nll", rows, summary)


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
    ap.add_argument("--eps-list", default="0.2,0.1,0.05,0.02,0.01,0.005")
    ap.add_argument("--ws", default="0,0.2,1.0")
    ap.add_argument("--refine", default="1,3,10")
    ap.add_argument("--shifts", default="0,0.25,0.5,1,2")
    ap.add_argument("--scales", default="0.25,0.5,1,2,4")
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
