"""Paper figures 1-5 from the outputs of experiments.py (and optionally a training log).

    python scripts/make_figures.py results/fig_calib                       # figs from one checkpoint
    python scripts/make_figures.py results/fig_calib --train-log runs/B_eps/metrics.jsonl

Writes <results>/figures/fig{1..5}_*.{pdf,png} and <results>/figures/summary.md.
Every point is a mean over examples with a bootstrap 95% CI; identity is carried by
colour + marker + a direct label, so the figures survive greyscale printing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e6e5e1", "#8a8986"
# Fixed categorical order (validated palette, adjacent CVD dE >= 9.1): the method first.
PROPOSAL_STYLE = {
    "full":       ("#2a78d6", "o", "full (defensive)"),
    "curvature":  ("#eb6834", "s", "curvature-scaled"),
    "recentered": ("#1baf7a", "^", "recentred only"),
    "naive":      ("#eda100", "D", "naive MC"),
}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5, "lines.markersize": 4.5,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300,
})


def load(results: Path, name: str):
    p = results / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def series(ax, x, y, lo=None, hi=None, *, color, marker, label, direct=True):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if lo is not None:
        yerr = np.vstack([y - np.asarray(lo, float), np.asarray(hi, float) - y]).clip(min=0)
        ax.errorbar(x, y, yerr=yerr, color=color, marker=marker, label=label, capsize=0, elinewidth=0.8,
                    markeredgecolor="white", markeredgewidth=0.6)
    else:
        ax.plot(x, y, color=color, marker=marker, label=label, markeredgecolor="white", markeredgewidth=0.6)
    if direct and len(x):
        if not hasattr(ax, "_direct"):
            ax._direct = []
        ax._direct.append((x[-1], y[-1], label))


def place_direct_labels(fig, min_gap_pt: float = 8.0):
    """Label line ends in neutral ink, nudged apart vertically so overlapping series stay legible."""
    fig.canvas.draw()
    gap = min_gap_pt * fig.dpi / 72.0
    for ax in fig.axes:
        items = getattr(ax, "_direct", [])
        if not items:
            continue
        pts = [(ax.transData.transform((x, y)), lab, (x, y)) for x, y, lab in items]
        pts.sort(key=lambda t: t[0][1])
        placed = []
        for disp, lab, xy in pts:
            ynew = disp[1] if not placed else max(disp[1], placed[-1] + gap)
            placed.append(ynew)
            ax.annotate(lab, xy, xytext=(4, (ynew - disp[1]) * 72.0 / fig.dpi), textcoords="offset points",
                        va="center", fontsize=6.5, color=INK2, annotation_clip=False)


def panel(ax, tag: str, title: str):
    ax.set_title(f"({tag}) {title}", loc="left", color=INK)


def save(fig, out: Path, name: str):
    place_direct_labels(fig)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


def legend_below(fig, ax, ncol=4):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol)


def adaptive_xscale(ax, values, axis="x"):
    v = np.asarray([x for x in values if np.isfinite(x) and x > 0], float)
    target = ax.xaxis if axis == "x" else ax.yaxis
    if len(v) and v.max() / v.min() > 20:
        (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    else:
        target.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        span = (v.max() - v.min()) if len(v) > 1 else 1.0
        decimals = min(4, max(0, int(math.ceil(-math.log10(max(span, 1e-12)))) + 1))
        target.set_major_formatter(matplotlib.ticker.FormatStrFormatter(f"%.{decimals}f"))


def read_log(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


# ---------------------------------------------------------------------------

def hbars(ax, names, vals, lo, hi, *, log=False, fmt="{:.3g}"):
    """One thin horizontal bar per proposal, value label in neutral ink."""
    y = np.arange(len(names))[::-1]
    for yi, n, v, l, h in zip(y, names, vals, lo, hi):
        col = PROPOSAL_STYLE[n][0]
        if not np.isfinite(v):
            ax.text(0.02, yi, "n/a (too few draws)", transform=ax.get_yaxis_transform(), va="center",
                    fontsize=6.5, color=INK2)
            continue
        ax.barh(yi, v, height=0.55, color=col, edgecolor="white", linewidth=1.0)
        end = v
        if np.isfinite(l) and np.isfinite(h):
            ax.plot([l, h], [yi, yi], color=INK2, lw=0.8)
            end = max(v, h)
        ax.annotate(fmt.format(v), (end, yi), xytext=(3, 0), textcoords="offset points", va="center",
                    fontsize=6.5, color=INK2, annotation_clip=False)
    ax.set_yticks(y)
    ax.set_yticklabels([PROPOSAL_STYLE[n][2] for n in names])
    ax.grid(axis="y", visible=False)
    if log:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=4))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.margins(x=0.25)


def centre_note(d):
    parts = [f"centre: {d.get('center', 'amortized')}", f"scale: {d.get('scale')}"]
    if "generator_resid_rel" in d:
        parts.append(f"generator rel. residual {d['generator_resid_rel']:.2g}")
    if "centre_resid_rms" in d:
        parts.append(f"refined-centre residual RMS {d['centre_resid_rms']:.2g}/coord")
    return ", ".join(parts)


def fig1(results, out, md):
    d = load(results, "proposals")
    if d is None:
        return
    rows = d["rows"]
    kmax = max(x["K"] for x in rows)
    names = list(PROPOSAL_STYLE)
    at_k = {x["proposal"]: x for x in rows if x["K"] == kmax}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3), layout="constrained")
    for name, (col, mk, lab) in PROPOSAL_STYLE.items():
        r = sorted((x for x in rows if x["proposal"] == name), key=lambda x: x["K"])
        series(axes[0], [x["K"] for x in r], [x["control_ess"] for x in r], [x["control_ess_lo"] for x in r],
               [x["control_ess_hi"] for x in r], color=col, marker=mk, label=lab, direct=False)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("draws K")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("control ESS  (ESS−1)/(K−1)")
    panel(axes[0], "a", "weight health vs K")
    hbars(axes[1], names, [max(at_k[n]["var_logw"], 1e-6) for n in names],
          [at_k[n]["var_logw_lo"] for n in names], [at_k[n]["var_logw_hi"] for n in names], log=True)
    axes[1].set_xlabel("s² = Var(log w)")
    panel(axes[1], "b", "log-weight variance")
    hbars(axes[2], names, [at_k[n]["log10_kstar"] for n in names], [at_k[n]["log10_kstar_lo"] for n in names],
          [at_k[n]["log10_kstar_hi"] for n in names], fmt="10^{:.3g}")
    axes[2].set_yticklabels([])
    axes[2].set_xlabel("log$_{10}$ draws needed")
    panel(axes[2], "c", "draws needed")
    legend_below(fig, axes[0])
    save(fig, out, "fig1_proposals")
    md.append(f"## Fig 1 - proposals (eps={d.get('eps'):.4g}, n={d['n']}, K={kmax}; {centre_note(d)})\n")
    md.append("Caption notes: bars at K = %d; s^2 is Var(log w) over Laplace-component draws (Thm 3.9); draws "
              "needed 1 + chi^2 ~ e^{s^2} / (1 - alpha) (log-normal, Prop. 3.7; defensive bound, Sec. 3.5).\n" % kmax)
    md.append("| proposal | control ESS [95% CI] | collapsed fraction | s^2 (Laplace draws) | log10 draws needed | "
              "plug-in log10(1+chi2) (<= log10 K) |\n|---|---|---|---|---|---|")
    for n in names:
        x = at_k[n]
        md.append(f"| {n} | {x['control_ess']:.3f} [{x['control_ess_lo']:.3f}, {x['control_ess_hi']:.3f}] "
                  f"| {x['collapsed_frac']:.2f} | {x['var_logw']:.3g} | {x['log10_kstar']:.3g} "
                  f"| {x['log10_1pchi2_plugin']:.3g} |")
    md.append("")


def fig2(results, out, md, train_log=None):
    d = load(results, "eps_sweep")
    if d is None:
        return
    rows = d["rows"]
    ncol = 3 if train_log else 2
    fig, axes = plt.subplots(1, ncol, figsize=(2.3 * ncol, 2.4), layout="constrained")
    for name, (col, mk, lab) in PROPOSAL_STYLE.items():
        r = sorted((x for x in rows if x["proposal"] == name), key=lambda x: -x["eps"])
        e = [x["eps"] for x in r]
        sl = d["slopes"][name]
        series(axes[0], e, [max(x["var_logw"], 1e-6) for x in r], [max(x["var_logw_lo"], 1e-6) for x in r],
               [x["var_logw_hi"] for x in r], color=col, marker=mk, label=f"{lab}, slope {sl['slope']:+.2f}")
        series(axes[1], e, [x["control_ess"] for x in r], [x["control_ess_lo"] for x in r],
               [x["control_ess_hi"] for x in r], color=col, marker=mk, label=lab, direct=False)
        axes[0]._direct[-1] = axes[0]._direct[-1][:2] + (lab,)
    mm = d["laplace_mismatch_half_frobenius_sq"]
    for key, ls, text in (("proposal_scale", (0, (4, 2)), "Laplace floor ½‖B‖², proposal scale"),
                          ("identity_scale", (0, (1, 2)), "Laplace floor, identity scale")):
        floor = max(mm[key]["mean"], 1e-6)
        axes[0].axhline(floor, color=INK2, lw=1.0, ls=ls)
        axes[0].annotate(text, (min(x["eps"] for x in rows), floor), xytext=(2, 3), textcoords="offset points",
                         fontsize=6.0, color=INK2)
    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("ε (decreasing →)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("s² = Var(log w), Laplace draws")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("control ESS")
    panel(axes[0], "a", "log-weight variance vs ε")
    panel(axes[1], "b", "weight health vs ε")
    legend_below(fig, axes[0], ncol=2)
    if train_log:
        log = [r for r in read_log(train_log) if "est/var_logw" in r and "sched/eps" in r]
        if log:
            e = np.array([r["sched/eps"] for r in log], float)
            v = np.array([r["est/var_logw"] for r in log], float)
            s = np.array([r.get("sched/control_ess_ema", np.nan) for r in log], float)
            axes[2].plot(e, np.clip(v, 1e-6, None), color=SERIES[0], marker="o", ms=2.5, lw=0.8,
                         label="Var(log w), full proposal")
            axes[2].set_xscale("log")
            axes[2].set_yscale("log")
            axes[2].invert_xaxis()
            axes[2].set_xlabel("ε during the annealing run")
            axes[2].set_ylabel("Var(log w)")
            panel(axes[2], "c", "during training")
            md.append(f"Training log: {len(log)} logged steps, eps {e.max():.3g} -> {e.min():.3g}, "
                      f"final control-ESS EMA {s[-1]:.3g}.\n")
    save(fig, out, "fig2_eps")
    md.append(f"## Fig 2 - variance vs eps (n={d['n']}, K={d['K']}; {centre_note(d)})\n")
    md.append("| proposal | slope d log Var / d log eps [95% CI] |\n|---|---|")
    for name, sl in d["slopes"].items():
        md.append(f"| {name} | {sl['slope']:+.3f} [{sl['ci95'][0]:+.3f}, {sl['ci95'][1]:+.3f}] |")
    mm = d["laplace_mismatch_half_frobenius_sq"]
    md.append(f"\nLaplace mismatch 0.5||S^1/2 H S^1/2 - I||_F^2: proposal scale {mm['proposal_scale']['mean']:.3g} "
              f"[{mm['proposal_scale']['lo']:.3g}, {mm['proposal_scale']['hi']:.3g}], identity scale "
              f"{mm['identity_scale']['mean']:.3g}. Theory's O(eps) reversal needs this to be small.\n")


def fig3(results, out, md, train_log=None):
    d, sweep = load(results, "prox"), load(results, "eps_sweep")
    if d is None:
        return
    raw = np.load(results / "prox_raw.npz")
    ws = sorted({float(k.split("|")[0][1:]) for k in raw.files})
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.2), layout="constrained")
    all_resid, all_steps = [], [0]
    for i, w in enumerate(ws):
        v = np.sort(raw[f"w{w}|resid_rel"])
        axes[0].step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=SERIES[i], label=f"w = {w:g}")
        steps = sorted({int(k.split("|")[1].split("_")[0][6:]) for k in raw.files if k.startswith(f"w{w}|refine")})
        med = [np.median(raw[f"w{w}|resid_rel"])] + [np.median(raw[f"w{w}|refine{n}_resid_rel"]) for n in steps]
        q1 = [np.percentile(raw[f"w{w}|resid_rel"], 25)] + [np.percentile(raw[f"w{w}|refine{n}_resid_rel"], 25) for n in steps]
        q3 = [np.percentile(raw[f"w{w}|resid_rel"], 75)] + [np.percentile(raw[f"w{w}|refine{n}_resid_rel"], 75) for n in steps]
        series(axes[1], [0] + steps, med, q1, q3, color=SERIES[i], marker=MARKERS[i], label=f"w = {w:g}")
        all_resid.extend(v.tolist())
        all_steps = sorted(set(all_steps) | set(steps))
    adaptive_xscale(axes[0], all_resid)
    axes[0].set_xlabel("‖∇φ(m)+m−x₀‖ / ‖m−x₀‖")
    axes[0].set_ylabel("fraction of samples")
    panel(axes[0], "a", "prox residual, Mode A (CDF)")
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set_xticks(all_steps)
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Mode-B refinement steps")
    axes[1].set_ylabel("rel. residual (median, IQR)")
    panel(axes[1], "b", "refinement")
    if sweep is not None:
        drawn = False
        for name in ("full", "curvature"):
            col, mk, lab = PROPOSAL_STYLE[name]
            r = sorted((x for x in sweep["rows"] if x["proposal"] == name and x.get("tgap_valid_frac", 1) >= 0.5
                        and np.isfinite(x["tgap_rms"])), key=lambda x: -x["eps"])
            if r:
                drawn = True
                series(axes[2], [x["eps"] for x in r], [x["tgap_rms"] for x in r], [x["tgap_rms_lo"] for x in r],
                       [x["tgap_rms_hi"] for x in r], color=col, marker=mk, label=lab)
        if not drawn:
            axes[2].text(0.5, 0.5, "weights collapsed at every ε:\nT̂$_ε$ not estimable", transform=axes[2].transAxes,
                         ha="center", va="center", fontsize=7, color=INK2)
        axes[2].set_xscale("log")
        adaptive_xscale(axes[2], [x["tgap_rms"] for x in sweep["rows"] if x["proposal"] in ("full", "curvature")], axis="y")
        axes[2].invert_xaxis()
        axes[2].set_xlabel("ε (decreasing →)")
        axes[2].set_ylabel("RMS  T̂$_ε$(x₀) − m(x₀)")
        panel(axes[2], "c", "T = prox = m gap")
    legend_below(fig, axes[0], ncol=len(ws))
    save(fig, out, "fig3_prox")
    md.append("## Fig 3 - prox residual\n")
    md.append("| w | median rel. residual | 90th pct | after max refinement (median) |\n|---|---|---|---|")
    for w in ws:
        steps = sorted({int(k.split("|")[1].split("_")[0][6:]) for k in raw.files if k.startswith(f"w{w}|refine")})
        last = np.median(raw[f"w{w}|refine{steps[-1]}_resid_rel"]) if steps else float("nan")
        md.append(f"| {w:g} | {np.median(raw[f'w{w}|resid_rel']):.3g} | {np.percentile(raw[f'w{w}|resid_rel'], 90):.3g} | {last:.3g} |")
    md.append("")
    if train_log:
        log = read_log(train_log)
        cal = [(r["step"], r["calib/resid_rel"]) for r in log if "calib/resid_rel" in r]
        gen = [(r["step"], r["gen/resid_rel"]) for r in log if "gen/resid_rel" in r]
        if cal or gen:
            fig, ax = plt.subplots(figsize=(3.2, 2.0), layout="constrained")
            if cal:
                series(ax, *zip(*cal), color=SERIES[0], marker=None, label="stage 0 (calibration)")
            if gen:
                series(ax, *zip(*gen), color=SERIES[1], marker=None, label="Algorithm 1 (generator)")
            ax.set_yscale("log")
            ax.set_xlabel("training step")
            ax.set_ylabel("relative prox residual")
            save(fig, out, "fig3d_prox_training")


def fig4(results, out, md):
    d = load(results, "mismatch")
    if d is None:
        return
    rows = d["rows"]
    alphas = sorted({x["alpha"] for x in rows})
    fig, axes = plt.subplots(2, 2, figsize=(5.4, 4.1), layout="constrained")
    for i, a in enumerate(alphas):
        for j, (kind, key, xlabel) in enumerate((("shift", "center_shift_std", "centre shift δ (proposal std / coord)"),
                                                 ("scale", "variance_multiplier", "variance multiplier κ"))):
            r = sorted((x for x in rows if x["alpha"] == a and x["kind"] == kind), key=lambda x: x[key])
            xs = [x[key] for x in r]
            lab = f"α = {a:g}"
            series(axes[0, j], xs, [max(x["var_logw"], 1e-6) for x in r], [max(x["var_logw_lo"], 1e-6) for x in r],
                   [x["var_logw_hi"] for x in r], color=SERIES[i], marker=MARKERS[i], label=lab, direct=j == 1)
            series(axes[1, j], xs, [x["control_ess"] for x in r], [x["control_ess_lo"] for x in r],
                   [x["control_ess_hi"] for x in r], color=SERIES[i], marker=MARKERS[i], label=lab, direct=False)
            if a == 0.0:
                base = next((x["var_logw"] for x in r if x["center_shift_std"] == 0 and x["variance_multiplier"] == 1), 0.0)
                axes[0, j].plot(xs, [max(base + x["pred_extra_var_logw"], 1e-6) for x in r], color=INK2,
                                lw=1.0, ls=(0, (4, 2)), label="Gaussian prediction (α = 0)")
            axes[1, j].set_xlabel(xlabel)
            if kind == "shift":
                for ax in axes[:, j]:
                    ax.set_xscale("symlog", linthresh=min(v for v in xs if v > 0) if any(v > 0 for v in xs) else 1e-3)
    for ax in axes[0]:
        ax.set_yscale("log")
        ax.set_ylabel("Var(log w)")
    for ax in axes[1]:
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("control ESS")
    panel(axes[0, 0], "a", "centre shift")
    panel(axes[0, 1], "b", "scale mismatch")
    panel(axes[1, 0], "c", "centre shift")
    panel(axes[1, 1], "d", "scale mismatch")
    legend_below(fig, axes[0, 0])
    save(fig, out, "fig4_mismatch")
    md.append(f"## Fig 4 - proposal mismatch (eps={d.get('eps'):.4g}, K={d['K']})\n")
    md.append("Measured Var(log w) is compared with the closed-form Gaussian prediction (dashed).\n")


def fig5(results, out, md):
    d = load(results, "nll")
    if d is None:
        return
    grid = d["grid"]
    kos = sorted({g["k_outer"] for g in grid})
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.3), layout="constrained")
    for i, ko in enumerate(kos):
        r = sorted((g for g in grid if g["k_outer"] == ko), key=lambda g: g["k_inner"])
        ki = [g["k_inner"] for g in r]
        series(axes[0], ki, [g["bits_per_dim"] for g in r], [g["lo"] for g in r], [g["hi"] for g in r],
               color=SERIES[i], marker=MARKERS[i], label=f"K$_{{out}}$ = {ko}")
        if all(g["seed_std"] is not None for g in r):
            series(axes[1], ki, [max(g["seed_std"], 1e-6) for g in r], color=SERIES[i], marker=MARKERS[i],
                   label=f"K$_{{out}}$ = {ko}", direct=False)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("inner draws K$_{in}$")
    axes[0].set_ylabel("NLL (bits / latent dim)")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("seed-to-seed std (bits/dim)")
    legend_below(fig, axes[0], ncol=len(kos))
    panel(axes[0], "a", "held-out latent NLL")
    panel(axes[1], "b", "estimator spread")
    save(fig, out, "fig5_nll")
    md.append(f"## Fig 5 - {d['label']} (eps={d.get('eps'):.4g}, n={d['n']})\n")
    md.append("| K_outer | K_inner | bits/dim [95% CI] | seed std | outer / inner control ESS |\n|---|---|---|---|---|")
    for g in grid:
        sd = "-" if g["seed_std"] is None else f"{g['seed_std']:.3g}"
        md.append(f"| {g['k_outer']} | {g['k_inner']} | {g['bits_per_dim']:.4f} [{g['lo']:.4f}, {g['hi']:.4f}] | {sd} "
                  f"| {g['outer_control_ess']:.3g} / {g['inner_control_ess']:.3g} |")
    md.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--train-log", default="", help="metrics.jsonl of an eps-annealing run (Fig 2c, Fig 3d)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    results = Path(a.results)
    out = Path(a.out) if a.out else results / "figures"
    md = [f"# PT-Flow mechanism figures\n\nSource: `{results}`\n"]
    fig1(results, out, md)
    fig2(results, out, md, a.train_log or None)
    fig3(results, out, md, a.train_log or None)
    fig4(results, out, md)
    fig5(results, out, md)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote", out / "summary.md")


if __name__ == "__main__":
    main()
