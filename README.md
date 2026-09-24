# PT-Flow

One-step Schrödinger-bridge generation by proximal tilting, trained from **W-Flow weights**
(B/2, L/2, XL/2) on class-conditional ImageNet 256 in SD-VAE latent space.

Training and inference follow the paper's theory directly:

- The generator is the proximal map of a learned potential: **T(x₀) = prox_φ(x₀) = m_η(x₀)**.
- The potential is fitted by the bridge likelihood objective, estimated with the prox-tilted estimator.
- ε is annealed by the ESS controller of Algorithm 1.
- Sampling is 1 NFE (Mode A), with the refined (B) and resampled (C) modes also available.

The W-Flow checkpoint is used **only as the initialization**. No W-Flow loss, memory bank, MAE
features or teacher is used.

## Theory → code

| Paper | Code |
|---|---|
| φ = 2ε u, ψ₁ = e^{−u} (App. B) | `ptflow/models/potential.py` `Potential.phi` (network outputs φ; every equation uses u = φ/2ε) |
| ψ₀ = G_{2ε} ∗ ψ₁ (eq. 3), tilted IS (eqs. 9–10, 15) | `ptflow/estimator.py` `tilted_estimate` |
| Defensive mixture q_def, antithetic ±z | `estimator.draw_noise`, `log_kernel_over_proposal` |
| Potential loss L_θ (eq. 14) at w = 0, conditioning dropout | `ptflow/losses.py` `potential_loss` |
| Generator loss ‖2ε g_w(m) + m − x₀‖² (eq. 16) | `losses.generator_loss` (`mode: full` = through ∇φ, `detach` = fixed-point target) |
| Scale head s_η, Hutchinson match to diag(I + ∇²φ) (Alg. 1 line 8) | `Generator.scale_layer`, `losses.hutchinson_diag` |
| CFG as potential arithmetic φ^w = (1+w)φ_c − wφ_∅ (eq. 17) | `potential.guided_grad`, `estimator.potential_phi_fn` |
| (A2) curvature monitor + hinge (App. L) | `losses.curvature_hinge` |
| Algorithm 1 steps A / B / C, ESS thresholds 0.3 / 0.05, cosine ε anneal | `train.py`, `ptflow/schedule.py` |
| Algorithm 2 Modes A / B / C | `ptflow/sampling.py` |
| Likelihood log ρ̂₁ = log ψ̂₁ − u (Thm 3.5, App. J) | `ptflow/likelihood.py` (nested MC) |
| SNIS posterior mean = bridge conditional mean T_ε(x₀) | `TiltedEstimate.posterior_mean` (logged as `est/tmean_gap_rms`) |

## How a run proceeds

1. **Initialize from W-Flow.**
   - The generator gets the W-Flow EMA weights, so m_η starts as the pretrained sampler.
   - W-Flow also draws 32 random discrete noise codes per sample. The theory needs one map x₀ ↦ prox(x₀), so the codes are folded into a single fixed conditioning bias (`init.noise_code: fixed:0`, or `mean`).
   - The scale head is new and zero-initialized (S = I).
   - The potential copies the first `depth` W-Flow blocks and has a zero readout, so φ ≡ 0 at step 0.
2. **Stage 0: calibration.**
   - A pretrained generator is not the prox of a zero potential. Starting Algorithm 1 from that pair puts the proposal far from the target mode, and the importance weights collapse. This is what stopped the earlier B_PT_Full run at step 368.
   - So with the generator frozen, θ is fitted to make m the prox: it minimizes ‖∇φ^w(m) + m − x₀‖² over θ for w ∈ [0, w_max]. Pairs with w > 0 also identify the unconditional potential through eq. 17.
   - Stage 0 ends when the relative residual is at most `calib_tol` and the generator-centred proposal's control ESS is at least 0.3. It also ends after `calib_max_steps`; the reason is written to `calibration.json`.
3. **Algorithm 1**, every step:
   - **(A)** n_η generator steps at frozen θ (eq. 16 plus the scale match), with w ~ U[0, w_max].
   - **(B)** one potential step at frozen η with w = 0 (eq. 14). It is skipped while the estimator is *broken*; the ESS is still measured, forward only.
   - **(C)** the controller:

     | State | Control ESS | Action |
     |---|---|---|
     | healthy | ≥ 0.3 | advance the cosine ε anneal |
     | degrading | 0.05 to 0.3 | hold ε, raise α, double n_η |
     | broken | < 0.05 | freeze θ, retrain η |

   The health signal is the control ESS (ESS − 1)/(K − 1). It is ESS/K with the 1/K floor removed; without that, 0.05 is unreachable for K < 20.

## Setup

```bash
pip install -r requirements.txt
cp scripts/env.example.sh ~/ptflow_env.sh      # edit paths, then:
source ~/ptflow_env.sh
python scripts/download_assets.py --models B L XL    # W-Flow checkpoints, SD-VAE, Inception
# latent cache (skip if you already have W-Flow's cache: same format)
python -m ptflow.data build-cache --imagenet /path/to/imagenet --out $IMAGENET_CACHE_PATH
python -m pytest -q                                  # CPU tests, ~30 s
```

Environment variables:

| Variable | Contents |
|---|---|
| `IMAGENET_CACHE_PATH` | Latent cache: `{train,val}_{moments,moments_flip,targets}.npy` |
| `WFLOW_ROOT` | Local copy of `jiaqihan99/W-Flow` |
| `VAE_PATH` | Local copy of `stabilityai/sd-vae-ft-mse` |
| `FID_REF_NPZ` | `jit_in256_stats.npz` |
| `TORCH_HUB_DIR` | Cache for the torch-fidelity Inception weights |

## Recommended order

```bash
# 0. Pre-flight (no training)
#    a. What folding the noise codes costs the initializer: FID with random vs fixed codes
sbatch --gpus=h200:8 scripts/eval.sbatch evaluate-wflow --config configs/B.yaml --noise-code random --cfg-scales 1.2 --num-samples 10000
sbatch --gpus=h200:8 scripts/eval.sbatch evaluate-wflow --config configs/B.yaml --noise-code fixed:0 --cfg-scales 1.2 --num-samples 10000
#    b. Can the W-Flow map be a prox map, and what ESS can a diagonal proposal reach even with a perfect potential?
python experiments.py jacobian --config configs/B.yaml --out results/B_preflight --n 8 --chunk 256

# 1. Train (calibration, then Algorithm 1); requeues itself before the wall time
sbatch --gpus=h200:8 scripts/train.sbatch configs/B.yaml runs/B
#    extra overrides: ... runs/B -o train.total_steps=80000 -o schedule.eps_min=0.05

# 2. FID-50K (Mode A = 1 NFE; also try B and C)
sbatch --gpus=h200:8 scripts/eval.sbatch evaluate --ckpt runs/B/checkpoints/state_XXXXXXXX.pt \
       --mode A --cfg-scales 1.0,1.2,1.4,1.6 --json-out runs/B/fid50k_A.json

# 3. Mechanism experiments on the same checkpoint
C=runs/B/checkpoints/state_XXXXXXXX.pt
python experiments.py proposals --ckpt $C --out results/B   # full proposal vs naive / recentred / diagonal
python experiments.py eps-sweep --ckpt $C --out results/B   # ESS, Var(log w), chi2 vs eps + Laplace mismatch
python experiments.py prox      --ckpt $C --out results/B   # residual quantiles, Mode-B refinement, |T_eps - m|
python experiments.py mismatch  --ckpt $C --out results/B   # centre shift / variance multiplier / alpha
python experiments.py nll       --ckpt $C --out results/B   # latent NLL over (K_outer x K_inner) x seeds
python experiments.py toy                 --out results/toy # exact T_eps vs prox by quadrature (O(eps) gap)
```

## What to watch

These are in `metrics.jsonl` (and W&B when `logging.use_wandb: true`); `status.json` holds the latest values.

| Metric | Meaning | Healthy |
|---|---|---|
| `calib/resid_rel` | ‖∇φ(m) + m − x₀‖ / ‖m − x₀‖ during stage 0 | falls toward `calib_tol` |
| `calib/control_ess`, `est/control_ess` | Importance-weight health (the controller input) | ≥ 0.3 |
| `gen/resid_rel` | How far m_η is from prox_φ (eq. 16) | small and stable |
| `est/tmean_gap_rms` | Per-coordinate RMS of T̂_ε(x₀) − m_η(x₀): the T = prox = m check | small; meaningful only when ESS is not near 0 |
| `est/var_logw`, `est/chi2` | s²_res and the χ² divergence of Thm 3.9 | — |
| `sched/eps`, `sched/health` | Anneal position and controller state (2 healthy, 1 degrading, 0 broken) | — |
| `curv/violation_frac` | Share of probes with curvature below −`curv_allow` | ≈ 0 |
| `fid/cfg*` | FID-10K of the EMA generator (Mode A) every `eval_every` steps | — |

If calibration exits on `calib_max_steps` with a large residual, or the controller stays *broken*, run
`experiments.py jacobian` and `eps-sweep`. They separate a map that is not a gradient field from an
off-diagonal curvature floor. A diagonal proposal cannot remove that floor for any ε (Thm 3.9 needs
‖B‖²_F = O(ε)).

## Layout

```
train.py            calibration + Algorithm 1 (torchrun, resume, periodic FID, requeue)
inference.py        sample grid; FID for modes A/B/C; W-Flow baseline with random/fixed codes
experiments.py      proposals, eps-sweep, prox, mismatch, nll, jacobian, toy
configs/            B.yaml L.yaml XL.yaml (W-Flow architectures), smoke.yaml (CPU tests)
ptflow/models/      dit.py (W-Flow LightningDiT), generator.py (m, s, W-Flow loading), potential.py (φ, ∇φ, HVP)
ptflow/             estimator, losses, schedule, sampling, likelihood, data, vae, fid, evaluate, ckpt, dist, config, logger
scripts/            train.sbatch, eval.sbatch, env.example.sh, download_assets.py
tests/              CPU tests (estimator exactness/unbiasedness, likelihood normalization, W-Flow loading, training/resume, experiments)
```

## Status and caveats

- **Tested on CPU only.** That covers:
  - the closed-form estimator checks and the unbiasedness check;
  - normalization of ρ̂₁ in 1-D;
  - HVPs against finite differences;
  - W-Flow loading against the upstream model code;
  - single- and two-process training with resume, and every experiment.
- **Not yet run on GPU or real data.** No PT-Flow ImageNet result exists yet: start with the pre-flight runs above.
- **ε is outside the asymptotic window.** The paper's defaults (ε 0.2 → 0.1) give εd = 400–800 at d = 4096, far outside its εd ≲ 1 window. The variance-reversal and Laplace statements are therefore motivation here, and the experiments measure how well they hold.
- **The likelihood is an estimate, not a bound.** It is a nested Monte-Carlo estimate of an analytically normalized *latent* density, with no certified bias direction. Report it over the budget grid and seeds, and don't compare it to pixel bits/dim.
