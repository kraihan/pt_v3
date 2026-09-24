"""PT-Flow training, initialized from W-Flow weights.

    torchrun --nproc_per_node=8 train.py --config configs/B.yaml --workdir runs/B

Stage 0 (calibration, once):  generator frozen at the W-Flow weights; fit the
    potential so that m_eta = prox_phi, i.e. minimize |grad phi^w(m) + m - x0|^2
    over theta.  Exit when the relative residual and the proposal ESS are healthy.
Algorithm 1 (every step):
    (A) n_eta generator steps at frozen theta on eq. 16 (+ Hutchinson scale match)
    (B) one potential step at frozen eta, w = 0, on eq. 14-15 (skipped when broken)
    (C) ESS controller: anneal eps / raise alpha / add eta steps / freeze theta
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch

from ptflow import ckpt, dist
from ptflow.build import build_generator, build_potential, init_from_wflow
from ptflow.config import load_config, save_config
from ptflow.data import build_dataset, infinite_loader
from ptflow.estimator import tilted_estimate, weight_stats
from ptflow.logger import Logger
from ptflow.losses import calibration_loss, curvature_hinge, generator_loss, potential_loss
from ptflow.schedule import build_schedule


@torch.no_grad()
def ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    e, m = list(ema.parameters()), list(model.parameters())
    torch._foreach_mul_(e, decay)
    torch._foreach_add_(e, m, alpha=1.0 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


def set_lr(opt, base: float, step: int, warmup: int) -> float:
    lr = base * min(1.0, (step + 1) / max(1, warmup))
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


def optimizer_step(opt, module, clip: float) -> float:
    dist.allreduce_grads(module)
    gnorm = torch.nn.utils.clip_grad_norm_(module.parameters(), clip, error_if_nonfinite=True)
    opt.step()
    return float(gnorm)


def sample_w(n: int, w_max: float, device, generator) -> torch.Tensor:
    """Guidance weights for the generator (Alg. 1 line 6): w ~ U[0, w_max]."""
    return torch.rand(n, generator=generator, device=device) * float(w_max)


def save_request(workdir: str, device) -> bool:
    flag = torch.tensor([float(dist.is_main() and Path(workdir, "REQUEST_SAVE").exists())], device=device)
    if dist.is_dist():
        torch.distributed.broadcast(flag, src=0)
    return bool(flag.item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("-o", "--override", action="append", default=[], help="key.sub=value")
    args = ap.parse_args()

    device = dist.init()
    cfg = load_config(args.config, args.override)
    tc, workdir = cfg["train"], args.workdir
    Path(workdir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(tc.get("seed", 0)))          # identical module init on every rank

    gen, pot = build_generator(cfg), build_potential(cfg)
    sched = build_schedule(cfg["schedule"])
    resume = ckpt.latest_checkpoint(workdir)
    payload, info, step = None, {}, 0
    if resume is not None:
        payload = ckpt.load(resume)
        gen.load_state_dict(payload["generator"])
        pot.load_state_dict(payload["potential"])
        sched.load_state_dict(payload["schedule"])
        info, step = payload.get("init_info", {}), int(payload["step"])
    elif cfg.get("init", {}).get("wflow_ckpt"):
        info = init_from_wflow(gen, pot, cfg)
    gen.to(device)
    pot.to(device)
    dist.broadcast_module(gen)
    dist.broadcast_module(pot)
    gen_ema, pot_ema = copy.deepcopy(gen).eval().requires_grad_(False), copy.deepcopy(pot).eval().requires_grad_(False)
    betas = tuple(tc.get("betas", (0.9, 0.95)))
    opt_g = torch.optim.AdamW(gen.parameters(), lr=tc["lr_gen"], betas=betas, weight_decay=tc.get("weight_decay", 0.0))
    opt_p = torch.optim.AdamW(pot.parameters(), lr=tc["lr_pot"], betas=betas, weight_decay=tc.get("weight_decay", 0.0))
    if payload is not None:
        gen_ema.load_state_dict(payload["generator_ema"])
        pot_ema.load_state_dict(payload["potential_ema"])
        opt_g.load_state_dict(payload["opt_gen"])
        opt_p.load_state_dict(payload["opt_pot"])
        del payload

    logger = Logger(workdir, cfg, use_wandb=cfg.get("logging", {}).get("use_wandb", False),
                    project=cfg.get("logging", {}).get("project", "ptflow"), name=cfg.get("name"))
    if dist.is_main():
        save_config(cfg, Path(workdir, "config.yaml"))
        logger.write_json("init.json", {"resumed_from": str(resume) if resume else None, **info})
        print(f"[init] {info if resume is None else f'resumed {resume}'}", flush=True)

    rng = torch.Generator(device=device).manual_seed(int(tc.get("seed", 0)) * 7919 + 104729 * dist.rank() + step)
    data = infinite_loader(build_dataset(cfg["data"], "train"), max(tc["batch_gen"], tc["batch_pot"]),
                           seed=int(tc.get("seed", 0)) + step, num_workers=int(cfg["data"].get("num_workers", 4)))
    K, n_pot, d = int(tc["K"]), int(tc["batch_pot"]), gen.dim
    shape = (gen.input_size, gen.input_size, gen.channels)
    clip, chunk = float(tc.get("grad_clip", 1.0)), int(tc.get("potential_chunk", 0))

    def payload_now():
        return {"step": step, "config": cfg, "init_info": info, "schedule": sched.state_dict(),
                "generator": ckpt.module_state(gen), "generator_ema": ckpt.module_state(gen_ema),
                "potential": ckpt.module_state(pot), "potential_ema": ckpt.module_state(pot_ema),
                "opt_gen": opt_g.state_dict(), "opt_pot": opt_p.state_dict()}

    def save():
        ckpt.save(workdir, step, payload_now(), keep_last=int(tc.get("keep_last", 2)), keep_every=int(tc.get("keep_every", 0)))

    t_last = time.time()
    while sched.phase == "calibrate" or sched.pt_step < int(tc["total_steps"]):
        x1, c = next(data)
        x1, c = x1.to(device, non_blocking=True), c.to(device, non_blocking=True)
        c = c[: tc["batch_gen"]]
        x0 = torch.randn((tc["batch_gen"], *shape), generator=rng, device=device)
        eps = sched.eps()
        metrics = {}
        calibrated_now = False

        if sched.phase == "calibrate":
            # ---- stage 0: make the W-Flow generator the prox of the potential -------------------
            metrics["lr/pot"] = set_lr(opt_p, tc.get("lr_calib", tc["lr_pot"]), sched.calib_step, tc.get("warmup", 500))
            w = sample_w(len(x0), tc["w_max"], device, rng)
            with torch.no_grad():
                m = gen(x0, c, w, with_scale=False)[0]
            loss, mc = calibration_loss(pot, m, x0, c, w)
            if tc.get("lambda_curv", 0) > 0:
                pen, mcv = curvature_hinge(pot, m[: tc.get("curv_batch", 8)], c[: tc.get("curv_batch", 8)],
                                           allow=tc.get("curv_allow", 0.5), generator=rng)
                loss = loss + tc["lambda_curv"] * pen
                mc.update(mcv)
            opt_p.zero_grad(set_to_none=True)
            loss.backward()
            mc["calib/gnorm"] = optimizer_step(opt_p, pot, clip)
            ema_update(pot_ema, pot, tc["ema_pot"])
            metrics.update(mc)
            resid = dist.mean_scalar(float(mc["calib/resid_rel"]), device)
            ess = None
            if sched.calib_step % sched.calib_check_every == 0:
                with torch.no_grad():
                    m0, s0 = gen(x0[:n_pot], c[:n_pot], 0.0)
                    est = tilted_estimate(pot.phi, x0[:n_pot], c[:n_pot], m0, s0, eps, K=K, alpha=sched.alpha_def,
                                          generator=rng, chunk=chunk)
                ess = dist.mean_scalar(float(weight_stats(est.log_w)["control_ess"].mean()), device)
                metrics["calib/control_ess"] = ess
            sched.observe_calibration(resid, ess)
            calibrated_now = sched.phase == "pt"
            if calibrated_now and dist.is_main():
                print(f"[calibration] {sched.calib_exit}", flush=True)
                logger.write_json("calibration.json", {"exit": sched.calib_exit, "step": step, **sched.metrics()})
        else:
            # ---- (A) generator steps at frozen theta ------------------------------------------------
            metrics["lr/gen"] = set_lr(opt_g, tc["lr_gen"], sched.pt_step, tc.get("warmup", 500))
            gen.train()
            for _ in range(sched.eta_steps()):
                w = sample_w(len(x0), tc["w_max"], device, rng)
                loss_g, mg = generator_loss(pot, gen, x0, c, w, mode=tc.get("generator_mode", "full"),
                                            lambda_scale=tc.get("lambda_scale", 0.1),
                                            probes=tc.get("hutchinson_probes", 1), generator=rng)
                opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                mg["gen/gnorm"] = optimizer_step(opt_g, gen, clip)
                ema_update(gen_ema, gen, tc["ema_gen"])
            metrics.update(mg)

            # ---- (B) potential step at frozen eta, guidance weight zero -------------------------------
            metrics["lr/pot"] = set_lr(opt_p, tc["lr_pot"], sched.pt_step, tc.get("warmup", 500))
            x0p, cp = x0[:n_pot], c[:n_pot]
            with torch.no_grad():
                m0, s0 = gen(x0p, cp, 0.0)
            c0 = pot.drop_labels(cp, tc.get("p_uncond", 0.1), generator=rng)
            c1 = pot.drop_labels(cp, tc.get("p_uncond", 0.1), generator=rng)
            update = sched.update_theta()
            with torch.enable_grad() if update else torch.no_grad():
                loss_p, est, mp = potential_loss(pot, x0p, c0, m0, s0, x1[:n_pot], c1, eps, K=K, alpha=sched.alpha(),
                                                 lambda_gauge=tc.get("lambda_gauge", 0.0), generator=rng, chunk=chunk)
                if update:
                    if tc.get("lambda_curv", 0) > 0:
                        nb = tc.get("curv_batch", 8)
                        pen, mcv = curvature_hinge(pot, est.y[:nb, 0], c0[:nb], allow=tc.get("curv_allow", 0.5), generator=rng)
                        loss_p = loss_p + tc["lambda_curv"] * pen
                        mp.update(mcv)
                    opt_p.zero_grad(set_to_none=True)
                    loss_p.backward()
                    mp["pot/gnorm"] = optimizer_step(opt_p, pot, clip)
                    ema_update(pot_ema, pot, tc["ema_pot"])
            metrics.update(mp)
            metrics["pot/updated"] = float(update)
            # T = prox = m monitor: SNIS bridge conditional mean vs the generator output.
            metrics["est/tmean_gap_rms"] = (est.posterior_mean() - m0).square().mean().sqrt()

            # ---- (C) monitor and anneal -------------------------------------------------------------
            ess = dist.mean_scalar(float(weight_stats(est.log_w)["control_ess"].mean()), device)
            sched.observe(ess)
        step += 1
        if calibrated_now:
            # Full, permanent checkpoint at the calibrated pair (generator still exactly W-Flow):
            # the reference point for the mechanism figures and the start of any secondary run.
            path = ckpt.save_named(workdir, f"calib_state_{step:08d}.pt", payload_now())
            if dist.is_main():
                print(f"[calibration] saved {path}", flush=True)

        if step % int(tc.get("log_every", 20)) == 0 or step == 1:
            metrics.update(sched.metrics())
            metrics["phase_pt"] = float(sched.phase == "pt")
            metrics["time/step_s"] = (time.time() - t_last) / (int(tc.get("log_every", 20)) if step > 1 else 1)
            t_last = time.time()
            logger.log(step, metrics)
            logger.write_json("status.json", {"step": step, "phase": sched.phase, "state": sched.state,
                                              "pt_step": sched.pt_step, **sched.metrics()})
            if dist.is_main():
                keys = ("calib/loss", "calib/resid_rel", "gen/resid_rel", "pot/loss", "est/control_ess", "sched/eps")
                print(f"[{step}] {sched.phase}/{sched.state} " + " ".join(
                    f"{k}={float(metrics[k]):.4g}" for k in keys if k in metrics), flush=True)
        if step % int(tc.get("save_every", 1000)) == 0:
            save()
        eval_every = int(tc.get("eval_every", 0))
        if eval_every and sched.phase == "pt" and sched.pt_step % eval_every == 0 and sched.pt_step > 0:
            from ptflow.evaluate import evaluate_fid
            # The exact EMA weights being scored, kept permanently so the best FID point can be reused.
            eval_ckpt = ckpt.save_named(workdir, f"eval_state_{step:08d}.pt", {
                "step": step, "config": cfg, "init_info": info, "schedule": sched.state_dict(),
                "generator_ema": ckpt.module_state(gen_ema), "potential_ema": ckpt.module_state(pot_ema)})
            for cs in tc.get("eval_cfg_scales", [1.0]):
                res = evaluate_fid(gen_ema, pot_ema, num_samples=int(tc.get("eval_samples", 10000)), cfg_scale=cs,
                                   mode="A", seed=0, batch=int(tc.get("eval_batch", 64)), device=device,
                                   ref_path=cfg["eval"]["fid_ref"])
                if dist.is_main():
                    logger.log(step, {f"fid/cfg{cs}": res["fid"], f"is/cfg{cs}": res["is_mean"]})
                    with Path(workdir, "fid_history.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"step": step, "pt_step": sched.pt_step, "cfg_scale": cs, "fid": res["fid"],
                                            "is": res["is_mean"], "num_samples": res["num_samples"],
                                            "ckpt": str(eval_ckpt)}) + "\n")
                    print(f"[fid] step {step} (pt {sched.pt_step}) cfg {cs}: {res['fid']:.3f} -> {eval_ckpt}", flush=True)
        if step % 10 == 0 and save_request(workdir, device):
            save()
            if dist.is_main():
                Path(workdir, "REQUEST_SAVE").unlink(missing_ok=True)
                Path(workdir, "SAVED_FOR_REQUEUE").write_text(str(step))
            dist.cleanup()
            return

    save()
    if dist.is_main():
        Path(workdir, "COMPLETED").write_text(str(step))
    logger.finish()
    dist.cleanup()


if __name__ == "__main__":
    main()
