"""Sampling and FID evaluation.

    # PT-Flow checkpoint, FID-50K, Mode A (1 NFE), several guidance scales
    torchrun --nproc_per_node=8 inference.py evaluate --ckpt runs/B/checkpoints/state_00040000.pt \
        --mode A --cfg-scales 1.0,1.2,1.4 --num-samples 50000 --json-out runs/B/fid_50k.json

    # the W-Flow initializer itself: its original random noise codes vs the fixed code we fold in
    torchrun --nproc_per_node=8 inference.py evaluate-wflow --config configs/B.yaml --noise-code random
    torchrun --nproc_per_node=8 inference.py evaluate-wflow --config configs/B.yaml --noise-code fixed:0

    # preview grid
    python inference.py sample --ckpt ... --mode A --cfg-scale 1.2 --classes 207,360,387,974 --out grid.png

cfg scales follow W-Flow's convention: cfg = 1 + w (w is the paper's guidance weight).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from ptflow import dist
from ptflow.build import load_checkpoint_models, load_wflow_only
from ptflow.config import load_config


def _floats(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def _write(path: str, obj) -> None:
    if path and dist.is_main():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def cmd_evaluate(a, device):
    from ptflow.evaluate import evaluate_fid
    gen, pot, sched, cfg, step = load_checkpoint_models(a.ckpt, device, ema=not a.raw)
    eps = a.eps if a.eps > 0 else sched.eps()
    results = []
    for cs in _floats(a.cfg_scales):
        r = evaluate_fid(gen, pot, num_samples=a.num_samples, cfg_scale=cs, mode=a.mode, eps=eps, alpha=a.alpha,
                         K=a.K, refine_steps=a.refine_steps, seed=a.seed, batch=a.batch, ref_path=a.fid_ref,
                         device=device, chunk=a.chunk)
        r.update(ckpt=a.ckpt, step=step, weights="raw" if a.raw else "ema", eps=eps)
        results.append(r)
        if dist.is_main():
            print(json.dumps(r), flush=True)
    _write(a.json_out, results)


def cmd_evaluate_wflow(a, device):
    from ptflow.evaluate import evaluate_fid
    cfg = load_config(a.config)
    ckpt = a.wflow_ckpt or cfg["init"]["wflow_ckpt"]
    gen, table = load_wflow_only(cfg["generator"], ckpt, device, noise_code=a.noise_code)
    results = []
    for cs in _floats(a.cfg_scales):
        r = evaluate_fid(gen, None, num_samples=a.num_samples, cfg_scale=cs, mode="A", seed=a.seed, batch=a.batch,
                         ref_path=a.fid_ref, device=device, code_table=table if a.noise_code == "random" else None)
        r.update(ckpt=ckpt, noise_code=a.noise_code)
        results.append(r)
        if dist.is_main():
            print(json.dumps(r), flush=True)
    _write(a.json_out, results)


@torch.no_grad()
def cmd_sample(a, device):
    import numpy as np
    from PIL import Image
    from ptflow.sampling import sample
    from ptflow.vae import decode, to_uint8
    gen, pot, sched, cfg, step = load_checkpoint_models(a.ckpt, device, ema=not a.raw)
    c = torch.tensor([int(x) for x in a.classes.split(",")], device=device).repeat_interleave(a.per_class)
    g = torch.Generator(device=device).manual_seed(a.seed)
    x0 = torch.randn((len(c), gen.input_size, gen.input_size, gen.channels), generator=g, device=device)
    eps = a.eps if a.eps > 0 else sched.eps()
    with torch.enable_grad():
        x1 = sample(a.mode, gen, pot, x0, c, a.cfg_scale - 1.0, eps=eps, refine_steps=a.refine_steps, K=a.K,
                    alpha=a.alpha, generator=g)
    img = to_uint8(decode(x1)).permute(0, 2, 3, 1).cpu().numpy()
    rows, cols = len(c) // a.per_class, a.per_class
    h, w = img.shape[1:3]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, im in enumerate(img):
        r, col = divmod(i, cols)
        grid[r * h:(r + 1) * h, col * w:(col + 1) * w] = im
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(a.out)
    print(f"saved {a.out} (step {step}, mode {a.mode}, cfg {a.cfg_scale})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--batch", type=int, default=64)
    common.add_argument("--json-out", default="")
    common.add_argument("--fid-ref", default=os.environ.get("FID_REF_NPZ", ""))
    common.add_argument("--num-samples", type=int, default=50000)
    common.add_argument("--cfg-scales", default="1.0")

    pt = argparse.ArgumentParser(add_help=False)
    pt.add_argument("--ckpt", required=True)
    pt.add_argument("--mode", default="A", choices=["A", "B", "C"])
    pt.add_argument("--raw", action="store_true", help="use raw weights instead of EMA")
    pt.add_argument("--eps", type=float, default=0.0, help="default: the checkpoint's current eps")
    pt.add_argument("--K", type=int, default=8)
    pt.add_argument("--alpha", type=float, default=0.1)
    pt.add_argument("--refine-steps", type=int, default=3)
    pt.add_argument("--chunk", type=int, default=0)

    sub.add_parser("evaluate", parents=[common, pt])
    w = sub.add_parser("evaluate-wflow", parents=[common])
    w.add_argument("--config", required=True)
    w.add_argument("--wflow-ckpt", default="")
    w.add_argument("--noise-code", default="random", help="random | fixed:<i> | mean")
    s = sub.add_parser("sample", parents=[pt])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--cfg-scale", type=float, default=1.0)
    s.add_argument("--classes", default="207,360,387,974,88,979,417,279")
    s.add_argument("--per-class", type=int, default=4)
    s.add_argument("--out", default="samples.png")
    a = ap.parse_args()

    device = dist.init()
    if a.cmd in ("evaluate", "evaluate-wflow") and not a.fid_ref:
        raise SystemExit("Set --fid-ref or $FID_REF_NPZ")
    {"evaluate": cmd_evaluate, "evaluate-wflow": cmd_evaluate_wflow, "sample": cmd_sample}[a.cmd](a, device)
    dist.cleanup()


if __name__ == "__main__":
    main()
