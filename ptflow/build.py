"""Model construction from a config, from W-Flow weights, or from a PT-Flow checkpoint."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from ptflow.models.generator import Generator, load_wflow_generator
from ptflow.models.potential import Potential, init_potential_from_generator
from ptflow.schedule import PTSchedule, build_schedule

_SHARED = ("num_classes", "input_size", "in_channels", "patch_size")


def build_generator(cfg: Dict) -> Generator:
    return Generator(**cfg["generator"])


def build_potential(cfg: Dict) -> Potential:
    kw = {k: v for k, v in cfg["potential"].items() if k != "init_from_generator"}
    for k in _SHARED:
        kw.setdefault(k, cfg["generator"].get(k, {"num_classes": 1000, "input_size": 32, "in_channels": 4,
                                                   "patch_size": 2}[k]))
    return Potential(**kw)


def init_from_wflow(gen: Generator, pot: Potential, cfg: Dict) -> Dict:
    """Requirement 1: start from W-Flow weights instead of a random initialization."""
    init = cfg["init"]
    info = load_wflow_generator(gen, init["wflow_ckpt"], weights=init.get("weights", "ema"),
                                noise_code=init.get("noise_code", "fixed:0"))
    info.pop("noise_code_table", None)
    if cfg["potential"].get("init_from_generator", True):
        info["potential_init"] = init_potential_from_generator(pot, gen)
    return info


def load_checkpoint_models(path: str, device: torch.device, *, ema: bool = True
                           ) -> Tuple[Generator, Potential, PTSchedule, Dict, int]:
    """Generator, potential and schedule (with its eps) from a PT-Flow training checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    gen, pot = build_generator(cfg), build_potential(cfg)
    gen.load_state_dict(payload["generator_ema" if ema else "generator"], strict=True)
    pot.load_state_dict(payload["potential_ema" if ema else "potential"], strict=True)
    sched = build_schedule(cfg["schedule"])
    sched.load_state_dict(payload["schedule"])
    for mod in (gen, pot):
        mod.to(device).eval().requires_grad_(False)
    return gen, pot, sched, cfg, int(payload["step"])


def load_wflow_only(wflow_cfg: Dict, ckpt: str, device: torch.device, noise_code: str = "fixed:0"
                    ) -> Tuple[Generator, Optional[torch.Tensor]]:
    """A raw W-Flow generator (for baseline FID), with its noise-code table for random codes."""
    gen = Generator(**wflow_cfg)
    info = load_wflow_generator(gen, ckpt, noise_code=noise_code if noise_code != "random" else "fixed:0")
    gen.to(device).eval().requires_grad_(False)
    table = info["noise_code_table"]
    return gen, (table.to(device) if table is not None else None)
