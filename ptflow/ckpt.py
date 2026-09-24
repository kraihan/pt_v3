"""Training checkpoints: atomic writes, rotation, resume."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ptflow import dist


def list_checkpoints(workdir: str) -> List[Path]:
    return sorted(Path(workdir, "checkpoints").glob("state_*.pt"))


def latest_checkpoint(workdir: str) -> Optional[Path]:
    ckpts = list_checkpoints(workdir)
    return ckpts[-1] if ckpts else None


def save(workdir: str, step: int, payload: Dict, *, keep_last: int = 2, keep_every: int = 0) -> None:
    """Rank 0 writes state_XXXXXXXX.pt through a temporary file; others wait."""
    dist.barrier()
    if dist.is_main():
        d = Path(workdir, "checkpoints")
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"state_{step:08d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        ckpts = list_checkpoints(workdir)
        for p in ckpts[:-keep_last] if keep_last > 0 else []:
            s = int(p.stem.split("_")[-1])
            if not (keep_every and s % keep_every == 0):
                p.unlink(missing_ok=True)
    dist.barrier()


def save_named(workdir: str, name: str, payload: Dict) -> Path:
    """Permanent checkpoint under a name outside the state_*.pt rotation (never deleted; not used for resume)."""
    if name.startswith("state_"):
        raise ValueError("Named checkpoints must not use the rotated state_ prefix")
    path = Path(workdir, "checkpoints", name)
    dist.barrier()
    if dist.is_main():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
    dist.barrier()
    return path


def load(path: str | Path) -> Dict:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def module_state(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}
