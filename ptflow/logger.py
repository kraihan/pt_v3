"""Metrics to <workdir>/metrics.jsonl (always) and Weights & Biases (optional)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Optional

import torch

from ptflow import dist


def _scalar(v):
    if torch.is_tensor(v):
        v = v.detach().float().mean().item()
    v = float(v)
    return v if math.isfinite(v) else str(v)


class Logger:
    def __init__(self, workdir: str, cfg: Optional[Dict] = None, *, use_wandb: bool = False,
                 project: str = "ptflow", name: Optional[str] = None):
        self.path = Path(workdir, "metrics.jsonl")
        self.wandb = None
        if dist.is_main():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if use_wandb:
                import wandb
                run_id = hashlib.sha1(str(Path(workdir).resolve()).encode()).hexdigest()[:16]
                wandb.init(project=project, name=name or Path(workdir).name, config=cfg, id=run_id, resume="allow")
                self.wandb = wandb

    def log(self, step: int, metrics: Dict) -> None:
        if not dist.is_main():
            return
        row = {k: _scalar(v) for k, v in metrics.items()}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, **row}) + "\n")
        if self.wandb is not None:
            self.wandb.log({k: v for k, v in row.items() if isinstance(v, float)}, step=step)

    def write_json(self, name: str, obj: Dict) -> None:
        if dist.is_main():
            p = self.path.parent / name
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
            tmp.replace(p)

    def finish(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()
