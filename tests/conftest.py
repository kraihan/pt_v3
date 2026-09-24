import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ptflow.config import load_config  # noqa: E402
from ptflow.models.generator import Generator  # noqa: E402


def write_wflow_checkpoint(path, gen_cfg, *, coords=3, classes=5, seed=0):
    """A random generator saved in W-Flow's state_*.pt layout (with noise-code embeddings)."""
    torch.manual_seed(seed)
    g = Generator(**gen_cfg)
    with torch.no_grad():
        for p in g.parameters():          # make zero-initialized readouts non-trivial
            p.add_(0.05 * torch.randn_like(p))
    sd = {k: v for k, v in g.state_dict().items() if not k.startswith("scale_layer.") and k != "code_bias"}
    for i in range(coords):
        sd[f"noise_embeds.{i}.weight"] = 0.02 * torch.randn(classes, gen_cfg["cond_dim"])
    torch.save({"step": 200000, "model": sd, "ema_model": sd}, path)
    return path


@pytest.fixture
def smoke_cfg():
    return load_config(str(ROOT / "configs" / "smoke.yaml"))


@pytest.fixture
def wflow_ckpt(tmp_path, smoke_cfg):
    return write_wflow_checkpoint(tmp_path / "state_00200000.pt", smoke_cfg["generator"])
