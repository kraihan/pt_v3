"""Config loading.  Strings may reference environment variables as ${NAME}."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Asset locations are environment variables (see README); these are only the names.
ENV_VARS = {
    "IMAGENET_CACHE_PATH": "latent cache dir with {train,val}_{moments,moments_flip,targets}.npy",
    "WFLOW_ROOT": "local copy of huggingface.co/jiaqihan99/W-Flow (contains checkpoints/)",
    "VAE_PATH": "local copy of huggingface.co/stabilityai/sd-vae-ft-mse",
    "FID_REF_NPZ": "ImageNet-256 Inception reference statistics (jit_in256_stats.npz)",
    "TORCH_HUB_DIR": "torch hub cache holding torch-fidelity's inception-v3-compat weights",
}


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def sub(m):
            if m.group(1) not in os.environ:
                raise KeyError(f"Config references ${{{m.group(1)}}} but it is not set in the environment")
            return os.environ[m.group(1)]
        return _VAR.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _set(cfg: Dict, dotted: str, raw: str) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:                    # YAML 1.1 reads "3e-3" as a string
            value = float(value)
        except ValueError:
            pass
    node[keys[-1]] = value


def load_config(path: str, overrides: List[str] | None = None, expand: bool = True) -> Dict:
    """Load YAML, apply ``a.b.c=value`` overrides, then expand ${ENV} references."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        k, v = item.split("=", 1)
        _set(cfg, k.strip(), v)
    return _expand(cfg) if expand else cfg


def env_path(name: str, required: bool = True) -> str:
    value = os.environ.get(name, "")
    if required and not value:
        raise KeyError(f"Set ${name}: {ENV_VARS.get(name, '')}")
    return value


def save_config(cfg: Dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(copy.deepcopy(cfg), sort_keys=False), encoding="utf-8")
