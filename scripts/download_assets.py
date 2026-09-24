"""Fetch the pretrained assets into the directories named by the environment variables.

    source ~/ptflow_env.sh
    python scripts/download_assets.py --models B L XL      # W-Flow checkpoints you need
The FID reference statistics (FID_REF_NPZ) are the ImageNet-256 stats W-Flow uses
(jit_in256_stats.npz); place that file yourself if you do not already have it.
"""

from __future__ import annotations

import argparse
import os

CKPTS = {"B": "checkpoints/latent_sota_B_ot/state_00200000.pt",
         "L": "checkpoints/latent_sota_L_ot/state_00200000.pt",
         "XL": "checkpoints/latent_sota_XL_ot/state_00180000.pt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["B"], choices=list(CKPTS))
    a = ap.parse_args()
    from huggingface_hub import snapshot_download

    snapshot_download("jiaqihan99/W-Flow", local_dir=os.environ["WFLOW_ROOT"],
                      allow_patterns=[CKPTS[m] for m in a.models])
    snapshot_download("stabilityai/sd-vae-ft-mse", local_dir=os.environ["VAE_PATH"])

    import torch
    torch.hub.set_dir(os.environ["TORCH_HUB_DIR"])
    from torch_fidelity.utils import create_feature_extractor
    create_feature_extractor("inception-v3-compat", ["2048", "logits_unbiased"], cuda=False)
    print("W-Flow:", [os.path.join(os.environ["WFLOW_ROOT"], CKPTS[m]) for m in a.models])
    print("VAE:", os.environ["VAE_PATH"], "| Inception cached in", os.environ["TORCH_HUB_DIR"])


if __name__ == "__main__":
    main()
