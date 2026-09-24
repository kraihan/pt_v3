# Copy to ~/ptflow_env.sh, edit the paths, and `source` it before training or evaluation.
# conda activate ptflow            # or your venv

export IMAGENET_CACHE_PATH=/scratch/$USER/ptflow/data/latents          # {train,val}_{moments,moments_flip,targets}.npy
export WFLOW_ROOT=/scratch/$USER/ptflow/wflow_hf                       # huggingface.co/jiaqihan99/W-Flow (has checkpoints/)
export VAE_PATH=/scratch/$USER/ptflow/sd-vae-ft-mse                    # huggingface.co/stabilityai/sd-vae-ft-mse
export FID_REF_NPZ=/scratch/$USER/ptflow/assets/fid_stats/jit_in256_stats.npz
export TORCH_HUB_DIR=/scratch/$USER/ptflow/torch_hub                   # torch-fidelity inception-v3-compat weights
export HF_HOME=/scratch/$USER/hf_cache
# export WANDB_API_KEY=...                                             # only if logging.use_wandb: true
