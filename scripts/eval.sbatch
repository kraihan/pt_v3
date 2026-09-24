#!/bin/bash
# FID-50K of a PT-Flow checkpoint:
#   sbatch --gpus=h200:8 scripts/eval.sbatch evaluate --ckpt runs/B/checkpoints/state_00040000.pt \
#          --mode A --cfg-scales 1.0,1.2,1.4 --json-out runs/B/fid50k_A.json
# W-Flow initializer baseline (random codes vs the fixed code PT-Flow uses):
#   sbatch --gpus=h200:8 scripts/eval.sbatch evaluate-wflow --config configs/B.yaml --noise-code random --cfg-scales 1.2
#SBATCH --job-name=ptflow-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
set -euo pipefail
source "${PTFLOW_ENV:-$HOME/ptflow_env.sh}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
torchrun --standalone --nproc_per_node="$(nvidia-smi -L | wc -l)" inference.py "$@"
