#!/bin/bash
# Usage:  sbatch --gpus=h200:8 scripts/train.sbatch configs/B.yaml runs/B [extra -o overrides...]
#SBATCH --job-name=ptflow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=%x_%j.out
set -euo pipefail
source "${PTFLOW_ENV:-$HOME/ptflow_env.sh}"
CONFIG=$1; WORKDIR=$2; shift 2
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p "$WORKDIR"
NGPU=$(nvidia-smi -L | wc -l)
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 15 min before the wall time: ask the trainer to checkpoint, then requeue this job.
trap 'touch "$WORKDIR/REQUEST_SAVE"' USR1
torchrun --standalone --nproc_per_node="$NGPU" train.py --config "$CONFIG" --workdir "$WORKDIR" "$@" &
wait $! || wait $!
if [[ -f "$WORKDIR/SAVED_FOR_REQUEUE" && ! -f "$WORKDIR/COMPLETED" ]]; then
    rm -f "$WORKDIR/SAVED_FOR_REQUEUE"
    scontrol requeue "$SLURM_JOB_ID"
fi
