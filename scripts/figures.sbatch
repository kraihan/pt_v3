#!/bin/bash
# All mechanism experiments (Figs 1-5) on one PT-Flow checkpoint, then the figures.
#
#   sbatch -p work1 --gpus=h200:1 scripts/figures.sbatch CKPT OUT_DIR [SCALE] [TRAIN_LOG]
#
# CKPT       calib_state_*.pt (generator = W-Flow, potential calibrated), eval_state_*.pt or state_*.pt
# SCALE      diagonal proposal scale: hutchinson (default; use for calib_state, whose scale head is still zero)
#            or learned (the generator's scale head, after Algorithm 1 has trained it)
# TRAIN_LOG  optional metrics.jsonl of an eps-annealing run (Fig 2c, Fig 3d)
#SBATCH --job-name=ptflow-figs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
set -euo pipefail
source "${PTFLOW_ENV:-$HOME/ptflow_env.sh}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
CKPT=$1; OUT=$2; SCALE=${3:-hutchinson}; TRAIN_LOG=${4:-}
COMMON=(--ckpt "$CKPT" --out "$OUT" --scale "$SCALE" --batch 8 --chunk 64 --alpha 0.1 --seed 0)

python experiments.py proposals "${COMMON[@]}" --n 256 --ks 8,16,64                          # Fig 1
python experiments.py eps-sweep "${COMMON[@]}" --n 128 --K 16 --probes 8 \
    --eps-list 0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001                                       # Fig 2
python experiments.py prox      "${COMMON[@]}" --n 256 --ws 0,0.2,1.0 --refine 1,3,10          # Fig 3
python experiments.py mismatch  "${COMMON[@]}" --n 128 --K 16 --alphas 0,0.1,0.5              # Fig 4
python experiments.py nll       "${COMMON[@]}" --n 128 --k-outer 4,16,64 --k-inner 4,16,64 --seeds 3   # Fig 5

python scripts/make_figures.py "$OUT" ${TRAIN_LOG:+--train-log "$TRAIN_LOG"}
echo "Figures: $OUT/figures  (summary: $OUT/figures/summary.md)"
