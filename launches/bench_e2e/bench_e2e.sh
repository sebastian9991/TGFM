#!/bin/bash
#SBATCH --output=logs/bench_e2e_%j.out
#SBATCH --error=logs/bench_e2e_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --constraint="ampere&dgx&80gb"
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --job-name=bench_e2e

# End-to-end per-step cost (Reviewer 2, Q1): LeGramJEPA (volume + SIGReg) vs
# InfoNCE over all modality pairs vs pairwise L_cross + SIGReg, plus the
# encoder-only baseline and the GraphCLIP model as published (k = 2).
# k = 2 (G-T) and k = 3 (G-T-I), real MM-Graph batches, global B swept
# 450..7200 on 4 GPUs under DDP exactly as in mm_main.py.
#
#   Phase 1  matched encoders (12-layer GPS, 1024 wide = GraphCLIP Base):
#            only the objective differs between arms.  -> the comparison table
#   Phase 2  GramJEPA as actually trained (tuned 6-layer configs) at its
#            training batch (450/rank x 4 = 1800).       -> "as trained" row
#
# Override with env vars, e.g.  SEEDS="0" BATCHES="1800 3600" sbatch ...

set -euo pipefail

REPO="${HOME}/misinfo/org/TGFM"
cd "${REPO}"

TARGET_BRANCH="${TARGET_BRANCH:-br_e2e_cost_bench}"
ENTRY="tgfm/tools/bench_e2e_cost.py"
SEEDS="${SEEDS:-0 1 2}"
BATCHES="${BATCHES:-450 900 1800 3600 7200}"
WARMUP="${WARMUP:-20}"
ITERS="${ITERS:-100}"
OUT_DIR="${OUT_DIR:-results/bench_e2e/${SLURM_JOB_ID:-local}}"

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Uncommitted changes in ${REPO}; commit or stash before submitting:" >&2
    git status --short >&2
    exit 1
fi
git fetch origin
git checkout "${TARGET_BRANCH}"
git pull origin "${TARGET_BRANCH}"

mkdir -p "${OUT_DIR}" logs
NGPU="${SLURM_GPUS_ON_NODE:-4}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} gpus=${NGPU}"
echo "commit=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Phase 1: matched encoders, all arms, B and k sweeps.
uv run --no-sync --group graphclip torchrun --standalone --nproc_per_node="${NGPU}" \
    "${ENTRY}" \
    --data real \
    --arms volume pairwise infonce encoder_only graphclip \
    --k 2 3 \
    --batches ${BATCHES} \
    --seeds ${SEEDS} \
    --warmup "${WARMUP}" --iters "${ITERS}" \
    --out "${OUT_DIR}/matched.json" \
    --emit-markdown "${OUT_DIR}/matched.md"

# Phase 2: GramJEPA at its tuned training configs and batch.
uv run --no-sync --group graphclip torchrun --standalone --nproc_per_node="${NGPU}" \
    "${ENTRY}" \
    --data real \
    --k2-config configs/gramJEPA/mm/gramjepa_gt_tuned.yaml \
    --k3-config configs/gramJEPA/mm/gramjepa_gti_tuned.yaml \
    --arms volume encoder_only \
    --k 2 3 \
    --batches 1800 \
    --seeds ${SEEDS} \
    --warmup "${WARMUP}" --iters "${ITERS}" \
    --out "${OUT_DIR}/as_trained.json" \
    --emit-markdown "${OUT_DIR}/as_trained.md"

echo "Results in ${OUT_DIR}"
