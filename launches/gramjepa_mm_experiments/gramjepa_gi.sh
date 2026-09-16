#!/bin/bash
#SBATCH --output=logs/gramjepa_gi_%j.out
#SBATCH --error=logs/gramjepa_gi_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --constraint="ampere&dgx&80gb"
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --job-name=gramjepa_gi

# GramJEPA (G-I): volume objective over (graph, image), DINOv2 node features,
# frozen image projection. Pretrain on sports+cloth+books-lp, linear probe on
# ele-fashion+books-nc.

set -euo pipefail

TARGET_BRANCH="l_br_multimodal_ablation"
REMOTE_BRANCH="br_multimodal_ablation"
CONFIG="configs/gramJEPA/mm/gramjepa_gi_tuned.yaml"
ENTRY="tgfm/experiments/leGTjepa/mm_main.py"

git fetch origin
git checkout "$TARGET_BRANCH"
git pull origin "$REMOTE_BRANCH"

echo "host=$(hostname) job=${SLURM_JOB_ID} gpus=${SLURM_GPUS_ON_NODE}"
echo "commit=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "config=${CONFIG}"

uv run --group graphclip torchrun --standalone \
    --nproc_per_node="${SLURM_GPUS_ON_NODE}" \
    "${ENTRY}" \
    --config-file "${CONFIG}"
