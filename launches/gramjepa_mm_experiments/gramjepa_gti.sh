#!/bin/bash
#SBATCH --output=logs/gramjepa_gti_%j.out
#SBATCH --error=logs/gramjepa_gti_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --constraint="ampere&dgx&80gb"
#SBATCH --mem=256G
#SBATCH --time=32:00:00
#SBATCH --job-name=gramjepa_gti

# GramJEPA (G-T-I): volume objective over (graph, text, image), [T5 || DINOv2]
# node features (graph_in_dim 1536), both partner projections frozen. Pretrain
# on sports+cloth+books-lp, linear probe on ele-fashion+books-nc.

set -euo pipefail

REPO="${HOME}/misinfo/org/TGFM"
cd "${REPO}"

TARGET_BRANCH="l_br_multimodal_ablation"
REMOTE_BRANCH="br_multimodal_ablation"
CONFIG="configs/gramJEPA/mm/gramjepa_gti_tuned.yaml"
ENTRY="tgfm/experiments/leGTjepa/mm_main.py"

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Uncommitted changes in ${REPO}; commit or stash before submitting:" >&2
    git status --short >&2
    exit 1
fi

git fetch origin
git checkout "$TARGET_BRANCH"
git pull origin "$REMOTE_BRANCH"

for required in "${ENTRY}" "${CONFIG}"; do
    if [ ! -f "${required}" ]; then
        echo "Missing on ${TARGET_BRANCH}: ${required}" >&2
        echo "Candidates on this branch:" >&2
        git ls-files ':/' | grep -iE 'mm_main\.py|mm/.*\.yaml' >&2 || true
        exit 1
    fi
done


echo "host=$(hostname) job=${SLURM_JOB_ID} gpus=${SLURM_GPUS_ON_NODE}"
echo "commit=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "config=${CONFIG}"

uv run --group graphclip torchrun --standalone \
    --nproc_per_node="${SLURM_GPUS_ON_NODE}" \
    "${ENTRY}" \
    --config-file "${CONFIG}"
