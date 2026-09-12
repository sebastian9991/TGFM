#!/bin/bash
#SBATCH --output=logs/graphclip_gi_%j.out
#SBATCH --error=logs/graphclip_gi_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --constraint="ampere&dgx&80gb"
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --job-name=graphclip_gi

set -euo pipefail

TARGET_BRANCH="l_br_multimodal_ablation"
REMOTE_BRANCH="br_multimodal_ablation"
CONFIG="configs/graphCLIP_mm/mm/graphclip_gi.yaml"
ENTRY="tgfm/experiments/leGTjepa/mm_main.py"

git fetch origin
git checkout "$TARGET_BRANCH"
git pull origin "$REMOTE_BRANCH"

# GraphCLIPMM imports models/gt.py and models/dp.py from the vendored repo.
export GRAPHCLIP_ROOT="${SLURM_SUBMIT_DIR:-$PWD}/third_party/GraphCLIP"
if [ ! -d "${GRAPHCLIP_ROOT}/models" ]; then
    echo "GraphCLIP repo missing at ${GRAPHCLIP_ROOT}" >&2
    exit 1
fi

echo "host=$(hostname) job=${SLURM_JOB_ID} gpus=${SLURM_GPUS_ON_NODE}"
echo "commit=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "config=${CONFIG} graphclip_root=${GRAPHCLIP_ROOT}"

uv run --group graphclip torchrun --standalone \
    --nproc_per_node="${SLURM_GPUS_ON_NODE}" \
    "${ENTRY}" \
    --config-file "${CONFIG}"
