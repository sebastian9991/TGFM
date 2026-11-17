#!/bin/bash
#SBATCH --output=logs/bench_e2e_largeB_%j.out
#SBATCH --error=logs/bench_e2e_largeB_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --constraint="ampere&dgx&80gb"
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --job-name=bench_e2e_largeB

# Large-B end-to-end benchmark (Reviewer 2, Q1): pushes the global batch past
# the Figure 4 crossover with the encoder included, using activation
# checkpointing on the GPS layers (same math, ~1 extra encoder forward).
#
#   Part A  REAL 4-GPU DDP, checkpointed encoder, global B = 3600/7200/14400
#           (900/1800/3600 per GPU). Each B is its own torchrun under a
#           timeout, because an OOM on one rank can hang DDP.
#   Part B  EMULATED larger clusters, one process per GPU, 900 per GPU held
#           fixed (the "add GPUs" regime): global B = 900*W for
#           W = 4, 8, 16, 32, 64. The encoder runs on 900 samples; the loss
#           gathers are tiled W times so InfoNCE / SIGReg see the global B.
#           No communication time. Label as emulation when reporting.
#
# Override with env vars, e.g.  SEEDS="0 1" REAL_BATCHES="3600 7200" sbatch ...

set -euo pipefail

REPO="${HOME}/misinfo/org/TGFM"
cd "${REPO}"

TARGET_BRANCH="${TARGET_BRANCH:-br_e2e_cost_bench}"
ENTRY="tgfm/tools/bench_e2e_cost.py"
SEEDS="${SEEDS:-0}"
REAL_BATCHES="${REAL_BATCHES:-3600 7200 14400}"
EMU_WORLDS="${EMU_WORLDS:-4 8 16 32 64}"
PER_GPU="${PER_GPU:-900}"
WARMUP="${WARMUP:-20}"
ITERS="${ITERS:-100}"
REAL_TIMEOUT="${REAL_TIMEOUT:-3h}"
OUT_DIR="${OUT_DIR:-results/bench_e2e_largeB/${SLURM_JOB_ID:-local}}"

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

COMMON=(--data real --k 2 3 --seeds ${SEEDS} --warmup "${WARMUP}" --iters "${ITERS}"
        --checkpoint-encoder)

# Part A: real DDP, one torchrun per B.
for B in ${REAL_BATCHES}; do
    echo "=== real DDP B=${B} ==="
    timeout "${REAL_TIMEOUT}" \
        uv run --no-sync --group graphclip torchrun --standalone --nproc_per_node="${NGPU}" \
        "${ENTRY}" "${COMMON[@]}" \
        --arms volume pairwise infonce encoder_only graphclip \
        --batches "${B}" \
        --out "${OUT_DIR}/real_B${B}.json" \
        --emit-markdown "${OUT_DIR}/real_B${B}.md" \
        || echo "real B=${B} failed or timed out (likely OOM hang); continuing"
done

# Part B: emulated W-GPU jobs, one per GPU, NGPU at a time.
run_emu() {
    local gpu=$1 W=$2
    local B=$((PER_GPU * W))
    CUDA_VISIBLE_DEVICES="${gpu}" uv run --no-sync --group graphclip python \
        "${ENTRY}" "${COMMON[@]}" \
        --arms volume pairwise infonce encoder_only \
        --emulate-world "${W}" \
        --batches "${B}" \
        --out "${OUT_DIR}/emu_W${W}_B${B}.json" \
        --emit-markdown "${OUT_DIR}/emu_W${W}_B${B}.md" \
        > "${OUT_DIR}/emu_W${W}.log" 2>&1 \
        || echo "emulated W=${W} failed; see ${OUT_DIR}/emu_W${W}.log"
}
gpu=0
for W in ${EMU_WORLDS}; do
    echo "=== emulated W=${W} (global B=$((PER_GPU * W))) on GPU ${gpu} ==="
    run_emu "${gpu}" "${W}" &
    gpu=$(( (gpu + 1) % NGPU ))
    if [ "${gpu}" -eq 0 ]; then wait; fi
done
wait

echo "Results in ${OUT_DIR}"
