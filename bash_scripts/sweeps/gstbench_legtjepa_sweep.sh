#!/bin/bash
# One sweep trial. wandb executes `command` without a shell, so SLURM
# variables have to be expanded here rather than in the sweep yaml.
set -e

# Reap orphans from a previous trial in this allocation, then wait for the
# GPUs to actually free: a crashed trial's ranks can hold memory past the
# launcher's exit and OOM the next one.
pkill -f 'legtjepa.py' 2>/dev/null || true
for i in $(seq 1 24); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "$used" -lt 2000 ] && break
    echo "GPUs not free yet (max ${used} MiB used), waiting... ($i)"
    sleep 5
done

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export DGLBACKEND=pytorch
ulimit -n "$(ulimit -Hn)"      # LazyGraphDataset workers open many files

uv run python tgfm/experiments/transfer_learning/legtjepa.py \
    --config-file configs/transfer_learning/legtjepa.yaml \
    --num_gpus "${SLURM_GPUS_ON_NODE:-1}"
