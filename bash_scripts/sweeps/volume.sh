#!/bin/bash
set -e

# Reap orphans from a previous trial in this allocation (only our user's procs).
pkill -f 'leGTjepa/main.py' 2>/dev/null || true

# Wait until the allocated GPUs are actually free (orphans can take ~NCCL
# timeout to die); bail out loudly rather than launch into occupied memory.
for i in $(seq 1 24); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "$used" -lt 2000 ] && break
    echo "GPUs not free yet (max ${used} MiB used), waiting... ($i)"
    sleep 5
done

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

uv run torchrun --standalone \
    --nproc_per_node=${SLURM_GPUS_ON_NODE:-4} \
    tgfm/experiments/leGTjepa/main.py \
    --config-file configs/legtjepa/volume.yaml
