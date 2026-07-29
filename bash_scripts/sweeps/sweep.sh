#!/bin/bash
set -e
uv run --group graphclip torchrun --standalone \
    --nproc_per_node=${SLURM_GPUS_ON_NODE:-4} \
    tgfm/experiments/leGTjepa/main.py \
    --config-file configs/legtjepa/bert.yaml
