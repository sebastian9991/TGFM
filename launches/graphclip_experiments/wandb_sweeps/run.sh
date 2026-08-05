#!/bin/bash
#SBATCH --output=logs/legtjepa_sweep_%A_%a.out
#SBATCH --error=logs/legtjepa_sweep_%A_%a.err
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --constraint="ampere&dgx&80gb"
#SBATCH --mem=512G
#SBATCH --time=15:00:00
#SBATCH --job-name=legtjepa_sweep

srun --gres-flags=allow-task-sharing \
    uv run wandb agent "$SWEEP_ID"
