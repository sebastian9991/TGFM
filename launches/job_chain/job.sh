#!/bin/bash
#SBATCH --partition=long
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --constraint="ampere&nvlink&80gb"
#SBATCH --cpus-per-task=24
#SBATCH --mem=512G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%j-out.txt
#SBATCH --error=logs/%j-err.txt
#SBATCH --job-name=unigraph-pretrain

set -e
echo "Date:     $(date)"
echo "Job ID:   $SLURM_JOB_ID"
echo "Nodes:    $SLURM_JOB_NODELIST"

echo "Attempt: #${SLURM_RESTART_COUNT:-0}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

export WANDB_RUN_GROUP="unigraph-pretrain-multinode"
export WANDB_TAGS="mila,a100l,2node,bert-base"
export WANDB_DIR="$HOME/scratch/wandb_runs"
mkdir -p "$WANDB_DIR" logs

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Slurm Nodes: $(($SLURM_NNODES))"



#Ensure that we do not cancel with time premption
if [[ -n "$PREV_JOBID" ]]; then
    prev_state=$(sacct -j "$PREV_JOBID" -X -n -o State | awk '{print $1}')
    echo "Previous job state: $prev_state"
    case "$prev_state" in
        COMPLETED|TIMEOUT) echo "Continuing." ;;
        *) echo "Aborting chain."; exit 1 ;;
    esac
fi

# Static rendezvous: no race condition between nodes for the store.
# Note the bash -c wrapper so SLURM_NODEID is evaluated in each task.
srun --gres-flags=allow-task-sharing bash -c "
    uv run torchrun \
        --nnodes=\$SLURM_NNODES \
        --node_rank=\$SLURM_NODEID \
        --nproc_per_node=\$SLURM_GPUS_ON_NODE \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        tgfm/experiments/unigraph/pretraining.py \
        --config-file configs/unigraph/base.yaml
    "
