#!/bin/bash
#SBATCH --partition=long
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100l:4
#SBATCH --constraint="ampere&nvlink&80gb"
#SBATCH --cpus-per-task=24
#SBATCH --mem=512G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%j-out.txt
#SBATCH --error=logs/%j-err.txt
#SBATCH --job-name=bench-e2e-B14400

# End-to-end per-step cost at global B = 14400: 4 nodes x 4 A100 = 16 GPUs,
# 900 subgraphs per GPU (the same per-GPU batch as the 4-GPU B = 3600 run).
# Adding GPUs at a fixed per-GPU batch keeps the encoder cost per GPU flat
# while InfoNCE / SIGReg see the all-gathered global batch. No activation
# checkpointing, so these rows are directly comparable to the B = 3600 run.
# Submit from the repo root with br_e2e_cost_bench checked out and pulled.

set -e
echo "Date:     $(date)"
echo "Job ID:   $SLURM_JOB_ID"
echo "Nodes:    $SLURM_JOB_NODELIST"
echo "Commit:   $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD))"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

# Override per submission, e.g. K=3 ARMS="encoder_only volume pairwise infonce" sbatch ...
K="${K:-2 3}"
ARMS="${ARMS:-volume pairwise infonce encoder_only graphclip}"
OUT_DIR="results/bench_e2e/multinode_B14400_k${K// /}_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR" logs

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Slurm Nodes: $(($SLURM_NNODES))"

# Static rendezvous; bash -c so SLURM_NODEID is evaluated in each task.
srun --gres-flags=allow-task-sharing bash -c "
    uv run --no-sync --group graphclip torchrun \
        --nnodes=\$SLURM_NNODES \
        --node_rank=\$SLURM_NODEID \
        --nproc_per_node=\$SLURM_GPUS_ON_NODE \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        tgfm/tools/bench_e2e_cost.py \
        --data real \
        --arms $ARMS \
        --k $K \
        --batches 14400 \
        --seeds 0 \
        --warmup 20 --iters 100 \
        --out $OUT_DIR/matched.json \
        --emit-markdown $OUT_DIR/matched.md
    "
echo "Results in $OUT_DIR"
