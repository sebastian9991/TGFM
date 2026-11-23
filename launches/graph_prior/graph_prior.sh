#!/bin/bash
#SBATCH --output=logs/graph_prior_%A_%a.out
#SBATCH --error=logs/graph_prior_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --constraint="ampere&dgx&80gb"
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --job-name=graph_prior
#SBATCH --array=0-17

# Graph-prior regularizer vs SIGReg: 6 arms x 3 seeds, one array task each.
# Each task pretrains (4 GPUs), then runs zero-shot node classification and
# zero-shot link prediction (AUC) on cora, wikics, history.
# Submit from the repo root, on the br_graph_prior_sigreg branch:
#     sbatch launches/graph_prior/graph_prior.sh
# One arm only, e.g. all seeds of gsig_t1:   sbatch --array=6-8 launches/graph_prior/graph_prior.sh
# Summary once tasks finish:   uv run python scripts/graph_prior_summary.py
#
# Arms (name  regularizer  t  sampler):
#   sigreg_rand    sigreg   -    random    the best volume config as is
#   sigreg_clust   sigreg   -    cluster   same batches as the graph arms
#   gsig_t0.1      gsigreg  0.1  cluster
#   gsig_t1        gsigreg  1    cluster
#   gsig_t5        gsigreg  5    cluster
#   gmmd_t1        gmmd     1    cluster
# Task id = 3 * arm + seed.

set -euo pipefail

ARMS=(
    "sigreg_rand sigreg 1.0 random"
    "sigreg_clust sigreg 1.0 cluster"
    "gsig_t0.1 gsigreg 0.1 cluster"
    "gsig_t1 gsigreg 1.0 cluster"
    "gsig_t5 gsigreg 5.0 cluster"
    "gmmd_t1 gmmd 1.0 cluster"
)
SEEDS=(0 1 2)
ENTRY="tgfm/experiments/leGTjepa/main.py"
BASE="configs/legtjepa/graph_prior/base.yaml"
CONF_DIR="configs/legtjepa/graph_prior/runs"

cd "${SLURM_SUBMIT_DIR:-$HOME/misinfo/org/TGFM}"
mkdir -p logs "$CONF_DIR"

task="${SLURM_ARRAY_TASK_ID:-0}"
read -r arm reg t sampler <<< "${ARMS[$((task / 3))]}"
seed="${SEEDS[$((task % 3))]}"
key="gprior_${arm}_s${seed}"
cfg="${CONF_DIR}/${key}.yaml"

sed -e "s/RUN_KEY:/${key}:/" \
    -e "s/LOG_NAME/${key}_train/" \
    -e "s/global_seed: SEED/global_seed: ${seed}/" \
    -e "s/regularizer: \"sigreg\"/regularizer: \"${reg}\"/" \
    -e "s/graph_kernel_t: 1.0/graph_kernel_t: ${t}/" \
    -e "s/batch_sampler: \"random\"/batch_sampler: \"${sampler}\"/" \
    "$BASE" > "$cfg"

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} task=${task} gpus=${SLURM_GPUS_ON_NODE:-?}"
echo "commit=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
echo "run=${key} regularizer=${reg} t=${t} sampler=${sampler} seed=${seed}"
grep -E "regularizer|graph_kernel_t|batch_sampler|global_seed|^    gprior" "$cfg"

# 1. Pretrain. main.py resumes from weights/<key>/legtjepa.pt if a previous
#    attempt was cut off; delete that folder to restart from scratch.
uv run torchrun --standalone --nproc_per_node="${SLURM_GPUS_ON_NODE:-4}" \
    "$ENTRY" --config-file "$cfg"

# 2. Zero-shot node classification and 3. link prediction (AUC), one GPU.
#    setup_logging truncates its file, so each stage gets its own log:
#    logs/<key>_train.log, logs/<key>_nc.log, logs/<key>_lp.log.
sed -e "s/${key}_train.log/${key}_nc.log/" "$cfg" > "${CONF_DIR}/${key}_nc.yaml"
sed -e "s/${key}_train.log/${key}_lp.log/" "$cfg" > "${CONF_DIR}/${key}_lp.yaml"
uv run python tgfm/evaluation/zero_shot_eval.py --config-file "${CONF_DIR}/${key}_nc.yaml"
uv run python tgfm/evaluation/zero_shot_link_pred.py \
    --config-file "${CONF_DIR}/${key}_lp.yaml" --evaluation-method AUC
echo "[done] ${key} $(date +%FT%T)"
