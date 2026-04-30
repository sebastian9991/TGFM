# TGFM

Text-Graph Foundation Model

## Getting Started

### Prerequisites

The project uses [uv](https://docs.astral.sh/uv/) to manage and lock project dependencies for a consistent and reproducible environment. If you do not have `uv` installed on your system, visit [this page](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

**Note**: If you have `pip`, you can invoke:

```sh
pip install uv
```

### Installation

```sh
# Clone the repo
git clone git@github.com:sebastian9991/TGFM.git

# Enter the repo directory
cd TGFM

# Install core dependencies into an isolated environment
uv sync

# The isolated env is .venv, you may source it like so:
source .venv/bin/activate
```

### Running mini-batching with PyG's loaders:

Given the size of our datasets we must leverage mini-batching in our GNN experiments. To do this we use PyG's `neighbor_loader`,
which requires additional libraries having undocumented build-time dependencies. As such, users are required to install them in their
own venv. seperate from `uv sync`.

PyTorch Sparse, Scatter and pyg-lib:

```sh
uv pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
```

For more information on installations of these additional libraries see [pyg-lib](https://github.com/pyg-team/pyg-lib) and [PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse).

## Usage

### Pre-Training

Due to the scale of the graph datasets and text attributes that come with them we recommend pre-training this in a distributed setup. Here is an example launch script:

```sh
#!/bin/bash
#SBATCH --nodes=x
#SBATCH --ntasks-per-node=1
#SBATCH --mem=400G
#SBATCH --job-name=unigraph-pretrain

set -e
echo "Date:     $(date)"
echo "Job ID:   $SLURM_JOB_ID"
echo "Nodes:    $SLURM_JOB_NODELIST"

echo "Attempt: #${SLURM_RESTART_COUNT:-0}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Slurm Nodes: $(($SLURM_NNODES))"

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
```

### Evaluation
