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

All experiment scripts will include an argument which points to a configuration file defining experimental, data and model arguments. As well as Meta arguments for constant values paths, seeds, etc. Here is an example:

```sh
MetaArguments:
  log_file_path: "legtjepa_volume.log"
  root_dir: "graph_clip_datasets"
  is_scratch_location: true
  verbose: true
  global_seed: 42

ExperimentArguments:
  exp_args:
    LeGTJEPA:
      model_args:
        model: "LeGTJEPA"
        align_objective: "volume"
        embed_dim: 384
        lr: 1.0e-4
        weight_decay: 1.0e-5
        device: 0
      data_args:
        task_name: "node"
        target_data: "cora+citeseer+wikics"
        eval_seeds: [0, 1, 2, 3, 4]
```

For more information on the arguments check: [args.py](tgfm/utils/args.py)

## Experiments

Every experiment is a two-stage pipeline: self-supervised pretraining, then a readout on the
frozen encoder. The encoder is never trained on target labels in any of them.

`align_objective` selects the alignment term: `mse` for the cross-modal squared-distance
objective (LeVLJEPA Eq. 5), `volume` for the Gramian volume $\\det G$ (GRAM, Cicchetti et
al., ICLR 2025). Both keep the per-modality SIGReg term and use no negatives.

The experiments live on two branches. Check out the matching branch before running:

| Experiment                           | Branch                   | Config                                |
| ------------------------------------ | ------------------------ | ------------------------------------- |
| Node classification (GraphCLIP TAGs) | `br_volume_alignment`    | `configs/legtjepa/base.yaml`          |
| Link prediction (GraphCLIP TAGs)     | `br_volume_alignment`    | `configs/legtjepa/base.yaml`          |
| Linear probing (MM-Graph)            | `br_multimodal_ablation` | `configs/gramJEPA/mm/gramjepa_*.yaml` |

Each script's module docstring carries its own launch notes; the commands below are the
canonical invocations.

### 1. Node classification

Zero-shot node classification on the GraphCLIP target TAGs, following the GraphCLIP Table 2
protocol: a random 20% test split per seed, mean accuracy ± std over `data_args.eval_seeds`.
Prediction is the nearest label sentence by cosine similarity — no classifier is fit, and the
target labels are never seen by the encoder.

```sh
git checkout br_volume_alignment
```

**Pretrain.** Distributed across the GPUs on one node:

```sh
uv run torchrun \
    --standalone \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    tgfm/experiments/leGTJEPA/main.py \
    --config-file configs/legtjepa/base.yaml
```

The run checkpoints to `<root_dir>/weights/<experiment>/legtjepa.pt` after every epoch and
resumes from it automatically if present. Set `align_objective: 'volume'` in the config for
the volumetric arm and `'mse'` for the baseline; the two arms differ in nothing else.

**Evaluate.** Same config file, so the evaluation rebuilds the architecture the checkpoint
was trained with:

```sh
uv run tgfm/evaluation/zero_shot_eval.py \
    --config-file configs/legtjepa/base.yaml
```

Scoring direction comes from `model_args.zeroshot_direction`:

| Value        | Score                                                    |
| ------------ | -------------------------------------------------------- |
| `text_pred`  | $\\cos(z_g,\\ h_t(z_t))$ — LeVLJEPA's reported direction |
| `graph_pred` | $\\cos(h_g(z_g),\\ z_t)$                                 |
| `direct`     | $\\cos(z_g,\\ z_t)$                                      |

Targets come from `data_args.target_data`, a `+`-joined list of dataset names; each needs a
prompt template in `EVAL_TEMPLATE`.

### 2. Link prediction

Zero-shot link prediction on the same pretrained encoder, no retraining. A candidate link
$(u, v)$ is scored by the cosine similarity of the two endpoint ego-subgraph embeddings.
Both operands come from the same encoder, so this score needs no predictor to be well
defined — `direct` is the principled direction here.

```sh
git checkout br_volume_alignment
```

```sh
# AUC: positives against an equal number of uniformly sampled non-edges
# (GraphCLIP Sec. 4.3.2: 50% of edges held out, mean +/- std over 5 seeds)
uv run tgfm/evaluation/zero_shot_link_pred.py \
    --config-file configs/legtjepa/base.yaml

# MRR / Hits: positives ranked against the held-out negatives
uv run tgfm/evaluation/zero_shot_link_pred.py \
    --config-file configs/legtjepa/base.yaml \
    --mrr
```

**Leakage caveat.** Node $u$'s ego-subgraph contains the edge $(u, v)$ whenever $v$ is a
neighbour, so a test edge is visible in the encoder input that produces its own score. This
is inherent to the GraphCLIP protocol and applies equally to their reported numbers, so the
comparison is controlled. `--mask-test-edges` re-parses the target graphs with test edges
deleted, for an honest-but-not-comparable variant to report alongside rather than instead:

```sh
uv run tgfm/evaluation/zero_shot_link_pred.py \
    --config-file configs/legtjepa/base.yaml \
    --mask-test-edges
```

### 3. Linear probing on MM-Graph

Three-modality pretraining (graph, text, image) on the MM-Graph / Mosaic of Modalities
benchmark, evaluated by a linear probe on frozen embeddings. Pretraining uses the
link-prediction graphs; the probe evaluates transfer to the held-out node-classification
graphs, whose labels the encoder never sees.

```sh
git checkout br_multimodal_ablation
```

**Data.** MM-Graph ships precomputed per-node features, so there is no tokenization or image
encoding step. Directory names on disk differ from the paper's display names:

| Paper         | Directory           | Task                |
| ------------- | ------------------- | ------------------- |
| Amazon-Sports | `sports-copurchase` | link prediction     |
| Amazon-Cloth  | `cloth-copurchase`  | link prediction     |
| Goodreads-LP  | `books-lp`          | link prediction     |
| Ele-Fashion   | `ele-fashion`       | node classification |
| Goodreads-NC  | `books-nc`          | node classification |

```sh
DEST=$SCRATCH/mm_graph_datasets

hf download mm-graph-org/mm-graph --repo-type dataset --local-dir "$DEST" \
  --include "sports-copurchase/*" "cloth-copurchase/*" "books-lp/*" \
            "ele-fashion/*" "books-nc/*" \
  --exclude "*/clip_feat.pt" "*/imagebind_feat.pt" "*/t5vit_feat.pt"
```

The `--exclude` drops the feature bundles the configs do not use. `t5dino_feat.pt` is the
default (`mm_feat_name: 't5dino'`): a single `N x 1536` float32 tensor, T5 text (768)
concatenated with DINOv2 image (768), **text first**. DINOv2 rather than CLIP or ImageBind is
deliberate — a text-aligned image encoder pre-collapses the text-image volume before the
objective acts on it, so the alignment would be inherited rather than earned.

**Pretrain.** One config per modality combination, under `configs/gramJEPA/mm/`:

```sh
ls configs/gramJEPA/mm/gramjepa_*.yaml
```

```sh
uv run torchrun \
    --standalone \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    tgfm/experiments/leGTjepa/mm_main.py \
    --config-file configs/gramJEPA/mm/gramjepa_<combination>.yaml
```

The combination is set by `model_args.graph_feat` (which node features the graph tower
consumes) together with `use_image`; run each config to fill one row of the ablation:

| Combination          | Towers             |
| -------------------- | ------------------ |
| graph + text         | graph, text        |
| graph + image        | graph, image       |
| graph + text + image | graph, text, image |

`mm_main.py` runs the probe in-loop every `data_args.eval_every_epochs` epochs and keeps the
epoch with the highest macro **validation** accuracy, writing it to
`<root_dir>/weights/<experiment>/legtjepa_best.pt` alongside the last-epoch `legtjepa.pt`.
Test labels influence neither choice.

**Probe.** Standalone evaluation of a saved checkpoint:

```sh
uv run tgfm/evaluation/mm_linear_probe.py \
    --config-file configs/gramJEPA/mm/gramjepa_<combination>.yaml \
    --representation both \
    --ckpt-name legtjepa_best.pt
```

The probe is a single `torch.nn.Linear(d, num_classes)` on standardized frozen features — no
nonlinearity, no fine-tuning — trained on the dataset's own `train_mask`, selected on
`val_mask`, scored on `test_mask`. It reports accuracy and macro-F1, and emits a `TABLE-ROW`
line per representation for pasting into the results table.

`--representation` picks which layer is probed:

- `projection` — the `graph_projection` output, what the objective acts on and what
  `encode_graph` returns. The default.
- `backbone` — `[mean-pool || center]` before that head, `2 * graph_hidden_dim` wide.
- `both` — both, from a single ego-subgraph pass.

`--ckpt-name` selects `legtjepa_best.pt` (validation-selected) or `legtjepa.pt` (last epoch).

## SLURM

Both pretraining entry points are `torchrun` scripts. An example multi-GPU launch:

```sh
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=400G
#SBATCH --job-name=gramjepa-pretrain

set -e
echo "Date:     $(date)"
echo "Job ID:   $SLURM_JOB_ID"
echo "Nodes:    $SLURM_JOB_NODELIST"
echo "Attempt:  #${SLURM_RESTART_COUNT:-0}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Note the bash -c wrapper so SLURM_NODEID is evaluated in each task.
srun --gres-flags=allow-task-sharing bash -c "
    uv run torchrun \
        --nnodes=\$SLURM_NNODES \
        --node_rank=\$SLURM_NODEID \
        --nproc_per_node=\$SLURM_GPUS_ON_NODE \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        tgfm/experiments/leGTjepa/mm_main.py \
        --config-file configs/gramJEPA/mm/gramjepa_gti.yaml
    "
```

### Sweeps

Hyperparameter sweeps are launched with a `wandb` agent inside `sbatch`. Each sweep config
pins `align_objective`, so one sweep covers one arm and the two are filterable on that axis:

```sh
wandb sweep bash_scripts/sweeps/random_large_sweep.yaml   # random search
wandb sweep bash_scripts/sweeps/bayes_volume_sweep.yaml   # bayes refinement

sbatch bash_scripts/sweeps/volume.sh <sweep-id>
```
