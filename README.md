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

````sh
# Clone the repo
git clone git@github.com:sebastian9991/TGFM.git

# Enter the repo directory
cd TGFM

# Install core dependencies into an isolated environment
uv sync

# The isolated env is .venv, you may source it like so:
source .venv/bin/activate


## Usage

### Running GNN Experiments

Given the size of our datasets we must leverage mini-batching in our GNN experiments. To do this we use PyG's `neighbor_loader`,
which requires additional libraries having undocumented build-time dependencies. As such, users are required to install them in their
own venv. seperate from `uv sync`.


PyTorch Sparse, Scatter and pyg-lib:

```sh
uv pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
````

For information on installations of these additional libraries see [pyg-lib](https://github.com/pyg-team/pyg-lib) and [PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse).
