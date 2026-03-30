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

### Text Processing

This project uses [nemo-curator](https://github.com/NVIDIA-NeMo/Curator) to process the CrediBench text into a usable and quality format for pre-training. Currently, nemo-curator fails with UV project API to define the project dependencies. Currently only installing packages into the active enironment works:

```sh
uv pip install "nemo-curator[text_cuda12]"
```
