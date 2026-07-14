import random
from pathlib import Path

import torch


def save_checkpoint(
    path: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    total_loss: float,
    prediction_loss: float,
    sig_loss: float,
) -> None:
    """Save a checkpoint. Called only on rank 0.

    Uses tmp-file + rename for atomicity: if the job dies mid-save, the old
    checkpoint is still valid and the partial .tmp file is just debris.
    """
    rng_states = {
        'python': random.getstate(),
        'torch_cpu': torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_states['torch_cuda_all'] = torch.cuda.random.get_rng_state_all()

    payload = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_loss': total_loss,
        'prediction_loss': prediction_loss,
        'sig_loss': sig_loss,
    }
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp_path)
    tmp_path.rename(path)
