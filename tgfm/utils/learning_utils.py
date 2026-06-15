from torch import Tensor


def batch_normalize(z: Tensor) -> Tensor:
    """Per-dimension standardization across the node (batch) axis.

    ``(z - mean) / std`` with unbiased std, exactly as in the reference.
    """
    return (z - z.mean(0)) / z.std(0)
