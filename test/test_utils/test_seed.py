import random

import numpy as np
import pytest
import torch

from tgfm.utils.seed import seed_everything


def test_random_module_reproducibility():
    """Verify that the standard random module is seeded correctly."""
    seed = 42
    seed_everything(seed)
    val1 = random.random()

    seed_everything(seed)
    val2 = random.random()

    assert val1 == val2


def test_numpy_reproducibility():
    """Verify that numpy random generators are seeded correctly."""
    seed = 123
    seed_everything(seed)
    arr1 = np.random.rand(5)

    seed_everything(seed)
    arr2 = np.random.rand(5)

    assert np.array_equal(arr1, arr2)


def test_torch_cpu_reproducibility():
    """Verify that PyTorch CPU tensors are seeded correctly."""
    seed = 999
    seed_everything(seed)
    tensor1 = torch.randn(3, 3)

    seed_everything(seed)
    tensor2 = torch.randn(3, 3)

    assert torch.equal(tensor1, tensor2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_torch_cuda_reproducibility():
    """Verify that PyTorch GPU tensors are seeded correctly."""
    seed = 7
    seed_everything(seed)
    tensor1 = torch.randn(3, 3, device="cuda")

    seed_everything(seed)
    tensor2 = torch.randn(3, 3, device="cuda")

    assert torch.equal(tensor1, tensor2)


def test_different_seeds_produce_different_results():
    """Verify that using different seeds produces different results."""
    seed_everything(1)
    val1 = random.random()

    seed_everything(2)
    val2 = random.random()

    assert val1 != val2
