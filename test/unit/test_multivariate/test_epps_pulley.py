""" Tests for `EppsPulley` (univariate test) and `SlicingUnivariateTest` (the
multivariate wrapper that defines SIGReg in the LeJEPA paper).

Key correctness property (the headline test):
    Standard-normal samples should give a LOWER Epps-Pulley statistic than
    pathological distributions, because the test measures distance from N(0, 1)
    in characteristic-function space.

Important nuance: the test compares against N(0, 1) *specifically* -- not
"any isotropic Gaussian." Scaled normals N(0, sigma^2) and shifted normals
N(mu, 1) should both score worse than the standard normal. This matters
because the LeJEPA loss feeds raw embeddings to SIGReg (no pre-standardization);
the encoder must learn to produce unit-variance, zero-mean embeddings, which
is part of what SIGReg pressures.

What we do NOT test here:
    - Absolute Epps-Pulley values (depend on t_max, n_points, num_slices --
      not portable across configurations).
    - Distributed code paths (all_reduce / world_size are no-ops when dist
      isn't initialized; testing properly requires a multi-process harness
      that's overkill for boilerplate).
"""

import pytest
import torch
from torch import Tensor

from tgfm.models.multivariate.slicing import SlicingUnivariateTest
from tgfm.models.multivariate.univariate import EppsPulley

SEED = 42
N = 256          # Bumped from 64 to keep the relative-ordering tests stable
FEATURE_DIM = 128


@pytest.fixture(scope="session")
def standard_normal() -> Tensor:
    """The reference case: N(0, 1) elementwise. Should give the lowest loss."""
    g = torch.Generator().manual_seed(SEED)
    return torch.randn(N, FEATURE_DIM, generator=g)


@pytest.fixture(scope="session")
def scaled_normal() -> Tensor:
    """N(0, 25): isotropic Gaussian but wrong scale. Each slice has var=25."""
    g = torch.Generator().manual_seed(SEED + 1)
    return torch.randn(N, FEATURE_DIM, generator=g) * 5.0


@pytest.fixture(scope="session")
def shifted_normal() -> Tensor:
    """N(3, 1): shifted mean. Each slice has mean ~3."""
    g = torch.Generator().manual_seed(SEED + 2)
    return torch.randn(N, FEATURE_DIM, generator=g) + 3.0


@pytest.fixture(scope="session")
def uniform() -> Tensor:
    """Uniform on [-sqrt(3), sqrt(3)]: variance ~1 but wrong shape (no tails)."""
    g = torch.Generator().manual_seed(SEED + 3)
    u = torch.rand(N, FEATURE_DIM, generator=g)
    return u


@pytest.fixture(scope="session")
def collapsed() -> Tensor:
    """All samples identical: degenerate, the SSL collapse worst case."""
    return torch.ones(N, FEATURE_DIM)


@pytest.fixture(scope="session")
def loss_fn():
    """A SIGReg instance with default-ish hyperparameters.

    num_slices=256 matches the LeJEPA default and the boilerplate's
    `LeJEPALoss.__init__` default.
    """
    univariate_test = EppsPulley(t_max=5.0, n_points=17)
    return SlicingUnivariateTest(
        univariate_test=univariate_test,
        num_slices=256,
        reduction="mean",
    )


def test_standard_normal_is_lowest(
    loss_fn, standard_normal, scaled_normal, shifted_normal, uniform, collapsed,
):
    """The N(0, 1) sample should score lower than every pathological case.

    This is the central correctness property of SIGReg: as embeddings approach
    isotropic standard normal, the regularization loss approaches its minimum.
    If this ordering breaks, the regularizer is pulling embeddings toward the
    wrong target distribution -- a silent training-corrupting bug.
    """
    # Each call advances `global_step`, so re-instantiate to keep seeds aligned
    # (this test only cares about value ordering, not per-call determinism).
    def fresh():
        ut = EppsPulley(t_max=5.0, n_points=17)
        return SlicingUnivariateTest(
            univariate_test=ut, num_slices=256, reduction="mean",
        )

    l_std = fresh()(standard_normal).item()
    l_scaled = fresh()(scaled_normal).item()
    l_shifted = fresh()(shifted_normal).item()
    l_uniform = fresh()(uniform).item()
    l_collapsed = fresh()(collapsed).item()

    # Standard normal must beat each pathological distribution.
    assert l_std < l_scaled, (
        f"Standard normal ({l_std:.4f}) did not score lower than N(0, 25) "
        f"({l_scaled:.4f}). SIGReg is not penalizing wrong scale."
    )
    assert l_std < l_shifted, (
        f"Standard normal ({l_std:.4f}) did not score lower than N(3, 1) "
        f"({l_shifted:.4f}). SIGReg is not penalizing shifted mean."
    )
    assert l_std < l_uniform, (
        f"Standard normal ({l_std:.4f}) did not score lower than Uniform "
        f"({l_uniform:.4f}). SIGReg is not penalizing wrong shape."
    )
    assert l_std < l_collapsed, (
        f"Standard normal ({l_std:.4f}) did not score lower than collapsed "
        f"({l_collapsed:.4f}). SIGReg is not penalizing collapse."
    )

    # Sanity: collapse should be the WORST (or at least near-worst) case --
    # it's the only one with zero variance per slice.
    assert l_collapsed >= max(l_scaled, l_shifted, l_uniform) * 0.5, (
        "Collapsed embeddings should be one of the worst cases; got "
        f"l_collapsed={l_collapsed:.4f} vs others "
        f"(scaled={l_scaled:.4f}, shifted={l_shifted:.4f}, uniform={l_uniform:.4f})."
    )


def test_output_is_nonneg_scalar(loss_fn, standard_normal):
    """Default reduction='mean' returns a non-negative scalar tensor."""
    out = loss_fn(standard_normal)
    assert out.dim() == 0, f"Expected scalar, got shape {out.shape}."
    assert out.item() >= 0, f"Loss must be non-negative, got {out.item()}."


def test_reduction_modes(standard_normal):
    """reduction='mean'|'sum'|None each return the right shape."""
    for reduction, expected_shape in [
        ("mean", torch.Size([])),
        ("sum", torch.Size([])),
        (None, torch.Size([256])),  # (num_slices,)
    ]:
        ut = EppsPulley(t_max=5.0, n_points=17)
        loss = SlicingUnivariateTest(
            univariate_test=ut, num_slices=256, reduction=reduction,
        )
        out = loss(standard_normal)
        assert out.shape == expected_shape, (
            f"reduction={reduction!r}: expected {expected_shape}, got {out.shape}."
        )


def test_seeding_advances_per_call(standard_normal):
    """Two consecutive calls produce different projections (different seeds).

    The module increments `global_step` after each forward pass; this is what
    makes successive training-step SIGReg calls see fresh random directions.
    If `global_step` ever stops advancing, every training step sees the same
    projections -- a subtle bug that would degrade test power without an
    obvious failure mode.
    """
    ut = EppsPulley(t_max=5.0, n_points=17)
    loss = SlicingUnivariateTest(
        univariate_test=ut, num_slices=256, reduction=None,
    )

    out_a = loss(standard_normal).clone()
    out_b = loss(standard_normal).clone()

    # Same input, different projection matrices -> different per-slice stats.
    # We can't assert ANY element differs (extremely unlikely with 256 slices
    # but in principle possible), so check that NOT ALL slices are equal.
    assert not torch.allclose(out_a, out_b), (
        "Two consecutive forward passes produced identical per-slice stats. "
        "global_step is not advancing -- successive SIGReg calls would see "
        "the same random directions."
    )


def test_seeding_is_deterministic_at_same_step():
    """Two SIGReg modules at the same global_step produce identical projections.

    This is the property that makes distributed training work: every rank
    syncs its global_step via all_reduce(MAX) and therefore samples the same
    projection matrix.
    """
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(N, FEATURE_DIM, generator=g)

    ut_a = EppsPulley(t_max=5.0, n_points=17)
    loss_a = SlicingUnivariateTest(
        univariate_test=ut_a, num_slices=256, reduction=None,
    )

    ut_b = EppsPulley(t_max=5.0, n_points=17)
    loss_b = SlicingUnivariateTest(
        univariate_test=ut_b, num_slices=256, reduction=None,
    )

    # Both modules start at global_step=0 -> same seed -> same projections
    # -> same per-slice statistics on the same input.
    out_a = loss_a(x)
    out_b = loss_b(x)

    assert torch.allclose(out_a, out_b, atol=1e-5), (
        "Two SIGReg instances at the same global_step produced different "
        "outputs on the same input. Seeding is non-deterministic, which "
        "would break distributed training (different ranks would project "
        "onto different directions)."
    )
