"""Test suite for tgfm.utils.diagnostics.collapse_diagnostics.

For isotropic Gaussian Embeddings we should see the following:

1. trac_cov decrease. The trace of the covariance matrix should decrease.

2. var_mean decrease. The per dimension veriance should decrease.

3. rank increase. The rank of the matrix should increase.

4. cos_std increase. The cosine similarity spread should increase.


We assert three tiers of strength:

  (1) DETERMINISTIC facts that hold for any continuous distribution with
      N > d (hard rank == d) or for the degenerate input (collapsed).
      These get exact asserts.

  (2) STATISTICAL quantities (trace, per-dim var, cos stats) that
      concentrate but fluctuate with the seed. These get value asserts
      with generous relative margins, derived analytically.

  (3) CROSS-FIXTURE ORDERINGS that encode the *intent* of each diagnostic
      (scaling moves variance but not rank; shift is erased by centering;
      dimensional collapse drops rank/eff_rank; representational collapse
      drops cos_std). These are the most robust regression guards.
"""

import math

import pytest
import torch
from torch import Tensor

from tgfm.utils.diagnostics import CollapseStats, collapse_diagnostics

SEED = 42
N = 256          # N > FEATURE_DIM so the sample covariance is full rank
FEATURE_DIM = 128
GAMMA = FEATURE_DIM / N          # Marchenko-Pastur aspect ratio (0.5 here)
LOW_RANK_K = 8                   # intrinsic dim of the dimensional-collapse fixture

# Expected near-orthogonal cosine spread for d iid zero-mean coords: ~1/sqrt(d).
COS_STD_EXPECTED = 1.0 / math.sqrt(FEATURE_DIM)   # ~0.0884


@pytest.fixture(scope="session")
def standard_normal() -> Tensor:
    """Reference case: N(0, 1) elementwise. var ~ 1, full rank."""
    g = torch.Generator().manual_seed(SEED)
    return torch.randn(N, FEATURE_DIM, generator=g)


@pytest.fixture(scope="session")
def scaled_normal() -> Tensor:
    """N(0, 25): isotropic but wrong scale. var ~ 25; rank/cos unchanged."""
    g = torch.Generator().manual_seed(SEED + 1)
    return torch.randn(N, FEATURE_DIM, generator=g) * 5.0


@pytest.fixture(scope="session")
def shifted_normal() -> Tensor:
    """N(3, 1): shifted mean. Centering erases the shift -> behaves like standard."""
    g = torch.Generator().manual_seed(SEED + 2)
    return torch.randn(N, FEATURE_DIM, generator=g) + 3.0


@pytest.fixture(scope="session")
def uniform() -> Tensor:
    """Uniform on [-sqrt(3), sqrt(3)]: variance ~1 but no Gaussian tails.

    NOTE: the original fixture returned torch.rand() = U[0,1) (var 1/12),
    which contradicts its own docstring. Fixed here to actually have var ~1.
    """
    g = torch.Generator().manual_seed(SEED + 3)
    u = torch.rand(N, FEATURE_DIM, generator=g)
    return (u - 0.5) * 2.0 * math.sqrt(3.0)


@pytest.fixture(scope="session")
def low_rank() -> Tensor:
    """Dimensional collapse: data living on a LOW_RANK_K-dim subspace of R^d.

    rank should be exactly LOW_RANK_K and eff_rank should be far below d,
    even though no two rows are identical (so this is distinct from full
    representational collapse).
    """
    g = torch.Generator().manual_seed(SEED + 4)
    latent = torch.randn(N, LOW_RANK_K, generator=g)
    basis = torch.randn(LOW_RANK_K, FEATURE_DIM, generator=g)
    return latent @ basis


@pytest.fixture(scope="session")
def collapsed() -> Tensor:
    """Representational collapse worst case: every row identical."""
    return torch.ones(N, FEATURE_DIM)


def test_returns_collapse_stats(standard_normal):
    c = collapse_diagnostics(z=standard_normal)
    assert isinstance(c, CollapseStats)
    # All fields are plain Python floats (no leaked tensors).
    for field in (
        c.rank, c.eff_rank, c.trace_cov,
        c.var_min, c.var_mean, c.var_max,
        c.cos_mean, c.cos_std,
    ):
        assert isinstance(field, float)


@pytest.mark.parametrize(
    "fixture_name",
    ["standard_normal", "scaled_normal", "shifted_normal", "uniform"],
)
def test_full_rank_when_N_exceeds_d(fixture_name, request):
    """N > d + continuous distribution => hard rank is exactly d.

    Centering removes one sample DoF (N -> N-1 effective), but N-1=255 >= d=128,
    so the d x d covariance stays full rank.
    """
    z = request.getfixturevalue(fixture_name)
    c = collapse_diagnostics(z=z)
    assert c.rank == FEATURE_DIM


def test_low_rank_detected_exactly(low_rank):
    c = collapse_diagnostics(z=low_rank)
    assert c.rank == LOW_RANK_K
    # eff_rank for a balanced k-dim subspace sits near k, and well under d.
    assert c.eff_rank < FEATURE_DIM / 2
    assert c.eff_rank <= LOW_RANK_K + 1e-6


def test_collapsed_case(collapsed):
    c = collapse_diagnostics(z=collapsed)
    # Centering -> all-zero rows.
    assert c.rank == 0
    # Empty eigenvalue support => entropy 0 => exp(0) = 1.0 (NOT 0).
    assert c.eff_rank == pytest.approx(1.0)
    assert c.trace_cov == pytest.approx(0.0, abs=1e-6)
    assert c.var_min == pytest.approx(0.0, abs=1e-6)
    assert c.var_mean == pytest.approx(0.0, abs=1e-6)
    assert c.var_max == pytest.approx(0.0, abs=1e-6)
    # Cosine is computed on centered (=> zero) vectors: no signal, not 1.0.
    assert c.cos_mean == pytest.approx(0.0, abs=1e-6)
    assert c.cos_std == pytest.approx(0.0, abs=1e-6)


def test_standard_normal_values(standard_normal):
    c = collapse_diagnostics(z=standard_normal)
    # trace ~ d * sigma^2 = 128.
    assert c.trace_cov == pytest.approx(FEATURE_DIM, rel=0.10)
    # mean per-dim var ~ 1.
    assert c.var_mean == pytest.approx(1.0, rel=0.10)

    assert 0.6 * FEATURE_DIM < c.eff_rank < FEATURE_DIM
    # Near-orthogonality in high-d: cos ~ N(0, 1/d).
    assert abs(c.cos_mean) < 0.05
    assert c.cos_std == pytest.approx(COS_STD_EXPECTED, rel=0.30)
    # var ordering sanity.
    assert c.var_min < c.var_mean < c.var_max


def test_scaled_normal_values(scaled_normal):
    c = collapse_diagnostics(z=scaled_normal)
    # Variance scales by 25; trace ~ 25 * d.
    assert c.trace_cov == pytest.approx(25.0 * FEATURE_DIM, rel=0.10)
    assert c.var_mean == pytest.approx(25.0, rel=0.10)
    # Cosine is scale-invariant.
    assert c.cos_std == pytest.approx(COS_STD_EXPECTED, rel=0.30)


def test_shifted_normal_matches_standard(shifted_normal):
    c = collapse_diagnostics(z=shifted_normal)
    # Centering removes the mean shift entirely -> same stats as standard.
    assert c.trace_cov == pytest.approx(FEATURE_DIM, rel=0.10)
    assert c.var_mean == pytest.approx(1.0, rel=0.10)
    assert abs(c.cos_mean) < 0.05


def test_uniform_values(uniform):
    c = collapse_diagnostics(z=uniform)
    assert c.trace_cov == pytest.approx(FEATURE_DIM, rel=0.10)
    assert c.var_mean == pytest.approx(1.0, rel=0.10)
    # ...and high-d near-orthogonality is distribution-agnostic (CLT).
    assert c.cos_std == pytest.approx(COS_STD_EXPECTED, rel=0.30)


def test_scaling_moves_variance_not_rank(standard_normal, scaled_normal):
    a = collapse_diagnostics(z=standard_normal)
    b = collapse_diagnostics(z=scaled_normal)
    assert b.trace_cov > a.trace_cov          # variance diagnostics see scale
    assert b.var_mean > a.var_mean
    assert a.rank == b.rank                    # rank does not
    assert b.cos_std == pytest.approx(a.cos_std, rel=0.20)  # cosine does not


def test_shift_is_invisible_after_centering(standard_normal, shifted_normal):
    a = collapse_diagnostics(z=standard_normal)
    b = collapse_diagnostics(z=shifted_normal)
    assert b.trace_cov == pytest.approx(a.trace_cov, rel=0.10)
    assert b.eff_rank == pytest.approx(a.eff_rank, rel=0.10)


def test_dimensional_collapse_drops_eff_rank(standard_normal, low_rank):
    healthy = collapse_diagnostics(z=standard_normal)
    collapsed = collapse_diagnostics(z=low_rank)
    assert collapsed.rank < healthy.rank
    assert collapsed.eff_rank < healthy.eff_rank


def test_representational_collapse_drops_cos_std(standard_normal, collapsed):
    healthy = collapse_diagnostics(z=standard_normal)
    dead = collapse_diagnostics(z=collapsed)
    # The collapse signature is vanishing cosine spread + zero rank,
    # NOT a cos_mean near 1 (centering forbids that).
    assert dead.cos_std < healthy.cos_std
    assert dead.rank < healthy.rank


def test_trace_var_consistency(standard_normal):
    """trace_cov uses unbiased (N-1) cov; var_mean uses population (N).

    => var_mean ~ (trace_cov / d) * (N-1)/N. Checks they were not wired
    to different tensors.
    """
    c = collapse_diagnostics(z=standard_normal)
    expected_var_mean = (c.trace_cov / FEATURE_DIM) * (N - 1) / N
    assert c.var_mean == pytest.approx(expected_var_mean, rel=0.05)
