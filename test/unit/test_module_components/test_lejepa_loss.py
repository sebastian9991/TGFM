from typing import Tuple

import pytest
import torch
from torch import Tensor

from tgfm.models.leJepa_loss import LeJEPALoss, LeJEPALossOutput

SEED = 42
B = 64
DIM = 128
V_GLOBAL = 8
V_LOCAL = 2
@pytest.fixture(scope="session")
def view_embeddings() -> Tuple[Tensor, Tensor]:
    """(z_global, z_local) with standard-normal entries.
    """
    g = torch.Generator().manual_seed(SEED)
    z_global = torch.randn(B, V_GLOBAL, DIM, generator=g)
    z_local = torch.randn(B, V_LOCAL, DIM, generator=g)
    return z_global, z_local


def test_output_shape_contract(view_embeddings):
    """LeJEPALossOutput fields are scalar tensors with the expected attachment.
    """
    z_global, z_local = view_embeddings
    loss = LeJEPALoss()
    out = loss(z_global, z_local)

    assert isinstance(out, LeJEPALossOutput)
    assert out.total.dim() == 0
    assert out.pred.dim() == 0
    assert out.sigreg.dim() == 0

    # pred and sigreg are detached -> requires_grad=False; total is not detached.
    assert not out.pred.requires_grad
    assert not out.sigreg.requires_grad
    # `total` doesn't require_grad here because the inputs don't either; the
    # important thing is that pred/sigreg explicitly do not.


def test_output_nonneg(view_embeddings):
    """pred is squared-L2 (>=0); sigreg is Epps-Pulley (>=0). total = convex combo."""
    z_global, z_local = view_embeddings
    out = LeJEPALoss()(z_global, z_local)
    assert out.pred.item() >= 0
    assert out.sigreg.item() >= 0
    assert out.total.item() >= 0


def test_gradient_flows_through_global_views(view_embeddings):
    """Gradients must reach z_global, including via the centroid mu.
    """
    z_global, z_local = view_embeddings
    z_global = z_global.clone().requires_grad_(True)
    z_local = z_local.clone().requires_grad_(True)

    out = LeJEPALoss()(z_global, z_local)
    out.total.backward()

    assert z_global.grad is not None
    assert z_local.grad is not None
    assert (z_global.grad.abs().sum() > 0).item(), (
        "No gradient flowed to z_global -- check that mu is NOT detached and "
        "that the SIGReg pathway from z_global is intact."
    )
    assert (z_local.grad.abs().sum() > 0).item(), (
        "No gradient flowed to z_local -- the prediction loss should pull "
        "locals toward mu."
    )


def test_lambda_zero_equals_pred(view_embeddings):
    """lambd=0 -> total = pred (sigreg term zeroed out)."""
    z_global, z_local = view_embeddings
    out = LeJEPALoss(lambd=0.0)(z_global, z_local)
    assert torch.allclose(out.total, out.pred, atol=1e-6), (
        f"At lambd=0, total ({out.total.item():.4f}) should equal pred "
        f"({out.pred.item():.4f})."
    )


def test_lambda_one_equals_sigreg(view_embeddings):
    """lambd=1 -> total = sigreg (pred term zeroed out)."""
    z_global, z_local = view_embeddings
    out = LeJEPALoss(lambd=1.0)(z_global, z_local)
    assert torch.allclose(out.total, out.sigreg, atol=1e-6), (
        f"At lambd=1, total ({out.total.item():.4f}) should equal sigreg "
        f"({out.sigreg.item():.4f})."
    )


def test_lambda_convex_combination(view_embeddings):
    """At arbitrary lambd, total = (1-lambd)*pred + lambd*sigreg.
    """
    z_global, z_local = view_embeddings
    lambd = 0.3

    out = LeJEPALoss(lambd=lambd)(z_global, z_local)
    expected = (1 - lambd) * out.pred + lambd * out.sigreg
    assert torch.allclose(out.total, expected, atol=1e-6), (
        f"Convex combination broken at lambd={lambd}: "
        f"total={out.total.item():.4f}, expected={expected.item():.4f}."
    )

def test_standard_normal_yields_low_sigreg():
    """N(0,1) embeddings should produce low sigreg vs scaled normal embeddings.

    Cross-cuts the LeJEPALoss -> SlicingUnivariateTest -> EppsPulley path
    end-to-end. Catches integration bugs (e.g. shape mangling before SIGReg
    is called, accidentally pre-standardizing) that the component-level
    sigreg tests don't see.
    """
    g = torch.Generator().manual_seed(SEED)
    z_global_std = torch.randn(B, V_GLOBAL, DIM, generator=g)
    z_local_std = torch.randn(B, V_LOCAL, DIM, generator=g)

    # Same shape, but scaled to N(0, 25), still isotropic Gaussian, wrong scale.
    z_global_bad = z_global_std * 5.0
    z_local_bad = z_local_std * 5.0

    # Fresh module each call so global_step is aligned (sigreg uses same projections).
    out_std = LeJEPALoss()(z_global_std, z_local_std)
    out_bad = LeJEPALoss()(z_global_bad, z_local_bad)

    assert out_std.sigreg < out_bad.sigreg, (
        f"Standard-normal embeddings sigreg ({out_std.sigreg.item():.4f}) was "
        f"not lower than N(0,25)-scaled sigreg ({out_bad.sigreg.item():.4f}). "
        "End-to-end SIGReg integration may be broken."
    )


def test_argument_order_matters(view_embeddings):
    """Swapping (z_global, z_local) should change the loss.

    `LeJEPALoss.forward(z_global, z_local)` is asymmetric: the default
    centroid is computed from z_global only, so swapping the arguments
    computes mu from what should have been the locals.
    """
    z_global, z_local = view_embeddings
    loss = LeJEPALoss()

    out_correct = loss(z_global, z_local)
    # Build a fresh loss so global_step doesn't drift between the two calls.
    loss2 = LeJEPALoss()
    out_swapped = loss2(z_local, z_global)

    assert not torch.allclose(out_correct.total, out_swapped.total, atol=1e-4), (
        "Swapping (z_global, z_local) gave identical losses. The loss is not "
        "honoring the asymmetry between global and local view sets -- check "
        "centroid construction and view-count handling."
    )
