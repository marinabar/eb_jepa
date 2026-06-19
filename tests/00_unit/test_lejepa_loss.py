"""LeJEPA objective: centroid invariance over V views + lambda * SIGReg, with the
exact convex-combination convention from MINIMAL.md.
"""

import math

import pytest
import torch

from eb_jepa.architectures import Projector
from eb_jepa.losses import LeJEPALoss


def make_loss(d_model=16, d_proj=8, lamb=0.02, **kw):
    proj = Projector(f"{d_model}-32-{d_proj}")
    return LeJEPALoss(projector=proj, lamb=lamb, num_slices=64, **kw)


class TestLeJEPALoss:
    @pytest.fixture(autouse=True)
    def _seed(self):
        torch.manual_seed(0)

    def test_output_structure(self):
        loss_fn = make_loss()
        out = loss_fn(torch.randn(4, 64, 16))
        for k in ("loss", "invariance_loss", "sigreg_loss"):
            assert k in out
            assert out[k].ndim == 0 and math.isfinite(out[k].item())

    def test_convex_combination_exact(self):
        # The returned components must combine with the stated convention.
        lamb = 0.3
        loss_fn = make_loss(lamb=lamb)
        out = loss_fn(torch.randn(4, 128, 16))
        expected = lamb * out["sigreg_loss"] + (1 - lamb) * out["invariance_loss"]
        assert torch.allclose(out["loss"], expected, atol=1e-6)

    def test_identical_views_zero_invariance(self):
        loss_fn = make_loss()
        one = torch.randn(1, 64, 16)
        views = one.repeat(4, 1, 1)  # all V views identical
        out = loss_fn(views)
        assert out["invariance_loss"].item() == pytest.approx(0.0, abs=1e-6)

    def test_lambda_extremes(self):
        views = torch.randn(4, 128, 16)
        out0 = make_loss(lamb=0.0)(views)
        assert torch.allclose(out0["loss"], out0["invariance_loss"], atol=1e-6)
        out1 = make_loss(lamb=1.0)(views)
        assert torch.allclose(out1["loss"], out1["sigreg_loss"], atol=1e-6)

    def test_gradients_reach_projector_and_inputs(self):
        loss_fn = make_loss()
        views = torch.randn(4, 64, 16, requires_grad=True)
        loss_fn(views)["loss"].backward()
        assert views.grad is not None and torch.isfinite(views.grad).all()
        grads = [p.grad for p in loss_fn.projector.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)

    def test_projector_output_not_l2_normalized_by_default(self):
        # LeJEPA must feed un-normalized projections to SIGReg.
        assert Projector("16-32-8").l2_norm is False
