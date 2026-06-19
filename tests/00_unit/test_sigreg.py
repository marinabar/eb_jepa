"""SIGReg (LeJEPA) numerics: the Gaussianity statistic should be small for an
isotropic Gaussian and large for non-Gaussian / non-unit-variance / spherical
distributions. See CLAUDE.md "Views and the LeJEPA objective".
"""

import math

import pytest
import torch
import torch.nn.functional as F

from eb_jepa.losses import SIGReg, _sigreg_all_reduce_mean


def _sigreg(x, seed=0, **kwargs):
    torch.manual_seed(seed)
    return SIGReg(**kwargs)(x).item()


class TestSIGRegNumerics:
    @pytest.fixture(autouse=True)
    def _seed(self):
        torch.manual_seed(123)

    def test_gaussian_much_smaller_than_non_gaussian(self):
        # Under H0 (true Gaussian) the statistic stays ~O(1) regardless of N; under
        # H1 it grows ~proportionally to N. Use a large N (slices halved to keep the
        # [N, S, knots] intermediate the same size) so even the mildly-non-Gaussian
        # uniform separates cleanly.
        n, d, s = 16384, 32, 128
        gauss = torch.randn(n, d)
        # unit-variance uniform (var of U(-a,a) is a^2/3 -> a = sqrt(3)); light tails
        uniform = (torch.rand(n, d) * 2 - 1) * math.sqrt(3.0)
        # unit-variance bimodal mixture at +/-1; strongly non-Gaussian
        bimodal = torch.randint(0, 2, (n, d)).float() * 2 - 1

        s_gauss = _sigreg(gauss, num_slices=s)
        s_uniform = _sigreg(uniform, num_slices=s)
        s_bimodal = _sigreg(bimodal, num_slices=s)

        assert s_gauss < s_uniform < s_bimodal
        assert s_uniform > 2 * s_gauss  # mild non-Gaussian, still clearly flagged
        assert s_bimodal > 3 * s_gauss  # strong non-Gaussian

    def test_non_unit_variance_is_penalized(self):
        n, d = 8192, 32
        gauss = torch.randn(n, d)
        scaled = gauss * 5.0  # variance 25, still Gaussian shape but wrong scale
        assert _sigreg(scaled) > 3 * _sigreg(gauss)

    def test_l2_normalized_sphere_is_penalized(self):
        # L2-normalizing onto the unit sphere is NOT isotropic-Gaussian -> SIGReg
        # should flag it. This is the empirical justification for the projector
        # output NOT being L2-normalized in LeJEPA.
        n, d = 8192, 32
        gauss = torch.randn(n, d)
        sphere = F.normalize(gauss, p=2, dim=-1)
        assert _sigreg(sphere) > _sigreg(gauss)

    def test_non_negative_and_finite(self):
        x = torch.randn(1024, 16)
        s = _sigreg(x)
        assert s >= 0.0
        assert math.isfinite(s)

    def test_deterministic_given_seed(self):
        x = torch.randn(1024, 16)
        assert _sigreg(x, seed=7) == _sigreg(x, seed=7)

    def test_gradients_flow(self):
        x = torch.randn(512, 16, requires_grad=True)
        torch.manual_seed(0)
        SIGReg(num_slices=64)(x).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_multiview_shape_accepted(self):
        # SIGReg must accept a [V, N, d_proj] tensor (sample dim at -2).
        views = torch.randn(4, 256, 16)
        torch.manual_seed(0)
        out = SIGReg(num_slices=32)(views)
        assert out.ndim == 0 and math.isfinite(out.item())

    def test_quadrature_weights_match_minimal(self):
        # Structural check vs MINIMAL.md: trapezoid weights * Gaussian window.
        sr = SIGReg(knots=17, t_max=3.0)
        dt = 3.0 / 16
        expected = torch.full((17,), 2 * dt)
        expected[0] = dt
        expected[-1] = dt
        expected = expected * torch.exp(-sr.t.square() / 2.0)
        assert torch.allclose(sr.weights, expected, atol=1e-6)


class TestSIGRegDistributedAndDtype:
    """Lock-step RNG (DDP correctness), the all-reduce helper, and bf16 stability."""

    def test_lock_step_projections_across_ranks(self):
        # Two SIGReg instances at the same (seed, step) must produce the IDENTICAL
        # statistic on the same input -> guarantees every DDP rank projects with
        # the same matrix A, which is required for the ECF all-reduce to be valid.
        x = torch.randn(1024, 16)
        rank0, rank1 = SIGReg(num_slices=64), SIGReg(num_slices=64)
        assert rank0(x).item() == rank1(x).item()

    def test_step_advances_changes_projection(self):
        x = torch.randn(1024, 16)
        sr = SIGReg(num_slices=64)
        first = sr(x).item()  # step 0
        second = sr(x).item()  # step 1 -> different projection matrix
        assert first != second

    def test_all_reduce_mean_is_noop_single_process(self):
        x = torch.randn(3, 4, requires_grad=True)
        assert _sigreg_all_reduce_mean(x) is x

    def test_bf16_finite_and_close_to_fp32(self):
        x32 = torch.randn(4096, 16)
        x16 = x32.to(torch.bfloat16)
        s32 = SIGReg(num_slices=64)(x32).item()
        s16 = SIGReg(num_slices=64)(x16).item()  # same seed+step -> same A
        assert math.isfinite(s16) and s16 > 0
        assert s16 == pytest.approx(s32, rel=0.25)
