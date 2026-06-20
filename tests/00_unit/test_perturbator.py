"""Unit tests for the perturbator (CPU, tiny dims, no rdkit required)."""

import math

import numpy as np
import torch

from eb_jepa.singlecell.perturbator.featurize import DrugFeaturizer
from eb_jepa.singlecell.perturbator.flow import (
    flow_matching_loss,
    ode_sample,
    predict_perturbed,
)
from eb_jepa.singlecell.perturbator.losses import sliced_wasserstein
from eb_jepa.singlecell.perturbator.matching import build_strata
from eb_jepa.singlecell.perturbator.model import Perturbator
from eb_jepa.singlecell.perturbator.visualize import (
    build_dose_track,
    monotonicity_score,
    rank_combos,
)


class TestFeaturize:
    def test_fixed_dim_deterministic(self):
        feat = DrugFeaturizer(n_bits=64, radius=2)
        # Deterministic + fixed-dim under either backend (rdkit or the hash fallback).
        a = feat.featurize("CCO", -7.0)
        b = feat.featurize("CCO", -7.0)
        assert a.shape == (feat.action_dim,)
        assert feat.action_dim == 64 + 6 + 2
        assert torch.allclose(a, b)  # deterministic
        # different SMILES -> different drug features
        c = feat.featurize("c1ccccc1", -7.0)
        assert not torch.allclose(a[: feat.drug_dim], c[: feat.drug_dim])

    def test_control_no_dose(self):
        feat = DrugFeaturizer(n_bits=32)
        # control: nan dose -> sentinel [0, 0] dose channels, same dim as treated
        ctrl = feat.featurize("DMSO", float("nan"))
        treated = feat.featurize("DMSO", -6.0)
        assert ctrl.shape == treated.shape == (feat.action_dim,)
        assert ctrl[-2].item() == 0.0 and ctrl[-1].item() == 0.0
        assert treated[-2].item() == 1.0 and treated[-1].item() == -6.0
        # None smiles -> zero drug features, still valid dim
        none_feat = feat.featurize(None, None)
        assert none_feat.shape == (feat.action_dim,)
        assert torch.count_nonzero(none_feat).item() == 0

    def test_batch(self):
        feat = DrugFeaturizer(n_bits=16)
        out = feat.featurize_batch(["CCO", "CCN", None], torch.tensor([-7.0, -6.0, float("nan")]))
        assert out.shape == (3, feat.action_dim)


class TestSlicedWasserstein:
    def test_identical_is_near_zero(self):
        torch.manual_seed(0)
        x = torch.randn(128, 6)
        g = torch.Generator().manual_seed(1)
        d = sliced_wasserstein(x, x.clone(), n_slices=256, generator=g)
        assert d.item() < 1e-5

    def test_shifted_is_positive(self):
        torch.manual_seed(0)
        x = torch.randn(128, 6)
        y = x + 3.0  # clear mean shift
        g = torch.Generator().manual_seed(1)
        d = sliced_wasserstein(x, y, n_slices=256, generator=g)
        assert d.item() > 1.0

    def test_symmetric_ish(self):
        torch.manual_seed(0)
        x = torch.randn(96, 5)
        y = torch.randn(96, 5) + 1.5
        g1 = torch.Generator().manual_seed(7)
        g2 = torch.Generator().manual_seed(7)
        d_xy = sliced_wasserstein(x, y, n_slices=256, generator=g1)
        d_yx = sliced_wasserstein(y, x, n_slices=256, generator=g2)
        assert abs(d_xy.item() - d_yx.item()) < 1e-4

    def test_unequal_cardinality(self):
        torch.manual_seed(0)
        x = torch.randn(50, 4)
        y = torch.randn(120, 4) + 2.0
        d = sliced_wasserstein(x, y, n_slices=128)
        assert d.item() > 0.5  # runs and is positive for shifted clouds


class TestMatching:
    def _meta(self):
        # 2 strata. stratum A: (L1, p1) has controls + drugX@dose1 + drugX@dose2.
        # stratum B: (L2, p1) has controls + drugY. stratum C: (L3, p2) NO controls.
        d = 4
        rows = []
        # stratum A controls (2)
        rows += [("L1", "p1", "DMSO_TF", None, float("nan"))] * 2
        # stratum A drugX dose1 (3)
        rows += [("L1", "p1", "drugX", "CCO", -7.0)] * 3
        # stratum A drugX dose2 (2)
        rows += [("L1", "p1", "drugX", "CCO", -6.0)] * 2
        # stratum B controls (2) + drugY (2)
        rows += [("L2", "p1", "DMSO_TF", None, float("nan"))] * 2
        rows += [("L2", "p1", "drugY", "CCN", -5.0)] * 2
        # stratum C: treated but no control -> skipped
        rows += [("L3", "p2", "drugZ", "CCC", -5.0)] * 2
        latents = torch.arange(len(rows) * d, dtype=torch.float32).reshape(len(rows), d)
        cell_line = [r[0] for r in rows]
        plate = [r[1] for r in rows]
        drug = [r[2] for r in rows]
        smiles = [r[3] for r in rows]
        log_conc = [r[4] for r in rows]
        return latents, cell_line, plate, drug, smiles, log_conc

    def test_grouping(self):
        latents, cl, pl, dr, sm, lc = self._meta()
        strata = build_strata(latents, cl, pl, dr, sm, lc)
        # stratum A -> 2 target groups (2 doses), stratum B -> 1; C skipped => 3
        assert len(strata) == 3
        keys = {(s.stratum, s.drug, round(s.log_conc, 3)) for s in strata}
        assert (("L1", "p1"), "drugX", -7.0) in keys
        assert (("L1", "p1"), "drugX", -6.0) in keys
        assert (("L2", "p1"), "drugY", -5.0) in keys
        # no stratum C
        assert all(s.stratum != ("L3", "p2") for s in strata)
        # sources are the controls of that stratum
        for s in strata:
            if s.stratum == ("L1", "p1"):
                assert s.source.shape[0] == 2
            if s.stratum == ("L2", "p1"):
                assert s.source.shape[0] == 2
        # target sizes
        sizes = {(s.stratum, s.drug, round(s.log_conc, 3)): s.target.shape[0] for s in strata}
        assert sizes[(("L1", "p1"), "drugX", -7.0)] == 3
        assert sizes[(("L1", "p1"), "drugX", -6.0)] == 2
        assert sizes[(("L2", "p1"), "drugY", -5.0)] == 2

    def test_skip_no_controls(self):
        # only treated cells, no controls anywhere -> no strata
        latents = torch.randn(4, 4)
        strata = build_strata(
            latents,
            ["L1"] * 4,
            ["p1"] * 4,
            ["drugX"] * 4,
            ["CCO"] * 4,
            [-7.0] * 4,
        )
        assert strata == []


class TestModel:
    def test_forward_shape(self):
        feat = DrugFeaturizer(n_bits=16)
        model = Perturbator(d_model=12, action_dim=feat.action_dim, depth=2, d_cond=16)
        src = torch.randn(7, 12)
        action = feat.featurize("CCO", -7.0)
        out = model(src, action)
        assert out.shape == (7, 12)

    def test_film_conditions(self):
        feat = DrugFeaturizer(n_bits=16)
        model = Perturbator(d_model=12, action_dim=feat.action_dim, depth=2, d_cond=16)
        # FiLM is zero-init (identity), so move it off zero — mimicking a trained
        # model — so the action actually modulates the hidden state and the head.
        with torch.no_grad():
            for p in model.head.parameters():
                p.add_(0.1 * torch.randn_like(p))
            for block in model.blocks:
                block.film.weight.add_(0.3 * torch.randn_like(block.film.weight))
        src = torch.randn(5, 12)
        a1 = feat.featurize("CCO", -7.0)
        a2 = feat.featurize("c1ccccc1", -4.0)
        o1 = model(src, a1)
        o2 = model(src, a2)
        assert not torch.allclose(o1, o2, atol=1e-4)  # action changes output

    def test_optimization_reduces_loss(self):
        torch.manual_seed(0)
        feat = DrugFeaturizer(n_bits=16)
        model = Perturbator(d_model=8, action_dim=feat.action_dim, depth=2, d_cond=16)
        src = torch.randn(64, 8)
        target = torch.randn(64, 8) + 2.5  # shifted target distribution
        action = feat.featurize("CCO", -6.0)
        opt = torch.optim.SGD(model.parameters(), lr=0.05)

        def loss_now():
            g = torch.Generator().manual_seed(0)
            return sliced_wasserstein(model(src, action), target, n_slices=128, generator=g)

        start = loss_now().item()
        for _ in range(40):
            opt.zero_grad()
            g = torch.Generator().manual_seed(123)
            loss = sliced_wasserstein(model(src, action), target, n_slices=128, generator=g)
            loss.backward()
            opt.step()
        end = loss_now().item()
        assert end < start - 0.1  # a few SGD steps clearly reduce the OT loss


class TestFlowMatching:
    def _model(self, d=8, action_dim=24):
        return Perturbator(
            d_model=d, action_dim=action_dim, depth=2, d_cond=16,
            time_conditioned=True,
        )

    def test_forward_backward_smoke(self):
        torch.manual_seed(0)
        feat = DrugFeaturizer(n_bits=16)
        model = self._model(d=8, action_dim=feat.action_dim)
        source = torch.randn(32, 8)
        target = torch.randn(40, 8) + 2.0  # unequal cardinality + shift
        action = feat.featurize("CCO", -6.0)
        gen = torch.Generator().manual_seed(1)
        loss = flow_matching_loss(model, source, target, action, generator=gen)
        assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() > 0
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, "no gradients flowed"
        assert any(g.abs().sum() > 0 for g in grads), "all gradients are zero"

    def test_velocity_requires_time_conditioning(self):
        feat = DrugFeaturizer(n_bits=16)
        direct = Perturbator(d_model=8, action_dim=feat.action_dim, depth=2, d_cond=16)
        x = torch.randn(4, 8)
        t = torch.zeros(4)
        action = feat.featurize("CCO", -6.0)
        try:
            direct.velocity(x, t, action)
            assert False, "velocity() must require time_conditioned=True"
        except RuntimeError:
            pass

    def test_optimization_reduces_flow_loss(self):
        torch.manual_seed(0)
        feat = DrugFeaturizer(n_bits=16)
        model = self._model(d=8, action_dim=feat.action_dim)
        source = torch.randn(64, 8)
        target = torch.randn(64, 8) + 2.5
        action = feat.featurize("CCO", -6.0)
        opt = torch.optim.Adam(model.parameters(), lr=0.02)

        def loss_now():
            g = torch.Generator().manual_seed(0)
            return flow_matching_loss(model, source, target, action, generator=g).item()

        start = loss_now()
        for _ in range(60):
            opt.zero_grad()
            g = torch.Generator().manual_seed(123)
            loss = flow_matching_loss(model, source, target, action, generator=g)
            loss.backward()
            opt.step()
        assert loss_now() < start - 0.05


class TestODESampler:
    def test_identity_at_init(self):
        # zero-init head -> velocity == 0 -> ODE is the identity map.
        feat = DrugFeaturizer(n_bits=16)
        model = Perturbator(
            d_model=8, action_dim=feat.action_dim, depth=2, d_cond=16,
            time_conditioned=True,
        )
        source = torch.randn(10, 8)
        action = feat.featurize("CCO", -6.0)
        for method in ("euler", "heun", "midpoint"):
            out = ode_sample(model, source, action, n_steps=8, method=method)
            assert out.shape == source.shape
            assert torch.allclose(out, source, atol=1e-5), f"{method} not identity at init"

    def test_constant_velocity_integration(self):
        # Force a constant velocity field (bias-only head, gamma/beta still 0): the
        # trunk output is the head bias for every (x, t), so the ODE over [0,1] must
        # move every point by exactly that bias regardless of step count / method.
        feat = DrugFeaturizer(n_bits=16)
        model = Perturbator(
            d_model=4, action_dim=feat.action_dim, depth=1, d_cond=8,
            time_conditioned=True,
        )
        with torch.no_grad():
            bias = torch.tensor([1.0, -2.0, 0.5, 3.0])
            model.head.bias.copy_(bias)  # weight stays 0 -> output == bias
        source = torch.zeros(5, 4)
        action = feat.featurize("CCO", -6.0)
        for method, steps in (("euler", 4), ("heun", 2), ("midpoint", 3)):
            out = ode_sample(model, source, action, n_steps=steps, method=method)
            assert torch.allclose(out, source + bias, atol=1e-4), method

    def test_predict_perturbed_dispatch(self):
        feat = DrugFeaturizer(n_bits=16)
        flow = Perturbator(
            d_model=6, action_dim=feat.action_dim, depth=1, d_cond=8,
            time_conditioned=True,
        )
        direct = Perturbator(d_model=6, action_dim=feat.action_dim, depth=1, d_cond=8)
        source = torch.randn(7, 6)
        action = feat.featurize("CCO", -6.0)
        a = predict_perturbed(flow, source, action, "flow_matching", n_steps=5)
        b = predict_perturbed(direct, source, action, "direct")
        assert a.shape == b.shape == (7, 6)


class TestMonotonicity:
    def test_perfect_monotone_track(self):
        # straight, increasing-magnitude track -> score ~ 1.
        c0 = np.zeros(3)
        direction = np.array([1.0, 0.0, 0.0])
        cents = np.stack([direction * m for m in (1.0, 2.0, 3.0, 4.0)])
        m = monotonicity_score(c0, cents)
        assert m["collinearity"] > 0.99
        assert m["magnitude_monotonicity"] == 1.0
        assert m["score"] > 0.99
        assert np.allclose(m["displacements"], [1.0, 2.0, 3.0, 4.0])

    def test_non_monotone_track_scores_lower(self):
        c0 = np.zeros(2)
        good = np.stack([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        # zig-zag, non-increasing magnitude
        bad = np.stack([[1.0, 0.0], [0.5, 1.0], [0.2, -0.5]])
        sg = monotonicity_score(c0, good)["score"]
        sb = monotonicity_score(c0, bad)["score"]
        assert sg > sb

    def test_single_dose_is_zero(self):
        m = monotonicity_score(np.zeros(3), np.ones((1, 3)))
        assert m["score"] == 0.0

    def test_build_dose_track_and_rank(self):
        # end-to-end on synthetic latents: a trained-ish flow model produces a
        # ranked set of tracks with the expected dataclass shape.
        torch.manual_seed(0)
        feat = DrugFeaturizer(n_bits=16)
        model = Perturbator(
            d_model=6, action_dim=feat.action_dim, depth=2, d_cond=16,
            time_conditioned=True,
        )
        control = torch.randn(40, 6)
        t1 = build_dose_track(
            model, feat, control, "drugA", "CCO", [-7.0, -6.0, -5.0],
            objective="flow_matching", ode_steps=5, ode_method="euler",
        )
        t2 = build_dose_track(
            model, feat, control, "drugB", "CCN", [-7.0, -6.0],
            objective="flow_matching", ode_steps=5, ode_method="euler",
        )
        assert t1.dose_centroids.shape == (3, 6)
        assert t1.control_centroid.shape == (6,)
        ranked = rank_combos([t1, t2])
        assert len(ranked) == 2
        assert ranked[0].metrics["score"] >= ranked[1].metrics["score"]
