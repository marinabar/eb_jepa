"""Unit tests for the perturbator (CPU, tiny dims, no rdkit required)."""

import math

import torch

from eb_jepa.singlecell.perturbator.featurize import DrugFeaturizer
from eb_jepa.singlecell.perturbator.losses import sliced_wasserstein
from eb_jepa.singlecell.perturbator.matching import build_strata
from eb_jepa.singlecell.perturbator.model import Perturbator


class TestFeaturize:
    def test_fixed_dim_deterministic(self):
        feat = DrugFeaturizer(n_bits=64, radius=2)
        assert not feat.has_rdkit, "test environment must run the rdkit-free fallback"
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
