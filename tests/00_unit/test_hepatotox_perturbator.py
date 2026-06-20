"""Unit tests for the hepatotox-featurized perturbator + validation helpers.

CPU, tiny dims, no rdkit required (the featurizers fall back to a deterministic hash).
"""

import numpy as np
import torch

from eb_jepa.singlecell.perturbator.featurize import DrugFeaturizer
from eb_jepa.singlecell.perturbator.flow import flow_matching_loss, predict_perturbed
from eb_jepa.singlecell.perturbator.hepatotox_features import (
    HepatotoxActionFeaturizer,
    HepatotoxPathwayFeaturizer,
)
from eb_jepa.singlecell.perturbator.model import Perturbator
from examples.tahoe_hepatotox.validate_perturbator import (
    balanced_accuracy_at_threshold,
    centroid_cosine,
    dose_response_slope,
    gap_closed,
    moa_hierarchy_cosine,
    roc_auc,
    spearman,
)


class TestHepatotoxActionFeaturizer:
    def test_action_dim_and_shape(self):
        path = HepatotoxPathwayFeaturizer()
        feat = HepatotoxActionFeaturizer(path)
        assert feat.action_dim == path.feature_dim + 2
        a = feat.featurize("CCO", -7.0)
        assert a.shape == (feat.action_dim,)
        # pathway block then [validity, log_conc]
        assert a[-2].item() == 1.0
        assert a[-1].item() == -7.0

    def test_control_sentinel_dose(self):
        feat = HepatotoxActionFeaturizer()
        ctrl = feat.featurize("DMSO", float("nan"))
        treated = feat.featurize("DMSO", -6.0)
        assert ctrl.shape == treated.shape == (feat.action_dim,)
        assert ctrl[-2].item() == 0.0 and ctrl[-1].item() == 0.0
        # None smiles -> zero pathway block, valid dim
        none = feat.featurize(None, None)
        assert none.shape == (feat.action_dim,)
        assert torch.count_nonzero(none).item() == 0

    def test_determinism_and_distinct_drugs(self):
        feat = HepatotoxActionFeaturizer()
        a = feat.featurize("CCO", -7.0)
        b = feat.featurize("CCO", -7.0)
        assert torch.allclose(a, b)
        c = feat.featurize("c1ccccc1", -7.0)
        assert not torch.allclose(a[: feat.drug_dim], c[: feat.drug_dim])

    def test_batch_matches_single(self):
        feat = HepatotoxActionFeaturizer()
        out = feat.featurize_batch(["CCO", "CCN", None], torch.tensor([-7.0, -6.0, float("nan")]))
        assert out.shape == (3, feat.action_dim)
        assert torch.allclose(out[0], feat.featurize("CCO", -7.0))

    def test_interface_parity_with_drug_featurizer(self):
        # both expose action_dim / featurize / featurize_batch identically
        for f in (DrugFeaturizer(n_bits=16), HepatotoxActionFeaturizer()):
            assert isinstance(f.action_dim, int)
            assert f.featurize("CCO", -6.0).shape == (f.action_dim,)
            assert f.featurize_batch(["CCO", "CCN"], [-6.0, -5.0]).shape == (2, f.action_dim)

    def test_set_predictor_passthrough(self):
        feat = HepatotoxActionFeaturizer()
        name = feat.feature_names[0]
        feat.set_predictor(name, lambda s: 0.5)
        assert feat.featurize("CCO", -6.0)[0].item() == 0.5


class TestFlowWithHepatotoxAction:
    def test_forward_backward(self):
        torch.manual_seed(0)
        feat = HepatotoxActionFeaturizer()
        model = Perturbator(
            d_model=8, action_dim=feat.action_dim, depth=2, d_cond=16,
            time_conditioned=True,
        )
        source = torch.randn(20, 8)
        target = torch.randn(24, 8) + 2.0
        action = feat.featurize("CCO", -6.0)
        gen = torch.Generator().manual_seed(1)
        loss = flow_matching_loss(model, source, target, action, generator=gen)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert any(g.abs().sum() > 0 for g in grads)

    def test_predict_perturbed_shape(self):
        feat = HepatotoxActionFeaturizer()
        model = Perturbator(
            d_model=6, action_dim=feat.action_dim, depth=1, d_cond=8,
            time_conditioned=True,
        )
        src = torch.randn(9, 6)
        action = feat.featurize("CCO", -6.0)
        out = predict_perturbed(model, src, action, "flow_matching", n_steps=4)
        assert out.shape == (9, 6)


class TestValidationHelpers:
    def test_roc_auc_perfect_and_random(self):
        # perfectly separable -> AUC 1
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = [0, 0, 1, 1]
        assert roc_auc(scores, labels) == 1.0
        # reversed -> AUC 0
        assert roc_auc([0.9, 0.8, 0.2, 0.1], labels) == 0.0
        # single class -> nan
        assert np.isnan(roc_auc([0.1, 0.2], [1, 1]))

    def test_balanced_accuracy(self):
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = [0, 0, 1, 1]
        bacc, thr = balanced_accuracy_at_threshold(scores, labels)
        assert bacc == 1.0
        assert 0.2 < thr < 0.8

    def test_gap_closed_and_cosine(self):
        assert abs(gap_closed(0.2, 1.0) - 0.8) < 1e-9
        assert gap_closed(0.5, 0.0) == 0.0  # no baseline gap
        a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        b = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
        assert abs(centroid_cosine(a, b) - 1.0) < 1e-6
        c = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        assert abs(centroid_cosine(a, c)) < 1e-6

    def test_dose_response_slope_monotone(self):
        ld = [-8.0, -7.0, -6.0, -5.0]
        sh = [0.0, 1.0, 2.0, 3.0]  # slope = 1 per log10
        dr = dose_response_slope(ld, sh)
        assert abs(dr["slope"] - 1.0) < 1e-6
        assert abs(dr["r"] - 1.0) < 1e-6
        # half-max (1.5) is between -7 and -6 -> ec50_log = -6.5
        assert abs(dr["ec50_log"] - (-6.5)) < 1e-6

    def test_dose_response_non_monotone_no_ec50(self):
        dr = dose_response_slope([-7.0, -6.0, -5.0], [2.0, 1.0, 3.0])
        assert np.isnan(dr["ec50_log"])
        assert np.isfinite(dr["slope"])

    def test_spearman(self):
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]  # perfectly monotone
        assert abs(spearman(x, y) - 1.0) < 1e-9
        assert abs(spearman(x, y[::-1]) + 1.0) < 1e-9
        assert np.isnan(spearman([1, 2], [3, 4]))  # too few points

    def test_moa_hierarchy_cosine(self):
        # two MoAs: A drugs point the same way, B drugs point opposite to A
        disp = torch.tensor([
            [1.0, 0.0], [1.0, 0.1],   # MoA A (aligned)
            [-1.0, 0.0], [-1.0, 0.1],  # MoA B (aligned to each other, opposite A)
        ])
        labels = ["A", "A", "B", "B"]
        m = moa_hierarchy_cosine(disp, labels)
        assert m["intra"] > m["inter"]
        assert m["separation"] > 0
        assert m["n_pairs_intra"] == 2  # (A,A) and (B,B)
        assert m["n_pairs_inter"] == 4  # cross pairs

    def test_moa_hierarchy_excludes_unlabelled(self):
        disp = torch.randn(4, 3)
        m = moa_hierarchy_cosine(disp, ["A", None, "A", ""])
        # only the two "A" drugs form an intra pair; no inter pairs
        assert m["n_pairs_intra"] == 1
        assert m["n_pairs_inter"] == 0
        assert np.isnan(m["separation"])
