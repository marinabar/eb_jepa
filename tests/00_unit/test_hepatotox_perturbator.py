"""Unit tests for the hepatotox-featurized perturbator + validation helpers.

CPU, tiny dims, no rdkit required (the featurizers fall back to a deterministic hash).
"""

import os

import numpy as np
import torch

from eb_jepa.singlecell.perturbator.featurize import DrugFeaturizer
from eb_jepa.singlecell.perturbator.flow import (
    flow_matching_loss,
    ode_sample,
    predict_perturbed,
)
from eb_jepa.singlecell.perturbator.hepatotox_features import (
    HEPATOTOX_DRUGS,
    LOW_DILI_DRUGS,
    HepatotoxActionFeaturizer,
    HepatotoxPathwayFeaturizer,
    dili_label_by_name,
    weak_dili_label_from_moa,
)
from eb_jepa.singlecell.perturbator.model import Perturbator
from examples.tahoe_hepatotox.validate_perturbator import (
    _rank_findings,
    balanced_accuracy_at_threshold,
    centroid_cosine,
    cv_axis_scores,
    dose_response_slope,
    gap_closed,
    mannwhitney_p,
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

    def test_ode_sample_return_path(self):
        # return_path yields [n_steps+1, N, d]; path[0]==source, path[-1]==final;
        # and the final state is byte-identical to the default (no return_path) call.
        feat = HepatotoxActionFeaturizer()
        model = Perturbator(
            d_model=5, action_dim=feat.action_dim, depth=2, d_cond=8,
            time_conditioned=True,
        ).eval()
        src = torch.randn(11, 5)
        action = feat.featurize("CCO", -6.0)
        steps = 7
        for method in ("euler", "heun", "midpoint"):
            final_only = ode_sample(model, src, action, n_steps=steps, method=method)
            final, path = ode_sample(
                model, src, action, n_steps=steps, method=method, return_path=True
            )
            assert path.shape == (steps + 1, 11, 5)
            assert torch.equal(path[0], src)          # path starts at the source
            assert torch.equal(path[-1], final)       # path ends at the final state
            assert torch.equal(final, final_only)     # default behaviour byte-identical


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
        rho, p = spearman(x, y)
        assert abs(rho - 1.0) < 1e-9
        rho_r, _ = spearman(x, y[::-1])
        assert abs(rho_r + 1.0) < 1e-9
        rho_few, p_few = spearman([1, 2], [3, 4])  # too few points
        assert np.isnan(rho_few) and np.isnan(p_few)
        # a noisy-but-monotone trend yields a finite p-value in (0, 1]
        rho2, p2 = spearman([1, 2, 3, 4, 5, 6], [1, 1, 2, 2, 3, 3])
        assert 0.0 < rho2 <= 1.0 and 0.0 <= p2 <= 1.0

    def test_moa_hierarchy_cosine(self):
        # two MoAs: A drugs point the same way, B drugs point opposite to A
        disp = torch.tensor([
            [1.0, 0.0], [1.0, 0.1],   # MoA A (aligned)
            [-1.0, 0.0], [-1.0, 0.1],  # MoA B (aligned to each other, opposite A)
        ])
        labels = ["A", "A", "B", "B"]
        m = moa_hierarchy_cosine(disp, labels, n_perm=500, seed=0)
        assert m["intra"] > m["inter"]
        assert m["separation"] > 0
        assert m["n_pairs_intra"] == 2  # (A,A) and (B,B)
        assert m["n_pairs_inter"] == 4  # cross pairs
        # perfectly separated -> small permutation p-value
        assert 0.0 < m["p_value"] <= 1.0

    def test_moa_hierarchy_excludes_unlabelled(self):
        disp = torch.randn(4, 3)
        m = moa_hierarchy_cosine(disp, ["A", None, "A", ""], n_perm=0)
        # only the two "A" drugs form an intra pair; no inter pairs
        assert m["n_pairs_intra"] == 1
        assert m["n_pairs_inter"] == 0
        assert np.isnan(m["separation"])

    def test_mannwhitney_p(self):
        # well separated groups -> small p
        p = mannwhitney_p([5, 6, 7, 8], [0, 1, 2, 3])
        assert 0.0 <= p < 0.1
        # overlapping groups -> large p
        p2 = mannwhitney_p([1, 2, 3, 4], [1, 2, 3, 4])
        assert p2 > 0.5
        assert np.isnan(mannwhitney_p([], [1, 2]))

    def test_dose_response_monotonicity(self):
        dr = dose_response_slope([-8, -7, -6, -5], [0.0, 1.0, 2.0, 3.0])
        assert dr["monotonicity"] == 1.0  # all steps increase
        dr2 = dose_response_slope([-8, -7, -6, -5], [3.0, 2.0, 1.0, 0.0])
        assert dr2["monotonicity"] == 0.0

    def test_cv_axis_scores(self):
        # linearly separable feature -> CV decision scores rank the classes correctly
        rng = np.random.default_rng(0)
        n = 12
        feats = np.concatenate([
            rng.normal(2.0, 0.3, (n, 3)),   # positives
            rng.normal(-2.0, 0.3, (n, 3)),  # negatives
        ])
        labels = np.array([1] * n + [0] * n)
        scores = cv_axis_scores(feats, labels, n_folds=3, seed=0)
        m = np.isfinite(scores)
        if m.sum() >= 4:  # sklearn present
            assert roc_auc(scores[m], labels[m]) > 0.8


class TestLabels:
    def test_dilirank_by_name_case_insensitive(self):
        assert dili_label_by_name("Acetaminophen") == "hepatotoxic"
        assert dili_label_by_name("ASPIRIN") == "low_concern"
        assert dili_label_by_name("not_a_real_drug") is None
        assert dili_label_by_name(None) is None

    def test_expanded_label_coverage(self):
        # the broadened DILIrank/LiverTox sets are substantial and disjoint
        assert len(HEPATOTOX_DRUGS) >= 80
        assert len(LOW_DILI_DRUGS) >= 60
        assert not (HEPATOTOX_DRUGS & LOW_DILI_DRUGS)

    def test_weak_moa_label(self):
        assert weak_dili_label_from_moa("topoisomerase II inhibitor") == "hepatotoxic"
        assert weak_dili_label_from_moa("histamine receptor antagonist") == "low_concern"
        assert weak_dili_label_from_moa(None) is None
        assert weak_dili_label_from_moa("unrelated mechanism") is None


class TestFindingsRanking:
    def test_rank_findings_orders_by_strength(self):
        results = {
            "dili": {"best": {
                "score": "top_shift_mag", "label_source": "dilirank",
                "roc_auc": 0.95, "auc_strength": 0.45, "balanced_accuracy": 0.9,
                "n_hepatotoxic": 5, "n_low_concern": 5,
            }},
            "moa_hierarchy": {
                "intra": 0.6, "inter": 0.55, "separation": 0.05, "p_value": 0.3,
                "n_pairs_intra": 3, "n_pairs_inter": 9,
            },
            "accuracy": {"mean_gap_closed": 0.2, "mean_centroid_cosine": 0.8, "n_strata": 10},
            "coverage": {"n_drugs": 20},
        }
        out = _rank_findings(results)
        assert out["ranked"][0]["analysis"] == "dili_classification"  # strongest
        assert "DILI AUROC=0.950" in out["headline"]
        # ranked strictly by descending strength
        strengths = [f["strength"] for f in out["ranked"]]
        assert strengths == sorted(strengths, reverse=True)


class TestFlowFieldFigure:
    def test_projection_and_plot_smoke(self, tmp_path):
        # synthetic control / treated / predicted clouds + a recorded flow path ->
        # PCA(2) projection + plot_flow_field writes PNG/PDF/SVG without error.
        import matplotlib
        matplotlib.use("Agg")
        from examples.tahoe_hepatotox.flow_field import plot_flow_field

        rng = np.random.default_rng(0)
        d, nc, nt, steps = 6, 40, 30, 5
        ctrl = rng.normal(0.0, 1.0, (nc, d))
        real = rng.normal(2.0, 1.0, (nt, d))
        # linear flow path control -> ~real centroid (straight lines)
        target_dir = real.mean(0) - ctrl.mean(0)
        ts = np.linspace(0.0, 1.0, steps + 1)
        path = np.stack([ctrl + t * target_dir for t in ts], 0)  # [T+1, nc, d]
        pred = path[-1]

        # PCA(2) on (control ∪ real), faithful linear map
        fit = np.concatenate([ctrl, real], 0)
        mu = fit.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(fit - mu, full_matrices=False)
        comp = Vt[:2].T

        def proj(x):
            return (x - mu) @ comp

        paths2d = proj(path.reshape(-1, d)).reshape(steps + 1, nc, 2)
        choice = {"drug": "Dabrafenib", "cell_line": "CVCL_0027", "dose": -5.5,
                  "n_control": nc, "n_treated": nt}
        metrics = {"sliced_w_pred": 0.4, "sliced_w_base": 1.2, "gap_closed": 0.67,
                   "shift_cosine": 0.95}
        out_prefix = str(tmp_path / "flow_smoke")
        paths = plot_flow_field(proj(ctrl), proj(real), proj(pred), paths2d,
                                choice, metrics, out_prefix, n_streamlines=20, seed=0)
        assert any(p.endswith(".png") for p in paths)
        assert any(p.endswith(".svg") for p in paths)
        for p in paths:
            assert os.path.exists(p)


class TestTrajectoryDeviceConsistency:
    def test_dose_track_cpu_smoke(self):
        # device-consistency smoke for the trajectory path (CPU): control latents,
        # action and model all on CPU -> build_dose_track runs without a device error.
        from eb_jepa.singlecell.perturbator.visualize import build_dose_track

        feat = HepatotoxActionFeaturizer()
        model = Perturbator(
            d_model=6, action_dim=feat.action_dim, depth=1, d_cond=8,
            time_conditioned=True,
        ).eval()
        control = torch.randn(16, 6)  # CPU
        track = build_dose_track(
            model, feat, control, "acetaminophen", "CCO", [-7.0, -6.0, -5.0],
            objective="flow_matching", ode_steps=3,
        )
        assert track.dose_centroids.shape == (3, 6)
        assert "score" in track.metrics
