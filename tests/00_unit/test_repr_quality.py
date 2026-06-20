"""Representation-quality metric helpers: the expected ordering must hold on
synthetic Z (isotropic Gaussian beats a rank-1 spiky matrix on isotropy + SIGReg;
identical view-sets give alignment ~ 1; kNN beats chance on separable clusters).
Tiny / CPU only.
"""
import math

import numpy as np
import torch

from examples.tahoe_baselines.repr_quality import (
    compute_wins,
    drop_view_item,
    gaussianity_metrics,
    invariance_metrics,
    isotropy_score,
    knn_balanced_accuracy,
    make_two_views,
    uniformity,
)


def _rank1(n=400, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n, 1, generator=g)
    return base @ torch.randn(1, d, generator=g)


def _iso(n=400, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


class TestInvariance:
    def test_identical_views_align(self):
        z = _iso()
        m = invariance_metrics(z, z.clone())
        assert m["alignment_cos"] > 0.999
        assert m["self_retrieval_top1"] > 0.999
        assert m["self_retrieval_mean_rank"] < 1.001

    def test_unrelated_views_misalign(self):
        za, zb = _iso(seed=1), _iso(seed=2)
        m = invariance_metrics(za, zb)
        assert abs(m["alignment_cos"]) < 0.2  # random => ~0 cosine
        assert m["self_retrieval_top1"] < 0.2  # no better than scattered

    def test_noisy_views_between(self):
        z = _iso()
        zb = z + 0.1 * _iso(seed=9)
        m = invariance_metrics(z, zb)
        assert 0.9 < m["alignment_cos"] < 1.0
        assert m["self_retrieval_top1"] > 0.5


class TestGaussianity:
    def test_isotropy_iso_vs_rank1(self):
        assert isotropy_score(_iso())["isotropy_score"] > 0.5
        assert isotropy_score(_rank1())["isotropy_score"] < 0.05

    def test_isotropy_condition_number(self):
        assert isotropy_score(_iso())["condition_number"] < \
            isotropy_score(_rank1())["condition_number"]

    def test_sigreg_iso_lower_than_rank1(self):
        gi = gaussianity_metrics(_iso(n=1024), num_slices=64, seed=0)
        gr = gaussianity_metrics(_rank1(n=1024), num_slices=64, seed=0)
        assert math.isfinite(gi["sigreg"]) and gi["sigreg"] >= 0
        assert gi["sigreg"] < gr["sigreg"]
        assert gi["isotropy_score"] > gr["isotropy_score"]


class TestKNNAndUniformity:
    def _clusters(self, n_per=80, k=4, d=16, noise=0.25, seed=0):
        g = torch.Generator().manual_seed(seed)
        centers = torch.randn(k, d, generator=g) * 3.0
        feats, labels = [], []
        for c in range(k):
            feats.append(centers[c] + noise * torch.randn(n_per, d, generator=g))
            labels += [f"c{c}"] * n_per
        return torch.cat(feats), labels

    def test_knn_beats_chance_on_clusters(self):
        X, y = self._clusters()
        acc = knn_balanced_accuracy(X, y, k=15, seed=0)
        assert acc > 0.8  # well-separated => near-perfect, chance is 0.25

    def test_knn_chance_on_random(self):
        X = _iso(n=200, d=16)
        y = [f"c{i % 4}" for i in range(200)]
        acc = knn_balanced_accuracy(X, y, k=15, seed=0)
        assert acc < 0.5  # no signal => ~chance (0.25)

    def test_uniformity_iso_more_spread_than_rank1(self):
        # Wang-Isola uniformity = log E[exp(-2 d^2)]: a well-spread isotropic rep
        # has large pairwise distances => MORE NEGATIVE uniformity than a rank-1
        # rep, which collapses to ~one direction on the sphere (uniformity ~ 0).
        assert uniformity(_iso()) < uniformity(_rank1())


class TestDropoutAndWins:
    def test_drop_view_keeps_subset_aligned(self):
        item = {
            "gene_token_ids": torch.arange(10, 30),
            "values": torch.arange(20).float(),
            "raw_counts": torch.arange(20).float() + 100,
        }
        rng = np.random.default_rng(0)
        out = drop_view_item(item, p=0.5, rng=rng)
        n = out["gene_token_ids"].numel()
        assert 1 <= n < 20
        # values/raw_counts stay aligned to the kept tokens (token = value here)
        kept = out["gene_token_ids"] - 10
        assert torch.equal(out["values"], kept.float())
        assert torch.equal(out["raw_counts"], kept.float() + 100)

    def test_make_two_views_independent(self):
        items = [{
            "gene_token_ids": torch.arange(50),
            "values": torch.ones(50),
            "raw_counts": torch.ones(50),
        }]
        a, b = make_two_views(items, p=0.5, seed=0)
        # two independent draws => different kept sets (overwhelmingly likely)
        assert not torch.equal(a[0]["gene_token_ids"], b[0]["gene_token_ids"])

    def test_compute_wins_orientation(self):
        table = {
            "Subliminal14": {"gaussianity/sigreg": 1.0, "knn/organ_acc": 0.9},
            "MAE-512": {"gaussianity/sigreg": 5.0, "knn/organ_acc": 0.95},
        }
        wins = compute_wins(table)
        # sigreg is lower-better => sub14 (1.0) wins
        assert wins["gaussianity/sigreg"]["sub14_wins"] is True
        # organ acc is higher-better => MAE (0.95) wins
        assert wins["knn/organ_acc"]["sub14_wins"] is False
        assert wins["knn/organ_acc"]["winner"] == "MAE-512"
