"""Probing harness (detached linear probes, imbalance-aware) and representation
analysis (covariance spectrum, effective rank, tSNE)."""

import torch

from eb_jepa.datasets.tahoe.dataset import TahoeCollator, TahoeConfig
from eb_jepa.singlecell.embeddings import GeneTokenEmbedding
from eb_jepa.singlecell.encoder import SingleCellEncoder
from eb_jepa.singlecell.probes import (
    extract_features,
    run_probe_suite,
    train_classification_probe,
    train_regression_probe,
)
from eb_jepa.singlecell.visualize import (
    covariance_spectrum,
    effective_rank,
    plot_spectrum,
    tsne_embed,
)


def _separable(n_per=40, n_classes=3, d=8, noise=0.3, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.eye(n_classes, d)
    feats, labels = [], []
    for c in range(n_classes):
        feats.append(centers[c] + noise * torch.randn(n_per, d, generator=g))
        labels += [f"class_{c}"] * n_per
    return torch.cat(feats), labels


class TestProbes:
    def test_classification_beats_chance(self):
        feats, labels = _separable()
        m = train_classification_probe(feats, labels, epochs=150)
        assert m["n_classes"] == 3
        assert m["balanced_accuracy"] > 0.7 > m["chance"]
        assert m["macro_f1"] > 0.7

    def test_classification_ignores_none_labels(self):
        feats, labels = _separable(n_per=20, n_classes=2)
        labels[0] = None  # should be dropped, not crash
        m = train_classification_probe(feats, labels, epochs=50)
        assert m["n_classes"] == 2

    def test_regression_recovers_linear_target(self):
        torch.manual_seed(0)
        feats = torch.randn(200, 8)
        w = torch.randn(8)
        target = feats @ w + 0.05 * torch.randn(200)
        m = train_regression_probe(feats, target, epochs=300)
        assert m["r2"] > 0.5 and m["explained_variance"] > 0.5

    def test_extract_features_and_suite(self):
        cfg = TahoeConfig(data_dir="", L=16, n_views=1, n_genes=50, gene_keep_frac=1.0)
        embed = GeneTokenEmbedding.random(50, 32, d_esmc=16, d_evo2=12)
        enc = SingleCellEncoder(embed, d_model=32, n_layers=2, n_heads=4)
        coll = TahoeCollator(cfg)

        def _cell(i):
            gen = torch.Generator().manual_seed(i)
            g = 6 + i % 4
            return {
                "gene_token_ids": torch.randperm(48, generator=gen)[:g] + 2,
                "values": torch.rand(g, generator=gen),
                "drug": "a" if i % 2 else "b",
                "sample": "s",
                "cell_line_id": "CVCL_0001" if i % 2 else "CVCL_0002",
                "organ": "Liver" if i % 2 else "Lung",
                "moa_fine": "x",
                "plate": "p1",
                "canonical_smiles": "CCO",
                "log_conc": -7.0,
            }

        loader = [coll([_cell(i) for i in range(8)]) for _ in range(2)]
        feats, meta = extract_features(enc, loader, device="cpu")
        assert feats.shape == (16, 32)
        assert "organ" in meta and len(meta["organ"]) == 16
        assert "gene_count" in meta and len(meta["gene_count"]) == 16
        results = run_probe_suite(feats, meta)
        assert "clf/organ" in results and "reg/gene_count" in results


class TestVisualize:
    def test_covariance_spectrum_descending_nonneg(self):
        eig = covariance_spectrum(torch.randn(200, 8))
        assert eig.shape == (8,)
        assert (eig >= 0).all()
        assert torch.all(eig[:-1] >= eig[1:] - 1e-5)

    def test_effective_rank_isotropic_vs_collapsed(self):
        iso = torch.randn(300, 8)
        collapsed = torch.randn(300, 1) @ torch.randn(1, 8)  # rank-1
        assert effective_rank(iso) > 4.0
        assert effective_rank(collapsed) < 2.0

    def test_tsne_shape(self):
        emb = tsne_embed(torch.randn(60, 8))
        assert emb.shape == (60, 2)

    def test_plot_spectrum_saves(self, tmp_path):
        eig = covariance_spectrum(torch.randn(100, 8))
        out = plot_spectrum(eig, str(tmp_path / "spec.png"))
        import os

        assert os.path.exists(out)
