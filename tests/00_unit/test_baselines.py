"""MAE / VAE / PCA baselines: densify collate, encode shapes, and learning."""

import torch

from eb_jepa.singlecell.baselines import (
    DensifyCollator,
    MAEBaseline,
    PCABaseline,
    VAEBaseline,
    build_baseline,
)

N_GENES = 64


def _cells(n=16):
    cells = []
    for i in range(n):
        gen = torch.Generator().manual_seed(i)
        g = 6 + i % 5
        cells.append(
            {
                "gene_token_ids": torch.randperm(N_GENES, generator=gen)[:g],
                "values": torch.rand(g, generator=gen),
                "drug": "d",
                "sample": "s",
                "cell_line_id": "CVCL_0001",
                "organ": "Liver",
                "moa_fine": "x",
                "plate": "p1",
                "log_conc": -7.0,
            }
        )
    return cells


def test_densify_collate():
    out = DensifyCollator(N_GENES)(_cells(8))
    assert out["dense"].shape == (8, N_GENES)
    assert len(out["organ"]) == 8 and out["log_conc"].shape == (8,)


def test_mae_overfit_and_encode():
    torch.manual_seed(0)
    dense = DensifyCollator(N_GENES)(_cells(16))["dense"]
    mae = MAEBaseline(N_GENES, hidden=64, latent=16, mask_frac=0.5)
    opt = torch.optim.Adam(mae.parameters(), lr=1e-2)
    first = mae(dense)["loss"].item()
    loss = mae(dense)["loss"]
    for _ in range(50):
        opt.zero_grad()
        loss = mae(dense)["loss"]
        loss.backward()
        opt.step()
    assert loss.item() < first
    assert mae.encode(dense).shape == (16, 16)


def test_vae_overfit_and_encode():
    torch.manual_seed(0)
    dense = DensifyCollator(N_GENES)(_cells(16))["dense"]
    vae = VAEBaseline(N_GENES, hidden=64, latent=16, kl_coeff=1e-3)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-2)
    out0 = vae(dense)
    assert "kl" in out0 and "recon_loss" in out0
    first = out0["recon_loss"].item()
    for _ in range(50):
        opt.zero_grad()
        vae(dense)["loss"].backward()
        opt.step()
    assert vae(dense)["recon_loss"].item() < first
    assert vae.encode(dense).shape == (16, 16)


def test_pca_fit_encode():
    dense = DensifyCollator(N_GENES)(_cells(40))["dense"]
    pca = PCABaseline(n_components=8).fit(dense)
    assert pca.encode(dense).shape == (40, 8)


def test_build_baseline_factory():
    assert isinstance(build_baseline("mae", N_GENES), MAEBaseline)
    assert isinstance(build_baseline("vae", N_GENES), VAEBaseline)
    assert isinstance(build_baseline("pca", N_GENES), PCABaseline)


# --------------------------------------------------------------------------- #
# Train-step smoke (the loop the Dalia train.py runs, in miniature)           #
# --------------------------------------------------------------------------- #
def _train_step_smoke(model):
    """One AdamW step on synthetic dense data; loss must be finite and decrease."""
    torch.manual_seed(0)
    dense = DensifyCollator(N_GENES)(_cells(24))["dense"]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = model(dense)["loss"].item()
    assert torch.isfinite(torch.tensor(first))
    loss = model(dense)["loss"]
    for _ in range(30):
        opt.zero_grad()
        loss = model(dense)["loss"]
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)
    assert loss.item() <= first  # learning, not diverging
    return model


def test_mae_train_step_and_keys():
    out = MAEBaseline(N_GENES, hidden=64, latent=16, mask_frac=0.5)(
        DensifyCollator(N_GENES)(_cells(8))["dense"]
    )
    assert {"loss", "recon_loss"} <= set(out)
    _train_step_smoke(MAEBaseline(N_GENES, hidden=64, latent=16, mask_frac=0.5))


def test_vae_train_step_and_keys():
    out = VAEBaseline(N_GENES, hidden=64, latent=16, kl_coeff=1e-3)(
        DensifyCollator(N_GENES)(_cells(8))["dense"]
    )
    assert {"loss", "recon_loss", "kl"} <= set(out)
    _train_step_smoke(VAEBaseline(N_GENES, hidden=64, latent=16, kl_coeff=1e-3))


def test_pca_encode_after_fit_shapes():
    dense = DensifyCollator(N_GENES)(_cells(40))["dense"]
    pca = PCABaseline(n_components=8).fit(dense)
    z = pca.encode(dense)
    assert z.shape == (40, 8) and torch.is_tensor(z)


# --------------------------------------------------------------------------- #
# Benchmark-table assembly smoke (synthetic reps; no cluster / no data)       #
# --------------------------------------------------------------------------- #
def _synthetic_feats_meta(n=60, d=16):
    """Structured + collapsed reps and per-cell labels for the metric harness."""
    g = torch.Generator().manual_seed(0)
    organ = ["Liver" if i % 2 else "Lung" for i in range(n)]
    cell = [f"CVCL_{i % 4:04d}" for i in range(n)]
    # rich (isotropic) rep with organ signal; collapsed rep = low-rank (variance
    # concentrated in 1 direction -> small effective rank).
    base = torch.randn(n, d, generator=g)
    signal = torch.tensor([[1.0 if o == "Liver" else -1.0] for o in organ])
    collapsed = signal @ torch.randn(1, d, generator=g) + 1e-3 * torch.randn(n, d, generator=g)
    feats = {
        "Subliminal14": base + 1.5 * signal,
        "PCA": collapsed,  # rank-~1: collapsed representation
    }
    meta = {
        "organ": organ,
        "cell_line_id": cell,
        "drug": ["DMSO_TF" if i % 3 else "DrugX" for i in range(n)],
        "moa_fine": ["m" if i % 2 else "n" for i in range(n)],
        "plate": [f"plate{i % 2}" for i in range(n)],
    }
    return feats, meta


def test_benchmark_table_assembly():
    from examples.tahoe_baselines.benchmark import representation_metrics

    feats, meta = _synthetic_feats_meta()
    table = {name: representation_metrics(reps, meta) for name, reps in feats.items()}
    # required scalar columns present + sane ranges
    for row in table.values():
        assert "effective_rank" in row and row["effective_rank"] >= 0
        assert "latent_dim" in row
        assert 0.0 <= row["organ/balanced_accuracy"] <= 1.0
        assert "organ/above_chance" in row
    # the structured rep should separate organ better than the collapsed one,
    # and have a higher effective rank.
    assert (table["Subliminal14"]["organ/balanced_accuracy"]
            >= table["PCA"]["organ/balanced_accuracy"])
    assert table["Subliminal14"]["effective_rank"] > table["PCA"]["effective_rank"]


def test_benchmark_write_table(tmp_path):
    from examples.tahoe_baselines.benchmark import representation_metrics, write_table

    feats, meta = _synthetic_feats_meta()
    table = {name: representation_metrics(reps, meta) for name, reps in feats.items()}
    csv_path, json_path = write_table(table, str(tmp_path))
    import json as _json
    import os

    assert os.path.exists(csv_path) and os.path.exists(json_path)
    loaded = _json.load(open(json_path))
    assert set(loaded) == set(table)


def test_benchmark_plots_smoke(tmp_path):
    from examples.tahoe_baselines.benchmark import representation_metrics
    from examples.tahoe_baselines.plots import make_all_plots

    feats, meta = _synthetic_feats_meta()
    table = {name: representation_metrics(reps, meta) for name, reps in feats.items()}
    paths = make_all_plots(table, feats, meta, str(tmp_path))
    import os

    assert paths["probe_bars"] and os.path.exists(paths["probe_bars"])
    assert paths["effrank"] and os.path.exists(paths["effrank"])
    assert paths["tsne_grid"] and os.path.exists(paths["tsne_grid"])
