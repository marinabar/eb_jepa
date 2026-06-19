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
