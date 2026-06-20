"""SingleCellEncoder + GeneTokenEmbedding + LeJEPALoss integration:
shapes, mask invariance, frozen ESMC/Evo2 tables, count modes, and a tiny overfit.
"""

import pytest
import torch

from eb_jepa.architectures import Projector
from eb_jepa.datasets.tahoe.dataset import TahoeCollator, TahoeConfig
from eb_jepa.losses import LeJEPALoss
from eb_jepa.singlecell.embeddings import CountEmbedding, GeneTokenEmbedding
from eb_jepa.singlecell.encoder import SingleCellEncoder, encode_views

N_GENES, D_MODEL, L = 50, 32, 16


def _cell(g, n_genes=N_GENES, with_bins=False, n_bins=10):
    torch_g = torch.Generator().manual_seed(g)
    tok = torch.randperm(n_genes - 2, generator=torch_g)[:g] + 2
    val = torch.rand(g, generator=torch_g) * 2
    cell = {
        "gene_token_ids": tok,
        "values": val,
        "drug": "drug_x",
        "sample": "smp_0",
        "cell_line_id": "CVCL_0001",
        "organ": "Liver",
        "moa_fine": "unclear",
        "plate": "plate1",
        "canonical_smiles": "CCO",
        "log_conc": -7.0,
    }
    if with_bins:
        cell["bin_ids"] = (val * 3).long().clamp(max=n_bins - 1)
    return cell


def _batch(cfg, n=8, with_bins=False):
    cells = [_cell(g=5 + i, with_bins=with_bins, n_bins=cfg.n_bins) for i in range(n)]
    return TahoeCollator(cfg)(cells)


def _build(count_mode="A", use_cls=False, readout="meanpool", n_layers=2):
    embed = GeneTokenEmbedding.random(
        N_GENES, D_MODEL, d_esmc=16, d_evo2=12, count_mode=count_mode, n_bins=10
    )
    enc = SingleCellEncoder(
        embed,
        d_model=D_MODEL,
        n_layers=n_layers,
        n_heads=4,
        use_cls=use_cls,
        readout=readout,
    )
    return enc


class TestCountEmbedding:
    def test_mode_a_mask_substitutes_vector(self):
        ce = CountEmbedding(D_MODEL, "A")
        val = torch.randn(4, L)
        mask = torch.zeros(4, L, dtype=torch.bool)
        mask[:, 0] = True
        out = ce(count_value=val, count_mask=mask)
        # masked positions equal the learned mask vector
        assert torch.allclose(out[:, 0], ce.mask_vector.expand(4, D_MODEL))

    def test_mode_b_table_has_mask_row(self):
        ce = CountEmbedding(D_MODEL, "B", n_bins=10)
        assert ce.table.num_embeddings == 11  # n_bins + MASK


class TestEncoder:
    def test_encode_views_shape(self):
        cfg = TahoeConfig(data_dir="", L=L, n_views=3, n_genes=N_GENES)
        enc = _build()
        reps = encode_views(enc, _batch(cfg, n=6))
        assert reps.shape == (3, 6, D_MODEL)
        assert torch.isfinite(reps).all()

    @torch.no_grad()
    def test_meanpool_ignores_padding(self):
        cfg = TahoeConfig(data_dir="", L=L, n_views=1, n_genes=N_GENES)
        enc = _build().eval()
        batch = _batch(cfg, n=6)
        reps1 = encode_views(enc, batch)
        # corrupt the padded positions' token ids; output must not change
        ids = batch["gene_token_ids"].clone()
        pad = ~batch["pad_mask"]
        ids[pad] = 3  # some valid gene id
        batch2 = dict(batch, gene_token_ids=ids)
        reps2 = encode_views(enc, batch2)
        assert torch.allclose(reps1, reps2, atol=1e-5)

    def test_frozen_tables_have_no_grad(self):
        cfg = TahoeConfig(data_dir="", L=L, n_views=2, n_genes=N_GENES)
        enc = _build()
        reps = encode_views(enc, _batch(cfg, n=4))
        reps.sum().backward()
        # ESMC/Evo2 lookups are buffers, not parameters
        names = {n for n, _ in enc.named_parameters()}
        assert not any("esmc_table" in n or "evo2_table" in n for n in names)
        # the learned projections do receive gradients
        assert enc.embed.evo2_proj.weight.grad is not None

    def test_cls_readout(self):
        cfg = TahoeConfig(data_dir="", L=L, n_views=2, n_genes=N_GENES)
        enc = _build(use_cls=True, readout="cls")
        reps = encode_views(enc, _batch(cfg, n=4))
        assert reps.shape == (2, 4, D_MODEL) and torch.isfinite(reps).all()

    def test_count_mode_b(self):
        cfg = TahoeConfig(
            data_dir="", L=L, n_views=2, n_genes=N_GENES, count_mode="B", n_bins=10
        )
        enc = _build(count_mode="B")
        reps = encode_views(enc, _batch(cfg, n=4, with_bins=True))
        assert reps.shape == (2, 4, D_MODEL) and torch.isfinite(reps).all()

    def test_cls_readout_requires_cls_token(self):
        embed = GeneTokenEmbedding.random(N_GENES, D_MODEL, d_esmc=16, d_evo2=12)
        with pytest.raises(ValueError):
            SingleCellEncoder(embed, d_model=D_MODEL, use_cls=False, readout="cls")


class TestFromCache:
    def test_from_cache_noncontiguous_token_ids(self, tmp_path):
        # Real Tahoe token_ids are non-contiguous (0-2 reserved, genes up to 62712).
        # from_cache must size tables by max(token_id)+1 and index by token_id.
        import json

        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        d_esmc, d_evo2 = 8, 6
        token_ids = [3, 5, 9]  # max 9 -> tables must have 10 rows
        is_coding = [True, False, True]
        esmc_row = [0, -1, 1]
        evo2_row = [0, 1, 2]
        np.save(tmp_path / "esmc.npy", np.random.randn(2, d_esmc).astype("float32"))
        np.save(tmp_path / "evo2.npy", np.random.randn(3, d_evo2).astype("float32"))
        pq.write_table(
            pa.table(
                {
                    "token_id": token_ids,
                    "is_coding": is_coding,
                    "esmc_row": esmc_row,
                    "evo2_row": evo2_row,
                    "ensembl_id": ["ENSG1", "ENSG2", "ENSG3"],
                }
            ),
            str(tmp_path / "index.parquet"),
        )
        (tmp_path / "metadata.json").write_text(json.dumps({"d_evo2": d_evo2}))

        embed = GeneTokenEmbedding.from_cache(tmp_path, d_model=16)
        assert embed.n_genes == 10  # max token id (9) + 1
        # indexing by the real (non-contiguous) token ids must not go out of bounds
        ids = torch.tensor([[3, 5, 9]])
        out = embed(ids, count_value=torch.zeros(1, 3))
        assert out.shape == (1, 3, 16) and torch.isfinite(out).all()
        # non-coding token (5) has a zero protein term
        assert embed.coding_mask[5].item() is False
        assert embed.coding_mask[3].item() is True


class TestFlopCount:
    def test_measure_encoder_flops_positive(self):
        from examples.tahoe_jepa.main import measure_encoder_flops

        cfg = TahoeConfig(data_dir="", L=L, n_views=2, n_genes=N_GENES)
        enc = _build()
        flops = measure_encoder_flops(enc, _batch(cfg, n=4))
        assert flops > 0


class TestLeJEPAIntegration:
    def test_tiny_overfit_decreases_loss(self):
        torch.manual_seed(0)
        cfg = TahoeConfig(
            data_dir="", L=L, n_views=2, n_genes=N_GENES, gene_keep_frac=0.8
        )
        enc = _build(n_layers=2)
        proj = Projector(f"{D_MODEL}-64-16")
        loss_fn = LeJEPALoss(projector=proj, lamb=0.05, num_slices=64)
        batch = _batch(cfg, n=16)  # fixed batch -> should overfit
        opt = torch.optim.Adam(
            list(enc.parameters()) + list(loss_fn.parameters()), lr=1e-3
        )
        losses = []
        for _ in range(60):
            opt.zero_grad()
            out = loss_fn(encode_views(enc, batch))
            out["loss"].backward()
            opt.step()
            losses.append(out["loss"].item())
        assert all(torch.isfinite(torch.tensor(losses)))
        assert losses[-1] < losses[0] * 0.9  # the objective is being optimized
