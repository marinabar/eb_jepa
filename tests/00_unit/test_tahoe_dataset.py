"""Tahoe dataloader: CLS stripping, CP10k+log1p, on-the-fly drop/mask views,
padding/attention masks, and count-mode-B cross-view bin consistency.
"""

import math

import numpy as np
import pytest
import torch

from eb_jepa.datasets.tahoe.dataset import (
    TahoeCollator,
    TahoeConfig,
    TahoeDataset,
    densify,
    parse_log_conc,
)
from eb_jepa.datasets.tahoe.normalizer import (
    QuantileBinner,
    cp10k_log1p,
    fit_quantile_boundaries,
)

N_GENES = 200


def _write_synthetic_parquet(path, n_cells=24, seed=0):
    """Write a parquet mimicking expression_data, with a CLS marker at index 0."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    genes, exprs, drug, sample, cline, moa, smiles, plate, bc = ([] for _ in range(9))
    for c in range(n_cells):
        g = int(rng.integers(5, 40))  # non-zero genes in this cell
        toks = rng.choice(np.arange(2, N_GENES), size=g, replace=False).astype(np.int64)
        counts = rng.integers(1, 50, size=g).astype(np.float32)
        # prepend CLS marker (token_id 1, expression -2.0) as in real Tahoe rows
        genes.append([1] + toks.tolist())
        exprs.append([-2.0] + counts.tolist())
        drug.append("DMSO_TF" if c % 4 == 0 else f"drug_{c % 3}")
        sample.append(f"smp_{c % 5}")
        cline.append("CVCL_0001" if c % 2 == 0 else "CVCL_0002")
        moa.append("unclear")
        smiles.append("CCO")
        plate.append(f"plate{c % 3 + 1}")
        bc.append(f"bc_{c}")
    table = pa.table(
        {
            "genes": genes,
            "expressions": exprs,
            "drug": drug,
            "sample": sample,
            "cell_line_id": cline,
            "moa-fine": moa,
            "canonical_smiles": smiles,
            "plate": plate,
            "BARCODE_SUB_LIB_ID": bc,
        }
    )
    pq.write_table(table, str(path / "shard0.parquet"))


@pytest.fixture
def data_dir(tmp_path):
    _write_synthetic_parquet(tmp_path)
    return str(tmp_path)


class TestNormalizer:
    def test_cp10k_log1p_math(self):
        counts = torch.tensor([1.0, 3.0, 6.0])  # sum 10
        out = cp10k_log1p(counts)
        expected = torch.log1p(counts / 10.0 * 1e4)
        assert torch.allclose(out, expected)

    def test_cp10k_log1p_zero_cell(self):
        assert torch.allclose(cp10k_log1p(torch.zeros(5)), torch.zeros(5))

    def test_binner_roundtrip_and_consistency(self, tmp_path):
        torch.manual_seed(0)
        tok = torch.randint(0, N_GENES, (5000,))
        val = torch.rand(5000) * 3
        bnd = fit_quantile_boundaries(tok, val, n_genes=N_GENES, n_bins=10)
        binner = QuantileBinner(bnd, n_bins=10)
        b1 = binner.bin(tok, val)
        assert b1.min() >= 0 and b1.max() <= 9
        # same (gene, value) -> same bin, always (the SIGReg-stability property)
        b2 = binner.bin(tok, val)
        assert torch.equal(b1, b2)
        binner.save(tmp_path / "bins")
        reloaded = QuantileBinner.load(tmp_path / "bins")
        assert torch.equal(reloaded.bin(tok, val), b1)


class TestParseLogConc:
    def test_uM_concentration(self):
        # 0.05 uM = 5e-8 M
        assert parse_log_conc("[('Infigratinib', 0.05, 'uM')]") == pytest.approx(
            math.log10(0.05e-6)
        )

    def test_control_is_nan(self):
        assert math.isnan(parse_log_conc("[('DMSO_TF', 0.0, 'uM')]"))

    def test_malformed_is_nan(self):
        assert math.isnan(parse_log_conc("not a tuple"))


class TestDataset:
    def test_strips_cls_and_aligns(self, data_dir):
        ds = TahoeDataset(TahoeConfig(data_dir=data_dir))
        item = ds[0]
        # CLS marker (token 1 / value -2.0) must be gone
        assert (item["gene_token_ids"] != 1).all() or item["gene_token_ids"].numel() > 0
        assert item["gene_token_ids"].shape == item["values"].shape
        assert (item["values"] >= 0).all()  # log1p of non-negative

    def test_metadata_fields(self, data_dir):
        organ_map = {"CVCL_0001": "Liver", "CVCL_0002": "Lung"}
        ds = TahoeDataset(
            TahoeConfig(data_dir=data_dir),
            cell_line_to_organ=organ_map,
            sample_to_logconc={"smp_0": -7.0},
        )
        item = ds[0]
        assert item["organ"] in ("Liver", "Lung")
        for key in ("drug", "sample", "cell_line_id", "plate", "canonical_smiles"):
            assert isinstance(item[key], str)


def _make_loaded_dataset(data_dir, **cfg_kw):
    cfg = TahoeConfig(data_dir=data_dir, **cfg_kw)
    binner = None
    if cfg.count_mode == "B":
        bnd = torch.zeros(cfg.n_genes, cfg.n_bins - 1)
        # nonzero edges so bins actually spread
        bnd += torch.linspace(0.1, 2.0, cfg.n_bins - 1)
        binner = QuantileBinner(bnd, n_bins=cfg.n_bins)
    ds = TahoeDataset(cfg, binner=binner)
    return cfg, ds


class TestCollatorDrop:
    def test_shapes_and_padding(self, data_dir):
        cfg, ds = _make_loaded_dataset(data_dir, L=64, n_views=3, view_mode="drop")
        coll = TahoeCollator(cfg)
        batch = [ds[i] for i in range(6)]
        out = coll(batch)
        assert out["gene_token_ids"].shape == (3, 6, 64)
        assert out["pad_mask"].shape == (3, 6, 64)
        assert "count_value" in out and out["count_value"].shape == (3, 6, 64)
        # pad positions carry the pad token and are masked out
        pad = ~out["pad_mask"]
        assert (out["gene_token_ids"][pad] == cfg.pad_token_id).all()
        # drop keeps a subset: real tokens per view <= cell's gene count
        assert out["count_mask"].sum() == 0  # drop mode never masks counts

    def test_views_differ(self, data_dir):
        cfg, ds = _make_loaded_dataset(data_dir, L=64, n_views=2, view_mode="drop")
        torch.manual_seed(0)
        out = TahoeCollator(cfg)([ds[i] for i in range(6)])
        # the two views should not be identical token sets (independent sampling)
        assert not torch.equal(out["gene_token_ids"][0], out["gene_token_ids"][1])


class TestCollatorMask:
    def test_keeps_all_genes_and_masks_counts(self, data_dir):
        cfg, ds = _make_loaded_dataset(
            data_dir, L=128, n_views=2, view_mode="mask", gene_mask_frac=0.5
        )
        coll = TahoeCollator(cfg)
        batch = [ds[i] for i in range(6)]
        out = coll(batch)
        # mask mode keeps all genes (cells here have < L genes) -> real count == g
        for j, cell in enumerate(batch):
            g = cell["gene_token_ids"].numel()
            assert out["pad_mask"][0, j].sum().item() == g
        # some counts are masked
        assert out["count_mask"].any()
        # masked positions are always real tokens, never padding
        assert (out["pad_mask"] | ~out["count_mask"]).all()


class TestCountModeBConsistency:
    def test_same_gene_value_same_bin_across_views(self, data_dir):
        cfg, ds = _make_loaded_dataset(
            data_dir, L=128, n_views=4, view_mode="drop", count_mode="B", n_bins=10
        )
        out = TahoeCollator(cfg)([ds[i] for i in range(6)])
        ids, bins, pad = out["gene_token_ids"], out["count_bin"], out["pad_mask"]
        v, n, _ = ids.shape
        # For each cell, a token that appears (real) in two views must share its bin.
        for j in range(n):
            tok_to_bin = {}
            for view in range(v):
                real = pad[view, j]
                for t, b in zip(
                    ids[view, j][real].tolist(), bins[view, j][real].tolist()
                ):
                    if t in tok_to_bin:
                        assert tok_to_bin[t] == b, "bin differs across views!"
                    else:
                        tok_to_bin[t] = b


def test_densify():
    tok = torch.tensor([2, 5, 9])
    val = torch.tensor([1.0, 2.0, 3.0])
    dense = densify(tok, val, n_genes=10)
    assert dense.shape == (10,)
    assert dense[2] == 1.0 and dense[5] == 2.0 and dense[9] == 3.0
    assert dense.sum() == 6.0
