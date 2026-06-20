"""Preprocess helpers: group-level split (no leakage), liver filter, stats fit,
and an end-to-end run() over a synthetic Tahoe layout (data/ + metadata/)."""

import glob

import numpy as np
import torch

from eb_jepa.datasets.tahoe import preprocess as pp
from eb_jepa.datasets.tahoe.normalizer import GeneHistogram, QuantileBinner
from eb_jepa.datasets.tahoe.preprocess import (
    assign_split,
    fit_boundaries_from_cells,
    fit_histogram_from_cells,
    group_level_split,
    liver_cell_lines,
)


class TestGroupSplit:
    def test_no_group_leakage(self):
        groups = [f"line_{i % 10}" for i in range(1000)]
        train, val = group_level_split(groups, val_frac=0.3, seed=0)
        assert train.isdisjoint(val)
        assert train | val == set(groups)
        # every cell lands in exactly one split, by its group
        for gk in groups:
            s = assign_split(gk, val)
            assert (s == "val") == (gk in val)

    def test_val_frac_respected(self):
        groups = [f"d{i}" for i in range(100)]
        _, val = group_level_split(groups, val_frac=0.2, seed=1)
        assert len(val) == 20

    def test_deterministic_by_seed(self):
        groups = [f"d{i}" for i in range(50)]
        a = group_level_split(groups, val_frac=0.2, seed=7)
        b = group_level_split(groups, val_frac=0.2, seed=7)
        c = group_level_split(groups, val_frac=0.2, seed=8)
        assert a[1] == b[1]
        assert a[1] != c[1]

    def test_ignores_none(self):
        train, val = group_level_split(["a", "b", None, "c"], val_frac=0.5, seed=0)
        assert None not in train and None not in val


def test_liver_cell_lines():
    cl2organ = {"CVCL_1": "Liver", "CVCL_2": "Lung", "CVCL_3": "Liver"}
    assert liver_cell_lines(cl2organ) == {"CVCL_1", "CVCL_3"}


def _write_tahoe_layout(root, n_shards=2, cells_per=12, n_genes=200, seed=0):
    """Synthetic mirror of the real Tahoe layout: data/*.parquet + metadata/*."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    lines = ["CVCL_0001", "CVCL_0002"]
    samples = ["smp_0", "smp_1", "smp_2"]
    for s in range(n_shards):
        cols = {
            k: []
            for k in (
                "genes",
                "expressions",
                "drug",
                "sample",
                "cell_line_id",
                "moa-fine",
                "canonical_smiles",
                "plate",
                "BARCODE_SUB_LIB_ID",
            )
        }
        for c in range(cells_per):
            g = int(rng.integers(5, 30))
            toks = rng.choice(np.arange(2, n_genes), size=g, replace=False).astype(
                np.int64
            )
            counts = rng.integers(1, 40, size=g).astype(np.float32)
            cols["genes"].append([1] + toks.tolist())  # CLS marker first
            cols["expressions"].append([-2.0] + counts.tolist())
            cols["drug"].append("DMSO_TF" if c % 3 == 0 else "drugA")
            cols["sample"].append(samples[c % 3])
            cols["cell_line_id"].append(lines[(s + c) % 2])
            cols["moa-fine"].append("unclear")
            cols["canonical_smiles"].append("CCO")
            cols["plate"].append("plate1")
            cols["BARCODE_SUB_LIB_ID"].append(f"bc_{s}_{c}")
        pq.write_table(pa.table(cols), str(root / "data" / f"train-{s}.parquet"))
    pq.write_table(
        pa.table({"Cell_ID_Cellosaur": lines, "Organ": ["Liver", "Lung"]}),
        str(root / "metadata" / "cell_line_metadata.parquet"),
    )
    pq.write_table(
        pa.table(
            {
                "sample": samples,
                "drugname_drugconc": [
                    "[('DMSO_TF', 0.0, 'uM')]",
                    "[('drugA', 0.05, 'uM')]",
                    "[('drugA', 0.5, 'uM')]",
                ],
            }
        ),
        str(root / "metadata" / "sample_metadata.parquet"),
    )


def test_preprocess_run_e2e(tmp_path):
    import pyarrow.parquet as pq

    root, out = tmp_path / "tahoe", tmp_path / "cache"
    _write_tahoe_layout(root)
    pp.run(
        data_dir=str(root),
        out_dir=str(out),
        n_genes=200,
        n_bins=8,
        quantile_cells=0,
        val_frac=0.5,
        seed=0,
    )
    # artifacts written
    assert (out / "maps.pt").exists() and (out / "split.pt").exists()
    assert (out / "quantile_bins.npy").exists() and (
        out / "quantile_bins.json"
    ).exists()
    maps = torch.load(out / "maps.pt")
    assert maps["cell_line_to_organ"]["CVCL_0001"] == "Liver"
    # group split with 2 lines @ val_frac 0.5 -> both splits populated
    assert (out / "train").exists() and (out / "val").exists()
    parts = glob.glob(str(out / "*" / "*.parquet"))
    assert parts
    t = pq.read_table(parts[0])
    assert "genes" in t.column_names and "value" in t.column_names
    # CLS stripped + genes/value aligned
    g0, v0 = t.column("genes")[0].as_py(), t.column("value")[0].as_py()
    assert len(g0) == len(v0) and 1 not in g0


def test_dataset_reads_preprocessed_cache(tmp_path):
    """TahoeDataset auto-detects the cache 'value' column (no re-strip/re-normalize)."""
    from eb_jepa.datasets.tahoe.dataset import TahoeConfig, TahoeDataset

    root, out = tmp_path / "tahoe", tmp_path / "cache"
    _write_tahoe_layout(root)
    pp.run(
        data_dir=str(root),
        out_dir=str(out),
        n_genes=200,
        n_bins=8,
        quantile_cells=0,
        val_frac=0.5,
        seed=0,
    )
    ds = TahoeDataset(TahoeConfig(data_dir=str(out / "train"), n_genes=200))
    assert ds._cached is True
    item = ds[0]
    assert item["gene_token_ids"].shape == item["values"].shape
    assert (item["values"] >= 0).all()  # already CP10k+log1p
    assert item["drug"] in ("DMSO_TF", "drugA")
    assert item["cell_line_id"].startswith("CVCL_")


def test_preprocess_liver_filter_e2e(tmp_path):
    root, out = tmp_path / "tahoe", tmp_path / "cache_liver"
    _write_tahoe_layout(root)
    pp.run(
        data_dir=str(root),
        out_dir=str(out),
        n_genes=200,
        n_bins=8,
        quantile_cells=0,
        liver_only=True,
        val_frac=0.5,
        split_by="sample",
        seed=0,
    )
    import pyarrow.parquet as pq

    # only the Liver line (CVCL_0001) survives
    for part in glob.glob(str(out / "*" / "*.parquet")):
        t = pq.read_table(part)
        assert set(t.column("cell_line_id").to_pylist()) <= {"CVCL_0001"}


def test_fit_boundaries_from_cells():
    torch.manual_seed(0)
    cells = [
        (torch.randint(0, 50, (30,)), torch.randint(1, 100, (30,)).float())
        for _ in range(200)
    ]
    binner = fit_boundaries_from_cells(cells, n_genes=50, n_bins=8)
    assert isinstance(binner, QuantileBinner)
    assert binner.boundaries.shape == (50, 7)
    bins = binner.bin(torch.arange(50), torch.ones(50))
    assert bins.min() >= 0 and bins.max() <= 7


def test_gene_histogram_matches_exact_quantiles():
    """Streaming histogram boundaries should approximate the exact fit closely."""
    torch.manual_seed(0)
    cells = [
        (torch.randint(0, 20, (40,)), torch.randint(1, 200, (40,)).float())
        for _ in range(500)
    ]
    n_bins = 16
    exact = fit_boundaries_from_cells(cells, n_genes=20, n_bins=n_bins)
    hist = fit_histogram_from_cells(cells, n_genes=20, n_hist_bins=4096)
    approx = hist.binner(n_bins)
    assert isinstance(hist, GeneHistogram)
    assert approx.boundaries.shape == exact.boundaries.shape == (20, n_bins - 1)
    # grid is fine (range 0..10 over 4096 bins) -> edges within ~1 grid step
    assert torch.allclose(approx.boundaries, exact.boundaries, atol=0.05)


def test_gene_histogram_save_load_and_rebin(tmp_path):
    """Saved histogram round-trips and re-derives boundaries for any n_bins."""
    torch.manual_seed(1)
    cells = [(torch.randint(0, 10, (30,)), torch.randint(1, 100, (30,)).float())]
    hist = fit_histogram_from_cells(cells, n_genes=10)
    hist.save(tmp_path / "hist")
    loaded = GeneHistogram.load(tmp_path / "hist")
    assert loaded.n_observed == hist.n_observed
    # same histogram -> identical boundaries; different n_bins changes width only
    assert torch.equal(loaded.quantile_boundaries(32), hist.quantile_boundaries(32))
    assert loaded.quantile_boundaries(64).shape == (10, 63)
