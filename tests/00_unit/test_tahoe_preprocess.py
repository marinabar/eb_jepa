"""Preprocess helpers: group-level split (no leakage), liver filter, stats fit."""

import torch

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
