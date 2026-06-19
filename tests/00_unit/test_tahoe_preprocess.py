"""Preprocess helpers: group-level split (no leakage), liver filter, stats fit."""

import torch

from eb_jepa.datasets.tahoe.normalizer import QuantileBinner
from eb_jepa.datasets.tahoe.preprocess import (
    assign_split,
    fit_boundaries_from_cells,
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
