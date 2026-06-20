"""Hallmark pathway assets + loader (eb_jepa.datasets.tahoe.pathways)."""

import torch

from eb_jepa.datasets.tahoe.pathways import (
    hallmark_metadata,
    hallmark_to_token_ids,
    load_hallmark,
    membership_matrix,
    symbol_to_token_map,
)


class TestHallmarkAssets:
    def test_fifty_sets_loaded(self):
        sets = load_hallmark()
        assert len(sets) == 50
        assert all(name.startswith("HALLMARK_") for name in sets)

    def test_known_set_present_and_sane(self):
        sets = load_hallmark()
        assert "HALLMARK_APOPTOSIS" in sets
        genes = sets["HALLMARK_APOPTOSIS"]
        assert len(genes) > 100  # ~161 in v2024.1
        assert "CASP3" in genes  # canonical apoptosis gene

    def test_metadata_provenance(self):
        meta = hallmark_metadata()
        assert meta["n_sets"] == 50
        assert meta["id_type"] == "gene_symbol"
        assert "msigdb" in meta["source"].lower()


class TestTokenMapping:
    def test_symbol_to_token_map_dedups_and_uppercases(self):
        gm = [("CASP3", 10), ("casp3", 99), ("TP53", 11), (None, 5)]
        m = symbol_to_token_map(gm)
        assert m["CASP3"] == 10  # first occurrence wins, case-folded
        assert m["TP53"] == 11
        assert None not in m

    def test_hallmark_to_token_ids_drops_unknown(self):
        # only two symbols are in-vocab -> only those map
        s2t = {"CASP3": 100, "ANXA1": 200}
        mapped = hallmark_to_token_ids(s2t)
        ids = mapped["HALLMARK_APOPTOSIS"]
        assert set(ids).issubset({100, 200})
        assert len(ids) >= 1
        assert ids == sorted(set(ids))

    def test_membership_matrix_shape_and_weights(self):
        s2t = {"CASP3": 100, "ANXA1": 200, "TP53": 300}
        m, names = membership_matrix(s2t, n_genes=512)
        assert m.shape == (50, 512)
        assert len(names) == 50 and names == sorted(names)
        ap = names.index("HALLMARK_APOPTOSIS")
        # CASP3/ANXA1 belong to apoptosis -> their columns are 1.0
        assert m[ap, 100] == 1.0 and m[ap, 200] == 1.0
        assert m.max() == 1.0 and m.min() == 0.0

    def test_membership_matrix_counts_pathway_score(self):
        s2t = {"CASP3": 5, "ANXA1": 7}
        m, names = membership_matrix(s2t, n_genes=16)
        counts = torch.zeros(16)
        counts[5] = 2.0
        counts[7] = 3.0
        scores = m @ counts  # pathway counts = weighted sum of member-gene counts
        ap = names.index("HALLMARK_APOPTOSIS")
        assert scores[ap] == 5.0
