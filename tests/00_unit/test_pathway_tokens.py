"""Pathway tokens (CLAUDE.md "Pathways"): collator emits per-cell hallmark counts +
per-view dropout masks; the encoder appends P pathway tokens (identity + count),
attends over them, but excludes them from the mean-pool readout. Constant shapes.
"""

import torch

from eb_jepa.datasets.tahoe.dataset import TahoeCollator, TahoeConfig
from eb_jepa.singlecell.embeddings import GeneTokenEmbedding, PathwayEmbedding
from eb_jepa.singlecell.encoder import SingleCellEncoder, encode_views

N_GENES, D_MODEL, L, P = 50, 32, 16, 8


def _cell(g):
    gen = torch.Generator().manual_seed(g)
    tok = torch.randperm(N_GENES - 2, generator=gen)[:g] + 2
    val = torch.rand(g, generator=gen) * 2
    return {
        "gene_token_ids": tok,
        "values": val,
        "drug": "d",
        "sample": "s",
        "cell_line_id": "CVCL_0001",
        "organ": "Liver",
        "moa_fine": "x",
        "plate": "plate1",
        "canonical_smiles": "CCO",
        "log_conc": -7.0,
    }


def _membership(seed=0):
    # random sparse [P, n_genes] hallmark-style membership (each gene in ~1 pathway)
    g = torch.Generator().manual_seed(seed)
    m = (torch.rand(P, N_GENES, generator=g) < 0.15).float()
    return m


def _cfg(**kw):
    kw.setdefault("pathway_drop_frac", 0.5)
    return TahoeConfig(data_dir="", L=L, n_genes=N_GENES, use_pathways=True, **kw)


def _collate(cfg, n=6, membership=None):
    cells = [_cell(5 + i) for i in range(n)]
    return TahoeCollator(cfg, membership=membership)(cells)


def _build(n_pathways=P, use_cls=False, readout="meanpool"):
    embed = GeneTokenEmbedding.random(N_GENES, D_MODEL, d_esmc=16, d_evo2=12)
    return SingleCellEncoder(
        embed,
        d_model=D_MODEL,
        n_layers=2,
        n_heads=4,
        use_cls=use_cls,
        readout=readout,
        n_pathways=n_pathways,
    )


class TestPathwayCollate:
    def test_emits_count_and_mask_shapes(self):
        cfg = _cfg(n_views=3)
        batch = _collate(cfg, n=6, membership=_membership())
        assert batch["pathway_count"].shape == (3, 6, P)
        assert batch["pathway_mask"].shape == (3, 6, P)
        assert batch["pathway_mask"].dtype == torch.bool

    def test_count_is_membership_times_dense(self):
        cfg = _cfg(n_views=2)
        m = _membership()
        cells = [_cell(5 + i) for i in range(4)]
        batch = TahoeCollator(cfg, membership=m)(cells)
        # recompute pathway count for cell 0 independently
        c0 = cells[0]
        dense = torch.zeros(N_GENES)
        dense[c0["gene_token_ids"].long()] = c0["values"].float()
        expected = m @ dense  # [P]
        assert torch.allclose(batch["pathway_count"][0, 0], expected, atol=1e-5)
        # the count is shared across views (a cell property)
        assert torch.allclose(batch["pathway_count"][0, 0], batch["pathway_count"][1, 0])

    def test_drop_frac_controls_keep_rate(self):
        cfg = _cfg(n_views=4, pathway_drop_frac=0.0)
        batch = _collate(cfg, n=8, membership=_membership())
        assert batch["pathway_mask"].all()  # nothing dropped at drop_frac=0


class TestPathwayEncoder:
    def test_encode_views_shape_with_pathways(self):
        cfg = _cfg(n_views=3)
        enc = _build()
        reps = encode_views(enc, _collate(cfg, n=6, membership=_membership()))
        assert reps.shape == (3, 6, D_MODEL) and torch.isfinite(reps).all()

    @torch.no_grad()
    def test_pathway_dropout_changes_representation(self):
        # pathway tokens influence gene tokens via attention, so dropping them
        # (mask=False) changes the pooled representation.
        cfg = _cfg(n_views=1)
        enc = _build().eval()
        batch = _collate(cfg, n=5, membership=_membership())
        all_on = dict(batch, pathway_mask=torch.ones_like(batch["pathway_mask"]))
        all_off = dict(batch, pathway_mask=torch.zeros_like(batch["pathway_mask"]))
        r_on = encode_views(enc, all_on)
        r_off = encode_views(enc, all_off)
        assert not torch.allclose(r_on, r_off, atol=1e-4)

    @torch.no_grad()
    def test_meanpool_excludes_pathways_and_padding(self):
        # The readout pools over gene positions only: corrupting padded gene slots OR
        # pathway counts must not change a meanpool that excludes them... but pathways
        # DO feed attention, so only padding is guaranteed inert. Check padding here.
        cfg = _cfg(n_views=1)
        enc = _build().eval()
        batch = _collate(cfg, n=5, membership=_membership())
        r1 = encode_views(enc, batch)
        ids = batch["gene_token_ids"].clone()
        ids[~batch["pad_mask"]] = 3
        r2 = encode_views(enc, dict(batch, gene_token_ids=ids))
        assert torch.allclose(r1, r2, atol=1e-5)

    def test_pathway_identity_receives_grad(self):
        cfg = _cfg(n_views=2)
        enc = _build()
        reps = encode_views(enc, _collate(cfg, n=4, membership=_membership()))
        reps.sum().backward()
        assert enc.pathway_embed.identity.weight.grad is not None

    def test_disabled_pathways_no_pathway_module(self):
        enc = _build(n_pathways=0)
        assert enc.pathway_embed is None

    @torch.no_grad()
    def test_constant_shape_across_dropout(self):
        # different dropout draws -> identical output shape (compile-safe)
        cfg = _cfg(n_views=2)
        enc = _build().eval()
        m = _membership()
        s1 = encode_views(enc, _collate(cfg, n=4, membership=m)).shape
        s2 = encode_views(enc, _collate(cfg, n=4, membership=m)).shape
        assert s1 == s2 == (2, 4, D_MODEL)


class TestPathwayEmbedding:
    def test_identity_plus_count(self):
        pe = PathwayEmbedding(P, D_MODEL)
        pcount = torch.rand(4, P) * 5
        out = pe(pcount)
        assert out.shape == (4, P, D_MODEL) and torch.isfinite(out).all()
