"""Unit tests for the hierarchical (pathway-level) Subliminal 1.4 encoder.

Runs on CPU in fp32 with random PC features + random hallmark membership —
no gene-embedding cache or GPU required. Covers: membership routing,
forward/backward parity with the flat model, the masked cross-attention
semantics, and warm-start compatibility (a flat sub14 checkpoint loads the
gene level key-for-key and leaves the pathway level fresh).
"""
from __future__ import annotations

import numpy as np
import torch

from eb_jepa.singlecell.sub14.collator import Sub14Collator
from eb_jepa.singlecell.sub14.features import (
    random_pathway_membership,
    random_pc_features,
)
from eb_jepa.singlecell.sub14.hierarchical import (
    HierarchicalSubliminal14,
    HierCrossAttention,
)
from eb_jepa.singlecell.sub14.model import Subliminal14
from eb_jepa.singlecell.sub14.optim import build_muon_adamw_optimizer
from eb_jepa.singlecell.sub14.sigreg import SIGReg


def _make_cells(pc, n_cells=6, min_g=40, max_g=120, seed=0):
    rng = np.random.default_rng(seed)
    cells = []
    for _ in range(n_cells):
        g = int(rng.integers(min_g, max_g))
        toks = rng.choice(pc.global_token_ids, size=g, replace=False)
        counts = rng.integers(1, 200, size=g).astype(np.float32)
        cells.append(
            {
                "gene_token_ids": torch.from_numpy(toks).long(),
                "raw_counts": torch.from_numpy(counts).float(),
                "values": torch.from_numpy(np.log1p(counts)).float(),
                "drug": "DMSO_TF" if rng.random() < 0.5 else "DrugX",
                "organ": "Liver" if rng.random() < 0.5 else "Lung",
                "cell_line_id": "CVCL_0001",
                "sample": "s1",
                "moa_fine": "moaA",
                "plate": "plate1",
                "canonical_smiles": "C",
                "log_conc": float("nan"),
            }
        )
    return cells


def _kwargs(pc, num_bins, gpb):
    return dict(
        n_pc_genes=pc.n_pc_genes,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        latent_dim=16,
        num_bins=num_bins,
        max_genes_per_cell=num_bins * gpb,
        dna_features=pc.dna_features,
        protein_features=pc.protein_features,
        attention_activation="sigmoid",
    )


def _collate(pc, num_bins, gpb, v, n_cells=6):
    coll = Sub14Collator(
        token_to_pc_local=pc.token_to_pc_local,
        n_pc_genes=pc.n_pc_genes,
        num_bins=num_bins,
        genes_per_bin=gpb,
        num_views=v,
        binomial_subsample={"enabled": True, "p_min": 0.5, "p_max": 0.9},
        seed=0,
    )
    return coll(_make_cells(pc, n_cells=n_cells))


def test_hier_cross_attention_masking():
    """A fully-masked query row emits exactly zero (sigmoid, no info)."""
    torch.manual_seed(0)
    attn = HierCrossAttention(16, 4, activation="sigmoid")
    q = torch.randn(2, 3, 16)
    kv = torch.randn(2, 5, 16)
    keep = torch.ones(2, 3, 5, dtype=torch.bool)
    keep[0, 1] = False  # query (0,1) attends to nothing
    out = attn(q, kv, keep)
    assert out.shape == (2, 3, 16)
    assert torch.allclose(out[0, 1], torch.zeros(16), atol=1e-6)
    assert not torch.allclose(out[0, 0], torch.zeros(16))


def test_hier_membership_routing():
    """gene_to_pathway buffer routes valid genes and zeroes the pad sentinel."""
    pc = random_pc_features(n_pc=120, vocab_size=2000, d_esmc=8, d_evo2=8, seed=0)
    pm = random_pathway_membership(pc, n_pathways=6, members_per_pathway=20, seed=1)
    model = HierarchicalSubliminal14(pathway_membership=pm.membership, **_kwargs(pc, 8, 4))
    # pad row (index n_pc_genes) is all-zero membership
    assert torch.all(~model.gene_to_pathway[pc.n_pc_genes])
    # a known member gene routes to its pathway
    p, g = int(pm.membership.nonzero()[0, 0]), int(pm.membership.nonzero()[0, 1])
    assert bool(model.gene_to_pathway[g, p])


def test_hier_forward_and_step():
    torch.manual_seed(0)
    pc = random_pc_features(n_pc=300, vocab_size=5000, d_esmc=16, d_evo2=24, seed=2)
    num_bins, gpb, v = 8, 4, 3
    pm = random_pathway_membership(pc, n_pathways=8, members_per_pathway=40, seed=3)
    model = HierarchicalSubliminal14(pathway_membership=pm.membership, **_kwargs(pc, num_bins, gpb))
    assert model.encoder.hier_positions == [0]  # 2 layers -> one block after layer 0

    batch = _collate(pc, num_bins, gpb, v)
    sigreg = SIGReg(num_slices=32)
    opt = build_muon_adamw_optimizer(model, muon_lr=2e-4, adamw_lr=2e-4)

    projs, sig = [], torch.zeros(())
    for vw in range(v):
        out = model(batch["gene_ids"][vw], batch["bin_ids"][vw], batch["padding_mask"][vw])
        assert out.cell_representation.shape == (6, 32)
        assert out.cell_projection.shape == (6, 16)
        projs.append(out.cell_projection)
        sig = sig + sigreg(out.cell_projection)
    inv = torch.stack(
        [
            1.0 - torch.nn.functional.cosine_similarity(projs[i], projs[j], dim=-1).mean()
            for i in range(v)
            for j in range(v)
            if i != j
        ]
    ).mean()
    loss = inv + 0.4 * (sig / v)
    assert torch.isfinite(loss)
    opt.zero_grad()
    loss.backward()
    # gene level AND every pathway-level tensor must receive gradient
    assert model.gene_embedding.protein.projection.weight.grad is not None
    hier_params = [p for n, p in model.named_parameters() if "hier_blocks" in n]
    assert hier_params and all(p.grad is not None for p in hier_params)
    opt.step()


def test_hier_changes_representation():
    """The pathway level actually perturbs the [CELL] readout (not a no-op)."""
    torch.manual_seed(0)
    pc = random_pc_features(n_pc=300, vocab_size=5000, d_esmc=16, d_evo2=24, seed=2)
    num_bins, gpb = 8, 4
    pm = random_pathway_membership(pc, n_pathways=8, members_per_pathway=40, seed=3)
    kw = _kwargs(pc, num_bins, gpb)
    flat = Subliminal14(**kw)
    hier = HierarchicalSubliminal14(pathway_membership=pm.membership, **kw)
    # copy the shared gene-level weights so the ONLY difference is the pathway level
    hier.load_state_dict(flat.state_dict(), strict=False)

    batch = _collate(pc, num_bins, gpb, v=1)
    gi, bi, pad = batch["gene_ids"][0], batch["bin_ids"][0], batch["padding_mask"][0]
    flat.eval(); hier.eval()
    with torch.no_grad():
        rf = flat.encode(gi, bi, pad)
        rh = hier.encode(gi, bi, pad)
    assert rf.shape == rh.shape
    assert not torch.allclose(rf, rh, atol=1e-4)  # hierarchy changed the readout


def test_hier_warmstart_from_flat_checkpoint(tmp_path):
    """A flat sub14 checkpoint warm-starts the gene level key-for-key; the
    pathway level stays freshly initialised."""
    from eb_jepa.singlecell.sub14.load_checkpoint import load_subliminal14_checkpoint

    pc = random_pc_features(n_pc=120, vocab_size=2000, d_esmc=8, d_evo2=8, seed=0)
    pm = random_pathway_membership(pc, n_pathways=6, members_per_pathway=20, seed=1)
    kw = _kwargs(pc, 8, 4)
    flat = Subliminal14(**kw)
    ckpt = tmp_path / "encoder_final.pt"
    torch.save({"model": flat.state_dict(), "step": 0}, ckpt)

    hier = HierarchicalSubliminal14(pathway_membership=pm.membership, **kw)
    report = load_subliminal14_checkpoint(hier, str(ckpt), map_location="cpu", verbose=False)

    # gene body / projector / count table / [CELL] reused; pathway level fresh.
    assert any(k.startswith("encoder.layers") for k in report.loaded)
    assert "encoder.cell_token" in report.loaded
    assert all("hier_blocks" not in k for k in report.loaded)
    assert all("hier_blocks" in k for k in report.missing_in_ckpt) or any(
        "hier_blocks" in k for k in report.missing_in_ckpt
    )
    # a reused gene-layer weight now matches the flat checkpoint exactly
    w = "encoder.layers.0.ffn.w_gate.weight"
    assert torch.allclose(hier.state_dict()[w], flat.state_dict()[w])
