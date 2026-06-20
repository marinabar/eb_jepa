"""Smoke / shape tests for the Subliminal 1.4 port (eb_jepa/singlecell/sub14).

Runs on CPU in fp32 with random PC features — no gene-embedding cache or
GPU required. Exercises the full path: collator -> model forward -> SIGReg
+ pairwise-cosine JEPA loss -> optimizer step.
"""
from __future__ import annotations

import numpy as np
import torch

from eb_jepa.singlecell.sub14.collator import Sub14Collator
from eb_jepa.singlecell.sub14.features import random_pc_features
from eb_jepa.singlecell.sub14.model import Subliminal14
from eb_jepa.singlecell.sub14.optim import build_muon_adamw_optimizer
from eb_jepa.singlecell.sub14.sigreg import SIGReg


def _make_cells(pc, n_cells=6, min_g=40, max_g=120, seed=0):
    """Synthetic eb_jepa-style sparse cells drawn from the PC vocabulary."""
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


def test_collator_shapes():
    pc = random_pc_features(n_pc=300, vocab_size=5000, d_esmc=16, d_evo2=24, seed=1)
    num_bins, gpb, v = 8, 4, 3
    coll = Sub14Collator(
        token_to_pc_local=pc.token_to_pc_local,
        n_pc_genes=pc.n_pc_genes,
        num_bins=num_bins,
        genes_per_bin=gpb,
        num_views=v,
        binomial_subsample={"enabled": True, "p_min": 0.5, "p_max": 0.9},
        seed=0,
    )
    cells = _make_cells(pc, n_cells=5)
    out = coll(cells)
    G = num_bins * gpb
    assert out["gene_ids"].shape == (v, 5, G)
    assert out["bin_ids"].shape == (v, 5, G)
    assert out["padding_mask"].shape == (v, 5, G)
    # valid (non-pad) gene ids are PC-local in [0, n_pc); pad sentinel = n_pc
    gid = out["gene_ids"]
    valid = ~out["padding_mask"]
    assert int(gid[valid].max()) < pc.n_pc_genes
    assert torch.all(gid[out["padding_mask"]] == pc.n_pc_genes)
    # bin ids of valid slots are in [0, num_bins); pad sentinel = num_bins
    assert int(out["bin_ids"][valid].max()) < num_bins
    assert torch.all(out["bin_ids"][out["padding_mask"]] == num_bins)


def test_model_forward_and_step():
    torch.manual_seed(0)
    pc = random_pc_features(n_pc=300, vocab_size=5000, d_esmc=16, d_evo2=24, seed=2)
    num_bins, gpb, v = 8, 4, 3
    model = Subliminal14(
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
    coll = Sub14Collator(
        token_to_pc_local=pc.token_to_pc_local,
        n_pc_genes=pc.n_pc_genes,
        num_bins=num_bins,
        genes_per_bin=gpb,
        num_views=v,
        binomial_subsample={"enabled": True, "p_min": 0.5, "p_max": 0.9},
        seed=0,
    )
    batch = coll(_make_cells(pc, n_cells=6))
    sigreg = SIGReg(num_slices=32)
    opt = build_muon_adamw_optimizer(model, muon_lr=2e-4, adamw_lr=2e-4)

    projs, sig = [], torch.zeros(())
    for vw in range(v):
        out = model(batch["gene_ids"][vw], batch["bin_ids"][vw], batch["padding_mask"][vw])
        assert out.cell_representation.shape == (6, 32)
        assert out.cell_projection.shape == (6, 16)
        projs.append(out.cell_projection)
        sig = sig + sigreg(out.cell_projection)

    # pairwise-cosine invariance + SIGReg
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
    # gene-identity projections should receive gradient
    assert model.gene_embedding.protein.projection.weight.grad is not None
    opt.step()


def test_thermometer_pad_is_zero():
    from eb_jepa.singlecell.sub14.embeddings import QuantileThermometerCountEmbedding

    emb = QuantileThermometerCountEmbedding(num_bins=4, d_model=8)
    idx = torch.tensor([[0, 1, 4]])  # 4 = pad sentinel
    out = emb(idx)
    assert out.shape == (1, 3, 8)
    assert torch.allclose(out[0, 2], torch.zeros(8))  # pad -> zero
    # thermometer: bin 1 == cumulative of bins 0..1
    cum = torch.cumsum(emb.bin_embeddings.weight, dim=0)
    assert torch.allclose(out[0, 1], cum[1])
