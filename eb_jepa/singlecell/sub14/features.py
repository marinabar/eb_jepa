"""Protein-coding gene features from eb_jepa's frozen gene-embedding cache.

Subliminal 1.4's gene identity = DNA (Evo2) + protein (ESMC) features for
protein-coding genes only. eb_jepa already ships exactly these tensors in
its ``gene_emb_cache`` (built by ``scripts/build_gene_embeddings.py``):
``esmc.npy`` (protein, coding only) + ``evo2.npy`` (DNA, all genes) +
``index.parquet`` (token_id, is_coding, esmc_row, evo2_row).

:func:`load_pc_features` filters that cache to the protein-coding subset
and returns, aligned to a PC-local vocabulary ``[0, n_pc)``:
- ``protein_features`` ``(n_pc, d_esmc)`` and ``dna_features``
  ``(n_pc, d_evo2)`` (Evo2 row, zeros if the gene had no usable
  transcript),
- ``token_to_pc_local`` ``(vocab_size,)`` int64 mapping a raw Tahoe
  ``token_id`` to its PC-local index (``-1`` for non-PC genes), used by
  the collator to slice each sparse cell onto the PC vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class PCFeatures:
    protein_features: torch.Tensor  # (n_pc, d_esmc)
    dna_features: torch.Tensor | None  # (n_pc, d_evo2) or None
    token_to_pc_local: np.ndarray  # (vocab_size,) int64, -1 = not protein-coding
    global_token_ids: np.ndarray  # (n_pc,) raw Tahoe token_ids of the PC genes
    n_pc_genes: int


def load_pc_features(cache_dir: str | Path) -> PCFeatures:
    """Load + filter the eb_jepa gene_emb_cache to the protein-coding subset."""
    import pyarrow.parquet as pq

    cache_dir = Path(cache_dir)
    index = pq.read_table(cache_dir / "index.parquet").to_pydict()
    esmc_raw = np.load(cache_dir / "esmc.npy")  # (N_coding, d_esmc)
    evo2_raw = np.load(cache_dir / "evo2.npy")  # (N_genes, d_evo2)

    token_ids = np.asarray(index["token_id"], dtype=np.int64)
    is_coding = np.asarray(index["is_coding"], dtype=bool)
    esmc_row = np.asarray(index["esmc_row"], dtype=np.int64)
    evo2_row = np.asarray(index["evo2_row"], dtype=np.int64)
    vocab_size = int(token_ids.max()) + 1

    # Protein-coding = flagged coding AND has a real ESMC row.
    pc_sel = is_coding & (esmc_row >= 0)
    pc_token_ids = token_ids[pc_sel]
    pc_esmc_row = esmc_row[pc_sel]
    pc_evo2_row = evo2_row[pc_sel]
    n_pc = int(pc_sel.sum())

    protein = torch.from_numpy(esmc_raw[pc_esmc_row]).float()  # (n_pc, d_esmc)
    d_evo2 = evo2_raw.shape[1]
    dna = torch.zeros(n_pc, d_evo2, dtype=torch.float32)
    has_evo2 = pc_evo2_row >= 0
    if has_evo2.any():
        dna[has_evo2] = torch.from_numpy(evo2_raw[pc_evo2_row[has_evo2]]).float()

    token_to_pc_local = np.full(vocab_size, -1, dtype=np.int64)
    token_to_pc_local[pc_token_ids] = np.arange(n_pc, dtype=np.int64)

    return PCFeatures(
        protein_features=protein,
        dna_features=dna,
        token_to_pc_local=token_to_pc_local,
        global_token_ids=pc_token_ids,
        n_pc_genes=n_pc,
    )


def random_pc_features(
    n_pc: int = 2000,
    vocab_size: int = 62713,
    d_esmc: int = 1152,
    d_evo2: int = 4096,
    seed: int = 0,
) -> PCFeatures:
    """Random PC features for smoke / unit tests (no real cache needed).

    Picks ``n_pc`` random token ids in ``[3, vocab_size)`` (ids 0-2 are
    reserved special tokens in Tahoe) as the protein-coding vocabulary.
    """
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    pc_token_ids = np.sort(rng.choice(np.arange(3, vocab_size), size=n_pc, replace=False))
    protein = torch.randn(n_pc, d_esmc, generator=g)
    dna = torch.randn(n_pc, d_evo2, generator=g)
    token_to_pc_local = np.full(vocab_size, -1, dtype=np.int64)
    token_to_pc_local[pc_token_ids] = np.arange(n_pc, dtype=np.int64)
    return PCFeatures(
        protein_features=protein,
        dna_features=dna,
        token_to_pc_local=token_to_pc_local,
        global_token_ids=pc_token_ids,
        n_pc_genes=n_pc,
    )
