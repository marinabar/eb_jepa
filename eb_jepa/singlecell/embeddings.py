"""Gene-token embeddings for the single-cell encoder.

A gene token embedding is the **sum** of three components projected to d_model
(CLAUDE.md "Token embeddings"):
  1. ESMC protein embedding (coding genes only; zero otherwise),
  2. Evo2 DNA embedding (all genes),
  3. count embedding (mode A continuous MLP, or mode B quantile-bin table).

ESMC/Evo2 vectors are **precomputed and frozen** (built offline by
``scripts/build_gene_embeddings.py``), indexed by ``token_id``. Only the linear
projections to d_model and the count embedding are learned.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class CountEmbedding(nn.Module):
    """Encode the CP10k+log1p count to d_model (mode A continuous or mode B bins).

    - mode A: MLP on the log scalar; masked counts use a learned MASK vector.
    - mode B: ``Embedding(n_bins + 1, d_model)``; the last row (index ``n_bins``)
      is the MASK bin (the collator sets masked positions to ``n_bins``).
    """

    def __init__(self, d_model: int, count_mode: str = "A", n_bins: int = 64):
        super().__init__()
        assert count_mode in ("A", "B")
        self.count_mode = count_mode
        self.n_bins = n_bins
        if count_mode == "A":
            self.mlp = nn.Sequential(
                nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model)
            )
            self.mask_vector = nn.Parameter(torch.zeros(d_model))
        else:
            self.table = nn.Embedding(n_bins + 1, d_model)  # +1 = MASK bin

    def forward(
        self,
        count_value: torch.Tensor | None = None,
        count_bin: torch.Tensor | None = None,
        count_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.count_mode == "A":
            emb = self.mlp(count_value.unsqueeze(-1))  # [..., d_model]
            if count_mask is not None:
                emb = torch.where(count_mask.unsqueeze(-1), self.mask_vector, emb)
            return emb
        return self.table(count_bin)  # masked positions already carry bin == n_bins


class GeneTokenEmbedding(nn.Module):
    """Compose frozen ESMC+Evo2 lookups (projected) with the count embedding.

    Args:
        n_genes: vocabulary size.
        d_model: model width.
        esmc_table: [n_genes, d_esmc] frozen (zeros for non-coding).
        evo2_table: [n_genes, d_evo2] frozen.
        coding_mask: [n_genes] bool (True if the gene has a protein term).
        count_mode / n_bins: passed to CountEmbedding.
    The ESMC/Evo2 tables are registered as non-trainable buffers.
    """

    def __init__(
        self,
        n_genes: int,
        d_model: int,
        esmc_table: torch.Tensor,
        evo2_table: torch.Tensor,
        coding_mask: torch.Tensor,
        count_mode: str = "A",
        n_bins: int = 64,
    ):
        super().__init__()
        assert esmc_table.shape[0] == n_genes and evo2_table.shape[0] == n_genes
        self.n_genes = n_genes
        self.d_model = d_model
        self.register_buffer("esmc_table", esmc_table.float(), persistent=False)
        self.register_buffer("evo2_table", evo2_table.float(), persistent=False)
        self.register_buffer("coding_mask", coding_mask.bool(), persistent=False)
        self.esmc_proj = nn.Linear(esmc_table.shape[1], d_model, bias=False)
        self.evo2_proj = nn.Linear(evo2_table.shape[1], d_model, bias=False)
        self.count_emb = CountEmbedding(d_model, count_mode, n_bins)

    def forward(
        self,
        gene_token_ids: torch.Tensor,
        count_value: torch.Tensor | None = None,
        count_bin: torch.Tensor | None = None,
        count_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """gene_token_ids: [..., L] long -> token embeddings [..., L, d_model]."""
        ids = gene_token_ids.long()
        esmc = self.esmc_proj(self.esmc_table[ids])  # [..., L, d_model]
        # zero the protein term for non-coding genes
        esmc = esmc * self.coding_mask[ids].unsqueeze(-1)
        evo2 = self.evo2_proj(self.evo2_table[ids])
        count = self.count_emb(count_value, count_bin, count_mask)
        return esmc + evo2 + count

    # ---- constructors ---------------------------------------------------
    @classmethod
    def from_cache(
        cls,
        cache_dir: str | Path,
        d_model: int,
        count_mode: str = "A",
        n_bins: int = 64,
    ) -> "GeneTokenEmbedding":
        """Load the frozen ESMC/Evo2 cache produced by build_gene_embeddings.py."""
        import pyarrow.parquet as pq

        cache_dir = Path(cache_dir)
        meta = json.loads((cache_dir / "metadata.json").read_text())
        index = pq.read_table(cache_dir / "index.parquet").to_pydict()
        esmc_raw = np.load(cache_dir / "esmc.npy")  # [N_coding, d_esmc]
        evo2_raw = np.load(cache_dir / "evo2.npy")  # [N_genes, d_evo2]
        n_genes = len(index["token_id"])
        d_esmc, d_evo2 = esmc_raw.shape[1], evo2_raw.shape[1]

        esmc_table = torch.zeros(n_genes, d_esmc, dtype=torch.float32)
        evo2_table = torch.zeros(n_genes, d_evo2, dtype=torch.float32)
        coding_mask = torch.zeros(n_genes, dtype=torch.bool)
        for tok, is_coding, esmc_row, evo2_row in zip(
            index["token_id"], index["is_coding"], index["esmc_row"], index["evo2_row"]
        ):
            evo2_table[tok] = torch.from_numpy(evo2_raw[evo2_row])
            if is_coding and esmc_row >= 0:
                esmc_table[tok] = torch.from_numpy(esmc_raw[esmc_row])
                coding_mask[tok] = True
        assert d_evo2 == meta.get("d_evo2", d_evo2)
        return cls(
            n_genes, d_model, esmc_table, evo2_table, coding_mask, count_mode, n_bins
        )

    @classmethod
    def random(
        cls,
        n_genes: int,
        d_model: int,
        d_esmc: int = 1280,
        d_evo2: int = 512,
        count_mode: str = "A",
        n_bins: int = 64,
        coding_frac: float = 0.7,
        seed: int = 0,
    ) -> "GeneTokenEmbedding":
        """Random frozen tables — for unit tests / smoke runs before the real cache."""
        g = torch.Generator().manual_seed(seed)
        esmc = torch.randn(n_genes, d_esmc, generator=g)
        evo2 = torch.randn(n_genes, d_evo2, generator=g)
        coding = torch.rand(n_genes, generator=g) < coding_frac
        esmc = esmc * coding.unsqueeze(-1)
        return cls(n_genes, d_model, esmc, evo2, coding, count_mode, n_bins)
