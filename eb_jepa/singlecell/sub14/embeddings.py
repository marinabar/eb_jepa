"""Gene-identity + count embeddings for Subliminal 1.4 (faithful port).

- :class:`ProteinCodingGeneEmbeddings`: gene identity = frozen DNA (Evo2)
  + protein (ESMC) feature tables, each through ``RMSNorm + Linear``,
  summed and RMSNorm'd. A sentinel pad row (index ``n_pc_genes``) emits
  zero. No learned per-gene table.
- :class:`QuantileThermometerCountEmbedding`: ``num_bins`` learned bin
  embeddings; a gene in bin ``k`` is encoded as the cumulative sum
  ``sum_{i<=k} e_i`` (thermometer). The pad index ``num_bins`` emits zero.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

# torch.nn.RMSNorm exists in torch >= 2.4; use it here so the frozen-feature
# projection is independent of eb_jepa's float32-cast RMSNorm.
_RMSNorm = nn.RMSNorm


class _PrecomputedGeneFeatureProjection(nn.Module):
    """RMSNorm + Linear projection of a frozen ``(n_genes, feat_dim)`` table."""

    def __init__(self, features: Tensor, d_model: int, *, freeze: bool, pad_row: bool) -> None:
        super().__init__()
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {tuple(features.shape)}")
        _, feat_dim = features.shape

        if pad_row:
            zero_row = torch.zeros(1, feat_dim, dtype=features.dtype)
            features = torch.cat([features, zero_row], dim=0)

        if freeze:
            # persistent=False: the frozen ESMC/Evo2 tables are rebuilt from the
            # gene_emb_cache at construction, so they stay out of the checkpoint
            # (keeps it small; matches eb_jepa's GeneTokenEmbedding).
            self.register_buffer("features", features, persistent=False)
        else:
            self.features = nn.Parameter(features)

        self.in_norm = _RMSNorm(feat_dim)
        self.projection = nn.Linear(feat_dim, d_model, bias=False)

    def forward(self, gene_ids: Tensor) -> Tensor:
        feats = self.features[gene_ids]
        feats = feats.to(dtype=self.projection.weight.dtype)
        return self.projection(self.in_norm(feats))


class ProteinCodingGeneEmbeddings(nn.Module):
    """Sum-pooled DNA + protein gene-identity embeddings.

    ``n_pc_genes`` is both the vocabulary size and the pad gene id; an
    extra zero row is appended to each feature table so ``forward(pad)``
    returns zero. At least one of ``dna_features`` / ``protein_features``
    must be given.
    """

    def __init__(
        self,
        n_pc_genes: int,
        d_model: int,
        *,
        dna_features: Optional[Tensor] = None,
        protein_features: Optional[Tensor] = None,
        freeze_features: bool = True,
    ) -> None:
        super().__init__()
        if dna_features is None and protein_features is None:
            raise ValueError("requires at least one of dna_features / protein_features")
        self.n_pc_genes = n_pc_genes
        self.pad_index = n_pc_genes

        for name, feats in (("dna_features", dna_features), ("protein_features", protein_features)):
            if feats is not None and feats.size(0) != n_pc_genes:
                raise ValueError(f"{name} has {feats.size(0)} rows but n_pc_genes={n_pc_genes}")

        self.dna: Optional[_PrecomputedGeneFeatureProjection] = None
        self.protein: Optional[_PrecomputedGeneFeatureProjection] = None
        if dna_features is not None:
            self.dna = _PrecomputedGeneFeatureProjection(
                dna_features, d_model, freeze=freeze_features, pad_row=True
            )
        if protein_features is not None:
            self.protein = _PrecomputedGeneFeatureProjection(
                protein_features, d_model, freeze=freeze_features, pad_row=True
            )
        self.out_norm = _RMSNorm(d_model)

    def forward(self, gene_ids: Tensor) -> Tensor:
        """``gene_ids``: (B, G) PC-local ids (with ``n_pc_genes`` for pad)."""
        out: Optional[Tensor] = None
        if self.dna is not None:
            out = self.dna(gene_ids)
        if self.protein is not None:
            prot = self.protein(gene_ids)
            out = prot if out is None else out + prot
        assert out is not None
        return self.out_norm(out)


class QuantileThermometerCountEmbedding(nn.Module):
    """Thermometer-coded count embedding over per-cell quantile bins."""

    def __init__(self, num_bins: int = 16, d_model: int = 768) -> None:
        super().__init__()
        if num_bins < 1:
            raise ValueError(f"num_bins must be >= 1, got {num_bins}")
        self.num_bins = num_bins
        self.d_model = d_model
        self.pad_index = num_bins  # sentinel; emits zero

        self.bin_embeddings = nn.Embedding(num_bins, d_model)
        nn.init.trunc_normal_(self.bin_embeddings.weight, std=0.02)

    def forward(self, bin_indices: Tensor) -> Tensor:
        """``bin_indices``: (B, G) in [0, num_bins-1], or ``num_bins`` for pad."""
        bin_emb = self.bin_embeddings.weight  # (K, D)
        cumulative = torch.cumsum(bin_emb, dim=0)  # (K, D)
        zero_row = torch.zeros(1, bin_emb.size(1), device=bin_emb.device, dtype=bin_emb.dtype)
        thermometer_table = torch.cat([cumulative, zero_row], dim=0)  # (K+1, D)
        return thermometer_table[bin_indices]
