"""Subliminal 1.4 model — encoder-only cell representation (faithful port).

Composes the protein-coding gene-identity embedding + thermometer count
embedding, runs the sigmoid-attention encoder, and projects the
``[CELL]`` representation through the JEPA / SIGReg projector. The
representation of interest (for probing / downstream) is the
*pre-projection* ``[CELL]`` output; the projection is used only by the
loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from eb_jepa.singlecell.layers import RMSNorm
from eb_jepa.singlecell.sub14.embeddings import (
    ProteinCodingGeneEmbeddings,
    QuantileThermometerCountEmbedding,
)
from eb_jepa.singlecell.sub14.encoder import EncoderSub14


@dataclass
class Subliminal14Output:
    cell_representation: Tensor  # (B, D) pre-projection [CELL] output
    cell_projection: Tensor  # (B, latent) projected (JEPA / SIGReg)


class Subliminal14(nn.Module):
    """Encoder-only cell representation model on protein-coding genes."""

    def __init__(
        self,
        *,
        n_pc_genes: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        latent_dim: int = 128,
        num_bins: int = 16,
        max_genes_per_cell: int = 512,
        dna_features: Optional[Tensor] = None,
        protein_features: Optional[Tensor] = None,
        freeze_features: bool = True,
        attention_activation: str = "sigmoid",
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.n_pc_genes = n_pc_genes
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.num_bins = num_bins
        self.max_genes_per_cell = max_genes_per_cell

        self.gene_embedding = ProteinCodingGeneEmbeddings(
            n_pc_genes=n_pc_genes,
            d_model=d_model,
            dna_features=dna_features,
            protein_features=protein_features,
            freeze_features=freeze_features,
        )
        self.count_embedding = QuantileThermometerCountEmbedding(num_bins=num_bins, d_model=d_model)

        self.encoder = EncoderSub14(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            expected_seq_len=max_genes_per_cell + 1,
            attention_activation=attention_activation,
            grad_checkpoint=grad_checkpoint,
        )

        # JEPA / SIGReg projector (1.4 spec): 2 hidden layers @ d_model with
        # RMSNorm + SiLU, bias-free latent output. NOT L2-normalised.
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, latent_dim, bias=False),
        )

    def forward(self, gene_ids: Tensor, bin_ids: Tensor, padding_mask: Tensor) -> Subliminal14Output:
        """Run one view.

        ``gene_ids``: (B, G) PC-local ids ([0, n_pc_genes], pad = n_pc_genes).
        ``bin_ids``: (B, G) thermometer bins ([0, num_bins], pad = num_bins).
        ``padding_mask``: (B, G) bool, True = padded slot.
        """
        gene_emb = self.gene_embedding(gene_ids)
        count_emb = self.count_embedding(bin_ids)
        embeddings = gene_emb + count_emb

        cls_pad = torch.zeros(
            padding_mask.size(0), 1, device=padding_mask.device, dtype=padding_mask.dtype
        )
        encoder_pad_mask = torch.cat([cls_pad, padding_mask], dim=1)

        cell_rep = self.encoder(embeddings, padding_mask=encoder_pad_mask)
        cell_proj = self.projection(cell_rep)
        return Subliminal14Output(cell_representation=cell_rep, cell_projection=cell_proj)

    @torch.no_grad()
    def encode(self, gene_ids: Tensor, bin_ids: Tensor, padding_mask: Tensor) -> Tensor:
        """Pre-projection cell representation (for probing / t-SNE)."""
        return self.forward(gene_ids, bin_ids, padding_mask).cell_representation
