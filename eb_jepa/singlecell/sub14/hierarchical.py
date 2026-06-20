"""Hierarchical (two-level) extension of the Subliminal 1.4 encoder.

Subliminal 1.4 is a single-level set transformer over gene tokens: genes
attend to each other and a learnable ``[CELL]`` token reads out the cell
representation. This module adds a **second, abstract level** built on the
50 MSigDB hallmark pathways — the inductive bias anticipated in CLAUDE.md
("a hierarchical encoder ... two attention levels, an abstract level over
gene-level tokens connected by pathway").

The construction follows the HRM / hierarchical-JEPA pattern, interleaved
with the existing gene-level blocks:

1. **pool** (genes -> pathways): one learned query token per hallmark
   pathway aggregates, by masked cross-attention, the hidden states of the
   genes that belong to it. A pathway only sees its own member genes.
2. **mix** (pathways <-> pathways): the pathway tokens run through one
   gene-level-identical pre-norm sigmoid-attention block, so distinct
   hallmark programs (apoptosis, glycolysis, ...) communicate at the
   abstract level.
3. **scatter** (pathways -> genes + CELL): every gene reads back, by masked
   cross-attention, from the pathway tokens it belongs to; the ``[CELL]``
   token reads from *all* present pathways. Genes that share a pathway thus
   exchange information through the abstract hierarchy, and the readout
   token is enriched by it.

The loss is unchanged — LeJEPA (pairwise-cosine invariance + SIGReg) on the
projected ``[CELL]`` token, exactly as in :mod:`eb_jepa.singlecell.sub14`.
Every gene-level module keeps the same name and shape as
:class:`~eb_jepa.singlecell.sub14.model.Subliminal14`, so a trained sub14
checkpoint warm-starts the hierarchical model key-for-key; only the new
``encoder.hier_blocks.*`` parameters are freshly initialised.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from eb_jepa.singlecell.layers import RMSNorm
from eb_jepa.singlecell.sub14.embeddings import (
    ProteinCodingGeneEmbeddings,
    QuantileThermometerCountEmbedding,
)
from eb_jepa.singlecell.sub14.encoder import PreNormSigmoidBlock
from eb_jepa.singlecell.sub14.model import Subliminal14Output


class HierCrossAttention(nn.Module):
    """Masked multi-head cross-attention between two token sets.

    Query and key/value come from different inputs, and a boolean
    ``keep_mask`` of shape ``(B, Lq, Lk)`` says which key each query may
    attend to (``True`` = attend). Like the gene-level encoder this uses the
    sigmoid activation by default (no sum-to-one constraint — a pathway may
    legitimately pool many or few genes, a gene may read many or few
    pathways); fully-masked query rows therefore emit zero, which is the
    correct "no information" residual.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        dropout: float = 0.0,
        expected_kv_len: int = 64,
        activation: str = "sigmoid",
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if activation not in ("sigmoid", "softmax"):
            raise ValueError(f"Unknown activation: {activation!r}; expected sigmoid | softmax")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = float(dropout)
        self.activation = activation

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        init_bias = -math.log(float(max(expected_kv_len, 1)))
        self.attn_bias = nn.Parameter(torch.full((n_heads,), init_bias))
        self._scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, query: Tensor, kv: Tensor, keep_mask: Tensor) -> Tensor:
        """``query``: (B, Lq, D); ``kv``: (B, Lk, D); ``keep_mask``: (B, Lq, Lk) bool."""
        b, lq, d = query.shape
        lk = kv.size(1)
        h, dh = self.n_heads, self.head_dim

        q = self.q_proj(query).view(b, lq, h, dh).transpose(1, 2)  # (B, H, Lq, Dh)
        k = self.k_proj(kv).view(b, lk, h, dh).transpose(1, 2)
        v = self.v_proj(kv).view(b, lk, h, dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self._scale  # (B, H, Lq, Lk)
        block = ~keep_mask[:, None, :, :]  # (B, 1, Lq, Lk) True = forbid

        if self.activation == "sigmoid":
            scores = scores + self.attn_bias.view(1, h, 1, 1)
            scores = scores.masked_fill(block, float("-inf"))
            attn = torch.sigmoid(scores)  # fully-masked rows -> all 0 -> output 0
        else:
            scores = scores.masked_fill(block, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            # rows with no allowed key -> softmax over all -inf is NaN; zero them.
            row_has_key = keep_mask.any(dim=-1)  # (B, Lq)
            attn = torch.where(row_has_key[:, None, :, None], attn, torch.zeros_like(attn))
        if self.training and self.dropout_p > 0.0:
            attn = torch.nn.functional.dropout(attn, p=self.dropout_p)

        out = torch.matmul(attn, v)  # (B, H, Lq, Dh)
        out = out.transpose(1, 2).contiguous().view(b, lq, d)
        return self.o_proj(out)


class PathwayHierarchyBlock(nn.Module):
    """One pool -> mix -> scatter round over the hallmark pathway level."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_pathways: int,
        *,
        dropout: float = 0.0,
        genes_per_view: int = 512,
        attention_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.n_pathways = n_pathways

        # Learned pathway-identity query tokens (apoptosis, glycolysis, ...),
        # the abstract-level analogue of the [CELL] token.
        self.pathway_identity = nn.Parameter(torch.zeros(1, n_pathways, d_model))
        nn.init.trunc_normal_(self.pathway_identity, std=0.02)

        # pool: pathways query their member genes.
        self.pool_q_norm = RMSNorm(d_model)
        self.pool_kv_norm = RMSNorm(d_model)
        self.pool_attn = HierCrossAttention(
            d_model, n_heads, dropout=dropout,
            expected_kv_len=max(genes_per_view // max(n_pathways, 1), 1),
            activation=attention_activation,
        )

        # mix: pathway <-> pathway self-attention (gene-level-identical block).
        self.pathway_mix = PreNormSigmoidBlock(
            d_model, n_heads, d_ff, dropout,
            expected_seq_len=n_pathways, attention_activation=attention_activation,
        )

        # scatter: genes + CELL read back from the pathway tokens.
        self.scatter_q_norm = RMSNorm(d_model)
        self.scatter_kv_norm = RMSNorm(d_model)
        self.scatter_attn = HierCrossAttention(
            d_model, n_heads, dropout=dropout,
            expected_kv_len=n_pathways, activation=attention_activation,
        )

    def forward(
        self,
        cell: Tensor,          # (B, 1, D)
        genes: Tensor,         # (B, G, D)
        gene_membership: Tensor,  # (B, G, P) bool, already AND-ed with gene validity
        pathway_present: Tensor,  # (B, P) bool, any member gene present in the view
    ) -> tuple[Tensor, Tensor]:
        b = genes.size(0)

        # 1. pool genes -> pathways (residual on the learned pathway identity).
        path = self.pathway_identity.expand(b, -1, -1)  # (B, P, D)
        pool_keep = gene_membership.transpose(1, 2)      # (B, P, G)
        pooled = self.pool_attn(self.pool_q_norm(path), self.pool_kv_norm(genes), pool_keep)
        path = path + pooled

        # 2. mix pathways <-> pathways; absent pathways masked out as keys.
        path = self.pathway_mix(path, key_padding_mask=~pathway_present)

        # 3. scatter pathways -> genes + CELL. Genes read their own pathways;
        #    the CELL token reads every pathway present in the view.
        seq = torch.cat([cell, genes], dim=1)  # (B, 1+G, D)
        cell_keep = pathway_present[:, None, :]  # (B, 1, P)
        gene_keep = gene_membership & pathway_present[:, None, :]  # (B, G, P)
        scatter_keep = torch.cat([cell_keep, gene_keep], dim=1)    # (B, 1+G, P)
        upd = self.scatter_attn(
            self.scatter_q_norm(seq), self.scatter_kv_norm(path), scatter_keep
        )
        seq = seq + upd
        return seq[:, :1, :], seq[:, 1:, :]


class HierarchicalEncoderSub14(nn.Module):
    """Gene-level sigmoid-attention encoder with interleaved pathway blocks.

    Identical to :class:`~eb_jepa.singlecell.sub14.encoder.EncoderSub14` in
    its gene-level modules (``cell_token``, ``layers``, ``final_norm`` — so a
    sub14 checkpoint warm-starts them key-for-key), plus ``hier_blocks``: a
    pathway-level block is run after each gene layer whose index is in
    ``hier_positions``.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        *,
        n_pathways: int,
        hier_positions: list[int],
        expected_seq_len: int = 512,
        genes_per_view: int = 512,
        attention_activation: str = "sigmoid",
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.cell_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cell_token, std=0.02)

        self.layers = nn.ModuleList(
            PreNormSigmoidBlock(
                d_model, n_heads, d_ff, dropout,
                expected_seq_len=expected_seq_len, attention_activation=attention_activation,
            )
            for _ in range(n_layers)
        )

        # A pathway block keyed by the gene-layer index it follows. Using a
        # ModuleDict keeps the module name stable regardless of which layers
        # carry a hierarchy block.
        self.hier_positions = sorted(set(hier_positions))
        if self.hier_positions and (self.hier_positions[0] < 0 or self.hier_positions[-1] >= n_layers):
            raise ValueError(f"hier_positions {self.hier_positions} out of range [0, {n_layers})")
        self.hier_blocks = nn.ModuleDict(
            {
                str(pos): PathwayHierarchyBlock(
                    d_model, n_heads, d_ff, n_pathways,
                    dropout=dropout, genes_per_view=genes_per_view,
                    attention_activation=attention_activation,
                )
                for pos in self.hier_positions
            }
        )
        self.final_norm = RMSNorm(d_model)

    def forward(
        self,
        embeddings: Tensor,        # (B, G, D) gene-token embeddings (CELL not prepended)
        *,
        gene_membership: Tensor,   # (B, G, P) bool, already AND-ed with gene validity
        padding_mask: Tensor,      # (B, 1+G) bool, True = ignore (CELL slot included)
    ) -> Tensor:
        from torch.utils.checkpoint import checkpoint

        b = embeddings.size(0)
        cell = self.cell_token.expand(b, -1, -1)  # (B, 1, D)
        x = torch.cat([cell, embeddings], dim=1)  # (B, 1+G, D)
        pathway_present = gene_membership.any(dim=1)  # (B, P)

        for i, layer in enumerate(self.layers):
            if self.grad_checkpoint and self.training:
                x = checkpoint(
                    lambda inp, m=padding_mask, lyr=layer: lyr(inp, key_padding_mask=m),
                    x, use_reentrant=False,
                )
            else:
                x = layer(x, key_padding_mask=padding_mask)
            if str(i) in self.hier_blocks:
                cell_tok, gene_tok = self.hier_blocks[str(i)](
                    x[:, :1, :], x[:, 1:, :], gene_membership, pathway_present
                )
                x = torch.cat([cell_tok, gene_tok], dim=1)

        x = self.final_norm(x)
        return x[:, 0, :]  # [CELL] token


class HierarchicalSubliminal14(nn.Module):
    """Subliminal 1.4 with the hallmark pathway hierarchy (warm-startable from sub14).

    Drop-in for :class:`~eb_jepa.singlecell.sub14.model.Subliminal14`: same
    constructor knobs and the same ``forward(gene_ids, bin_ids, padding_mask)``
    -> :class:`Subliminal14Output` contract (so the training loop, eval, and
    SIGReg loss are unchanged). Extra args: ``pathway_membership``
    ``(P, n_pc_genes)`` 0/1 and ``hier_positions`` (gene-layer indices after
    which to run a pathway block).
    """

    def __init__(
        self,
        *,
        n_pc_genes: int,
        pathway_membership: Tensor,  # (P, n_pc_genes) 0/1
        hier_positions: Optional[list[int]] = None,
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

        if pathway_membership.size(1) != n_pc_genes:
            raise ValueError(
                f"pathway_membership has {pathway_membership.size(1)} gene columns "
                f"but n_pc_genes={n_pc_genes}"
            )
        n_pathways = pathway_membership.size(0)
        self.n_pathways = n_pathways

        # gene_to_pathway[gene_id] -> (P,) membership row. Row n_pc_genes is the
        # pad sentinel (all zero), matching the collator's pad gene id.
        pad_row = torch.zeros(1, n_pathways, dtype=torch.bool)
        gene_to_pathway = torch.cat(
            [pathway_membership.bool().t().contiguous(), pad_row], dim=0
        )  # (n_pc_genes + 1, P)
        self.register_buffer("gene_to_pathway", gene_to_pathway, persistent=False)

        if hier_positions is None:
            # Default: one pathway block after every gene layer except the last,
            # so at least one gene layer mixes the scattered pathway info before
            # readout. For a 2-layer model this is [0].
            hier_positions = list(range(max(n_layers - 1, 0))) or [0]

        self.gene_embedding = ProteinCodingGeneEmbeddings(
            n_pc_genes=n_pc_genes,
            d_model=d_model,
            dna_features=dna_features,
            protein_features=protein_features,
            freeze_features=freeze_features,
        )
        self.count_embedding = QuantileThermometerCountEmbedding(num_bins=num_bins, d_model=d_model)

        self.encoder = HierarchicalEncoderSub14(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            n_pathways=n_pathways,
            hier_positions=hier_positions,
            expected_seq_len=max_genes_per_cell + 1,
            genes_per_view=max_genes_per_cell,
            attention_activation=attention_activation,
            grad_checkpoint=grad_checkpoint,
        )

        # Identical projector to Subliminal14 (warm-starts key-for-key).
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
        """One view. Args match :meth:`Subliminal14.forward`.

        ``gene_ids``/``bin_ids``/``padding_mask``: (B, G); pad gene id is
        ``n_pc_genes``, pad bin is ``num_bins``, ``padding_mask`` True = pad.
        """
        gene_emb = self.gene_embedding(gene_ids)
        count_emb = self.count_embedding(bin_ids)
        embeddings = gene_emb + count_emb

        gene_valid = ~padding_mask  # (B, G)
        gene_membership = self.gene_to_pathway[gene_ids] & gene_valid.unsqueeze(-1)  # (B, G, P)

        cls_pad = torch.zeros(
            padding_mask.size(0), 1, device=padding_mask.device, dtype=padding_mask.dtype
        )
        encoder_pad_mask = torch.cat([cls_pad, padding_mask], dim=1)  # (B, 1+G)

        cell_rep = self.encoder(
            embeddings, gene_membership=gene_membership, padding_mask=encoder_pad_mask
        )
        cell_proj = self.projection(cell_rep)
        return Subliminal14Output(cell_representation=cell_rep, cell_projection=cell_proj)

    @torch.no_grad()
    def encode(self, gene_ids: Tensor, bin_ids: Tensor, padding_mask: Tensor) -> Tensor:
        """Pre-projection cell representation (for probing / t-SNE)."""
        return self.forward(gene_ids, bin_ids, padding_mask).cell_representation
