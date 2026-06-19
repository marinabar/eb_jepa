"""Set-based transformer encoder for single cells (CLAUDE.md "Encoder architecture").

Unordered gene set -> no positional encoding. Stack of pre-norm RMSNorm + GQA +
SwiGLU blocks; residual scaled x1/sqrt(2*n_layers); stochastic depth p=0.1 only
when n_layers >= 16; dropout 0. Readout is masked mean-pool (default) or CLS. The
returned representation is **pre-projection** (the JEPA projector lives in the loss).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from eb_jepa.singlecell.embeddings import GeneTokenEmbedding
from eb_jepa.singlecell.layers import RMSNorm, TransformerBlock


def _xavier_init(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SingleCellEncoder(nn.Module):
    """Gene-set transformer: token embeddings -> N blocks -> readout (pre-projection).

    Args:
        gene_embedding: a GeneTokenEmbedding (frozen ESMC/Evo2 + learned count).
        d_model, n_layers, n_heads, n_kv_heads: transformer size.
        use_cls: prepend a learned [CLS] token.
        readout: "meanpool" (masked mean over gene tokens) or "cls".
        drop_path: stochastic-depth prob; auto 0.1 if n_layers>=16 and left None.
        grad_checkpoint: checkpoint each block to save activation memory.
    """

    def __init__(
        self,
        gene_embedding: GeneTokenEmbedding,
        d_model: int = 1024,
        n_layers: int = 12,
        n_heads: int = 16,
        n_kv_heads: int | None = None,
        use_cls: bool = False,
        readout: str = "meanpool",
        drop_path: float | None = None,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        assert readout in ("meanpool", "cls")
        if use_cls is False and readout == "cls":
            raise ValueError("readout='cls' requires use_cls=True")
        self.embed = gene_embedding
        self.d_model = d_model
        self.use_cls = use_cls
        self.readout = readout
        self.grad_checkpoint = grad_checkpoint

        if drop_path is None:
            drop_path = 0.1 if n_layers >= 16 else 0.0
        residual_scale = (2.0 * n_layers) ** -0.5  # x1/sqrt(2*n_layers)

        self.blocks = nn.ModuleList(
            TransformerBlock(
                d_model,
                n_heads,
                n_kv_heads,
                residual_scale=residual_scale,
                drop_path=drop_path,
            )
            for _ in range(n_layers)
        )
        self.norm = RMSNorm(d_model)
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Xavier on qkv/o/proj (frozen ESMC/Evo2 buffers are untouched)
        self.blocks.apply(_xavier_init)

    def forward(
        self,
        gene_token_ids: torch.Tensor,
        pad_mask: torch.Tensor,
        count_value: torch.Tensor | None = None,
        count_bin: torch.Tensor | None = None,
        count_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        gene_token_ids / pad_mask / count_*: [B, L]. Returns [B, d_model] (pre-proj).
        ``pad_mask`` is True at real tokens.
        """
        x = self.embed(gene_token_ids, count_value, count_bin, count_mask)  # [B, L, d]
        gene_mask = pad_mask
        if self.use_cls:
            cls = self.cls_token.expand(x.shape[0], 1, -1)
            x = torch.cat([cls, x], dim=1)
            cls_col = torch.ones(x.shape[0], 1, dtype=torch.bool, device=x.device)
            attn_mask = torch.cat([cls_col, pad_mask], dim=1)
        else:
            attn_mask = pad_mask

        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                x = checkpoint(block, x, attn_mask, use_reentrant=False)
            else:
                x = block(x, attn_mask)
        x = self.norm(x)

        if self.readout == "cls":
            return x[:, 0]
        # masked mean over the gene tokens (exclude CLS, exclude padding)
        if self.use_cls:
            x = x[:, 1:]
        m = gene_mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


def encode_views(encoder: SingleCellEncoder, batch: dict) -> torch.Tensor:
    """Run all V views through the shared encoder. Returns [V, N, d_model].

    ``batch`` is the TahoeCollator output ([V, N, L] per-token tensors). Views are
    flattened into the batch dim so the encoder sees constant [V*N, L] shapes.
    """
    ids = batch["gene_token_ids"]  # [V, N, L]
    v, n, l = ids.shape
    flat_ids = ids.reshape(v * n, l)
    flat_pad = batch["pad_mask"].reshape(v * n, l)
    cv = batch.get("count_value")
    cb = batch.get("count_bin")
    cm = batch.get("count_mask")
    reps = encoder(
        flat_ids,
        flat_pad,
        count_value=cv.reshape(v * n, l) if cv is not None else None,
        count_bin=cb.reshape(v * n, l) if cb is not None else None,
        count_mask=cm.reshape(v * n, l) if cm is not None else None,
    )
    return reps.reshape(v, n, -1)
