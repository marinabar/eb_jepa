"""Transformer primitives for the set-based single-cell encoder.

Building blocks specified in CLAUDE.md ("Encoder architecture"):
- pre-norm RMSNorm (no bias)
- SwiGLU FFN (d_ff ~= 2/3 * 2 * d_model)
- grouped-query attention (GQA 4:1) over an *unordered* gene set (no positional
  encoding / RoPE), using torch SDPA (Flash Attention picked automatically on
  Hopper/Blackwell) with a key-padding mask for variable-length cells padded to L
- stochastic depth (DropPath)

The encoder that stacks these blocks lives in ``encoder.py`` (milestone M2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no bias, no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize in float32 for stability, then cast back (bf16-friendly).
        out = self._norm(x.float()).type_as(x)
        return out * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Shazeer 2020), no biases.

    Hidden size defaults to ``d_ff ~= 2/3 * 2 * dim`` rounded up to ``multiple_of``,
    matching the GLU-variant convention (the 2/3 keeps the parameter count of the
    3-matrix SwiGLU comparable to a 2-matrix ReLU FFN of width 4*dim).
    """

    def __init__(self, dim: int, hidden_dim: int | None = None, multiple_of: int = 256):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 / 3 * 2 * dim)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up-projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down-projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to match the number of query heads (GQA).

    Args:
        x: [B, n_kv_heads, L, head_dim]
        n_rep: number of query heads per kv head (= n_heads // n_kv_heads)
    Returns:
        [B, n_kv_heads * n_rep, L, head_dim]
    """
    if n_rep == 1:
        return x
    b, n_kv, l, hd = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, l, hd)
        .reshape(b, n_kv * n_rep, l, hd)
    )


class GQAttention(nn.Module):
    """Grouped-query self-attention over a token set, via scaled_dot_product_attention.

    No positional encoding: gene tokens form an unordered set. Variable-length cells
    are padded to L and a ``key_padding_mask`` (True = real token) hides pad keys.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int | None = None):
        super().__init__()
        if n_kv_heads is None:
            n_kv_heads = max(1, n_heads // 4)  # GQA 4:1
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, dim]
            key_padding_mask: [B, L] bool, True for real tokens, False for padding.
        Returns:
            [B, L, dim]
        """
        b, l, _ = x.shape
        q = self.wq(x).view(b, l, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, l, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, l, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        attn_mask = None
        if key_padding_mask is not None:
            # SDPA boolean mask: True = participate in attention. Broadcast the
            # key-padding mask over heads and query positions: [B, 1, 1, L].
            attn_mask = key_padding_mask[:, None, None, :]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(b, l, -1)
        return self.wo(out)


class DropPath(nn.Module):
    """Stochastic depth: randomly drop the residual branch per sample (Huang 2016)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        # one Bernoulli mask per sample, broadcast over the rest of the dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x / keep * mask


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm -> GQA -> residual; RMSNorm -> SwiGLU -> residual.

    ``residual_scale`` implements the ×1/sqrt(2*n_layers) residual scaling from
    CLAUDE.md; ``drop_path`` is the stochastic-depth probability (0 to disable).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        ffn_hidden_dim: int | None = None,
        residual_scale: float = 1.0,
        drop_path: float = 0.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.attn = GQAttention(dim, n_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn = SwiGLU(dim, ffn_hidden_dim)
        self.residual_scale = residual_scale
        self.drop_path = DropPath(drop_path)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.drop_path(
            self.residual_scale * self.attn(self.attn_norm(x), key_padding_mask)
        )
        x = x + self.drop_path(self.residual_scale * self.ffn(self.ffn_norm(x)))
        return x
