"""GQA attention and the transformer block: shapes, and correct key-padding-mask
behaviour (outputs at real tokens must not depend on padded tokens).
"""

import pytest
import torch

from eb_jepa.singlecell.layers import (
    DropPath,
    GQAttention,
    RMSNorm,
    SwiGLU,
    TransformerBlock,
    repeat_kv,
)


class TestRepeatKV:
    def test_shape_expansion(self):
        x = torch.randn(2, 3, 7, 4)  # [B, n_kv_heads, L, head_dim]
        out = repeat_kv(x, n_rep=4)
        assert out.shape == (2, 12, 7, 4)

    def test_identity_when_one(self):
        x = torch.randn(2, 8, 5, 4)
        assert torch.equal(repeat_kv(x, 1), x)

    def test_repeated_heads_are_copies(self):
        x = torch.randn(1, 2, 3, 4)
        out = repeat_kv(x, 3)  # heads 0,1,2 == kv head 0; heads 3,4,5 == kv head 1
        assert torch.equal(out[:, 0], out[:, 1]) and torch.equal(out[:, 1], out[:, 2])
        assert torch.equal(out[:, 3], out[:, 4]) and torch.equal(out[:, 4], out[:, 5])


class TestGQAMask:
    @pytest.fixture(autouse=True)
    def _seed(self):
        torch.manual_seed(0)

    def test_gqa_4to1_default_kv_heads(self):
        attn = GQAttention(dim=64, n_heads=8)
        assert attn.n_kv_heads == 2 and attn.n_rep == 4

    def test_output_shape(self):
        attn = GQAttention(dim=32, n_heads=4)
        x = torch.randn(3, 10, 32)
        assert attn(x).shape == (3, 10, 32)

    @torch.no_grad()
    def test_padding_does_not_affect_valid_outputs(self):
        attn = GQAttention(dim=32, n_heads=4).eval()
        b, l, d = 2, 12, 32
        x = torch.randn(b, l, d)
        mask = torch.ones(b, l, dtype=torch.bool)
        mask[:, 8:] = False  # last 4 tokens are padding

        out1 = attn(x, key_padding_mask=mask)
        x2 = x.clone()
        x2[:, 8:] = torch.randn(b, 4, d)  # arbitrarily change padded tokens
        out2 = attn(x2, key_padding_mask=mask)

        # outputs at real tokens must be identical
        assert torch.allclose(out1[:, :8], out2[:, :8], atol=1e-5)

    @torch.no_grad()
    def test_block_padding_does_not_affect_valid_outputs(self):
        block = TransformerBlock(dim=32, n_heads=4, residual_scale=0.5).eval()
        b, l, d = 2, 12, 32
        x = torch.randn(b, l, d)
        mask = torch.ones(b, l, dtype=torch.bool)
        mask[:, 9:] = False

        out1 = block(x, key_padding_mask=mask)
        x2 = x.clone()
        x2[:, 9:] = torch.randn(b, 3, d)
        out2 = block(x2, key_padding_mask=mask)
        assert torch.allclose(out1[:, :9], out2[:, :9], atol=1e-5)

    @torch.no_grad()
    def test_no_nan_when_some_rows_fully_real(self):
        attn = GQAttention(dim=16, n_heads=4).eval()
        x = torch.randn(1, 6, 16)
        mask = torch.tensor([[True, True, True, False, False, False]])
        out = attn(x, key_padding_mask=mask)
        assert torch.isfinite(out).all()


class TestDropPath:
    def test_eval_is_identity(self):
        dp = DropPath(0.5).eval()
        x = torch.randn(8, 3, 16)
        assert torch.equal(dp(x), x)

    def test_zero_prob_is_identity_in_train(self):
        dp = DropPath(0.0).train()
        x = torch.randn(8, 3, 16)
        assert torch.equal(dp(x), x)

    def test_train_per_sample_mask_and_expectation(self):
        torch.manual_seed(0)
        dp = DropPath(0.5).train()
        x = torch.ones(2000, 4)
        out = dp(x)
        # each sample row is dropped (all 0) or kept-and-scaled (all 1/keep = 2.0)
        assert torch.logical_or(
            (out == 0).all(dim=1),
            torch.isclose(out, torch.full_like(out, 2.0)).all(dim=1),
        ).all()
        # expectation is preserved (mean ~ 1.0)
        assert abs(out.mean().item() - 1.0) < 0.1


class TestResidualScale:
    @torch.no_grad()
    def test_residual_scale_zero_is_identity(self):
        # residual_scale=0 zeroes both residual branches -> block is identity.
        torch.manual_seed(0)
        block = TransformerBlock(dim=16, n_heads=4, residual_scale=0.0).eval()
        x = torch.randn(2, 5, 16)
        assert torch.allclose(block(x), x, atol=1e-6)


class TestNormAndFFN:
    def test_rmsnorm_unit_rms(self):
        x = torch.randn(4, 128) * 7.0
        norm = RMSNorm(128)
        out = norm(x)
        # with weight=1, the RMS over the feature dim is ~1
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)

    def test_swiglu_shape_and_hidden(self):
        ffn = SwiGLU(256)
        assert ffn(torch.randn(3, 5, 256)).shape == (3, 5, 256)
        assert ffn.hidden_dim % 256 == 0  # rounded to multiple_of
