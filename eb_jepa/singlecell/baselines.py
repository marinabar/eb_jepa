"""Baseline encoders for the headline comparison: MAE, VAE, PCA.

JEPA must beat *well-tuned* MAE / VAE / PCA on the probing metrics (CLAUDE.md
"Success criteria"). Baselines operate on the **densified** fixed gene vocabulary
(``densify``: token_ids+values -> [n_genes]) rather than the sparse token set used
by the JEPA. Each exposes ``.encode(dense) -> latent`` so they share the M3 probing
harness; MAE/VAE also return a reconstruction/ELBO loss for training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from eb_jepa.datasets.tahoe.dataset import densify

_META_KEYS = ("drug", "sample", "cell_line_id", "organ", "moa_fine", "plate")


def _meta_out(dense: torch.Tensor, batch: list[dict]) -> dict:
    out: dict = {"dense": dense}
    for k in _META_KEYS:
        out[k] = [c.get(k) for c in batch]
    out["log_conc"] = torch.tensor(
        [c.get("log_conc", float("nan")) for c in batch], dtype=torch.float32
    )
    return out


class DensifyCollator:
    """Collate sparse cells into dense [N, n_genes] vectors + probing metadata."""

    def __init__(self, n_genes: int):
        self.n_genes = n_genes

    def __call__(self, batch: list[dict]) -> dict:
        dense = torch.stack(
            [densify(c["gene_token_ids"], c["values"], self.n_genes) for c in batch]
        )
        return _meta_out(dense, batch)


class HVGDensifyCollator:
    """Collate sparse cells into the dense [N, n_hvg] HVG-panel vector + metadata.

    Restricts the dense input to a FIXED panel of N highly-variable genes so the
    MATCHED baselines share Subliminal-14's context-window size (genes per cell).
    ``hvg_local_map`` (size = token-id index space, -1 for off-panel genes) maps each
    raw token_id to its local panel column; off-panel genes are dropped.
    """

    def __init__(self, hvg_local_map: torch.Tensor, n_hvg: int):
        self.hvg_local_map = hvg_local_map.long()
        self.n_hvg = int(n_hvg)

    def __call__(self, batch: list[dict]) -> dict:
        dense = torch.zeros((len(batch), self.n_hvg), dtype=torch.float32)
        for j, c in enumerate(batch):
            tok = c["gene_token_ids"].long()
            if tok.numel() == 0:
                continue
            local = self.hvg_local_map[tok]
            keep = local >= 0
            if keep.any():
                dense[j, local[keep]] = c["values"].to(torch.float32)[keep]
        return _meta_out(dense, batch)


def _mlp(sizes, act=nn.GELU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class MAEBaseline(nn.Module):
    """Masked autoencoder on the dense gene vector.

    A random fraction of genes is zeroed at the input; the decoder reconstructs the
    full vector and the loss is MSE on the **masked** entries (denoising/MAE style).
    The representation is the bottleneck latent (pre-decoder).
    """

    def __init__(
        self, n_genes: int, hidden: int = 512, latent: int = 256, mask_frac: float = 0.5
    ):
        super().__init__()
        self.mask_frac = mask_frac
        self.encoder = _mlp([n_genes, hidden, latent])
        self.decoder = _mlp([latent, hidden, n_genes])

    def encode(self, dense: torch.Tensor) -> torch.Tensor:
        return self.encoder(dense)

    def forward(self, dense: torch.Tensor) -> dict:
        mask = torch.rand_like(dense) < self.mask_frac  # True = hidden
        corrupted = dense.masked_fill(mask, 0.0)
        latent = self.encoder(corrupted)
        recon = self.decoder(latent)
        denom = mask.sum().clamp(min=1)
        loss = (((recon - dense) ** 2) * mask).sum() / denom
        return {"loss": loss, "recon_loss": loss}


class VAEBaseline(nn.Module):
    """Vanilla VAE on the dense gene vector (Gaussian latent, MSE reconstruction)."""

    def __init__(
        self, n_genes: int, hidden: int = 512, latent: int = 256, kl_coeff: float = 1e-3
    ):
        super().__init__()
        self.kl_coeff = kl_coeff
        self.encoder = _mlp([n_genes, hidden, hidden])
        self.to_mu = nn.Linear(hidden, latent)
        self.to_logvar = nn.Linear(hidden, latent)
        self.decoder = _mlp([latent, hidden, n_genes])

    def encode(self, dense: torch.Tensor) -> torch.Tensor:
        return self.to_mu(self.encoder(dense))  # posterior mean = representation

    def forward(self, dense: torch.Tensor) -> dict:
        h = self.encoder(dense)
        mu, logvar = self.to_mu(h), self.to_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)
        recon_loss = F.mse_loss(recon, dense)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return {
            "loss": recon_loss + self.kl_coeff * kl,
            "recon_loss": recon_loss,
            "kl": kl,
        }


class PCABaseline:
    """PCA baseline (sklearn). Fit on a sample of dense vectors, then ``encode``."""

    def __init__(self, n_components: int = 256):
        self.n_components = n_components
        self._pca = None

    def fit(self, dense: torch.Tensor) -> "PCABaseline":
        from sklearn.decomposition import PCA

        self._pca = PCA(n_components=self.n_components)
        self._pca.fit(dense.detach().cpu().numpy())
        return self

    def encode(self, dense: torch.Tensor) -> torch.Tensor:
        if self._pca is None:
            raise RuntimeError("PCABaseline.fit must be called before encode")
        z = self._pca.transform(dense.detach().cpu().numpy())
        return torch.from_numpy(z).float()


def build_baseline(model_type: str, n_genes: int, **kw):
    """Factory: 'mae' | 'vae' | 'pca'."""
    if model_type == "mae":
        return MAEBaseline(n_genes, **kw)
    if model_type == "vae":
        return VAEBaseline(n_genes, **kw)
    if model_type == "pca":
        return PCABaseline(**kw)
    raise ValueError(f"Unknown baseline {model_type!r}; expected mae|vae|pca")
