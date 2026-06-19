"""Representation analysis & visualization (CLAUDE.md "Representation visualizations").

- covariance spectrum of the representation (collapse diagnostics);
- effective rank (entropy of the normalized eigenspectrum) — the scalar we sweep
  against lambda to show rich -> collapsed and justify the chosen lambda;
- tSNE / UMAP 2-D embeddings colored by class (organ, cell line, ...).

Numeric helpers are dependency-light (torch + sklearn); UMAP and plotting are
lazily imported so the core has no hard matplotlib/umap dependency.
"""

from __future__ import annotations

import numpy as np
import torch


def covariance_spectrum(features: torch.Tensor) -> torch.Tensor:
    """Eigenvalues (descending) of the feature covariance matrix [d]."""
    x = features.float()
    x = x - x.mean(0, keepdim=True)
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0)
    return torch.flip(eig, dims=[0])


def effective_rank(features: torch.Tensor) -> float:
    """Effective rank = exp(entropy of the normalized eigenspectrum).

    ~d for an isotropic (rich) representation, ~1 for a collapsed one. This is the
    collapse metric to plot against the SIGReg weight lambda.
    """
    eig = covariance_spectrum(features)
    total = eig.sum()
    if total <= 0:
        return 0.0
    p = eig / total
    p = p[p > 0]
    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy))


def tsne_embed(
    features: torch.Tensor,
    n_components: int = 2,
    seed: int = 0,
    perplexity: float = 30.0,
):
    """2-D t-SNE embedding [N, n_components] (sklearn)."""
    from sklearn.manifold import TSNE

    x = features.detach().cpu().numpy()
    perp = min(perplexity, max(2.0, (x.shape[0] - 1) / 3.0))
    return TSNE(
        n_components=n_components, random_state=seed, perplexity=perp
    ).fit_transform(x)


def umap_embed(features: torch.Tensor, n_components: int = 2, seed: int = 0):
    """2-D UMAP embedding (requires umap-learn; raises a clear error if absent)."""
    try:
        import umap
    except ImportError as e:
        raise ImportError(
            "umap_embed requires 'umap-learn' (uv pip install umap-learn)"
        ) from e
    return umap.UMAP(n_components=n_components, random_state=seed).fit_transform(
        features.detach().cpu().numpy()
    )


def plot_embedding(emb2d, labels, path: str, title: str = ""):
    """Scatter a 2-D embedding colored by categorical label; save to ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    emb2d = np.asarray(emb2d)
    classes = sorted({x for x in labels if x is not None})
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, c in enumerate(classes):
        m = np.array([x == c for x in labels])
        ax.scatter(emb2d[m, 0], emb2d[m, 1], s=6, color=cmap(i % 20), label=str(c))
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_spectrum(eigenvalues, path: str, title: str = "covariance spectrum"):
    """Log-scale plot of the (descending) eigenspectrum; save to ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eig = np.asarray(eigenvalues)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(eig) + 1), eig + 1e-12)
    ax.set_yscale("log")
    ax.set_xlabel("component")
    ax.set_ylabel("eigenvalue")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
