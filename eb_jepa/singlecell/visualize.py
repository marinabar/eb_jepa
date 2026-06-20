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


_INK = "#1d2433"
_SUB = "#7a8699"
_OTHER = "#d7dde5"


def _palette(n: int):
    """A perceptually-spread qualitative palette of length >= n."""
    import matplotlib.pyplot as plt

    base = []
    for name in ("tab20", "tab20b", "tab20c"):
        base.extend(plt.get_cmap(name).colors)
    return [base[i % len(base)] for i in range(n)]


def plot_tsne_grid(
    emb2d,
    labels_by_class: dict,
    path: str,
    step: int | None = None,
    top_k: int = 14,
    point_size: float = 7.0,
):
    """Elegant multi-panel t-SNE, one panel per class coloring (design system).

    ``labels_by_class``: {class_name -> per-point labels}. High-cardinality classes
    show the ``top_k`` most frequent categories in colour; the rest (and ``None``)
    are drawn as faint grey "other". Returns ``path``.
    """
    import collections

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    emb2d = np.asarray(emb2d)
    names = list(labels_by_class.keys())
    ncol = 2 if len(names) > 1 else 1
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(7.0 * ncol, 6.2 * nrow), facecolor="white"
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        labels = list(labels_by_class[name])
        freq = collections.Counter(x for x in labels if x is not None)
        shown = [c for c, _ in freq.most_common(top_k)]
        cmap = {c: col for c, col in zip(shown, _palette(len(shown)))}
        is_other = np.array([x not in cmap for x in labels])
        if is_other.any():
            ax.scatter(
                emb2d[is_other, 0], emb2d[is_other, 1],
                s=point_size * 0.7, c=_OTHER, linewidths=0, alpha=0.55, zorder=1,
            )
        for c in shown:
            m = np.array([x == c for x in labels])
            ax.scatter(
                emb2d[m, 0], emb2d[m, 1],
                s=point_size, color=cmap[c], linewidths=0, alpha=0.85, zorder=2,
                label=str(c),
            )
        n_cat = len(freq)
        ax.set_title(
            f"{name}", loc="left", fontsize=14, fontweight="bold", color=_INK, pad=10
        )
        ax.text(
            0.0, 1.005,
            f"{n_cat} categories" + (f"  ·  top {top_k} shown" if n_cat > top_k else ""),
            transform=ax.transAxes, fontsize=8.5, color=_SUB,
        )
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#c7cfdb"); s.set_linewidth(0.8)
        ax.set_aspect("equal", adjustable="datalim")
        leg = ax.legend(
            markerscale=2.2, fontsize=7.5, loc="upper right", frameon=False,
            handletextpad=0.3, labelspacing=0.3, ncol=1, borderaxespad=0.2,
        )
        for t in leg.get_texts():
            t.set_color(_INK)

    for ax in axes[len(names):]:
        ax.axis("off")

    suptitle = "t-SNE of cell representations (pre-projection)"
    if step is not None:
        suptitle += f"  ·  step {step}"
    fig.suptitle(
        suptitle, x=0.04, y=0.995, ha="left", fontsize=18, fontweight="bold", color=_INK
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
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
