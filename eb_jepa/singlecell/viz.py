"""Visualization toolkit for the single-cell LeJEPA encoder & perturbator.

Every function here takes plain numpy arrays (embeddings ``[N, D]`` + metadata),
so it plugs directly onto the two objects we actually produce:

  * encoder **pre-projection** latents ``z`` (probing / structure / collapse maps)
  * perturbator **displacements** ``Δ = centroid(z_perturbed) − centroid(z_control)``
    estimated per ``(cell_line, plate)`` stratum (see CLAUDE.md "Control matching").

Nothing here trains or loads data: given embeddings + labels it renders the
figures listed in CLAUDE.md ("Representation visualizations & analyses"). The
heavy 2-D embedder (``umap-learn``) is imported lazily with a PCA / t-SNE
fallback, so the toolkit runs with only numpy + matplotlib + scikit-learn.

The single source of truth for *what* each plot demonstrates is ``VIZ.md``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# dimensionality reduction (UMAP with graceful fallback)
# --------------------------------------------------------------------------- #
def embed_2d(
    X: np.ndarray,
    method: str = "umap",
    seed: int = 0,
    **kwargs,
) -> np.ndarray:
    """Project ``X`` ``[N, D]`` to ``[N, 2]``.

    ``method`` is one of ``{"umap", "tsne", "pca"}``. ``umap`` falls back to
    ``pca`` if ``umap-learn`` is not installed. PCA is linear, so it preserves
    relative *directions* of displacement vectors — prefer it for the
    perturbation field; prefer UMAP/t-SNE for cluster-separation maps.
    """
    X = np.asarray(X, dtype=np.float64)
    if method == "umap":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(random_state=seed, **kwargs)
            return reducer.fit_transform(X)
        except Exception:
            method = "pca"  # fall through
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = kwargs.pop("perplexity", min(30, max(5, X.shape[0] // 4)))
        return TSNE(
            n_components=2, random_state=seed, perplexity=perplexity, init="pca"
        ).fit_transform(X)
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=seed).fit_transform(X)


# --------------------------------------------------------------------------- #
# scalar collapse diagnostics (used to *justify* the chosen lambda)
# --------------------------------------------------------------------------- #
def covariance_eigenspectrum(X: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Eigenvalues of ``cov(X)`` sorted descending (the representation spectrum).

    A flat spectrum ⇒ isotropic (the LeJEPA / SIGReg target). A spectrum that
    crashes to ~0 after a few modes ⇒ dimensional collapse.
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0, keepdims=True)
    cov = (Xc.T @ Xc) / max(1, X.shape[0] - 1)
    eig = np.linalg.eigvalsh(cov)[::-1].clip(min=0)
    if normalize and eig.sum() > 0:
        eig = eig / eig.sum()
    return eig


def effective_rank(X: np.ndarray) -> float:
    """Participation ratio ``(Σλ)² / Σλ²`` — a smooth, scale-free rank.

    ≈ 1 under complete/dimensional collapse, ≈ D for a perfectly isotropic
    representation. A single number to put on a collapse plot.
    """
    eig = covariance_eigenspectrum(X, normalize=False)
    s1, s2 = eig.sum(), (eig**2).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


# --------------------------------------------------------------------------- #
# direction geometry (the core of "drugs all go the same way")
# --------------------------------------------------------------------------- #
def unit(v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + eps)


def cosine_matrix(V: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of row vectors ``V`` ``[M, D]`` → ``[M, M]``."""
    U = unit(np.asarray(V, dtype=np.float64))
    return U @ U.T


def dendrogram_order(V: np.ndarray, metric: str = "cosine"):
    """Hierarchical (average-linkage) leaf order + linkage matrix for ``V``.

    Returns ``(order, linkage)``; reorder a cosine matrix by ``order`` to expose
    block structure, or feed ``linkage`` to ``scipy.cluster.hierarchy.dendrogram``.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist

    Z = linkage(pdist(np.asarray(V, dtype=np.float64), metric=metric), method="average")
    return leaves_list(Z), Z


def group_centroids(
    X: np.ndarray, groups: Sequence
) -> tuple[np.ndarray, list]:
    """Mean embedding per group label. Returns ``(centroids [G, D], group_names)``."""
    X = np.asarray(X, dtype=np.float64)
    names = list(dict.fromkeys(groups))  # stable unique
    cents = np.stack([X[np.asarray(groups) == g].mean(0) for g in names])
    return cents, names


def displacement_vectors(
    X: np.ndarray,
    drug_of_cell: Sequence,
    control_value,
) -> tuple[np.ndarray, list, np.ndarray]:
    """Δ for every non-control drug = its centroid − the control centroid.

    Returns ``(deltas [K, D], drug_names [K], control_centroid [D])``. This is
    the perturbator's target geometry; in production ``X`` is the *pre-projection*
    encoder latent and the centroids are taken within one ``(cell_line, plate)``.
    """
    X = np.asarray(X, dtype=np.float64)
    drug_of_cell = np.asarray(drug_of_cell)
    c0 = X[drug_of_cell == control_value].mean(0)
    names = [d for d in dict.fromkeys(drug_of_cell.tolist()) if d != control_value]
    deltas = np.stack([X[drug_of_cell == d].mean(0) - c0 for d in names])
    return deltas, names, c0


# --------------------------------------------------------------------------- #
# Riemannian / geodesic helper (manifold-aware distance)
# --------------------------------------------------------------------------- #
def geodesic_distances(X: np.ndarray, n_neighbors: int = 15) -> np.ndarray:
    """Graph (Isomap-style) geodesic distances on the kNN graph of ``X``.

    Straight-line (Euclidean) distance overstates separation on a curved data
    manifold; the geodesic follows the manifold. Comparing the two tells us how
    curved the latent is and whether dose trajectories are manifold-straight.
    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import shortest_path

    g = kneighbors_graph(
        np.asarray(X, dtype=np.float64), n_neighbors=n_neighbors, mode="distance"
    )
    g = g.maximum(g.T)  # symmetrize
    return shortest_path(g, method="D", directed=False)


# --------------------------------------------------------------------------- #
# JEPA-specific latent diagnostics (straightening, representation similarity)
# --------------------------------------------------------------------------- #
def path_straightness(path: np.ndarray) -> float:
    """Chord/arc ratio of an ordered latent trajectory ``[T, D]`` → (0, 1].

    1 = perfectly straight. LeWM reports that JEPA latents *straighten*
    trajectories with no explicit term; we use this to show a dose path is
    straighter in the encoder latent than in raw-expression / PCA space.
    """
    path = np.asarray(path, dtype=np.float64)
    arc = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    chord = np.linalg.norm(path[-1] - path[0])
    return float(chord / arc) if arc > 0 else 1.0


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA similarity of two representations of the *same* N rows → [0, 1].

    Use it for a layer×layer matrix ("how the representation forms with depth")
    or to compare encoder vs baseline (MAE/VAE/PCA) representations.
    """
    X = np.asarray(X, dtype=np.float64) - np.asarray(X, dtype=np.float64).mean(0)
    Y = np.asarray(Y, dtype=np.float64) - np.asarray(Y, dtype=np.float64).mean(0)
    num = np.linalg.norm(X.T @ Y, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / den) if den > 0 else 0.0


def intrinsic_dimension_twonn(X: np.ndarray, discard_fraction: float = 0.1) -> float:
    """TwoNN intrinsic-dimension estimate (Facco et al. 2017).

    ``d ≈ M / Σ log(μ_i)`` with ``μ_i = r2_i / r1_i`` the ratio of each point's
    2nd- to 1st-nearest-neighbour distance. Used for the scaling-law axis: how the
    latent's intrinsic dimension grows with encoder width/data, independent of the
    ambient ``d_model``.
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=np.float64)
    dist, _ = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)
    mu = dist[:, 2] / np.maximum(dist[:, 1], 1e-12)
    mu = np.sort(mu[mu > 1.0])
    mu = mu[: int(len(mu) * (1 - discard_fraction))]  # drop a heavy tail
    return float(len(mu) / np.sum(np.log(mu))) if len(mu) else 0.0


def random_slice_quantiles(X: np.ndarray, n_slices: int = 6, seed: int = 0):
    """Standardized samples of random 1-D projections of ``X`` ``[N, D]``.

    Returns ``[n_slices, N]`` (each row zero-mean/unit-var). Feed to a Q-Q plot
    against N(0,1): that *is* the SIGReg target — random latent slices should be
    standard Gaussian. A spike-collapsed latent fails this visibly.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    A = rng.standard_normal((X.shape[1], n_slices))
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    s = X @ A  # [N, n_slices]
    s = (s - s.mean(0)) / (s.std(0) + 1e-12)
    return s.T


# --------------------------------------------------------------------------- #
# cross-dataset alignment (our latent vs an independent reference, e.g. DepMap)
# --------------------------------------------------------------------------- #
def procrustes_align(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Orthogonal Procrustes alignment of ``X`` onto ``Y`` (rows = same items).

    Returns ``(X_aligned, Y_std, disparity)`` where ``disparity`` ∈ [0, 1] is the
    residual after optimal scale+rotation (lower = the two spaces agree more).
    Use it to test whether our JEPA cell-line geometry matches an independent
    DepMap expression / dependency geometry over the *same* cell lines.
    """
    from scipy.spatial import procrustes

    mtx1, mtx2, disparity = procrustes(
        np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    )
    return mtx1, mtx2, float(disparity)
