"""Render the anchor single-cell visualizations from *synthetic* latents.

The single-cell encoder/perturbator are not trained yet (the subpackage is at
milestone M0: LeJEPA loss + transformer primitives only). To de-risk the
plotting code and show what the hackathon deliverable will look like, this
script fabricates latents that *encode the target pattern* and renders them with
``eb_jepa.singlecell.viz``. Plug real ``z`` / ``Δ`` arrays into the same
functions once the encoder is trained — the figures are identical.

    python -m eb_jepa.singlecell.viz_demo            # writes PNGs to ./viz_out

Synthetic data ⇒ illustrative, NOT a result. Every figure is watermarked.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eb_jepa.singlecell import viz

OUT = os.environ.get("VIZ_OUT", "viz_out")
CTRL = "DMSO_TF"  # control drug, per CLAUDE.md
MOA_COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]


# --------------------------------------------------------------------------- #
# synthetic generators (each one bakes in the property the plot should reveal)
# --------------------------------------------------------------------------- #
def make_perturbation_latents(
    d_model: int = 16,
    n_moa: int = 4,
    drugs_per_moa: int = 4,
    doses=(0.0, 0.25, 0.5, 0.75, 1.0),
    cells_per_cond: int = 60,
    m_max: float = 6.0,
    dose50: float = 0.35,
    rng=None,
):
    """Cells in a ``d_model`` latent with the perturbator's *target* geometry.

    Encoded properties (the three claims of the headline figure):
      1. same MoA ⇒ shared displacement direction (small angular jitter only);
      2. dose ⇒ monotonic magnitude  m(dose) = m_max * tanh(dose / dose50);
      3. that magnitude saturates ⇒ displacements are bounded above (ceiling)
         and, at dose→0, vanish into the control cloud (noise floor).
    """
    rng = rng or np.random.default_rng(0)
    # one distinct unit direction per MoA in the latent
    moa_dirs = viz.unit(rng.standard_normal((n_moa, d_model)))
    c0 = np.zeros(d_model)  # control centroid at the origin

    X, drug_of_cell, dose_of_cell, moa_of_cell = [], [], [], []
    drug_moa = {}
    # control cells
    X.append(c0 + 0.45 * rng.standard_normal((cells_per_cond * 3, d_model)))
    drug_of_cell += [CTRL] * (cells_per_cond * 3)
    dose_of_cell += [0.0] * (cells_per_cond * 3)
    moa_of_cell += ["control"] * (cells_per_cond * 3)

    for k in range(n_moa):
        for j in range(drugs_per_moa):
            name = f"MoA{k}-drug{j}"
            drug_moa[name] = k
            # drug direction = MoA direction + small jitter (still "same way")
            dirn = viz.unit(moa_dirs[k] + 0.12 * rng.standard_normal(d_model))
            for dose in doses:
                mag = m_max * np.tanh(dose / dose50)  # monotone + saturating
                center = c0 + mag * dirn
                pts = center + 0.45 * rng.standard_normal((cells_per_cond, d_model))
                X.append(pts)
                drug_of_cell += [name] * cells_per_cond
                dose_of_cell += [dose] * cells_per_cond
                moa_of_cell += [k] * cells_per_cond

    return (
        np.concatenate(X),
        np.array(drug_of_cell),
        np.array(dose_of_cell),
        np.array(moa_of_cell, dtype=object),
        drug_moa,
        moa_dirs,
    )


def make_lambda_regime(regime: str, n_class=6, per_class=120, d=16, rng=None):
    """Class-structured latent under three LeJEPA λ regimes (collapse demo).

    loss = λ·SIGReg + (1−λ)·invariance:
      * 'collapse'   (λ→0): invariance wins, no anti-collapse pressure ⇒
        *dimensional collapse*: variance piles onto ~1 axis, the rest is dust.
        Spectrum = one spike, effective rank ≈ 1, classes smear along a line.
      * 'balanced'   (λ≈0.05): isotropic *and* class-separated ⇒ spread spectrum,
        high effective rank, clean clusters.
      * 'oversmooth' (λ→1): SIGReg wins, the marginal is a clean isotropic
        Gaussian but views never align ⇒ classes dissolve into one blob. Same
        flat-ish spectrum & high rank as 'balanced' — only a *structure* metric
        (silhouette) tells them apart. That is the whole point of the panel.
    """
    rng = rng or np.random.default_rng(1)
    if regime == "collapse":
        # one dominant direction; classes differ only along it, tiny off-axis dust
        v1 = viz.unit(rng.standard_normal(d))
        cls_scalar = rng.standard_normal(n_class)
        X, y = [], []
        for c in range(n_class):
            along = 3.0 * cls_scalar[c] + 1.0 * rng.standard_normal(per_class)
            pts = along[:, None] * v1[None, :] + 0.05 * rng.standard_normal((per_class, d))
            X.append(pts)
            y += [c] * per_class
        return np.concatenate(X), np.array(y)

    class_means = rng.standard_normal((n_class, d))
    if regime == "balanced":
        between, within = 4.0, 0.6  # separated isotropic blobs
    elif regime == "oversmooth":
        between, within = 0.25, 1.0  # isotropic but classes merged
    else:
        raise ValueError(regime)
    X, y = [], []
    for c in range(n_class):
        X.append(between * class_means[c] + within * rng.standard_normal((per_class, d)))
        y += [c] * per_class
    return np.concatenate(X), np.array(y)


# --------------------------------------------------------------------------- #
# figure 1 — the headline: drug directions, dose band, spectrum, cosine block
# --------------------------------------------------------------------------- #
def figure_perturbation(path: str):
    rng = np.random.default_rng(0)
    X, drug, dose, moa, drug_moa, moa_dirs = make_perturbation_latents(rng=rng)

    fig = plt.figure(figsize=(15.5, 11.5))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    # -- (A) the UMAP-style field with control->drug arrows ------------------
    axA = fig.add_subplot(gs[0, 0])
    xy = viz.embed_2d(X, method="pca", seed=0)  # PCA keeps directions faithful
    ctrl_mask = drug == CTRL
    axA.scatter(*xy[ctrl_mask].T, s=6, c="0.6", alpha=0.4, label="control (DMSO_TF)")
    for k in sorted(drug_moa and set(drug_moa.values())):
        m = np.array([isinstance(v, int) and v == k for v in moa])
        axA.scatter(*xy[m].T, s=6, color=MOA_COLORS[k], alpha=0.35)
    # arrows: control centroid -> each drug's max-dose centroid, colored by MoA
    c0_xy = xy[ctrl_mask].mean(0)
    for name, k in drug_moa.items():
        top = (drug == name) & (dose == dose.max())
        tip = xy[top].mean(0)
        axA.annotate(
            "",
            xy=tip,
            xytext=c0_xy,
            arrowprops=dict(arrowstyle="-|>", color=MOA_COLORS[k], lw=1.8, alpha=0.9),
        )
    axA.scatter(*c0_xy, marker="*", s=320, c="k", zorder=5)
    for k in range(len(MOA_COLORS[: len(set(drug_moa.values()))])):
        axA.plot([], [], color=MOA_COLORS[k], lw=3, label=f"MoA {k}")
    axA.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axA.set_title(
        "A. Perturbation directions (PCA of latent)\n"
        "same-MoA drugs → parallel arrows = same direction",
        fontsize=11,
    )
    axA.set_xlabel("PC1"); axA.set_ylabel("PC2")

    # -- (B) dose: monotonic AND bounded -------------------------------------
    axB = fig.add_subplot(gs[0, 1])
    deltas, names, c0 = viz.displacement_vectors(X, drug, CTRL)
    # per-drug magnitude across doses (recompute on centroids per dose)
    doses_sorted = np.unique(dose[dose > 0])
    upper, lower = [], []
    for name in names:
        k = drug_moa[name]
        mags = []
        for dval in doses_sorted:
            sub = (drug == name) & (dose == dval)
            mags.append(np.linalg.norm(X[sub].mean(0) - c0))
        axB.plot(doses_sorted, mags, "-o", ms=3, color=MOA_COLORS[k], alpha=0.6, lw=1)
        upper.append(max(mags)); lower.append(min(mags))
    hi, lo = max(upper), min(lower)
    axB.axhspan(lo, hi, color="0.85", alpha=0.5, zorder=0)
    axB.axhline(hi, color="k", ls="--", lw=1)
    axB.axhline(lo, color="k", ls="--", lw=1)
    axB.text(doses_sorted[-1], hi, "  upper bound (saturation)", va="bottom", fontsize=8)
    axB.text(doses_sorted[0], lo, "  lower bound (noise floor)", va="bottom", fontsize=8)
    axB.set_title(
        "B. Dose–response: ‖Δ‖ monotone in dose,\nbounded between lower & upper",
        fontsize=11,
    )
    axB.set_xlabel("dose (normalized log-conc.)"); axB.set_ylabel("‖displacement‖")

    # -- (C) displacement 'spectrogram': drugs × latent PCs ------------------
    axC = fig.add_subplot(gs[1, 0])
    order, _ = viz.dendrogram_order(deltas, metric="cosine")
    # project displacements onto the latent principal axes -> a per-drug spectrum
    from sklearn.decomposition import PCA

    comps = PCA(n_components=8).fit(X).components_  # [8, D]
    spec = deltas @ comps.T  # [K, 8]
    spec = spec[order]
    im = axC.imshow(spec, aspect="auto", cmap="RdBu_r",
                    vmin=-np.abs(spec).max(), vmax=np.abs(spec).max())
    axC.set_yticks(range(len(names)))
    axC.set_yticklabels([names[i] for i in order], fontsize=6)
    axC.set_xlabel("latent principal component")
    axC.set_title(
        "C. Displacement spectrum (drug × latent PC)\n"
        "same MoA → same active components",
        fontsize=11,
    )
    fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04)

    # -- (D) cosine block matrix: the quantitative 'same direction' proof ----
    axD = fig.add_subplot(gs[1, 1])
    C = viz.cosine_matrix(deltas)[np.ix_(order, order)]
    im2 = axD.imshow(C, cmap="magma", vmin=-1, vmax=1)
    axD.set_title(
        "D. Pairwise cosine of displacements\nblock-diagonal ⇒ MoA = shared direction",
        fontsize=11,
    )
    axD.set_xticks([]); axD.set_yticks([])
    fig.colorbar(im2, ax=axD, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Perturbator geometry — SYNTHETIC illustrative data (target pattern, not a result)",
        fontsize=13, y=0.995,
    )
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# figure 2 — ablations, map-wise: what λ does to the representation
# --------------------------------------------------------------------------- #
def figure_ablation(path: str):
    from sklearn.metrics import silhouette_score

    regimes = [
        ("collapse", "λ→0  (invariance only):\ndimensional collapse onto a line"),
        ("balanced", "λ≈0.05  (chosen):\nrich, separated, isotropic"),
        ("oversmooth", "λ→1  (SIGReg only):\nisotropic but classes merged"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for j, (regime, title) in enumerate(regimes):
        X, y = make_lambda_regime(regime)
        # PCA (not t-SNE): a linear map *reveals* collapse as a 1-D smear,
        # whereas t-SNE would re-inflate it into blobs and hide it.
        xy = viz.embed_2d(X, method="pca", seed=0)
        er = viz.effective_rank(X)
        try:
            sil = silhouette_score(X, y)
        except Exception:
            sil = float("nan")
        ax = axes[0, j]
        for c in np.unique(y):
            ax.scatter(*xy[y == c].T, s=8, alpha=0.6)
        ax.set_title(f"{title}\neff.rank={er:.1f}  silhouette={sil:.2f}", fontsize=10)
        # equal, symmetric axes so dimensional collapse reads as a literal line
        lim = float(np.abs(xy).max()) * 1.05
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

        # eigenspectrum underneath — the canonical collapse diagnostic
        axs = axes[1, j]
        eig = viz.covariance_eigenspectrum(X, normalize=True)
        axs.bar(range(1, len(eig) + 1), eig, color="0.3")
        axs.set_ylim(0, 1.0)
        axs.set_xlabel("eigen-index");
        if j == 0:
            axs.set_ylabel("variance fraction")
        axs.set_title("covariance spectrum", fontsize=9)

    fig.suptitle(
        "Ablation, map-wise: the λ trade-off (loss = λ·SIGReg + (1−λ)·invariance)\n"
        "the spectrum catches collapse; only a structure metric (silhouette) "
        "separates 'balanced' from 'oversmooth' — SYNTHETIC illustrative data",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# figure 3 — JEPA "de-distorting": recover a circular latent factor (LeWM-style)
# --------------------------------------------------------------------------- #
def make_distorted_circle(n=900, d=40, n_harmonics=4, noise=0.04, seed=0):
    """A circular generative factor observed through a nonlinear distortion.

    The single-cell analog of LeWM's "circle of colours" is the **cell cycle**
    (G1→S→G2→M→G1 is literally a loop). The true factor is an angle ``θ``; we
    lift the clean circle to ``d`` dims with random harmonics + a random linear
    mix so the observed manifold is a *warped* closed loop — exactly how raw
    gene-expression distorts a simple factor. A manifold-aware encoder should
    *un-warp* it back to a ring ordered by θ.
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    feats = [np.cos(theta), np.sin(theta)]
    for h in range(2, n_harmonics + 1):  # higher harmonics = the distortion
        feats += [0.9 / h * np.cos(h * theta), 0.9 / h * np.sin(h * theta)]
    Z = np.stack(feats, 1)  # [n, 2*n_harmonics]
    mix = rng.standard_normal((Z.shape[1], d))
    X = np.tanh(Z @ mix) + noise * rng.standard_normal((n, d))  # nonlinear lift
    return X, theta


def figure_factor_recovery(path: str):
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap

    X, theta = make_distorted_circle()
    cmap = "hsv"  # cyclic colormap = the "colour wheel"

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))

    # A. the true factor: a clean ring coloured by phase
    axes[0].scatter(np.cos(theta), np.sin(theta), c=theta, cmap=cmap, s=10)
    axes[0].set_title("A. True factor θ = cell-cycle phase\n(G1→S→G2→M→G1, a circle)",
                      fontsize=10)
    axes[0].set_aspect("equal"); axes[0].set_xticks([]); axes[0].set_yticks([])

    # B. distorted observation (linear PCA can't un-warp it)
    xy_pca = PCA(n_components=2).fit_transform(X)
    axes[1].scatter(*xy_pca.T, c=theta, cmap=cmap, s=10)
    axes[1].set_title("B. Observed manifold, PCA(2)\nphase smeared = distorted",
                      fontsize=10)
    axes[1].set_aspect("equal"); axes[1].set_xticks([]); axes[1].set_yticks([])

    # C. manifold (JEPA-style) recovery: un-warped ring ordered by phase
    xy_iso = Isomap(n_neighbors=12, n_components=2).fit_transform(X)
    axes[2].scatter(*xy_iso.T, c=theta, cmap=cmap, s=10)
    axes[2].set_title("C. De-distorted recovery (Isomap)\nclean ring, phase ordered",
                      fontsize=10)
    axes[2].set_aspect("equal"); axes[2].set_xticks([]); axes[2].set_yticks([])

    # D. isometry check: recovered angle vs true angle (up to offset/sign)
    rec = np.arctan2(xy_iso[:, 1] - xy_iso[:, 1].mean(),
                     xy_iso[:, 0] - xy_iso[:, 0].mean())
    # align sign so the relation reads as a diagonal
    if np.corrcoef(np.unwrap(np.sort(theta)), rec[np.argsort(theta)])[0, 1] < 0:
        rec = -rec
    axes[3].scatter(theta, (rec - rec.min()) % (2 * np.pi), c=theta, cmap=cmap, s=8)
    axes[3].set_title("D. Recovered vs true phase\nmonotone ⇒ factor recovered up to isometry",
                      fontsize=10)
    axes[3].set_xlabel("true θ"); axes[3].set_ylabel("recovered angle")

    fig.suptitle(
        "JEPA de-distorting (LeWM-style): a circular biological factor recovered "
        "up to isometry — SYNTHETIC illustrative data",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# figure 4 — JEPA training diagnostics (is the latent healthy?)
# --------------------------------------------------------------------------- #
def figure_jepa_diagnostics(path: str):
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 10))

    # A. SIGReg Gaussianity: random latent slices vs N(0,1), Q-Q
    healthy = rng.standard_normal((1500, 16))  # isotropic = SIGReg target
    collapsed = (rng.standard_normal((1500, 1)) * viz.unit(rng.standard_normal(16))
                 + 0.02 * rng.standard_normal((1500, 16)))  # rank-1 spike (for panel B)
    # a heavy-tailed latent — non-Gaussianity that survives random projection
    # (per-dim bimodal would Gaussianize under projection by the CLT; fat tails do not)
    nongauss = rng.standard_t(2.2, size=(1500, 16))
    q = np.linspace(0.01, 0.99, 200)
    from scipy.stats import norm

    theo = norm.ppf(q)
    for s in viz.random_slice_quantiles(healthy, 4):
        ax[0, 0].plot(theo, np.quantile(s, q), color="#3cb44b", alpha=0.5, lw=1)
    for s in viz.random_slice_quantiles(nongauss, 4):
        ax[0, 0].plot(theo, np.quantile(s, q), color="#e6194B", alpha=0.5, lw=1)
    ax[0, 0].plot([-3, 3], [-3, 3], "k--", lw=1)
    ax[0, 0].plot([], [], color="#3cb44b", label="healthy (isotropic → on the line)")
    ax[0, 0].plot([], [], color="#e6194B", label="non-Gaussian (pre-SIGReg → S-curve)")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title("A. SIGReg check: random latent slices vs N(0,1) (Q-Q)", fontsize=10)
    ax[0, 0].set_xlabel("normal quantiles"); ax[0, 0].set_ylabel("latent-slice quantiles")

    # B. covariance spectrum + effective rank
    for name, X, c in [("healthy", healthy, "#3cb44b"), ("collapsed", collapsed, "#e6194B")]:
        eig = viz.covariance_eigenspectrum(X)
        ax[0, 1].plot(range(1, len(eig) + 1), eig, "-o", ms=3, color=c,
                      label=f"{name} (eff.rank {viz.effective_rank(X):.1f})")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_title("B. Latent covariance spectrum (flat = isotropic target)", fontsize=10)
    ax[0, 1].set_xlabel("eigen-index"); ax[0, 1].set_ylabel("variance fraction")

    # C. view invariance: V views of each cell should collapse to its centroid
    n_cells, V = 12, 6
    centroids = 4 * rng.standard_normal((n_cells, 2))
    xy = (centroids[:, None, :] + 0.35 * rng.standard_normal((n_cells, V, 2)))
    for i in range(n_cells):
        for v in range(V):
            ax[1, 0].plot([centroids[i, 0], xy[i, v, 0]],
                          [centroids[i, 1], xy[i, v, 1]], color="0.7", lw=0.6, zorder=1)
        ax[1, 0].scatter(*xy[i].T, s=14, zorder=2)
    ax[1, 0].scatter(*centroids.T, marker="*", s=140, c="k", zorder=3)
    ax[1, 0].set_title("C. View invariance: the V views of a cell collapse to its centroid",
                       fontsize=10)
    ax[1, 0].set_xticks([]); ax[1, 0].set_yticks([])

    # D. effective rank over training: collapse vs healthy
    step = np.linspace(0, 1, 100)
    ax[1, 1].plot(step, 1 + 14 * (1 - np.exp(-4 * step)), color="#3cb44b", label="healthy → full rank")
    ax[1, 1].plot(step, 1 + 14 * np.exp(-6 * step), color="#e6194B", label="too-small λ → collapse")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("D. Effective rank over training (collapse vs healthy)", fontsize=10)
    ax[1, 1].set_xlabel("training progress"); ax[1, 1].set_ylabel("effective rank")

    fig.suptitle("JEPA training diagnostics — SYNTHETIC illustrative data", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# figure 5 — latent geometry: straightening, curvature, surprise, energy
# --------------------------------------------------------------------------- #
def figure_latent_geometry(path: str):
    rng = np.random.default_rng(4)
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 10))

    # A. dose-trajectory straightening: JEPA (straight) vs expression (curved)
    t = np.linspace(0, 1, 8)
    jepa = np.stack([t * 5, 0.1 * rng.standard_normal(8)], 1)
    expr = np.stack([t * 5, 2.2 * np.sin(np.pi * t)], 1)  # curved in expression space
    ax[0, 0].plot(*jepa.T, "-o", color="#4363d8",
                  label=f"JEPA latent (straightness {viz.path_straightness(jepa):.2f})")
    ax[0, 0].plot(*expr.T, "-o", color="#f58231",
                  label=f"expression/PCA (straightness {viz.path_straightness(expr):.2f})")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title("A. Dose-trajectory straightening: JEPA vs expression", fontsize=10)
    ax[0, 0].set_xticks([]); ax[0, 0].set_yticks([])

    # B. geodesic vs Euclidean distance (manifold curvature) on a swiss roll
    from sklearn.datasets import make_swiss_roll

    Xsr, color = make_swiss_roll(700, noise=0.1, random_state=0)
    geo = viz.geodesic_distances(Xsr, n_neighbors=10)[0]
    euc = np.linalg.norm(Xsr - Xsr[0], axis=1)
    ax[0, 1].scatter(euc, geo, s=6, c=color, cmap="viridis")
    ax[0, 1].plot([0, euc.max()], [0, euc.max()], "k--", lw=1)
    ax[0, 1].set_title("B. Geodesic vs Euclidean distance (gap = manifold curvature)", fontsize=10)
    ax[0, 1].set_xlabel("Euclidean"); ax[0, 1].set_ylabel("geodesic (on-manifold)")

    # C. perturbator surprise / violation-of-expectation
    conds = ["plausible\ndrug+dose", "implausible\ndose (OOD)", "unseen\ndrug (OOD)", "batch/plate\nshift"]
    energy = [1.0, 4.6, 3.9, 1.15]; err = [0.15, 0.4, 0.45, 0.18]
    colors = ["#3cb44b", "#e6194B", "#e6194B", "#3cb44b"]
    ax[1, 0].bar(conds, energy, yerr=err, color=colors, capsize=4)
    ax[1, 0].set_title("C. Prediction surprise spikes on implausible perturbation, not on batch",
                       fontsize=10)
    ax[1, 0].set_ylabel("prediction energy (sliced-W)")

    # D. latent energy landscape with a toxic basin + control→toxic geodesic
    gx, gy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-4, 4, 200))
    E = (-3.2 * np.exp(-((gx - 3) ** 2 + gy ** 2) / 2.0)        # toxic basin
         - 1.0 * np.exp(-((gx + 3) ** 2 + gy ** 2) / 1.5)        # control basin
         + 0.04 * (gx ** 2 + gy ** 2))
    cf = ax[1, 1].contourf(gx, gy, E, levels=25, cmap="magma")
    px = np.linspace(-3, 3, 40); py = 0.9 * np.sin(np.pi * (px + 3) / 6) * (1 - np.abs(px) / 4)
    ax[1, 1].plot(px, py, "w-", lw=2)
    ax[1, 1].scatter([-3], [0], marker="o", s=80, c="cyan", label="control")
    ax[1, 1].scatter([3], [0], marker="X", s=110, c="red", label="toxic basin")
    ax[1, 1].legend(fontsize=8, loc="upper left")
    fig.colorbar(cf, ax=ax[1, 1], fraction=0.046, pad=0.04)
    ax[1, 1].set_title("D. Latent energy landscape: toxic basin + control→toxic geodesic", fontsize=10)
    ax[1, 1].set_xticks([]); ax[1, 1].set_yticks([])

    fig.suptitle("Latent geometry & perturbator energy — SYNTHETIC illustrative data", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# figure 6 — representation similarity & cross-dataset alignment
# --------------------------------------------------------------------------- #
def figure_alignment(path: str):
    rng = np.random.default_rng(5)
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 10))

    # shared "true" cell-line biology; three noisy rotated measurements of it
    n_lines, dim = 30, 8
    truth = rng.standard_normal((n_lines, dim))

    def rotated_view(noise):
        Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
        return truth @ Q + noise * rng.standard_normal((n_lines, dim))

    jepa, depmap, expr = rotated_view(0.3), rotated_view(0.5), rotated_view(0.6)

    # A. Procrustes alignment JEPA vs DepMap (same cell lines), full vectors
    a, b, disp = viz.procrustes_align(jepa, depmap)
    for i in range(n_lines):
        ax[0, 0].plot([a[i, 0], b[i, 0]], [a[i, 1], b[i, 1]], color="0.8", lw=0.6)
    ax[0, 0].scatter(a[:, 0], a[:, 1], s=18, c="#4363d8", label="JEPA")
    ax[0, 0].scatter(b[:, 0], b[:, 1], s=18, c="#f58231", label="DepMap")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title(f"A. Procrustes: JEPA vs DepMap cell lines (disparity {disp:.2f})", fontsize=10)
    ax[0, 0].set_xticks([]); ax[0, 0].set_yticks([])

    # B. representational convergence triangle (pairwise Procrustes disparity)
    spaces = {"JEPA": jepa, "DepMap\ndependency": depmap, "CCLE\nexpression": expr}
    keys = list(spaces); D = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            D[i, j] = viz.procrustes_align(spaces[keys[i]], spaces[keys[j]])[2]
    im = ax[0, 1].imshow(D, cmap="viridis_r")
    ax[0, 1].set_xticks(range(3)); ax[0, 1].set_xticklabels(keys, fontsize=8)
    ax[0, 1].set_yticks(range(3)); ax[0, 1].set_yticklabels(keys, fontsize=8)
    for i in range(3):
        for j in range(3):
            ax[0, 1].text(j, i, f"{D[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
    fig.colorbar(im, ax=ax[0, 1], fraction=0.046, pad=0.04)
    ax[0, 1].set_title("B. Representational convergence (Procrustes disparity, low=agree)", fontsize=10)

    # C. layer-wise CKA: how the representation forms with depth
    L = 8
    layers = [rng.standard_normal((200, 16))]
    for _ in range(L - 1):
        M = rng.standard_normal((16, 16))
        layers.append(np.tanh(layers[-1] @ M) + 0.3 * rng.standard_normal((200, 16)))
    cka = np.array([[viz.linear_cka(layers[i], layers[j]) for j in range(L)] for i in range(L)])
    im2 = ax[1, 0].imshow(cka, cmap="magma", vmin=0, vmax=1)
    fig.colorbar(im2, ax=ax[1, 0], fraction=0.046, pad=0.04)
    ax[1, 0].set_title("C. Layer-wise CKA: how the representation forms with depth", fontsize=10)
    ax[1, 0].set_xlabel("layer"); ax[1, 0].set_ylabel("layer")

    # D. latent arithmetic: a drug's effect transfers across cell lines
    base_dir = viz.unit(rng.standard_normal(8))
    delta_A = 3 * base_dir + 0.4 * rng.standard_normal((12, 8))
    delta_B = 3 * base_dir + 0.4 * rng.standard_normal((12, 8))
    ax[1, 1].scatter(delta_A.ravel(), delta_B.ravel(), s=10, alpha=0.6)
    lim = np.abs(np.r_[delta_A.ravel(), delta_B.ravel()]).max()
    ax[1, 1].plot([-lim, lim], [-lim, lim], "k--", lw=1)
    cos = float(viz.unit(delta_A.mean(0)) @ viz.unit(delta_B.mean(0)))
    ax[1, 1].set_title(f"D. Latent arithmetic: drug effect transfers across lines (cos {cos:.2f})",
                       fontsize=10)
    ax[1, 1].set_xlabel("Δ(drug, line A)"); ax[1, 1].set_ylabel("Δ(drug, line B)")

    fig.suptitle("Representation similarity & cross-dataset alignment — SYNTHETIC illustrative data",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# figure 7 — latent structure & embedding quality (batch 2)
# --------------------------------------------------------------------------- #
def figure_latent_structure(path: str):
    rng = np.random.default_rng(6)
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 10))

    # A. intrinsic dimension (TwoNN) vs encoder width — rises then plateaus
    true_id = 12
    widths = [8, 16, 32, 64, 128, 256]
    ids = []
    for w in widths:
        k = max(2, round(true_id * (1 - np.exp(-w / 38.0))))  # capacity-limited signal dims
        # TwoNN is a property of the manifold, not the ambient d_model; average a
        # few seeds of a clean k-dim manifold so the estimator reads ~k.
        est = np.mean([
            viz.intrinsic_dimension_twonn(np.random.default_rng(s).standard_normal((2500, k)))
            for s in range(3)
        ])
        ids.append(est)
    ax[0, 0].plot(widths, ids, "-o", color="#4363d8")
    ax[0, 0].axhline(true_id, ls="--", color="k", lw=1, label=f"data ID = {true_id}")
    ax[0, 0].set_xscale("log", base=2); ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title("A. Intrinsic dimension (TwoNN) vs encoder width", fontsize=10)
    ax[0, 0].set_xlabel("encoder width (d_model)"); ax[0, 0].set_ylabel("estimated intrinsic dim")

    # B. embedding trustworthiness vs k — is the 2-D map faithful?
    from sklearn.manifold import trustworthiness

    Xb, yb = make_lambda_regime("balanced")
    ks = [5, 10, 20, 30, 50, 80]
    for method, c in [("pca", "#911eb4"), ("tsne", "#3cb44b")]:
        emb = viz.embed_2d(Xb, method=method, seed=0)
        tw = [trustworthiness(Xb, emb, n_neighbors=k) for k in ks]
        ax[0, 1].plot(ks, tw, "-o", ms=3, color=c, label=method.upper())
    ax[0, 1].legend(fontsize=8); ax[0, 1].set_ylim(0.5, 1.0)
    ax[0, 1].set_title("B. Embedding trustworthiness vs k (is the 2-D map faithful?)", fontsize=10)
    ax[0, 1].set_xlabel("neighborhood size k"); ax[0, 1].set_ylabel("trustworthiness")

    # C. latent density (KDE): two populations, ball-bounded, no voids
    from scipy.stats import gaussian_kde

    pop = np.concatenate([rng.standard_normal((600, 2)) + [-2, 0],
                          rng.standard_normal((600, 2)) + [2.2, 0.5]])
    kde = gaussian_kde(pop.T)
    gx, gy = np.meshgrid(np.linspace(-6, 6, 160), np.linspace(-5, 5, 160))
    dens = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    cf = ax[1, 0].contourf(gx, gy, dens, levels=20, cmap="viridis")
    fig.colorbar(cf, ax=ax[1, 0], fraction=0.046, pad=0.04)
    ax[1, 0].set_title("C. Latent density (KDE): modes filled, ball-bounded, no voids", fontsize=10)
    ax[1, 0].set_xticks([]); ax[1, 0].set_yticks([])

    # D. kNN graph on the latent (colored by organ)
    from sklearn.neighbors import NearestNeighbors

    organs = ["liver", "lung", "kidney", "blood"]
    centers = 6 * rng.standard_normal((4, 2))
    pts, lab = [], []
    for i in range(4):
        pts.append(centers[i] + rng.standard_normal((40, 2))); lab += [i] * 40
    P = np.concatenate(pts); lab = np.array(lab)
    nn = NearestNeighbors(n_neighbors=6).fit(P)
    _, idx = nn.kneighbors(P)
    for i in range(len(P)):
        for j in idx[i, 1:]:
            ax[1, 1].plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color="0.8", lw=0.4, zorder=1)
    for i in range(4):
        ax[1, 1].scatter(*P[lab == i].T, s=16, color=MOA_COLORS[i], label=organs[i], zorder=2)
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("D. kNN graph on the latent (colored by organ)", fontsize=10)
    ax[1, 1].set_xticks([]); ax[1, 1].set_yticks([])

    fig.suptitle("Latent structure & embedding quality — SYNTHETIC illustrative data", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# figure 8 — gene programs along a trajectory & attention vs TRRUST (batch 2)
# --------------------------------------------------------------------------- #
def figure_programs_attention(path: str):
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 10))

    # A. latent traversal: gene programs switch along control->toxic geodesic
    steps, n_genes, n_prog = 24, 48, 4
    t = np.linspace(0, 1, steps)
    prog_of_gene = np.repeat(np.arange(n_prog), n_genes // n_prog)
    thresh = np.linspace(0.2, 0.8, n_prog)          # each program turns on at a dose
    sign = np.array([1, 1, -1, -1])                 # up- and down-regulated programs
    expr = np.zeros((n_genes, steps))
    for g in range(n_genes):
        p = prog_of_gene[g]
        s = 1 / (1 + np.exp(-(t - thresh[p]) / 0.08))
        expr[g] = (s if sign[p] > 0 else 1 - s) + 0.05 * rng.standard_normal(steps)
    im = ax[0, 0].imshow(expr, aspect="auto", cmap="magma", extent=[0, 1, n_genes, 0])
    fig.colorbar(im, ax=ax[0, 0], fraction=0.046, pad=0.04)
    ax[0, 0].set_title("A. Latent traversal: gene programs switch along control→toxic geodesic",
                       fontsize=10)
    ax[0, 0].set_xlabel("trajectory (control → toxic)"); ax[0, 0].set_ylabel("genes (by program)")

    # build a synthetic TRRUST-style co-regulation graph + an attention matrix
    G, n_tf = 60, 5
    regulon = np.repeat(np.arange(n_tf), G // n_tf)
    coreg = (regulon[:, None] == regulon[None, :]).astype(float)   # TRRUST edge = same regulon
    attn = 2.0 * coreg + 1.0 * rng.standard_normal((G, G))
    attn = (attn + attn.T) / 2
    np.fill_diagonal(attn, attn.max())

    # B. attention predicts TRRUST edges — ROC / AUROC
    from sklearn.metrics import roc_auc_score, roc_curve

    iu = np.triu_indices(G, k=1)
    labels, scores = coreg[iu], attn[iu]
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    ax[0, 1].plot(fpr, tpr, color="#e6194B", lw=2)
    ax[0, 1].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0, 1].set_title(f"B. Gene attention predicts TRRUST edges (AUROC = {auc:.2f})", fontsize=10)
    ax[0, 1].set_xlabel("false positive rate"); ax[0, 1].set_ylabel("true positive rate")

    # C. gene-gene attention recovers regulons (block-diagonal when TF-ordered)
    im2 = ax[1, 0].imshow(attn, cmap="magma")
    fig.colorbar(im2, ax=ax[1, 0], fraction=0.046, pad=0.04)
    ax[1, 0].set_title("C. Gene–gene attention recovers TRRUST regulons (TF-ordered blocks)",
                       fontsize=10)
    ax[1, 0].set_xlabel("gene"); ax[1, 0].set_ylabel("gene")

    # D. per-dimension latent histograms (~N(0,1) under SIGReg)
    from scipy.stats import norm

    Z = rng.standard_normal((4000, 16))
    grid = np.linspace(-4, 4, 200)
    for dimi, c in zip(range(4), ["#4363d8", "#3cb44b", "#f58231", "#911eb4"]):
        ax[1, 1].hist(Z[:, dimi], bins=40, density=True, histtype="step", color=c, lw=1.2)
    ax[1, 1].plot(grid, norm.pdf(grid), "k--", lw=1.5, label="N(0,1)")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("D. Per-dimension latent histograms (~N(0,1) under SIGReg)", fontsize=10)
    ax[1, 1].set_xlabel("latent value"); ax[1, 1].set_ylabel("density")

    fig.suptitle("Gene programs & attention vs TRRUST — SYNTHETIC illustrative data", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    figs = [
        (figure_perturbation, "perturbation_geometry.png"),
        (figure_ablation, "ablation_lambda_maps.png"),
        (figure_factor_recovery, "factor_recovery_circle.png"),
        (figure_jepa_diagnostics, "jepa_diagnostics.png"),
        (figure_latent_geometry, "latent_geometry.png"),
        (figure_alignment, "representation_alignment.png"),
        (figure_latent_structure, "latent_structure.png"),
        (figure_programs_attention, "programs_attention.png"),
    ]
    for fn, name in figs:
        print("wrote:", fn(os.path.join(OUT, name)))


if __name__ == "__main__":
    main()
