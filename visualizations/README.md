# Visualizations

Figure gallery for the single-cell LeJEPA **encoder** and **perturbator**
(hepatotoxicity JEPA on Tahoe-100M). Each panel's title states exactly what it
shows. The plotting code is `eb_jepa/singlecell/viz.py`; the figures are rendered
by `eb_jepa/singlecell/viz_demo.py` and re-created cell-by-cell in
[`gallery.ipynb`](gallery.ipynb).

> ⚠️ The encoder is not trained yet (the subpackage is at milestone M0). These
> use **synthetic latents that encode the target pattern** — they validate the
> plotting pipeline and show the deliverable. Drop real `z` / `Δ` arrays into the
> same `viz.py` calls and the figures become results. Full plan: `eb_jepa/singlecell/VIZ.md`.

| figure | what it shows |
|---|---|
| `perturbation_geometry.png` | drug directions (parallel arrows per MoA), dose monotone & bounded, displacement spectrum, cosine block matrix |
| `ablation_lambda_maps.png` | the λ trade-off map-wise: collapse-to-line / rich / isotropic-but-merged, each with its covariance spectrum |
| `factor_recovery_circle.png` | cell-cycle ring recovered up to isometry (true → PCA-distorted → Isomap-recovered → phase diagonal) |
| `jepa_diagnostics.png` | SIGReg Q-Q, covariance spectrum + effective rank, view-invariance, effective rank over training |
| `latent_geometry.png` | dose-trajectory straightening, geodesic vs Euclidean, prediction surprise, latent energy landscape with toxic basin |
| `representation_alignment.png` | Procrustes JEPA↔DepMap, representational-convergence matrix, layer-wise CKA, latent arithmetic transfer |
| `latent_structure.png` | intrinsic dimension (TwoNN) vs encoder width, embedding trustworthiness, latent density KDE, kNN graph by organ |
| `programs_attention.png` | gene programs along the control→toxic geodesic, attention→TRRUST ROC, attention recovers regulons, per-dim ~N(0,1) |

External ground truth used by the alignment/attention panels: **TRRUST** (signed
TF→target regulatory graph) and **DepMap** (cell-line metadata, gene dependency,
drug sensitivity).

## Regenerate

```bash
uv pip install --system numpy matplotlib scikit-learn scipy   # if needed
PYTHONPATH=. VIZ_OUT=visualizations python -m eb_jepa.singlecell.viz_demo
```
