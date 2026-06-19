# Single-cell visualization plan — what each plot proves, and what feeds it

This is the design spec for the figures that demonstrate the JEPA encoder and
perturbator are *accurate*, not just low-loss. It maps every visualization to
our use case (hepatotoxicity JEPA on Tahoe-100M), the external ground-truth
datasets (**TRRUST**, **DepMap**), and the ablation it disambiguates.

Code lives in `eb_jepa/singlecell/viz.py` (reusable, numpy-in / figure-out).
A synthetic render of the two anchor figures is produced by
`python -m eb_jepa.singlecell.viz_demo` (writes to `viz_out/`).

## 0. Status & the two arrays everything consumes

The single-cell subpackage is at **M0**: LeJEPA loss (`losses.py`) + set-transformer
primitives (`singlecell/layers.py`). No trained encoder / cached data yet, so the
demo figures use **synthetic latents that encode the target pattern** — they
validate the plotting code and show the deliverable, they are not results.

Once the encoder is trained, every function below consumes one of two arrays:

| array | shape | what it is |
|---|---|---|
| `z` | `[N_cells, d_model]` | encoder **pre-projection** latent (CLS or mean-pool). The representation of interest — never the projector output. |
| `Δ` | `[K_drugs, d_model]` | perturbator displacement `centroid(z_perturbed) − centroid(z_control)`, estimated **within each `(cell_line, plate)` stratum** (CLAUDE.md "Control matching"). |

Plus per-cell metadata already in the dataloader: `cell_line_id`, `Organ`, `drug`,
`moa-fine`, `plate`, `sample`, `dose`.

## 1. External datasets as ground truth

**TRRUST** (grnpedia.org/trrust) — manually curated transcriptional regulatory
network: ~8,400 human **TF → target** interactions over ~800 TFs, each **signed**
(Activation / Repression / Unknown) with PMIDs. It is a directed, signed
gene–gene graph. We use it to test whether the *learned* gene geometry recovers
*known* regulation:
- **gene–gene cosine / dendrogram**: genes in one regulon (same TF) should cluster;
  co-activated targets should be closer than random pairs; repressed targets anti-correlated.
- **attention maps**: does a TF token attend to its TRRUST targets above chance?
  Score = AUROC(attention weight vs TRRUST edge indicator). A positive result is a
  strong "the encoder learned biology" claim.
- **causal / knockout** (later): the directed graph is the test oracle for whether
  perturbing a TF in latent space propagates to its targets.

**DepMap** (depmap.org) — Broad Cancer Dependency Map. Three tables we use:
- **Model metadata** (`OncotreeLineage`, Cellosaurus RRID, primary disease): an
  *independent* organ/lineage annotation. Cross-checks our liver filtering
  (HepG2, Huh-7, … are in DepMap) and gives trusted labels to color UMAP/t-SNE.
- **CRISPR gene-effect (dependency)** per gene × line: validate that a gene's
  learned importance (masking sensitivity / probe) tracks real essentiality.
- **PRISM / GDSC drug sensitivity** + dose-response (AUC, IC50): an *external*
  monotonic-dose and MoA signal for the perturbator. Drugs DepMap groups by
  viability profile should share latent displacement direction.

These two give us a **Procrustes / probing oracle**: align our unsupervised JEPA
geometry to an independent measured geometry over the *same* cell lines / genes /
drugs, and report the residual. Agreement = accuracy that is not circular.

## 2. Encoder representation maps

| viz | our use case | "accurate" signal | dataset link |
|---|---|---|---|
| **UMAP / t-SNE** of `z`, colored by `Organ`, `cell_line_id`, `drug`, `moa-fine` | does the unsupervised latent organize cells by biology we never supplied? | clean separation by line/organ; **biological hierarchy** — sublines of one organ form a sub-region (CLAUDE.md). | DepMap `OncotreeLineage` for trusted color/labels. |
| **PCA PC1/PC2** of `z` | linear view; preserves *directions* (use it for displacement fields). | top PCs align with a known axis (depth/pluripotency, cell-cycle). | regress PC vs DepMap/expressed-gene-count. |
| **Covariance eigenspectrum** + **effective rank** | the LeJEPA/SIGReg isotropy target; **collapse demonstration**. | flat-ish spectrum, high effective rank = healthy; a spike / rank≈1 = dimensional collapse. | — (model-intrinsic). |
| **Cosine similarity heatmap + dendrogram** (genes) | gene-token embedding geometry (ESMC+Evo2+count) and contextual gene reps. | regulons block-diagonalize; TRRUST co-regulated pairs > random. | **TRRUST** edges as the oracle. |
| **Attention maps** | which genes the set-transformer routes information between. | TF→target attention recovers TRRUST edges (AUROC); pathway tokens attend to member genes. | **TRRUST**. |
| **Geodesic vs Euclidean** (Isomap graph distance) | the latent is a curved manifold; straight-line distance overstates separation. | dose trajectories are *manifold-straight* (geodesic ≈ path length); curvature quantified. | references: temporal-straightening, koopman-jepa. |

## 3. Perturbator geometry — the headline (Figure 1)

`viz_demo.figure_perturbation` → `viz_out/perturbation_geometry.png`. Four panels,
one claim each:

- **A. Direction field.** PCA(2) of `z`; control cloud in grey at the centroid;
  arrows control → each drug's max-dose centroid, colored by `moa-fine`. Same-MoA
  arrows are **parallel** ⇒ *different drugs go the same direction*. (PCA, not
  UMAP, because a linear map keeps displacement directions faithful.)
- **B. Dose: monotone & bounded.** `‖Δ(drug, dose)‖` vs log-dose, one saturating
  curve per drug; a shaded band marks the **lower bound** (noise floor, dose→0)
  and **upper bound** (saturation ceiling). This is the literal "bounded upper /
  lower" claim. Saturation is `m_max·tanh(dose/dose50)` in the demo; real curves
  cross-checked against DepMap/GDSC dose-response.
- **C. Displacement "spectrum" (drug × latent PC).** Each drug's `Δ` projected onto
  the latent principal axes → a per-drug spectral signature; rows ordered by a
  cosine dendrogram. Same-MoA drugs light up the **same components** — the
  "spectrogram for drugs" idea, made quantitative on the representation eigenbasis.
- **D. Cosine block matrix.** Pairwise cosine of all `Δ`, dendrogram-ordered.
  Block-diagonal ⇒ MoA = a shared direction. This is the **perturbation hierarchy**
  (same `moa-fine` ⇒ similar displacement) as a number, not a vibe.

**Cross-dataset (Procrustes).** Stack our `Δ` (or cell-line centroids) and the
matching **DepMap** drug-sensitivity / dependency vectors over the same items;
`viz.procrustes_align` returns the residual disparity after optimal rotation+scale.
Low disparity = our learned displacement geometry agrees with measured biology.

**Drug chemical spectra.** Separate from latent: a heatmap of drugs × {RDKit
descriptors, Morgan-fingerprint bits} — the input-side "spectrogram". Pair it with
panel C to show the model maps *chemical* similarity → *latent-displacement*
similarity (the drug-featurization claim, CLAUDE.md II).

## 4. Ablations, map-wise — what each knob does *outside the metrics*

The point: a scalar (loss, linear-probe acc) hides *how* a knob reshapes the space.
Each ablation has a **map signature**. Fix a shared validation set; render the same
panels per setting; read the geometry.

`viz_demo.figure_ablation` → `viz_out/ablation_lambda_maps.png` demonstrates the λ
sweep: **λ→0** invariance-only ⇒ *dimensional collapse onto a line* (one spectral
spike, eff.rank≈1); **λ≈0.05** ⇒ rich separated isotropic clusters (spread spectrum,
high silhouette); **λ→1** SIGReg-only ⇒ a clean isotropic Gaussian whose *classes
have dissolved* (flat spectrum AND high rank — looks healthy — but silhouette≈0).
Takeaway baked into the figure: **the spectrum catches collapse; only a structure
metric separates "balanced" from "oversmoothed"** — which is why we pick λ by a map,
not by the loss alone.

| ablation (config knob) | map-wise signature to look for |
|---|---|
| **λ** (SIGReg weight) | collapse-to-line → separated clusters → over-isotropic blob (Fig 2). |
| **# views V** | more views ⇒ tighter invariance ⇒ clusters contract; too few ⇒ noisy centroids, fuzzy arrows. |
| **drop vs mask** view mode | drop changes which genes are *seen* (cluster geometry shifts); mask keeps all tokens (smoother, count-driven). Overlay both UMAPs. |
| **readout** meanpool vs CLS | CLS not polluted by padding (cleaner edges); meanpool smoother. Compare silhouette + edge sharpness. |
| **count mode** A continuous vs B quantile bins | bin boundaries can quantize the manifold (banding in UMAP); continuous is smooth. |
| **L** (tokens/view) | small L ⇒ undersampled cells ⇒ inflated within-cluster variance; watch eff.rank vs L. |
| **encoder scale** (d_model, layers) | the scaling-law axis: separation & eff.rank should grow then plateau — overlay UMAPs across scales. |
| **pathways on/off** | adding an apoptosis pathway token should *sharpen* the apoptosis axis (directional, visible in panel C). |

## 5. One-line "is it accurate" summary per figure

- UMAP/t-SNE separates by organ/line we never labeled → representation is biological.
- Eigenspectrum flat + high eff.rank → no collapse, isotropy target met.
- Gene cosine/dendrogram + attention recover **TRRUST** edges → learned real regulation.
- Procrustes to **DepMap** low disparity → geometry agrees with independent measurement.
- Perturbator field parallel within MoA, dose monotone & bounded, cosine block-diagonal
  → drug effects are directional, dose-ordered, and hierarchical by mechanism.

## 6. Rendered synthetic gallery (`python -m eb_jepa.singlecell.viz_demo`)

Six figures; every panel title states exactly what it shows. Synthetic data ⇒
illustrative target patterns, not results. Drop real `z` / `Δ` arrays into the
same `viz.py` calls once the encoder trains.

- **`perturbation_geometry.png`** — A: perturbation directions (parallel arrows per
  MoA); B: ‖Δ‖ monotone & bounded vs dose; C: displacement spectrum (drug × latent PC);
  D: cosine block matrix of displacements.
- **`ablation_lambda_maps.png`** — λ→0 collapse-to-line / λ≈0.05 rich / λ→1 merged,
  each with its covariance spectrum (eff.rank + silhouette annotated).
- **`factor_recovery_circle.png`** — **de-distorting (LeWM-style)**: a circular factor
  (cell-cycle phase) observed through a nonlinear warp; PCA stays distorted, a
  manifold/JEPA encoder recovers the clean ring; recovered-vs-true phase is the
  diagonal ⇒ factor recovered **up to isometry**. Real factors → primitives:
  cell-cycle = circle, dose = ray, differentiation = curve, expressed-gene count = axis.
- **`jepa_diagnostics.png`** — A: SIGReg Q-Q of random latent slices vs N(0,1)
  (isotropic on-line, heavy-tailed peels off — note a *single* random slice
  Gaussianizes by CLT, so this pairs with the spectrum); B: covariance spectrum +
  effective rank; C: view-invariance threads (V views → cell centroid); D: effective
  rank over training (collapse vs healthy).
- **`latent_geometry.png`** — A: dose-trajectory straightening (JEPA vs expression);
  B: geodesic vs Euclidean distance (manifold curvature); C: prediction-surprise bars
  (spike on implausible perturbation, flat on batch shift); D: latent energy landscape
  with a toxic basin + control→toxic geodesic.
- **`representation_alignment.png`** — A: Procrustes JEPA↔DepMap (disparity); B:
  representational-convergence matrix (JEPA / DepMap-dependency / CCLE-expression);
  C: layer-wise CKA (how the representation forms with depth); D: latent arithmetic
  (a drug's effect transfers across cell lines).
- **`latent_structure.png`** — A: intrinsic dimension (TwoNN) vs encoder width
  (rise→plateau, the scaling-law axis); B: embedding trustworthiness vs k (is the
  2-D map faithful?); C: latent density KDE (modes filled, ball-bounded, no voids);
  D: kNN graph on the latent (colored by organ).
- **`programs_attention.png`** — A: latent traversal — gene programs switch along the
  control→toxic geodesic; B: gene attention predicts **TRRUST** edges (ROC/AUROC);
  C: gene–gene attention recovers TRRUST regulons (TF-ordered blocks); D: per-dimension
  latent histograms (~N(0,1) under SIGReg).

New `viz.py` helpers backing these: `path_straightness`, `linear_cka`,
`intrinsic_dimension_twonn`, `random_slice_quantiles` (+ existing `procrustes_align`,
`geodesic_distances`, `covariance_eigenspectrum`, `effective_rank`, `cosine_matrix`,
`dendrogram_order`). Trustworthiness / KDE / kNN-graph / ROC use scikit-learn & scipy
directly in the demo.
