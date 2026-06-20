# CLAUDE.md

## Introduction

This project trains a **JEPA (Joint-Embedding Predictive Architecture)** model for a hackathon. The general hackathon brief is in `sujet.pdf`. Our solution: a JEPA of drug-induced perturbations on liver cells, aimed at assessing **hepatotoxicity**.

Data comes from the public **Tahoe-100M** dataset (https://huggingface.co/datasets/tahoebio/Tahoe-100M). The project has two main parts: training an **encoder**, then training a **perturbator** on top of that encoder.

Two-phase plan:
1. **Pretrain** on all cell lines (every tissue) to learn a general representation.
2. **Specialize** on liver: finetune the encoder on liver cell lines and refine the perturbator to predict hepatotoxic drug effects.

Your job: keep the codebase clean and modular, and run the experiments on the cluster. Read this file carefully and understand the codebase before acting.

---

# I. Encoder

The encoder takes a **cell profile** — transcriptomic data plus (optionally) pathway tokens — and produces a latent representation of the cell's biological state.

## Token embeddings

Each gene is a **token** fed to the JEPA transformer backbone. Tahoe cells carry ~60k genes. A gene token embedding is the **sum** of three projected components:

1. **Protein embedding (coding genes only):** average pooling of **ESMC 600M** per-residue embeddings over the protein (canonical isoform). 
2. **DNA embedding (all genes):** **Evo 2** embedding of the gene's nucleotide sequence (canonical transcript / gene body), extracted at the appropriate layer to yield biologically meaningful representations — the exact layer/pooling is selected by a small validation rather than assumed.
3. **Count embedding (all genes):** an embedding of the RNA count, normalized and encoded exactly as described in **Normalization / Count embedding** (CP10k + log1p, then mode A continuous or mode B quantile binning). That subsection is the single source of truth and applies identically to pathway counts.

**Composition rule by gene type:**
- **Coding gene:** protein (ESMC) + DNA (Evo 2) + count.
- **Non-coding gene:** DNA (Evo 2) + count (no protein term).

ESMC and Evo 2 embeddings are **precomputed once and cached**, indexed by `token_id → ensembl_id`. Because ESMC, DNA, and count have different dimensions, each is **linearly projected to `d_model`** before being summed.

## Views and the LeJEPA objective

A LeJEPA encoder is trained by aligning, in latent space, several **views** of the same cell. Our V views are formed by either:
- **drop**: keeping a subset of gene tokens, or
- **mask**: keeping all tokens but replacing the *count* embedding of some genes with a learned **mask** embedding (the true count is hidden).

We benchmark both view-construction strategies (drop vs mask).

**Training follows LeJEPA, not I-JEPA**, implemented per the official reference (`galilai-group/lejepa`, `MINIMAL.md` — this is the single source of truth for the loss). All V views pass through the **same shared encoder → projector**, producing `proj` of shape `[V, N, d_proj]` (V views, N cells in the batch, projector dim). The objective is a convex combination of two terms with a **single** constant trade-off λ:

```python
# proj: [V, N, d_proj] — projected embeddings of the V views (NOT L2-normalized)
inv_loss    = (proj.mean(0) - proj).square().mean()         # invariance: each view → centroid over views
sigreg_loss = sigreg(proj)                                  # SIGReg (sliced Epps–Pulley, see below)
lejepa_loss = cfg.lamb * sigreg_loss + (1 - cfg.lamb) * inv_loss
```

1. **Invariance** = squared deviation of each view's projection from the **mean over views** (centroid form; it generalizes naturally to V views — do **not** use a pairwise 2-view MSE).
2. **SIGReg** (Sketched Isotropic Gaussian Regularization) constrains the projected distribution toward an isotropic Gaussian via a **sliced Epps–Pulley** Gaussianity test over random 1-D projections — provably minimizing downstream prediction risk.

λ is **small** (reference λ ≈ 0.02 → strong invariance bias) and **constant**. LeJEPA is **heuristics-free**: **no teacher–student / EMA target, no stop-gradient, no separate predictor network, and no loss-coefficient/teacher schedulers** (a standard LR warmup+cosine schedule *is* allowed; only loss-side schedulers are forbidden).

**SIGReg (exact, per `MINIMAL.md`).** Symmetric Epps–Pulley quadrature on `t ∈ [0, t_max]` (reference `t_max=3`, `knots=17`), Gaussian window `φ(t)=exp(-t²/2)`, trapezoid weights folded into `weights = trapz_w · φ`. Each step draws **fresh** random projections `A ∈ R^{d_proj × S}` (S slices), L2-normalized per column. The empirical characteristic function is split into real/imag parts; the statistic compares the per-sample-mean `cos` to `φ` and the mean `sin` to 0, scaled by the batch size N:

```python
class SIGReg(nn.Module):
    def __init__(self, num_slices=256, knots=17, t_max=3.0):
        super().__init__()
        self.num_slices = num_slices
        t = torch.linspace(0, t_max, knots)
        dt = t_max / (knots - 1)
        w = torch.full((knots,), 2 * dt); w[[0, -1]] = dt          # trapezoid
        window = torch.exp(-t.square() / 2)
        self.register_buffer("t", t); self.register_buffer("phi", window)
        self.register_buffer("weights", w * window)

    def forward(self, proj):                                       # proj: [..., N, d_proj]
        A = torch.randn(proj.size(-1), self.num_slices, device=proj.device)
        A = A / A.norm(p=2, dim=0)                                 # L2-normalize each slice
        x_t = (proj @ A).unsqueeze(-1) * self.t                    # [..., N, S, knots]
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights) * proj.size(-2)).mean()       # × N (test-statistic scaling)
```

**Distributed.** `cos().mean(-3)` / `sin().mean(-3)` average the ECF over the per-rank batch; under multi-GPU these means must be **all-reduced (AVG)** across ranks so the test sees the global batch, and the projection RNG must stay **lock-step** across ranks. `S` (slices), `knots`, and `t_max` are config parameters — LeJEPA scales `S` large (e.g. ~1024) for high-dimensional representations.

Both losses act **after a projector** (https://arxiv.org/abs/2304.12210). The projector output is **not** L2-normalized — SIGReg targets an isotropic Gaussian, so L2-normalizing onto a sphere would contradict the target. The **representation of interest is pre-projector** — the final CLS or mean-pooled gene representation — used for probing and all downstream tasks. An **online linear probe** may be trained jointly but always on a **detached** copy of the pre-projection latent (no gradient to the encoder). The training loss is a strong label-free model-selection signal (high Spearman correlation with downstream accuracy), usable to tune λ and architecture.

The architecture must be sized to the dataset and the GPU budget.

## Encoder architecture

**Set-based transformer:** genes form an **unordered set**, so **no positional encoding / RoPE** on gene tokens; the **CLS** token (when enabled) is the only token with a dedicated identity. Choices are dictated by sequence length (62,710 genes reduced to L per view), `torch.compile` (constant shapes), and the 8×B200 budget.

**Block:** pre-norm RMSNorm (no bias), SwiGLU FFN (d_ff ≈ 2/3·2·d_model), GQA 4:1 (`n_kv_heads = n_heads//4`, KV shared via `repeat_kv`), SDPA backend (Flash Attention 3 auto on Hopper/Blackwell, sdpa fallback, no eager). Dropout = 0 (view drop/mask is the regularizer). Xavier init on qkv/o/proj, residual scaled ×1/√(2·n_layers). Stochastic depth (p=0.1) only if n_layers ≥ 16.

**Forward (one view):** precomputed gene embeddings (ESMC + Evo 2 projected → d_model, summed) + count embedding (mode A: MLP on the log scalar; mode B: `n_bins → d_model` table) → optional pathway tokens (identity + count) → optional `[CLS]` prefix → n_layers × {RMSNorm, GQA, residual; RMSNorm, SwiGLU, residual} → readout → **JEPA projector** (MLP d_model→d_proj, hidden **BatchNorm1d**; output **not** L2-normalized — SIGReg targets an isotropic Gaussian, see *Views and the LeJEPA objective*; used **only** by the losses).

**Readout** (flag, identical forward): `meanpool` (default, `z = mean(h[mask])`, consistent with single-cell literature) or `cls` (`z = h[:,0]`, not polluted by padding). Both benchmarked. The representation of interest (probing, downstream) is z **pre-projection**.

**Memory budget** (L=4096, batch=32, V=4, d_model=1024, bf16): the cost is V forward passes through the shared encoder (no separate predictor). Roughly ~25 GB/rank on B200 (183 GB HBM) per view-pass, with margin to push L=8192 or batch=64; beyond that → gradient checkpointing or FSDP. The cluster is **8× B200** connected by NVLink; first target is DDP + gradient checkpointing.

A later extension could add a **hierarchical encoder** with inductive biases built on hallmark pathways (à la hierarchical JEPA / HRM: two attention levels, an abstract level over gene-level tokens connected by pathway).

## Probing

To validate the representations, we train several probes on the **pre-projection** latent. Direct probes: organ, cell line, drug, sample, etc. We also train **regression probes** for hallmark-pathway prediction, and a regression on total number of expressed genes (a proxy for pluripotency).

Probes are trained **detached** from the encoder so they send it no supervision signal.

We want interpretable logging: explained-vs-total variance for pathway regressions, imbalance-aware metrics (not raw accuracy) for classification, etc.

## Pathways

Optionally, views may carry extra **pathway tokens** characterizing the cell's biological profile, using the **hallmark pathways** (official published version). A pathway "count" is the weighted sum of its genes' counts (hallmark weights). A pathway token embeds: a **learned pathway-identity** embedding (e.g. one for apoptosis, one for growth), plus a **count vector and count embedding** encoded *exactly* as gene counts (see Normalization) for gene/pathway consistency. Pathway tokens are dropped/masked like genes but with a **distinct probability**. Adding chosen pathways biases the latent toward what we want represented (e.g. an apoptosis pathway in some views → finer apoptosis representation). **Anticipate the implementation but do not build it yet.**

## Baselines / comparisons

To show JEPA's superiority over other architectures, the project must also support alternative encoder backbones: **MAE, VAE, and plain PCA**.

---

# II. Perturbator

The perturbator is the second part. Given an **action** — the drug, featurized coherently with the chosen task, combined with its **dose** — it predicts the latent representation of the perturbed cell.

We do **not** have paired control+perturbation data, only **control distributions** and **perturbed distributions**. Learning therefore proceeds by **optimal transport** (e.g. a **sliced Wasserstein** distance), mapping an unperturbed distribution to a perturbed one.

The perturbator is a **transformer** operating on the encoder's outputs, conditioned on the small-molecule representation (identity + dose). It may consume the **full** encoder output (all gene tokens, not just the pooled latent) — to be tested.

**Encoder is frozen** for the first version of the perturbator; finetuning the encoder jointly is a later experiment.

## Optimal-transport objective and control matching

For each prediction, the loss is the distance from the predicted cell to the target distribution, estimated with a **sliced Wasserstein** distance whose **number of slices is configurable**. Target embeddings are kept in a memory of arrival-distribution embeddings.

**Control matching (resolved, exact).** Control = cells with `drug == "DMSO_TF"`. The OT problem is defined **per stratum `(cell_line_id, plate)`**:
- **Source** distribution = the `DMSO_TF` (control) cells of that exact `(cell_line, plate)`.
- **Target** distribution = the cells treated with drug *d* at dose *c* on that **same** `(cell_line, plate)`.

Matching on the same plate **and** the same line guarantees that batch/technical effects are shared between source and target, so the perturbator learns the drug effect, not a plate artifact. The perturbator maps the source latent distribution, conditioned on (drug identity + dose), to the target latent distribution, minimizing the sliced Wasserstein distance.

## Drug featurization

The hard part. Two approaches:

1. **Hepatotoxicity-focused** (with the liver-finetuned encoder): chemistry features most informative for hepatotoxicity — CYP-enzyme metabolite prediction, BSEP, NRF2, mitochondrial signals, etc. Use Lucas's **virtual pathways** repo for this.
2. **General encoder + perturbator:** drugs featurized from **RDKit** descriptors, **Morgan fingerprints**, and — crucially — a meaningful embedding of their **target**, so the model can match the drug embedding against gene embeddings and infer the target across the architecture.

## Architectures / objectives to test

- **Conditioning:** FiLM inside a transformer; alternating self- and cross-attention; other relevant designs.
- **Objective:** direct prediction of the perturbed state (perturbator meanpool/CLS = encoder's perturbed meanpool/CLS), **or** flow matching with rigorous ODE inference for validation.

## Dataset

We use **Tahoe-100M** (`tahoebio/Tahoe-100M`): ~95.6M drug-perturbed single cells across **50 cancer cell lines**, **379 distinct drugs** (~1,100 drug×dose perturbations), vocabulary of **62,710 genes** (Ensembl release 109 / GRCh38). On Dalia it lives at `/lustre/work/vivatech-unaite/shared/tahoe-100m` (~429 GB parquet).

### Tables (HuggingFace configs)

- **`expression_data`** (main, ~95.6M rows, one per cell): `genes` (`list<int64>` gene token_ids), `expressions` (`list<float32>` raw UMI counts, aligned 1:1 with `genes`), `drug` (str; control = `"DMSO_TF"`), `sample` (str → `sample_metadata`), `cell_line_id` (str, Cellosaurus `CVCL_*` → `cell_line_metadata.Cell_ID_Cellosaur`), `moa-fine` (str), `canonical_smiles` (str), `pubchem_cid` (str), `plate` (str, `"plate1"`…`"plate14"`), `BARCODE_SUB_LIB_ID` (str → `obs_metadata`). The **first element of both `genes` and `expressions` is a CLS marker, stripped by position** (index 0); the remaining `expressions` are raw counts.
- **`gene_metadata`** (62,710 rows): `token_id` (int64), `gene_symbol` (str), `ensembl_id` (str). The token_id ↔ ensembl_id map driving the ESMC/Evo2 embedding cache. **Gene `token_id`s span 3..62712** (62,710 distinct values); ids 0–2 are reserved special tokens — **CLS = token id 1** (its expression value is the sentinel `-2.0`). Any table indexed by raw `token_id` (count bins, gene-embedding lookups) must therefore have size **62713** (`max token_id + 1`), not 62710.
- **`sample_metadata`** (PK `sample`): `plate`, `drug`, `drugname_drugconc`, mean QC stats. **`drugname_drugconc`** is the dose source — a string repr of a list of tuples, e.g. `"[('Infigratinib', 0.05, 'uM')]"` (control: `"[('DMSO_TF', 0.0, 'uM')]"`).
- **`drug_metadata`** (379 rows, PK `drug`): `targets`, `moa-broad`, `moa-fine`, `human-approved`, `clinical-trials`, `canonical_smiles`, `pubchem_cid` (float64 here).
- **`cell_line_metadata`** (one row per driver gene per line; 50 lines): `cell_name`, `Cell_ID_DepMap`, **`Cell_ID_Cellosaur`** (join key to `expression_data.cell_line_id`), **`Organ`** (e.g. `"Liver"`), driver-mutation fields.
- **`obs_metadata`** (~100.6M rows, per-cell QC, PK `BARCODE_SUB_LIB_ID`): `gene_count`, `tscp_count`, `mread_count`, `pcnt_mito`, `S_score`, `G2M_score`, `phase`, `pass_filter`, `cell_line` (CVCL). Source of QC filters and the gene-count probe target.
- **`pseudobulk_differential_expression`**: precomputed DESeq2-style DE (drug vs DMSO per line): `log2FoldChange`, `padj`, `concentration`, `Cell_ID_Cellosaur`, `drug`, … Use as a **probe/eval target**, not training input.

### Normalization / Count embedding (dataloader)

Fixed transform: per-cell depth normalization (**CP10k**) then **log1p**, applied to raw counts on **non-zero** genes; the same stats apply to pathway counts.

Value encoding is configurable between two benchmarked modes:
- **(A) Continuous:** MLP projection of the log scalar + a learned mask vector.
- **(B) Quantile binning:** per-gene quantile bins computed globally (default **64 bins**) + a dedicated `[MASK]` token. Global binning guarantees that a given `(gene, value)` maps to the **same bin across all drop/mask views**, keeping SIGReg stable.

The mode-B bins are **precomputed and saved** on Dalia at `/lustre/work/vivatech-unaite/shared/tahoe-cache/`: `quantile_bins.npy` (`[62713, 63]` edges, indexed by raw `token_id`) + `quantile_bins.json`, the full streaming `gene_count_histogram.npy` (`[62713, 4096]`, so any `n_bins` can be re-derived without rescanning), and `quantile_stats.pt`. Fit over **all 95.6M cells** (139.6 B observations) via `eb_jepa.datasets.tahoe.normalizer.GeneHistogram`; 54,884 of 62,710 genes are ever expressed (unseen genes get all-zero edges → bin 0). Bin occupancy on the busiest genes is uniform to ±0.0001 of the ideal 1/64.

### Liver filtering

There is no organ field in `expression_data`. Build the hepatic set upstream by joining `cell_line_id` → `cell_line_metadata.Cell_ID_Cellosaur` and keeping `Organ == "Liver"`, then filter the stream on that set of `CVCL_*` ids. These are **cancer lines** (HepG2, Huh-7…), not primary tissue.

### Raw-format parsing

Cells are **sparse**: `genes` = token_ids of non-zero genes, `expressions` = aligned raw counts. **Strip the first element of both by position** (the CLS marker; do not filter by a hardcoded token_id/value). Keep the sparse representation (1 gene = 1 token) for the JEPA; only **densify** into the fixed 62,710-gene vocabulary for the MAE/VAE/PCA baselines.

### Encoder dataloader output

`__getitem__` returns the **whole cell** (variable length): `gene_token_ids`, the CP10k+log1p value (mode-agnostic; `bin_id` or log scalar produced at collate time), plus probing metadata (`cell_line_id`, `organ` from the prebuilt CVCL→Organ map, `drug`, `moa_fine` read from column `moa-fine`, `plate`, `sample`, `log_conc`, `canonical_smiles`). The V views (drop/mask, distinct gene vs pathway probabilities) are generated **on-the-fly in the collate**, never precomputed. The collate also **samples/pads** each view to a fixed budget of **L tokens** (+ attention mask for padding) for constant, compilable shapes; **L is configurable**.

### Schema facts to respect

- Control = `drug == "DMSO_TF"` (in `expression_data`), matched by `(plate, cell_line_id)` for the perturbator (see Control matching above).
- **Dose**: parse `sample_metadata.drugname_drugconc` (join on `sample`) with `ast.literal_eval` → `[(name, conc, unit)]`; assert `unit == "uM"`, convert `log_conc = log10(conc · 1e-6)`. Controls have `conc == 0.0` → mask/sentinel the dose (never `log10(0)`).
- The perturbator action is featurized via `canonical_smiles`, **not** a hash of the drug name.
- The `cell_line_metadata` join key is `Cell_ID_Cellosaur` (there is no `cell_line_id` column there); `obs_metadata` names its Cellosaurus column `cell_line`.
- `pubchem_cid` dtype differs across tables (str in `expression_data`, float64 in `drug_metadata`) — cast to a nullable string.

### Gene-embedding cache (ESMC + Evo2)

Built **once, offline on the cluster**, keyed by `token_id → ensembl_id` (from `gene_metadata`, release 109/GRCh38). Per gene: resolve the canonical transcript (Ensembl REST, pinned to release 109), its biotype (`protein_coding` → coding), the canonical protein sequence (coding only) and the canonical transcript DNA (all genes). Then:
- **ESMC** (`esmc-600m`, **1152-dim** per residue): mean-pool over residues → one vector per coding gene. Non-coding genes get no protein term.
- **Evo2** (`arcinstitute/evo2_7b_base`, **d_evo2 = 4096**): run the DNA sequence and extract **layer 24, mean-pooled** → one vector per gene.

Store two frozen lookup tables — `esmc.npy` `[n_coding=20063, 1152]` (coding genes only) and `evo2.npy` `[n_genes=62710, 4096]` (all genes) — plus `index.parquet` `(token_id, ensembl_id, is_coding, esmc_row, evo2_row)` and `metadata.json` (~1.1 GB total). The encoder learns `Linear(1152→d_model)` and `Linear(4096→d_model)` over these frozen vectors and sums with the count embedding; `GeneTokenEmbedding.from_cache` sizes the lookup tables to `max(token_id)+1 = 62713` so they are indexed directly by raw token_id. The cache is **fetched once** with `python scripts/fetch_gene_embeddings.py --out <dir>` (public MinIO, no creds) and lives on Dalia at `/lustre/work/vivatech-unaite/shared/gene_emb_cache`. **Validate** before trusting it: confirm `genes`/`expressions` are 1:1 aligned and the first element is always the CLS marker on a sample of real rows. (61,428 of 62,710 genes have a nonzero Evo2 vector; the rest had no usable transcript sequence and fall back to count-only.)

### Preprocessing & cache (cluster)

Prep pass over `/lustre/work/vivatech-unaite/shared/tahoe-100m`: stream → filter liver (specialize phase) → subsample → write to local NVMe (Arrow/Parquet or sparse memmap) each cell as `(gene_token_id, CP10k+log1p-normalized value)` — value **continuous and mode-agnostic**.

Normalization stats (per-gene quantile boundaries, pathway stats) are computed on the cached sample and stored separately; binning (mode B) or projection (mode A) is applied **at load time**, so we can change mode or K **without re-caching**.

Generalization splits are defined **before** caching and at **group level** (held-out line / organ / drug, never cell-level, to avoid leakage). Training reads from cache with high `num_workers`; views are generated on the fly.

---

# III. Code and experiments

The project must stay clean and structured from the start — no spaghetti code, no time-sinking bugs. Be deliberate about architecture choices; avoid bugs proactively.

- Base repo: https://github.com/marinabar/eb_jepa — maintain it cleanly via git.
- Use **uv**, clean packaging, **wandb** for experiment logging.
- **Every run fully configurable from a single config file** (number of views, latent dim, λ, SW slice count, L, readout, view mode, count mode, etc.) — modular and anticipated from the start.
- Torch: use the right modules for embeddings/attention; keep **constant shapes** so the model is `torch.compile`-able (notably during training).
- Unit tests for sensitive components in `test/00_unit/…`.
- Must run on **multiple NVLink-connected GPUs**; mind both compute and memory utilization.

## Objectives

Establish rigorous **scaling laws** for the encoder (in compute and in data: amount and **diversity**), with clean visualizations, inferred scaling-law parameters, and a **rigorous diversity metric**. For a fixed architecture at several scales, compute the **FLOP budget of the trained parts only**. Use a **fixed validation set shared across all scales**.

These scaling laws are set against the paper reporting an **absence of scaling laws for single-cell encoders**: the goal is to see whether our setup (multi-view, controlled/measured data diversity) reveals a scaling regime where that paper found none — or confirms its absence.

## Success criteria

The headline deliverable: **JEPA beats well-tuned MAE / VAE / PCA baselines on rigorous metrics** (scib optional but acceptable). "Well-tuned" matters — the baselines must be properly tuned for the comparison to be credible.

## Representation visualizations & analyses

- **tSNE/UMAP** colored by class (cell line, organ, …); spectrum of the representation covariance matrix; explicit **collapse demonstrations** when the SIGReg weight is too high or too low (visualize rich → collapsed, then justify the chosen λ).
- **Structure & hierarchy:** check the representation exposes a biological hierarchy (e.g. several tissues of one organ form a sub-region of that organ's representation); test word2vec-style **transfer vectors** (latent arithmetic) and with/without-drug, with/without-disease separation.
- **Perturbator-specific:** dose **monotonicity** (latent trajectory varies monotonically with dose) and **perturbation hierarchy** (same-MoA `moa-fine` drugs produce similar latent displacements).
- **Advanced / exploratory:** causal field theory for gene-knockout modeling in internal representations (use Lucas's transformer causal-field-theory code when available).
- **Pedagogy:** large explanatory schematics of inputs and architecture for the hackathon deliverable.

## Visualization design system

Every figure must be **elegant** — publication-grade for the hackathon deliverable, never a default-matplotlib dump. **All visuals are saved to the repo's `visualizations/` folder** (the single canonical home); figures generated on the cluster are staged at `/lustre/work/vivatech-unaite/ljung/visualizations/` and synced into `visualizations/`.

House style (apply consistently):
- **Palette:** ink `#1d2433` (text/lines), muted `#7a8699` (subtitles/labels/ticks), accent `#2a6f97` (primary fill), grid `#e9edf2`; alternate accent `#3f8bb5`. Categorical colorings use a perceptually-even qualitative palette.
- **Layout:** white background; top & right spines removed; thin `#c7cfdb` axes; light **x-only** grid; no y-ticks on density plots. Titles **bold, left-aligned** with a small grey one-line subtitle giving the key stat/context.
- **Form:** gradient-filled densities (alpha fade to 0); `dpi≥200`; tight `bbox`; DejaVu Sans (or a cleaner serif when available). Provide **PDF/SVG** alongside PNG for anything entering the written deliverable.
- **Honesty:** display-only smoothing (e.g. Gaussian over the count "comb" from discrete UMIs) is fine but must never alter the saved data; annotate sample sizes and what each cell/point represents.

## Benchmarks

Use the single-cell **scib** library to compare against standard atlas representations.

---

# Cluster access

You'll get access to different GPU clusters per the instructions given. Always aim for **maximum GPU utilization** and sound memory usage.

## Dalia (IDRIS) — the hackathon GB200 cluster

The hackathon runs on **Dalia** at IDRIS (Hack The World(s) / Vivatech event). Access only works from the competition network.

**Connect:** `ssh -i ~/.ssh/unaite_ljung ljung@dalia.idris.fr` (ed25519 key, user `ljung`). Login node is `dalia2`.

**Architecture mismatch (critical):** the login node is **x86_64**, but the compute nodes are **aarch64** (Grace ARM) with **NVIDIA GB200** GPUs (sm_100, 198 GB HBM each). Binaries/venvs built on the login node fail on compute with "Exec format error". **Build the Python env *on* a compute node** via `srun`. Compute nodes have internet.

**SLURM:** one partition `defq`, 18 nodes × 4 GB200 (72 GPUs). All nodes are held by an **ACTIVE reservation `Vivatech`** (account `vivatech`), so every job MUST pass `--reservation=Vivatech` or it sits PENDING with "ReqNodeNotAvail". Default `srun` gives 1 CPU — pass `--cpus-per-task=N`. `--pty` does **not** work over non-interactive SSH; omit it.

```bash
srun --reservation=Vivatech --partition=defq --gres=gpu:b200:4 \
     --cpus-per-task=64 --time=HH:MM:SS \
     /lustre/work/vivatech-unaite/ljung/venv-arm/bin/python script.py
```

**Storage:** home `/lustre/home/extusers/ljung` is tiny (3 GB, tight **inode** quota — unpacking Python there blows the inode limit). Work dir `/lustre/work/vivatech-unaite/ljung` has 10 TB, no quota — put venvs, caches, and data there. The group `vivatech-unaite` is shared with teammates (mbarannik = Marina, lpahlawan, rsiahaang) and has a shared dir `/lustre/work/vivatech-unaite/shared`. Always redirect caches off home: export `XDG_CACHE_HOME`, `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `XDG_DATA_HOME`, `HF_HOME` under the work dir.

**Prebuilt aarch64 env:** `/lustre/work/vivatech-unaite/ljung/venv-arm/bin/python` (torch 2.11.0+cu128, verified on a GB200). aarch64 `uv` is at `/lustre/work/vivatech-unaite/ljung/bin-arm/uv`.

**Tahoe-100M data** lives at `/lustre/work/vivatech-unaite/shared/tahoe-100m` (downloaded from HF `tahoebio/Tahoe-100M`). The `/data/tahoe-100m` path mentioned elsewhere does **not** exist on Dalia.

---

# Maintaining this document (living spec)

This file is the **single source of truth** and a **living document**. Whenever reality contradicts what is written here — most often when the actual **Tahoe-100M schema / data reading** differs from the description (column meanings, sparse layout, CLS marker, dose parsing, control/plate semantics, the liver join, …), but equally for any architecture, loss, or training detail discovered during implementation — **update this file in place** to reflect the corrected knowledge.

State the corrected fact **directly and assertively** ("the dataset is structured like X", "counts are stored as Y"). Do **not** preserve or contrast with the old wording — no "it's not A but B", no changelog of superseded versions. Keep edits surgical and consistent with the surrounding sections so the document always reads as one clean, accurate spec, never an accumulation of corrections.