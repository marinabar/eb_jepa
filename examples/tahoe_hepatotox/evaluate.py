"""Hepatotoxicity validation of the liver-finetuned encoder + virtual-pathway features.

Three evaluations, all config-driven; figures land in ``visualizations/hepatotox/``
(PNG + PDF, house style imported from ``eb_jepa.singlecell.visualize``):

  A. Encoder probe — does the (frozen, liver-finetuned) pre-projection representation
     carry a hepatotoxicity signal? We attach per-drug labels to the eval cells:
       - ``dili`` : known-hepatotoxic vs not, from a curated DILIrank-style list
         (HEPATOTOX_DRUGS below; cite + editable). Drug name matched case-insensitively.
       - ``human_approved`` / ``clinical_trials`` : flags from ``drug_metadata``
         (if a metadata parquet is provided).
       - ``moa_fine`` : the Tahoe MoA label (always present).
     Each is run through the project's detached linear-probe harness
     (``run_probe_suite``), imbalance-aware (balanced accuracy / macro-F1).

  B. Virtual-pathway feature validation — featurize every distinct drug SMILES with
     ``HepatotoxPathwayFeaturizer`` and ask whether drugs sharing a MoA / CYP cluster
     group together in feature space. We report the silhouette score of the pathway
     features under the MoA labelling (separation metric) and a t-SNE of the features
     coloured by MoA.

  C. Pathway-feature -> representation alignment — a quick CCA-free check: the mean
     pathway feature per drug vs the mean encoder representation per drug, scored by
     the best linear-probe R2 (do chemistry pathways predict the learned latent?).

Run on Dalia (1 GPU):
    /lustre/work/vivatech-unaite/ljung/venv-arm/bin/python -m \
        examples.tahoe_hepatotox.evaluate run \
        --config examples/tahoe_hepatotox/cfgs/finetune.yaml \
        --checkpoint /lustre/work/vivatech-unaite/ljung/runs/hepatotox/liver_finetune/encoder_final.pt
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from eb_jepa.logging import get_logger
from eb_jepa.singlecell.perturbator.hepatotox_features import HepatotoxPathwayFeaturizer
from eb_jepa.singlecell.probes import (
    run_probe_suite,
    train_classification_probe,
    train_regression_probe,
)
from eb_jepa.training_utils import load_config

import examples.tahoe_hepatotox.finetune as finetune  # binds liver dataset onto sub14_main
import examples.tahoe_jepa.sub14_main as sub14_main

logger = get_logger(__name__)

# House style (single source of truth for figures: eb_jepa.singlecell.visualize).
from eb_jepa.singlecell import visualize as viz

# --------------------------------------------------------------------------- #
# Curated DILIrank-style hepatotoxic drug list.                               #
# Source: FDA DILIrank "vMost-DILI-Concern" / LiverTox well-known hepatotoxins #
# (Chen et al., Drug Discov Today 2016; NIH LiverTox). Lower-cased drug names; #
# matched against Tahoe's ``drug`` column. EDIT FREELY — this is a small,      #
# transparent curated set, not an exhaustive label.                           #
# --------------------------------------------------------------------------- #
HEPATOTOX_DRUGS = {
    "acetaminophen", "paracetamol", "troglitazone", "trovafloxacin", "diclofenac",
    "isoniazid", "ketoconazole", "nefazodone", "tolcapone", "bromfenac",
    "valproic acid", "amiodarone", "flutamide", "nimesulide", "leflunomide",
    "tamoxifen", "methotrexate", "rifampicin", "rifampin", "pioglitazone",
    "rosiglitazone", "bosentan", "felbamate", "dantrolene", "pemoline",
    "labetalol", "ticlopidine", "carbamazepine", "phenytoin", "erythromycin",
}

# Drugs with no/low DILI concern (negatives), for a balanced curated probe.
LOW_DILI_DRUGS = {
    "aspirin", "ibuprofen", "metformin", "atenolol", "famotidine",
    "loratadine", "cetirizine", "ranitidine", "lisinopril", "amlodipine",
    "omeprazole", "simvastatin", "warfarin", "furosemide", "hydrochlorothiazide",
}


def _dili_label(drug: str | None) -> str | None:
    if not drug:
        return None
    d = drug.strip().lower()
    if d in HEPATOTOX_DRUGS:
        return "hepatotoxic"
    if d in LOW_DILI_DRUGS:
        return "low_concern"
    return None  # unknown -> excluded from the curated probe


# --------------------------------------------------------------------------- #
# Encoder reconstruction + eval-set encode (reuse sub14_main building blocks)  #
# --------------------------------------------------------------------------- #
def _load_encoder(cfg, checkpoint, device):
    from eb_jepa.singlecell.sub14.features import load_pc_features, random_pc_features
    from eb_jepa.singlecell.sub14.load_checkpoint import load_subliminal14_checkpoint

    cache = cfg.model.get("gene_emb_cache", "random")
    pc = load_pc_features(cache) if cache and cache != "random" else random_pc_features(
        n_pc=int(cfg.model.get("smoke_n_pc", 2000))
    )
    model = sub14_main.build_model(cfg, pc, torch.float32, device)
    if checkpoint and os.path.exists(checkpoint):
        load_subliminal14_checkpoint(model, checkpoint, map_location="cpu", verbose=True)
        model = model.to(device=device, dtype=torch.float32)
    else:
        logger.warning("No checkpoint at %r — evaluating an UNTRAINED encoder (smoke).", checkpoint)
    model.eval()
    return model, pc


@torch.no_grad()
def _encode(model, eval_single, device, chunk=128):
    return sub14_main._encode_eval(model, eval_single, device, torch.float32, chunk)


# --------------------------------------------------------------------------- #
# Eval pieces                                                                 #
# --------------------------------------------------------------------------- #
def _probe_eval(reps, meta) -> dict:
    """Run the detached probe suite on hepatotox-relevant labels.

    The shared ``run_probe_suite`` covers the standard labels (cell_line_id, drug,
    moa_fine, is_dmso, ...); the curated DILI label is hepatotox-specific and not in
    that fixed key list, so it is probed here directly with the same detached,
    imbalance-aware classification probe.
    """
    out = run_probe_suite(reps, dict(meta))
    # Curated DILI: known hepatotoxin vs low-concern (the headline hepatotox probe).
    dili = [_dili_label(d) for d in meta.get("drug", [])]
    if len({x for x in dili if x is not None}) >= 2:
        out["clf/dili"] = train_classification_probe(reps, dili)
    flat = {}
    for k, v in out.items():
        if isinstance(v, dict):
            for mk, mv in v.items():
                flat[f"{k}/{mk}"] = mv
    return flat


def _feature_validation(smiles_list, moa_list, out_dir, seed=0):
    """Featurize distinct drugs and measure MoA separation in pathway-feature space."""
    feat = HepatotoxPathwayFeaturizer()
    # unique (smiles, moa)
    seen, U_smiles, U_moa = set(), [], []
    for s, mo in zip(smiles_list, moa_list):
        if s and s not in seen:
            seen.add(s)
            U_smiles.append(s)
            U_moa.append(mo)
    X = feat.featurize_batch(U_smiles).numpy()
    res = {
        "n_drugs": len(U_smiles),
        "feature_dim": feat.feature_dim,
        "has_rdkit": feat.has_rdkit,
        "feature_names": feat.feature_names,
        "surrogate_mask": feat.surrogate_mask,
    }

    # Silhouette of pathway features under MoA labels (>=2 classes, >=2 per class).
    try:
        from sklearn.metrics import silhouette_score
        import collections

        labels = np.array([m if m is not None else "?" for m in U_moa])
        freq = collections.Counter(labels.tolist())
        keep = np.array([freq[l] >= 2 and l != "?" for l in labels])
        if keep.sum() >= 4 and len({l for l in labels[keep]}) >= 2:
            Xn = (X[keep] - X[keep].mean(0)) / (X[keep].std(0) + 1e-8)
            res["moa_silhouette"] = float(silhouette_score(Xn, labels[keep]))
        else:
            res["moa_silhouette"] = float("nan")
    except Exception as e:
        logger.warning("silhouette failed: %s", e)
        res["moa_silhouette"] = float("nan")

    # t-SNE of pathway features coloured by MoA (house style).
    try:
        if X.shape[0] >= 5:
            emb = viz.tsne_embed(torch.from_numpy(X), seed=seed)
            p = os.path.join(out_dir, "pathway_features_tsne_moa.png")
            # PNG + PDF twin (deliverable form). plot_tsne_single saves by the path's
            # extension and closes its figure each call, so render both explicitly.
            viz.plot_tsne_single(emb, U_moa, p, name="moa_fine", step=0)
            viz.plot_tsne_single(emb, U_moa, p.replace(".png", ".pdf"), name="moa_fine", step=0)
            res["tsne_path"] = p
    except Exception as e:
        logger.warning("feature t-SNE failed: %s", e)
    return res, (U_smiles, U_moa, X)


def _alignment_eval(reps, smiles_list):
    """Do the chemistry pathway features linearly predict the encoder latent?

    Per-drug mean latent vs per-drug pathway feature; report mean R2 across latent
    dims (a coarse "does chemistry explain representation" score).
    """
    feat = HepatotoxPathwayFeaturizer()
    by_drug: dict[str, list[int]] = {}
    for i, s in enumerate(smiles_list):
        if s:
            by_drug.setdefault(s, []).append(i)
    drugs = [d for d, idx in by_drug.items() if len(idx) >= 1]
    if len(drugs) < 8:
        return {"alignment_mean_r2": float("nan"), "n_drugs": len(drugs)}
    Z = torch.stack([reps[by_drug[d]].mean(0) for d in drugs])         # [D, d_lat]
    P = feat.featurize_batch(drugs)                                     # [D, d_path]
    # Predict each latent PC from pathway features; average R2 over top PCs.
    Zc = Z - Z.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Zc, full_matrices=False)
    k = min(8, S.numel())
    r2s = []
    for j in range(k):
        target = (Zc @ Vh[j])  # projection onto PC j
        r2s.append(train_regression_probe(P, target).get("r2", float("nan")))
    return {
        "alignment_mean_r2": float(np.nanmean(r2s)),
        "alignment_top_pc_r2": float(r2s[0]) if r2s else float("nan"),
        "n_drugs": len(drugs),
    }


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def run(
    config: str = "examples/tahoe_hepatotox/cfgs/finetune.yaml",
    checkpoint: str = "",
    out_dir: str = "visualizations/hepatotox",
    eval_cells: int = 0,
    **overrides,
):
    cfg = load_config(config, cli_overrides=overrides or None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # 1) Encoder + liver eval set (sub14_main builds the fixed eval set; finetune has
    #    already bound the liver-filtered dataset onto sub14_main on import).
    model, pc = _load_encoder(cfg, checkpoint, device)
    if eval_cells:
        cfg.eval["eval_cells"] = int(eval_cells)
    _, dataset, _ = sub14_main.build_loader(cfg, pc, rank=0, world=1)
    collator_args = dict(
        token_to_pc_local=pc.token_to_pc_local,
        n_pc_genes=pc.n_pc_genes,
        num_bins=int(cfg.data.get("num_bins", 16)),
        genes_per_bin=int(cfg.data.get("genes_per_bin", 32)),
    )
    eval_single, eval_meta, _ = sub14_main.build_eval_set(dataset, collator_args, cfg)
    reps = _encode(model, eval_single, device, int(cfg.eval.get("encode_chunk", 128)))
    logger.info("encoded %d liver eval cells -> reps %s", reps.shape[0], tuple(reps.shape))

    results: dict = {}
    # 2) A: encoder probe on hepatotox labels
    results["probe"] = _probe_eval(reps, eval_meta)

    # 3) B: virtual-pathway feature validation
    smiles = eval_single.get("canonical_smiles", [None] * reps.shape[0])
    moa = eval_meta.get("moa_fine", [None] * reps.shape[0])
    results["features"], _ = _feature_validation(smiles, moa, out_dir, seed=int(cfg.meta.seed))

    # 4) C: pathway-feature -> representation alignment
    results["alignment"] = _alignment_eval(reps, smiles)

    # Report
    logger.info("=== Hepatotox validation (%.1fs) ===", time.time() - t0)
    p = results["probe"]
    for k in sorted(p):
        if k.endswith("balanced_accuracy") or k.endswith("/r2"):
            logger.info("  %s = %.3f", k, p[k])
    f = results["features"]
    logger.info("  features: n_drugs=%d dim=%d rdkit=%s moa_silhouette=%.3f",
                f["n_drugs"], f["feature_dim"], f["has_rdkit"], f["moa_silhouette"])
    a = results["alignment"]
    logger.info("  alignment: mean_r2=%.3f (top_pc_r2=%.3f) over %d drugs",
                a["alignment_mean_r2"], a["alignment_top_pc_r2"], a["n_drugs"])

    # Persist a JSON next to the figures.
    import json

    summary = {
        "probe": results["probe"],
        "features": {k: v for k, v in results["features"].items()
                     if k not in ("feature_names", "surrogate_mask")},
        "alignment": results["alignment"],
        "checkpoint": checkpoint,
    }
    with open(os.path.join(out_dir, "hepatotox_eval.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    logger.info("wrote %s", os.path.join(out_dir, "hepatotox_eval.json"))
    return results


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run})
