"""Probing suite: validate the pre-projection representation (CLAUDE.md "Probing").

Probes are trained **detached** from the encoder (no gradient flows back). We
extract the encoder's pre-projection latent once on a clean single full view per
cell, then fit linear probes:
  - classification: organ / cell_line / drug / sample / moa_fine — scored with
    imbalance-aware metrics (balanced accuracy, macro-F1), not raw accuracy.
  - regression: total expressed-gene count (pluripotency proxy); pathway
    regressions plug in the same way (explained vs total variance).

The same harness scores the JEPA encoder and the MAE/VAE/PCA baselines for the
headline comparison.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn

from eb_jepa.singlecell.encoder import encode_views


@torch.no_grad()
def extract_features(encoder, loader, device="cpu", max_batches: int | None = None):
    """Run the frozen encoder over a loader; return (features [N,d], meta dict).

    Uses view 0 of each batch as the clean representation (configure the loader with
    n_views=1, view_mode='drop', gene_keep_frac=1.0 for a full-cell view). Also
    returns ``gene_count`` (real tokens per cell) as a regression target.
    """
    encoder.eval()
    feats, meta = [], defaultdict(list)
    label_keys = ("organ", "cell_line_id", "drug", "sample", "moa_fine", "plate")
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        dev_batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        reps = encode_views(encoder, dev_batch)[0]  # view 0 -> [N, d]
        feats.append(reps.cpu())
        for k in label_keys:
            if k in batch:
                meta[k].extend(batch[k])
        meta["gene_count"].extend(batch["pad_mask"][0].sum(-1).cpu().tolist())
        if "log_conc" in batch:
            meta["log_conc"].extend(batch["log_conc"].cpu().tolist())
    return torch.cat(feats), dict(meta)


class LinearProbe(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.fc(x)


def _labels_to_ids(labels):
    """Map a list of hashable labels to contiguous int ids; returns (ids, classes)."""
    classes = sorted({x for x in labels if x is not None})
    idx = {c: i for i, c in enumerate(classes)}
    ids = torch.tensor([idx.get(x, -1) for x in labels], dtype=torch.long)
    return ids, classes


def _split(n: int, val_frac: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(n * val_frac))
    return perm[n_val:], perm[:n_val]  # (train_idx, val_idx)


def train_classification_probe(
    features: torch.Tensor,
    labels: list,
    val_frac: float = 0.2,
    seed: int = 0,
    C: float = 1.0,
    max_iter: int = 2000,
):
    """Linear logistic-regression probe fit **to convergence** (sklearn lbfgs) — the
    optimum of the convex problem, NOT fixed-budget SGD, so the metric isn't biased
    by a training schedule. Refit from scratch each eval on the held-out features.
    Imbalance-aware (class_weight=balanced; balanced-accuracy + macro-F1 on a holdout).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score

    ids, classes = _labels_to_ids(labels)
    keep = ids >= 0
    X = features[keep].float().numpy()
    y = ids[keep].numpy()
    n_classes = len(classes)
    nan = {
        "n_classes": n_classes,
        "balanced_accuracy": float("nan"),
        "macro_f1": float("nan"),
    }
    if n_classes < 2 or X.shape[0] < 4:
        return nan
    tr, va = _split(X.shape[0], val_frac, seed)
    tr, va = tr.numpy(), va.numpy()
    if len(set(y[tr].tolist())) < 2:  # need >=2 classes in the train split
        return {**nan, "chance": 1.0 / n_classes}
    clf = LogisticRegression(max_iter=max_iter, C=C, class_weight="balanced")
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[va])
    return {
        "n_classes": n_classes,
        "balanced_accuracy": float(balanced_accuracy_score(y[va], pred)),
        "macro_f1": float(f1_score(y[va], pred, average="macro", zero_division=0)),
        "chance": 1.0 / n_classes,
    }


def train_regression_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    val_frac: float = 0.2,
    seed: int = 0,
    ridge: float = 1e-2,
):
    """Closed-form ridge-regression probe: the **exact least-squares optimum** solved
    in one shot (NOT iterative — no training-budget bias). Features standardized on
    the train split; bias term added (unregularized); solves
    ``(XᵀX + ridge·I) w = Xᵀy``; reports R2 / explained variance on the holdout.
    """
    from sklearn.metrics import explained_variance_score, r2_score

    targets = targets.float()
    finite = torch.isfinite(targets)
    X = features[finite].double()
    y = targets[finite].double()
    n = X.shape[0]
    if n < 4:
        return {"r2": float("nan"), "explained_variance": float("nan")}
    tr, va = _split(n, val_frac, seed)
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp(min=1e-6)
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    Xtr_b = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1, dtype=Xtr.dtype)], 1)
    Xva_b = torch.cat([Xva, torch.ones(Xva.shape[0], 1, dtype=Xva.dtype)], 1)
    d = Xtr_b.shape[1]
    reg = ridge * torch.eye(d, dtype=Xtr_b.dtype)
    reg[-1, -1] = 0.0  # do not regularize the bias term
    w = torch.linalg.solve(Xtr_b.T @ Xtr_b + reg, Xtr_b.T @ ytr)
    pred = (Xva_b @ w).numpy()
    yt = yva.numpy()
    return {
        "r2": float(r2_score(yt, pred)),
        "explained_variance": float(explained_variance_score(yt, pred)),
    }


def run_probe_suite(
    features: torch.Tensor, meta: dict, epochs: int | None = None
) -> dict:
    """Run the standard classification + regression probes; return a metrics dict.

    ``epochs`` is accepted for backward compatibility but ignored: the probes are
    fit closed-form / to convergence (logistic-regression and ridge), so there is no
    training-budget to cap.
    """
    del epochs  # no longer used (probes are not iterative)
    results = {}
    for key in ("organ", "cell_line_id", "drug", "sample", "moa_fine"):
        if key in meta and len(set(meta[key])) >= 2:
            results[f"clf/{key}"] = train_classification_probe(features, meta[key])
    if "gene_count" in meta:
        results["reg/gene_count"] = train_regression_probe(
            features, torch.tensor(meta["gene_count"])
        )
    return results
