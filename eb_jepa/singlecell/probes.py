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


_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def _standardize(X, tr):
    """Center (per-dim, train stats) + scale by a SINGLE GLOBAL scalar, then numpy.

    Crucially NOT per-dim sd: dividing by a near-zero per-dim sd amplifies collapsed
    (zero-variance) dimensions and makes the probe explode (r2 << 0). A single global
    scale keeps collapsed dims small so the ridge simply ignores them.
    """
    X = X.float()
    mu = X[tr].mean(0, keepdim=True)
    s = (X[tr] - mu).std().clamp(min=1e-6)
    return ((X - mu) / s).numpy()


def train_classification_probe(features: torch.Tensor, labels: list, seed: int = 0):
    """Closed-form ridge classifier (RidgeClassifierCV: ridge on one-hot, LOO-selected
    alpha), refit from scratch each eval. Stable + sane at init (-> chance), no SGD /
    training-budget bias, fast (no lbfgs). Imbalance-aware metrics on a holdout."""
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.metrics import balanced_accuracy_score, f1_score

    ids, classes = _labels_to_ids(labels)
    keep = ids >= 0
    X, y = features[keep], ids[keep].numpy()
    n_classes = len(classes)
    nan = {
        "n_classes": n_classes,
        "balanced_accuracy": float("nan"),
        "macro_f1": float("nan"),
        "chance": 1.0 / max(1, n_classes),
    }
    if n_classes < 2 or X.shape[0] < 8:
        return nan
    tr, va = _split(X.shape[0], 0.2, seed)
    tr, va = tr.numpy(), va.numpy()
    if len(set(y[tr].tolist())) < 2:  # need >=2 classes in the train split
        return nan
    Xs = _standardize(X, tr)
    clf = RidgeClassifierCV(alphas=_ALPHAS, class_weight="balanced").fit(Xs[tr], y[tr])
    pred = clf.predict(Xs[va])
    return {
        "n_classes": n_classes,
        "balanced_accuracy": float(balanced_accuracy_score(y[va], pred)),
        "macro_f1": float(f1_score(y[va], pred, average="macro", zero_division=0)),
        "chance": 1.0 / n_classes,
    }


def train_regression_probe(
    features: torch.Tensor, targets: torch.Tensor, seed: int = 0
):
    """Closed-form ridge regression probe (RidgeCV, LOO-selected alpha), refit from
    scratch each eval. Stable + sane at init (no signal -> r2~0, never << 0). Reports
    R2 + val MSE on a z-scored target (1.0 = predicting the mean)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_squared_error, r2_score

    targets = targets.float()
    finite = torch.isfinite(targets)
    X, y = features[finite], targets[finite]
    if X.shape[0] < 8:
        return {"loss": float("nan"), "r2": float("nan")}
    tr, va = _split(X.shape[0], 0.2, seed)
    tr, va = tr.numpy(), va.numpy()
    ymu, ysd = y[tr].mean(), y[tr].std().clamp(min=1e-6)
    yz = ((y - ymu) / ysd).numpy()
    Xs = _standardize(X, tr)
    m = RidgeCV(alphas=_ALPHAS).fit(Xs[tr], yz[tr])
    pred = m.predict(Xs[va])
    return {
        "loss": float(mean_squared_error(yz[va], pred)),
        "r2": float(r2_score(yz[va], pred)),
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
    # DMSO-only (control) vs treated, derived from the drug label
    if "drug" in meta and "is_dmso" not in meta:
        meta = {
            **meta,
            "is_dmso": ["DMSO" if d == "DMSO_TF" else "treated" for d in meta["drug"]],
        }
    results = {}
    clf_keys = (
        "organ",
        "cell_line_id",
        "drug",
        "sample",
        "moa_fine",
        "driver_gene",  # cell-line driver mutation
        "driver_mech",  # driver mechanism
        "driver_type",  # oncogene vs tumour-suppressor
        "is_dmso",  # control vs treated
    )
    for key in clf_keys:
        if key in meta and len({x for x in meta[key] if x is not None}) >= 2:
            results[f"clf/{key}"] = train_classification_probe(features, meta[key])
    # pathway/<name> regression targets (e.g. HALLMARK scores) when provided
    for key, target in meta.items():
        if key.startswith("pathway/"):
            results[f"reg/{key}"] = train_regression_probe(
                features, torch.as_tensor(target, dtype=torch.float32)
            )
    return results
