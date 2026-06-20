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


def _linear_probe_1epoch(X, y, n_out, task, seed=0, lr=5e-2, batch_size=16):
    """Fit a fresh linear probe for exactly ONE epoch on a train split of the eval
    features, from a FIXED seed initialisation (identical every eval -> probe losses
    are directly comparable across evals and scales; no continuous/accumulated
    training). Features standardized (deterministic). Returns (val_loss, val_out,
    val_targets). task: 'clf' (cross-entropy) or 'reg' (MSE)."""
    X = X.float()
    mu = X.mean(0, keepdim=True)
    sd = X.std(0, keepdim=True).clamp(min=1e-6)
    X = (X - mu) / sd
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(X.shape[0], generator=g)
    n_val = max(1, int(0.2 * X.shape[0]))
    vi, ti = perm[:n_val], perm[n_val:]
    Xtr, ytr, Xva, yva = X[ti], y[ti], X[vi], y[vi]
    torch.manual_seed(seed)  # SAME probe initialisation at every eval
    probe = nn.Linear(X.shape[1], n_out)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    ce = task == "clf"

    def _loss(out, tgt):
        return (
            nn.functional.cross_entropy(out, tgt)
            if ce
            else nn.functional.mse_loss(out.squeeze(-1), tgt)
        )

    order = torch.randperm(Xtr.shape[0], generator=g)
    for s in range(0, Xtr.shape[0], batch_size):  # exactly one epoch
        b = order[s : s + batch_size]
        opt.zero_grad()
        _loss(probe(Xtr[b]), ytr[b]).backward()
        opt.step()
    with torch.no_grad():
        vout = probe(Xva)
        vloss = float(_loss(vout, yva))
    return vloss, vout, yva


def train_classification_probe(features: torch.Tensor, labels: list, seed: int = 0):
    """1-epoch-from-scratch linear probe (fixed init); report the held-out probe
    **loss** (cross-entropy) + imbalance-aware accuracy. Cheap and comparable across
    evals (refit each eval, never accumulated)."""
    from sklearn.metrics import balanced_accuracy_score, f1_score

    ids, classes = _labels_to_ids(labels)
    keep = ids >= 0
    X, y = features[keep], ids[keep]
    n_classes = len(classes)
    if n_classes < 2 or X.shape[0] < 8:
        return {
            "n_classes": n_classes,
            "loss": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
        }
    vloss, vout, yva = _linear_probe_1epoch(X, y, n_classes, "clf", seed)
    pred, yt = vout.argmax(-1).numpy(), yva.numpy()
    return {
        "n_classes": n_classes,
        "loss": vloss,
        "balanced_accuracy": float(balanced_accuracy_score(yt, pred)),
        "macro_f1": float(f1_score(yt, pred, average="macro", zero_division=0)),
        "chance": 1.0 / n_classes,
    }


def train_regression_probe(
    features: torch.Tensor, targets: torch.Tensor, seed: int = 0
):
    """1-epoch-from-scratch linear probe (fixed init); report the held-out probe
    **loss** (MSE on a z-scored target -> 1.0 = predicting the mean) + R2."""
    from sklearn.metrics import r2_score

    targets = targets.float()
    finite = torch.isfinite(targets)
    X, y = features[finite], targets[finite]
    if X.shape[0] < 8:
        return {"loss": float("nan"), "r2": float("nan")}
    ymu, ysd = y.mean(), y.std().clamp(min=1e-6)
    vloss, vout, yva = _linear_probe_1epoch(X, (y - ymu) / ysd, 1, "reg", seed)
    yt, yp = yva.numpy(), vout.squeeze(-1).numpy()
    return {"loss": vloss, "r2": float(r2_score(yt, yp))}


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
    # gene-count + any pathway/<name> regression targets (e.g. HALLMARK scores)
    for key, target in meta.items():
        if key == "gene_count" or key.startswith("pathway/"):
            results[f"reg/{key}"] = train_regression_probe(
                features, torch.as_tensor(target, dtype=torch.float32)
            )
    return results
