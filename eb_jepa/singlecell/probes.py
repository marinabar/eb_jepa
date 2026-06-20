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


def train_classification_probe(
    features: torch.Tensor,
    labels: list,
    epochs: int = 200,
    lr: float = 1e-2,
    val_frac: float = 0.2,
    seed: int = 0,
):
    """Train a detached linear probe; return imbalance-aware metrics on a holdout."""
    from sklearn.metrics import balanced_accuracy_score, f1_score

    ids, classes = _labels_to_ids(labels)
    keep = ids >= 0
    features, ids = features[keep], ids[keep]
    n_classes = len(classes)
    if n_classes < 2:
        return {
            "n_classes": n_classes,
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
        }

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(features.shape[0], generator=g)
    n_val = max(1, int(features.shape[0] * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    probe = LinearProbe(features.shape[1], n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    xtr, ytr = features[tr_idx], ids[tr_idx]
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(probe(xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = probe(features[val_idx]).argmax(-1)
    y_true = ids[val_idx].numpy()
    y_pred = pred.numpy()
    return {
        "n_classes": n_classes,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "chance": 1.0 / n_classes,
    }


def train_regression_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = 300,
    lr: float = 1e-2,
    val_frac: float = 0.2,
    seed: int = 0,
):
    """Train a detached linear regression probe; return R2 / explained variance."""
    from sklearn.metrics import explained_variance_score, r2_score

    targets = targets.float()
    finite = torch.isfinite(targets)
    features, targets = features[finite], targets[finite]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(features.shape[0], generator=g)
    n_val = max(1, int(features.shape[0] * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    # standardize the target for stable training
    mu, sd = targets[tr_idx].mean(), targets[tr_idx].std().clamp(min=1e-6)
    probe = LinearProbe(features.shape[1], 1)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    xtr, ytr = features[tr_idx], ((targets[tr_idx] - mu) / sd).unsqueeze(-1)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(probe(xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = probe(features[val_idx]).squeeze(-1) * sd + mu
    y_true = targets[val_idx].numpy()
    y_pred = pred.numpy()
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
    }


def run_probe_suite(
    features: torch.Tensor, meta: dict, epochs: int | None = None
) -> dict:
    """Run the standard classification + regression probes; return a metrics dict.

    ``epochs`` (optional) caps probe-training length for fast periodic evals; when
    ``None`` each probe trainer uses its own default.
    """
    clf_kw = {} if epochs is None else {"epochs": epochs}
    reg_kw = {} if epochs is None else {"epochs": epochs}
    results = {}
    for key in ("organ", "cell_line_id", "drug", "sample", "moa_fine"):
        if key in meta and len(set(meta[key])) >= 2:
            results[f"clf/{key}"] = train_classification_probe(
                features, meta[key], **clf_kw
            )
    if "gene_count" in meta:
        results["reg/gene_count"] = train_regression_probe(
            features, torch.tensor(meta["gene_count"]), **reg_kw
        )
    return results
