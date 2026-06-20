"""Shared plumbing for the baseline benchmark: data stream, the FIXED shared
validation set, and per-model encoders.

Fairness is the whole point of this harness (CLAUDE.md "Success criteria": JEPA
must beat *well-tuned* MAE / VAE / PCA on a fixed, shared eval set). Every model
— Subliminal-14 and the three baselines — is encoded on the *same* cells, drawn
deterministically from the same stream, so the probe table compares like with
like. The baselines consume the densified 62,713-gene vector; sub14 consumes its
protein-coding quantile-thermometer views. Both reduce to a pre-projection /
bottleneck representation ``[N, d]`` that feeds the shared probe suite.
"""
from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

from eb_jepa.datasets.tahoe.dataset import (
    TahoeConfig,
    TahoeIterableDataset,
    densify,
)

# Index space = max Tahoe token_id (62712) + 1.
N_GENES_INDEX = 62713

if TYPE_CHECKING:
    from eb_jepa.singlecell.baselines import (
        MAEBaseline,
        PCABaseline,
        VAEBaseline,
    )

    # The four representation models handled polymorphically: the three neural
    # baselines expose ``encode`` + nn.Module persistence; PCABaseline is an
    # sklearn-style fit/encode object persisted via pickle (not an nn.Module).
    Baseline = Union["MAEBaseline", "VAEBaseline", "PCABaseline"]


# --------------------------------------------------------------------------- #
# Data stream                                                                 #
# --------------------------------------------------------------------------- #
def load_maps(maps_path: Optional[str]) -> dict:
    """Load the {cell_line_to_organ, sample_to_logconc} maps if present."""
    if maps_path and os.path.exists(maps_path):
        return torch.load(maps_path)
    return {}


def build_stream(cfg, *, rank: int = 0, world: int = 1, shuffle: bool = True):
    """A TahoeIterableDataset over the configured shards (same reader as sub14)."""
    fields = TahoeConfig.__dataclass_fields__
    data_cfg = TahoeConfig(**{k: cfg.data[k] for k in cfg.data if k in fields})
    maps = load_maps(cfg.data.get("maps_path"))
    return TahoeIterableDataset(
        data_cfg,
        binner=None,
        cell_line_to_organ=maps.get("cell_line_to_organ"),
        sample_to_logconc=maps.get("sample_to_logconc"),
        rank=rank,
        world_size=world,
        shuffle=shuffle,
    )


_META_KEYS = ("organ", "cell_line_id", "drug", "moa_fine", "sample", "plate")


def fixed_eval_items(dataset: TahoeIterableDataset, n_cells: int) -> list[dict]:
    """The FIXED, deterministic shared validation cells (same for every model).

    ``sample_items`` spreads the draw across evenly-spaced shards (diverse cell
    lines / drugs) and is seed-free / shuffle-free, so the returned list is
    identical across models and across train/benchmark invocations on the same
    ``data_dir``.
    """
    return dataset.sample_items(n_cells)


def eval_meta(items: list[dict]) -> dict:
    """Per-cell probe labels straight off the cell items (+ derived gene_count)."""
    meta: dict = {k: [c.get(k) for c in items] for k in _META_KEYS}
    meta["gene_count"] = [int(c["gene_token_ids"].numel()) for c in items]
    meta["log_conc"] = [float(c.get("log_conc", float("nan"))) for c in items]
    return meta


def densify_items(items: list[dict], n_genes: int) -> torch.Tensor:
    """Stack the fixed eval cells into the dense [N, n_genes] matrix (baselines)."""
    return torch.stack(
        [densify(c["gene_token_ids"], c["values"], n_genes) for c in items]
    )


# --------------------------------------------------------------------------- #
# HVG-restricted dense input (MATCHED baselines)                              #
# --------------------------------------------------------------------------- #
def load_hvg_panel(hvg_path: str) -> torch.Tensor:
    """Load the saved HVG panel (``hvg_{N}.npy``) as a sorted int64 token_id tensor."""
    panel = np.load(hvg_path)
    return torch.from_numpy(np.asarray(panel, dtype=np.int64))


def build_hvg_local_map(
    panel: torch.Tensor, n_index: int = N_GENES_INDEX
) -> torch.Tensor:
    """Map raw token_id -> local HVG column (size ``n_index``, -1 for off-panel genes).

    ``panel`` is the sorted list of the N selected token_ids; local column j holds
    ``panel[j]``. Genes outside the panel map to -1 and are dropped at densify time.
    """
    local = torch.full((n_index,), -1, dtype=torch.long)
    local[panel.long()] = torch.arange(panel.numel(), dtype=torch.long)
    return local


def densify_hvg(
    items: list[dict], hvg_local_map: torch.Tensor, n_hvg: int
) -> torch.Tensor:
    """Stack the cells into the dense [N, n_hvg] HVG-panel matrix.

    Each cell's sparse (token_id, value) pairs are scattered into the n_hvg-wide
    vector via ``hvg_local_map``; off-panel genes (local index -1) are dropped.
    """
    out = torch.zeros((len(items), int(n_hvg)), dtype=torch.float32)
    for j, c in enumerate(items):
        tok = c["gene_token_ids"].long()
        if tok.numel() == 0:
            continue
        local = hvg_local_map[tok]  # [-1 for off-panel]
        keep = local >= 0
        if keep.any():
            out[j, local[keep]] = c["values"].to(torch.float32)[keep]
    return out


# --------------------------------------------------------------------------- #
# Per-model encoders -> pre-representation features [N, d]                     #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_baseline(
    model: "Baseline", dense: torch.Tensor, device, chunk: int = 512
) -> torch.Tensor:
    """Encode the dense matrix with an MAE/VAE (``.encode``) or fitted PCA.

    PCA encodes on CPU/numpy; the neural baselines encode on ``device`` in chunks.
    Returns float CPU features ``[N, latent]``.
    """
    if not isinstance(model, torch.nn.Module):  # PCABaseline (sklearn-style)
        return model.encode(dense).float()
    net: torch.nn.Module = model
    was_training = net.training
    net.eval()
    p = next(net.parameters())
    reps = []
    for s in range(0, dense.shape[0], chunk):
        x = dense[s : s + chunk].to(device=device, dtype=p.dtype)
        reps.append(net.encode(x).float().cpu())
    if was_training:
        net.train()
    return torch.cat(reps, dim=0)


# --------------------------------------------------------------------------- #
# Checkpoint loaders                                                          #
# --------------------------------------------------------------------------- #
def load_baseline_checkpoint(path: str, device) -> "Baseline":
    """Load a trained MAE/VAE (.pt) or fitted PCA (.pkl) baseline for benchmarking.

    PCA is an sklearn-style object restored from its pickle (no ``.to`` /
    ``load_state_dict``); MAE/VAE are nn.Modules rebuilt at their saved scale and
    given their weights.
    """
    from eb_jepa.singlecell.baselines import build_baseline

    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)  # PCABaseline (numpy / CPU)
    ckpt = torch.load(path, map_location="cpu")
    model = build_baseline(ckpt["type"], int(ckpt["n_genes"]), **ckpt["kwargs"])
    assert isinstance(model, torch.nn.Module)  # .pt path is always MAE/VAE
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def load_sub14_checkpoint(path: str, cfg, pc, device):
    """Rebuild the Subliminal14 model at the checkpoint scale + load weights.

    ``cfg.model`` supplies the architecture (d_model/n_layers/...); the frozen PC
    features come from ``pc`` (the gene_emb_cache filtered to protein-coding genes).
    """
    from eb_jepa.singlecell.sub14.model import Subliminal14

    num_bins = int(cfg.data.get("num_bins", 16))
    genes_per_bin = int(cfg.data.get("genes_per_bin", 32))
    model = Subliminal14(
        n_pc_genes=pc.n_pc_genes,
        d_model=int(cfg.model.d_model),
        n_heads=int(cfg.model.n_heads),
        n_layers=int(cfg.model.n_layers),
        d_ff=int(cfg.model.d_ff),
        dropout=float(cfg.model.get("dropout", 0.1)),
        latent_dim=int(cfg.model.get("proj_dim", 128)),
        num_bins=num_bins,
        max_genes_per_cell=num_bins * genes_per_bin,
        dna_features=pc.dna_features,
        protein_features=pc.protein_features,
        freeze_features=bool(cfg.model.get("freeze_features", True)),
        attention_activation=str(cfg.model.get("attention_activation", "sigmoid")),
    )
    raw = torch.load(path, map_location="cpu")
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


@torch.no_grad()
def encode_sub14(model, items: list[dict], pc, *, num_bins: int, genes_per_bin: int,
                 device, seed: int = 0, chunk: int = 256) -> torch.Tensor:
    """Encode the same fixed cells with the sub14 model on a single clean view.

    Uses ``Sub14Collator`` with ``num_views=1`` and no binomial subsample (the
    deterministic full-cell view used everywhere for probing / t-SNE), then runs
    ``model.encode`` to the pre-projection [CELL] representation ``[N, d_model]``.
    """
    from eb_jepa.singlecell.sub14.collator import Sub14Collator

    coll = Sub14Collator(
        token_to_pc_local=pc.token_to_pc_local,
        n_pc_genes=pc.n_pc_genes,
        num_bins=num_bins,
        genes_per_bin=genes_per_bin,
        num_views=1,
        binomial_subsample=None,
        seed=seed,
    )
    view = coll(items)
    gene_ids, bin_ids, pad = view["gene_ids"][0], view["bin_ids"][0], view["padding_mask"][0]
    was_training = model.training
    model.eval()
    reps = []
    for s in range(0, gene_ids.size(0), chunk):
        sl = slice(s, s + chunk)
        r = model.encode(gene_ids[sl].to(device), bin_ids[sl].to(device), pad[sl].to(device))
        reps.append(r.float().cpu())
    if was_training:
        model.train()
    return torch.cat(reps, dim=0)
