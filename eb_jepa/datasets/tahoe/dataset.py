"""Tahoe-100M sparse single-cell dataset + on-the-fly LeJEPA view collation.

``TahoeDataset`` reads cells from parquet (the raw ``expression_data`` config or a
preprocessed cache — same columns), strips the CLS marker, applies CP10k+log1p,
and returns one whole, variable-length cell plus probing metadata.

``TahoeCollator`` generates the V views on the fly (drop or mask), samples/pads
each view to a fixed budget of ``L`` tokens with an attention mask, and encodes
counts (mode A continuous scalar, or mode B quantile ``bin_id``). Constant
``[V, N, L]`` shapes keep the encoder ``torch.compile``-able.

See CLAUDE.md "Dataset" for the schema and the view/normalization rules.
"""

from __future__ import annotations

import ast
import glob
import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from eb_jepa.datasets.tahoe.normalizer import QuantileBinner, cp10k_log1p

# Columns read from expression_data (hyphen in "moa-fine" is intentional).
_EXPR_COLUMNS = [
    "genes",
    "expressions",
    "drug",
    "sample",
    "cell_line_id",
    "moa-fine",
    "canonical_smiles",
    "plate",
    "BARCODE_SUB_LIB_ID",
]


@dataclass
class TahoeConfig:
    """Single config for the Tahoe dataloader (see examples/tahoe_jepa/cfgs)."""

    data_dir: str = ""  # directory of *.parquet (cache or raw expression_data)
    split: str = "train"
    # views
    L: int = 4096  # tokens per view (fixed, for compile)
    n_views: int = 4
    view_mode: str = "drop"  # "drop" | "mask"
    gene_keep_frac: float = 0.6  # drop: fraction of a cell's genes kept per view
    gene_mask_frac: float = 0.3  # mask: fraction of kept genes whose count is hidden
    # counts
    count_mode: str = "A"  # "A" continuous | "B" quantile bins
    n_bins: int = 50  # mode B
    n_genes: int = 62710  # vocabulary size (densify + binning)
    pad_token_id: int = 0  # padding gene id (ignored via attention mask)
    # loader
    batch_size: int = 32
    num_workers: int = 0
    pin_mem: bool = False
    seed: int = 0
    max_cells: int = 0  # 0 = all; >0 truncates (debug/smoke)


def _read_parquet_dir(data_dir: str, columns: list[str]):
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files under {data_dir!r}")
    tables = [pq.read_table(f, columns=columns) for f in files]
    import pyarrow as pa

    return pa.concat_tables(tables)


def parse_log_conc(drugname_drugconc: str) -> float:
    """Parse ``"[('Drug', 0.05, 'uM')]"`` -> log10 molar concentration.

    Controls (conc == 0, e.g. DMSO_TF) return ``nan`` (no log10(0)); callers
    should treat nan as "no dose" / mask it.
    """
    try:
        parsed = ast.literal_eval(drugname_drugconc)
        _, conc, unit = parsed[0]
    except (ValueError, SyntaxError, IndexError, TypeError):
        return float("nan")
    if unit != "uM":  # spec: concentrations are in micromolar
        return float("nan")
    if conc is None or conc <= 0:
        return float("nan")
    return math.log10(float(conc) * 1e-6)


class TahoeDataset(Dataset):
    """Whole-cell sparse reader. ``__getitem__`` returns one variable-length cell.

    Args:
        config: TahoeConfig.
        binner: optional QuantileBinner; if given, mode-B ``bin_ids`` are computed
            per cell (deterministic per (gene, value), hence identical across views).
        cell_line_to_organ: optional {CVCL id -> organ} map (built from
            cell_line_metadata) used to attach the ``organ`` probe label.
        sample_to_logconc: optional {sample -> log10 molar} map (built from
            sample_metadata.drugname_drugconc) for the dose.
    """

    def __init__(
        self,
        config: TahoeConfig,
        binner: Optional[QuantileBinner] = None,
        cell_line_to_organ: Optional[dict] = None,
        sample_to_logconc: Optional[dict] = None,
    ):
        self.config = config
        self.binner = binner
        self.cell_line_to_organ = cell_line_to_organ or {}
        self.sample_to_logconc = sample_to_logconc or {}
        self._table = _read_parquet_dir(config.data_dir, _EXPR_COLUMNS)
        if config.max_cells and config.max_cells < self._table.num_rows:
            self._table = self._table.slice(0, config.max_cells)
        # keep column handles for per-row access
        self._genes = self._table.column("genes")
        self._expr = self._table.column("expressions")

    def __len__(self) -> int:
        return self._table.num_rows

    def _col(self, name: str, i: int):
        return self._table.column(name)[i].as_py()

    def __getitem__(self, i: int) -> dict:
        # strip the CLS marker at position 0 of both aligned arrays
        token_ids = self._genes[i].values.to_numpy(zero_copy_only=False)[1:]
        counts = self._expr[i].values.to_numpy(zero_copy_only=False)[1:]
        token_ids = torch.from_numpy(token_ids.copy()).long()
        counts = torch.from_numpy(counts.copy()).float()
        assert token_ids.shape == counts.shape, "genes/expressions misaligned"

        values = cp10k_log1p(counts)  # CP10k + log1p, mode-agnostic
        sample = self._col("sample", i)
        cell_line_id = self._col("cell_line_id", i)
        item = {
            "gene_token_ids": token_ids,
            "values": values,
            "drug": self._col("drug", i),
            "sample": sample,
            "cell_line_id": cell_line_id,
            "organ": self.cell_line_to_organ.get(cell_line_id),
            "moa_fine": self._col("moa-fine", i),
            "plate": self._col("plate", i),
            "canonical_smiles": self._col("canonical_smiles", i),
            "barcode": self._col("BARCODE_SUB_LIB_ID", i),
            "log_conc": self.sample_to_logconc.get(sample, float("nan")),
        }
        if self.binner is not None:
            item["bin_ids"] = self.binner.bin(token_ids, values)
        return item


def _sample_indices(g: int, keep: int, generator=None) -> torch.Tensor:
    """Random subset of ``keep`` indices out of ``g`` (no replacement, keep<=g)."""
    perm = torch.randperm(g, generator=generator)
    return perm[:keep]


class TahoeCollator:
    """Build V views (drop/mask) padded to L tokens, with count encoding.

    Returns a dict of batched tensors with shape ``[V, N, L]`` for the per-token
    fields plus ``[N]`` metadata. ``pad_mask`` is True at real tokens (use as the
    SDPA key-padding mask and the meanpool mask). ``count_mask`` is True where the
    count is hidden (mask-mode); the encoder substitutes a learned MASK there.
    """

    def __init__(self, config: TahoeConfig):
        self.cfg = config
        assert config.view_mode in ("drop", "mask")
        assert config.count_mode in ("A", "B")

    def __call__(self, batch: list[dict]) -> dict:
        cfg = self.cfg
        v, n, l = cfg.n_views, len(batch), cfg.L
        mode_b = cfg.count_mode == "B"

        gene_ids = torch.full((v, n, l), cfg.pad_token_id, dtype=torch.long)
        pad_mask = torch.zeros((v, n, l), dtype=torch.bool)
        count_mask = torch.zeros((v, n, l), dtype=torch.bool)
        values = torch.zeros((v, n, l), dtype=torch.float32) if not mode_b else None
        bins = torch.zeros((v, n, l), dtype=torch.long) if mode_b else None

        for j, cell in enumerate(batch):
            tok = cell["gene_token_ids"]
            val = cell["values"]
            bid = cell.get("bin_ids")
            g = tok.numel()
            if g == 0:
                continue
            for view in range(v):
                if cfg.view_mode == "drop":
                    keep = max(1, min(l, round(g * cfg.gene_keep_frac)))
                    sel = _sample_indices(g, keep)
                    masked = None
                else:  # mask: keep all genes (capped at L), hide some counts
                    keep = min(l, g)
                    sel = _sample_indices(g, keep) if g > l else torch.arange(g)
                    n_mask = int(round(keep * cfg.gene_mask_frac))
                    masked = _sample_indices(keep, n_mask) if n_mask > 0 else None

                k = sel.numel()
                gene_ids[view, j, :k] = tok[sel]
                pad_mask[view, j, :k] = True
                if mode_b:
                    bins[view, j, :k] = bid[sel]
                else:
                    values[view, j, :k] = val[sel]
                if masked is not None and masked.numel() > 0:
                    count_mask[view, j, masked] = True
                    if mode_b:
                        # dedicated MASK bin = n_bins (table has n_bins+1 rows)
                        bins[view, j, masked] = cfg.n_bins

        out = {
            "gene_token_ids": gene_ids,
            "pad_mask": pad_mask,
            "count_mask": count_mask,
            "drug": [c["drug"] for c in batch],
            "sample": [c["sample"] for c in batch],
            "cell_line_id": [c["cell_line_id"] for c in batch],
            "organ": [c["organ"] for c in batch],
            "moa_fine": [c["moa_fine"] for c in batch],
            "plate": [c["plate"] for c in batch],
            "canonical_smiles": [c["canonical_smiles"] for c in batch],
            "log_conc": torch.tensor(
                [c["log_conc"] for c in batch], dtype=torch.float32
            ),
        }
        if mode_b:
            out["count_bin"] = bins
        else:
            out["count_value"] = values
        return out


def densify(
    token_ids: torch.Tensor, values: torch.Tensor, n_genes: int
) -> torch.Tensor:
    """Scatter a sparse cell into a dense [n_genes] vector (for MAE/VAE/PCA baselines)."""
    dense = torch.zeros(n_genes, dtype=torch.float32)
    dense[token_ids.long()] = values.to(torch.float32)
    return dense


def init_tahoe_data(
    config: TahoeConfig,
    binner: Optional[QuantileBinner] = None,
    cell_line_to_organ: Optional[dict] = None,
    sample_to_logconc: Optional[dict] = None,
):
    """Build train/val DataLoaders with the on-the-fly view collator.

    Standalone (bypasses the two_rooms/maze PipelineManager): returns a standard
    DataLoader whose ``collate_fn`` generates the LeJEPA views. ``val`` reuses the
    same dataset path with shuffle off (a dedicated val split dir is wired via a
    separate config in M2/M3).
    """
    cfg = config
    dataset = TahoeDataset(cfg, binner, cell_line_to_organ, sample_to_logconc)
    collator = TahoeCollator(cfg)
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=True,
        collate_fn=collator,
    )
    train_loader = DataLoader(dataset, shuffle=(cfg.split == "train"), **loader_kwargs)
    return train_loader, dataset
