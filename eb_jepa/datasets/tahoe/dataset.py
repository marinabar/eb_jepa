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
from torch.utils.data import DataLoader, Dataset, IterableDataset

from eb_jepa.datasets.tahoe.normalizer import QuantileBinner, cp10k_log1p

# Per-cell metadata columns (hyphen in "moa-fine" is intentional). The value
# column is detected per-source: raw expression_data has "expressions" (raw counts
# with a leading CLS marker); the preprocessed cache has "value" (CP10k+log1p,
# already CLS-stripped).
_META_COLUMNS = [
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
    n_bins: int = 64  # mode B
    # pathway tokens (CLAUDE.md "Pathways"): P hallmark tokens appended per view,
    # each = learned identity + hallmark count, dropped per view with prob drop_frac.
    use_pathways: bool = False
    pathway_membership: str = ""  # path to a saved {"M":[P,n_genes],"names":[...]} .pt
    pathway_drop_frac: float = 0.5  # per-pathway dropout probability per view
    n_genes: int = (
        62713  # index space = max Tahoe token_id (62712) + 1 (densify/binning)
    )
    pad_token_id: int = 0  # padding gene id (ignored via attention mask)
    # loader
    batch_size: int = 32
    num_workers: int = 0
    pin_mem: bool = False
    seed: int = 0
    max_cells: int = 0  # 0 = all; >0 truncates (debug/smoke)
    streaming: bool = False  # IterableDataset: stream shards (full-dataset training)
    shuffle_buffer: int = 16384  # streaming: per-worker reservoir for row shuffling


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


def _make_item(genes, vals, cached, meta, binner, cell_line_to_organ, sample_to_logconc):
    """Build one cell's item dict from raw arrays + metadata (shared by datasets).

    ``meta`` is keyed by the raw parquet column names. ``cached`` selects whether the
    CLS marker has already been stripped and values are pre-normalized (cache) or
    raw counts with a leading CLS marker (raw expression_data).
    """
    if not cached:
        genes, vals = genes[1:], vals[1:]  # strip CLS marker at position 0
    token_ids = torch.from_numpy(genes.copy()).long()
    v = torch.from_numpy(vals.copy()).float()
    assert token_ids.shape == v.shape, "genes/values misaligned"
    values = v if cached else cp10k_log1p(v)
    sample = meta.get("sample")
    cell_line_id = meta.get("cell_line_id")
    item = {
        "gene_token_ids": token_ids,
        "values": values,
        "drug": meta.get("drug"),
        "sample": sample,
        "cell_line_id": cell_line_id,
        "organ": (cell_line_to_organ or {}).get(cell_line_id),
        "moa_fine": meta.get("moa-fine"),
        "plate": meta.get("plate"),
        "canonical_smiles": meta.get("canonical_smiles"),
        "barcode": meta.get("BARCODE_SUB_LIB_ID"),
        "log_conc": (sample_to_logconc or {}).get(sample, float("nan")),
    }
    if binner is not None:
        item["bin_ids"] = binner.bin(token_ids, values)
    return item


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
        import pyarrow.parquet as pq

        self.config = config
        self.binner = binner
        self.cell_line_to_organ = cell_line_to_organ or {}
        self.sample_to_logconc = sample_to_logconc or {}
        files = sorted(
            glob.glob(os.path.join(config.data_dir, "**", "*.parquet"), recursive=True)
        )
        if not files:
            raise FileNotFoundError(f"No parquet files under {config.data_dir!r}")
        names = set(pq.read_schema(files[0]).names)
        # cache: pre-normalized, CLS-stripped "value"; raw: "expressions" with CLS.
        self._cached = "value" in names
        value_col = "value" if self._cached else "expressions"
        self._meta_cols = [c for c in _META_COLUMNS if c in names]
        self._table = _read_parquet_dir(
            config.data_dir, ["genes", value_col] + self._meta_cols
        )
        if config.max_cells and config.max_cells < self._table.num_rows:
            self._table = self._table.slice(0, config.max_cells)
        # keep column handles for per-row access
        self._genes = self._table.column("genes")
        self._value = self._table.column(value_col)

    def __len__(self) -> int:
        return self._table.num_rows

    def _col(self, name: str, i: int):
        if name not in self._meta_cols:
            return None
        return self._table.column(name)[i].as_py()

    def __getitem__(self, i: int) -> dict:
        genes = self._genes[i].values.to_numpy(zero_copy_only=False)
        vals = self._value[i].values.to_numpy(zero_copy_only=False)
        meta = {c: self._table.column(c)[i].as_py() for c in self._meta_cols}
        return _make_item(
            genes, vals, self._cached, meta, self.binner,
            self.cell_line_to_organ, self.sample_to_logconc,
        )


class TahoeIterableDataset(IterableDataset):
    """Streaming reader for full-dataset training (no in-RAM table).

    Shards the parquet files disjointly across (DDP rank, DataLoader worker) so the
    union of all streams is one pass over the data with no overlap, reads one shard
    at a time, and yields whole-cell items (same schema as ``TahoeDataset``). A
    per-worker reservoir of ``config.shuffle_buffer`` items mixes rows across shards
    so batches are not all-same-shard (important for SIGReg/invariance diversity).
    Call ``set_epoch`` each epoch to reshuffle the shard order.
    """

    def __init__(
        self,
        config: TahoeConfig,
        binner: Optional[QuantileBinner] = None,
        cell_line_to_organ: Optional[dict] = None,
        sample_to_logconc: Optional[dict] = None,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
    ):
        import pyarrow.parquet as pq

        self.config = config
        self.binner = binner
        self.cell_line_to_organ = cell_line_to_organ or {}
        self.sample_to_logconc = sample_to_logconc or {}
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.epoch = 0
        self.files = sorted(
            glob.glob(os.path.join(config.data_dir, "**", "*.parquet"), recursive=True)
        )
        if not self.files:
            raise FileNotFoundError(f"No parquet files under {config.data_dir!r}")
        names = set(pq.read_schema(self.files[0]).names)
        self._cached = "value" in names
        self._value_col = "value" if self._cached else "expressions"
        self._meta_cols = [c for c in _META_COLUMNS if c in names]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _read_shard(self, f):
        import pyarrow.parquet as pq

        t = pq.read_table(f, columns=["genes", self._value_col] + self._meta_cols)
        genes_col = t.column("genes")
        val_col = t.column(self._value_col)
        meta_cols = {c: t.column(c) for c in self._meta_cols}
        for i in range(t.num_rows):
            genes = genes_col[i].values.to_numpy(zero_copy_only=False)
            vals = val_col[i].values.to_numpy(zero_copy_only=False)
            meta = {c: meta_cols[c][i].as_py() for c in self._meta_cols}
            yield _make_item(
                genes, vals, self._cached, meta, self.binner,
                self.cell_line_to_organ, self.sample_to_logconc,
            )

    def __iter__(self):
        import random

        from torch.utils.data import get_worker_info

        info = get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        gid = self.rank * nw + wid  # global stream id
        gn = self.world_size * nw  # number of disjoint streams
        files = list(self.files)
        if self.shuffle:
            random.Random(self.config.seed + self.epoch).shuffle(files)
        my_files = files[gid::gn]

        buf_cap = max(1, int(self.config.shuffle_buffer)) if self.shuffle else 1
        rng = random.Random(self.config.seed + self.epoch + 7919 * (gid + 1))
        buf = []
        for f in my_files:
            for item in self._read_shard(f):
                if buf_cap <= 1:
                    yield item
                    continue
                buf.append(item)
                if len(buf) >= buf_cap:
                    j = rng.randrange(len(buf))
                    buf[j], buf[-1] = buf[-1], buf[j]
                    yield buf.pop()
        rng.shuffle(buf)
        for item in buf:
            yield item

    def sample_items(self, n_cells: int, max_per_file: int = 200):
        """Read a fixed, diverse subset of ``n_cells`` items (for the eval/t-SNE set).

        Spreads the draw across evenly-spaced shards so cell lines/drugs are mixed.
        Deterministic (no shuffle) so the eval set is identical across snapshots.
        """
        import numpy as np

        k = max(1, min(len(self.files), -(-n_cells // max_per_file)))
        idxs = sorted(set(int(round(x)) for x in np.linspace(0, len(self.files) - 1, k)))
        per = -(-n_cells // len(idxs))
        items = []
        for fi in idxs:
            c = 0
            for item in self._read_shard(self.files[fi]):
                items.append(item)
                c += 1
                if c >= per or len(items) >= n_cells:
                    break
            if len(items) >= n_cells:
                break
        return items


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

    def __init__(self, config: TahoeConfig, membership: torch.Tensor | None = None):
        self.cfg = config
        assert config.view_mode in ("drop", "mask")
        assert config.count_mode in ("A", "B")
        # membership: [P, n_genes] float hallmark-weight matrix (row p, col token_id).
        # A cell's pathway counts = membership @ dense(cell counts). Required iff
        # use_pathways; the count is from the FULL cell (a cell property), and per-view
        # corruption is the dropout of the pathway TOKEN, not a recomputed count.
        self.membership = membership
        if config.use_pathways:
            assert membership is not None, "use_pathways requires a membership matrix"
            self.n_pathways = membership.shape[0]

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

        if cfg.use_pathways:
            # Pathway count per cell = membership @ dense(full-cell normalized counts).
            # Densify each cell's sparse (token_id -> value) into [n_genes], stack, and
            # project onto the [P, n_genes] hallmark matrix -> [N, P]. The count uses the
            # whole cell (not the view subset) and is shared across views; only the
            # token's presence is corrupted per view.
            dense = torch.zeros((n, cfg.n_genes), dtype=torch.float32)
            for j, cell in enumerate(batch):
                tok = cell["gene_token_ids"]
                if tok.numel():
                    dense[j, tok.long()] = cell["values"].float()
            pcount = dense @ self.membership.T  # [N, P]
            p = pcount.shape[1]
            # broadcast the per-cell count to V views; per-(view,cell,pathway) dropout
            pathway_count = pcount.unsqueeze(0).expand(v, n, p).contiguous()
            keep_prob = 1.0 - cfg.pathway_drop_frac
            pathway_mask = torch.rand((v, n, p)) < keep_prob  # True = token present
            out["pathway_count"] = pathway_count
            out["pathway_mask"] = pathway_mask
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
    rank: int = 0,
    world_size: int = 1,
):
    """Build the train DataLoader with the on-the-fly view collator.

    ``config.streaming`` selects the reader: the streaming ``TahoeIterableDataset``
    (full-dataset training; shards across ``rank``/workers internally — do NOT add a
    DistributedSampler) or the in-RAM ``TahoeDataset`` (small subsets / probing).
    The ``collate_fn`` generates the LeJEPA views on the fly.
    """
    cfg = config
    membership = None
    if cfg.use_pathways:
        if not cfg.pathway_membership:
            raise ValueError("use_pathways=True requires data.pathway_membership path")
        payload = torch.load(cfg.pathway_membership)
        membership = payload["M"].float()
        assert membership.shape[1] == cfg.n_genes, (
            f"membership n_genes {membership.shape[1]} != cfg.n_genes {cfg.n_genes}"
        )
    collator = TahoeCollator(cfg, membership=membership)
    if getattr(cfg, "streaming", False):
        dataset = TahoeIterableDataset(
            cfg, binner, cell_line_to_organ, sample_to_logconc,
            rank=rank, world_size=world_size, shuffle=(cfg.split == "train"),
        )
        train_loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_mem,
            drop_last=True,
            collate_fn=collator,
            persistent_workers=cfg.num_workers > 0,
        )
        return train_loader, dataset

    dataset = TahoeDataset(cfg, binner, cell_line_to_organ, sample_to_logconc)
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=True,
        collate_fn=collator,
        shuffle=(cfg.split == "train"),
    )
    return train_loader, dataset
