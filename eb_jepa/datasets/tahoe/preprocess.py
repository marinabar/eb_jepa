"""Tahoe-100M preprocessing & caching (cluster job).

Runs once over ``/data/tahoe-100m`` to produce a training-ready cache:
  1. Build metadata maps: cell_line (CVCL) -> Organ, and sample -> log10 dose.
  2. (Optional) liver filter for the specialize phase.
  3. Group-level train/val split (held-out cell_line / organ / drug — NEVER
     cell-level, to avoid leakage), assigned BEFORE caching.
  4. Stream + subsample expression_data, write per-split parquet shards storing the
     CP10k+log1p value (continuous, mode-agnostic).
  5. Fit per-gene quantile boundaries (mode B) over a configurable sample of cells
     (``--quantile_cells``, default 10M), accumulated with a streaming per-gene
     histogram so memory is independent of the sample size. The histogram, the
     boundaries, and a small stats file are all saved separately so mode/K can
     change without re-caching (re-derive boundaries from the saved histogram).

Two entrypoints (the stats are computed ONCE and reused):
  - ``fit_stats``: compute the per-gene quantile histogram/bins + the group split +
    metadata maps over the WHOLE dataset, cache them to a stats dir. Vectorized; the
    split groups come from the metadata tables (no expression scan to enumerate them).
  - ``run``: write the per-split cache (CP10k+log1p ``value``, CLS stripped). With
    ``--stats_dir <fit_stats out>`` it reuses the precomputed bins/split/maps and
    skips the stats pass; otherwise it computes stats on the shards it writes.

Pure helpers (maps, split, stats) are unit-tested; the streaming scans are cluster
operations. Run on the GPU box (data at /data/tahoe-100m, use uv).

Usage (cluster):
    # 1. stats once over the whole dataset
    python -m eb_jepa.datasets.tahoe.preprocess fit_stats \
        --data_dir /data/tahoe-100m --out_dir /data/tahoe-stats --n_bins 64
    # 2. build a (subset) training cache that reuses those stats
    python -m eb_jepa.datasets.tahoe.preprocess run \
        --data_dir /data/tahoe-100m --out_dir /data/tahoe-cache \
        --stats_dir /data/tahoe-stats --max_shards 400
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import torch

from eb_jepa.datasets.tahoe.dataset import parse_log_conc
from eb_jepa.datasets.tahoe.normalizer import (
    GeneHistogram,
    QuantileBinner,
    cp10k_log1p,
    fit_quantile_boundaries,
)


# --------------------------------------------------------------------------- #
# Metadata maps                                                               #
# --------------------------------------------------------------------------- #
def build_cell_line_to_organ(cell_line_metadata_path: str) -> dict:
    """{Cell_ID_Cellosaur (CVCL) -> Organ}. Dedup (one row per driver gene per line)."""
    import pyarrow.parquet as pq

    t = pq.read_table(cell_line_metadata_path, columns=["Cell_ID_Cellosaur", "Organ"])
    out = {}
    for cvcl, organ in zip(
        t.column("Cell_ID_Cellosaur").to_pylist(), t.column("Organ").to_pylist()
    ):
        if cvcl is not None and cvcl not in out:
            out[cvcl] = organ
    return out


def build_sample_to_logconc(sample_metadata_path: str) -> dict:
    """{sample -> log10 molar dose} parsed from drugname_drugconc (nan for controls)."""
    import pyarrow.parquet as pq

    t = pq.read_table(sample_metadata_path, columns=["sample", "drugname_drugconc"])
    return {
        s: parse_log_conc(d)
        for s, d in zip(
            t.column("sample").to_pylist(), t.column("drugname_drugconc").to_pylist()
        )
    }


def liver_cell_lines(cell_line_to_organ: dict) -> set:
    """Set of hepatic CVCL ids (Organ == 'Liver')."""
    return {cvcl for cvcl, organ in cell_line_to_organ.items() if organ == "Liver"}


# --------------------------------------------------------------------------- #
# Group-level split (no leakage)                                              #
# --------------------------------------------------------------------------- #
def group_level_split(groups, val_frac: float = 0.1, seed: int = 0):
    """Partition UNIQUE group keys into (train, val) so no group spans both splits.

    ``groups`` is any iterable of per-cell group keys (e.g. cell_line ids, drugs,
    or organs). Returns ``(train_groups, val_groups)`` as sets of unique keys; a
    cell is val iff its group key is in ``val_groups``.
    """
    uniq = sorted({g for g in groups if g is not None})
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(uniq), generator=g).tolist()
    n_val = max(1, int(round(len(uniq) * val_frac))) if uniq else 0
    val = {uniq[i] for i in perm[:n_val]}
    train = {uniq[i] for i in perm[n_val:]}
    return train, val


def assign_split(group_key, val_groups: set) -> str:
    return "val" if group_key in val_groups else "train"


# --------------------------------------------------------------------------- #
# Quantile-boundary fitting from a streamed sample                            #
# --------------------------------------------------------------------------- #
def fit_boundaries_from_cells(cells, n_genes: int, n_bins: int = 64) -> QuantileBinner:
    """Fit per-gene quantile bins from an iterable of (token_ids, raw_counts) cells.

    Pools CP10k+log1p values per gene across the sample, then computes boundaries.
    """
    tok_chunks, val_chunks = [], []
    for token_ids, counts in cells:
        token_ids = torch.as_tensor(token_ids).long()
        values = cp10k_log1p(torch.as_tensor(counts).float())
        tok_chunks.append(token_ids)
        val_chunks.append(values)
    tok = torch.cat(tok_chunks)
    val = torch.cat(val_chunks)
    boundaries = fit_quantile_boundaries(tok, val, n_genes=n_genes, n_bins=n_bins)
    return QuantileBinner(boundaries, n_bins=n_bins)


def fit_histogram_from_cells(
    cells,
    n_genes: int,
    n_hist_bins: int = 4096,
    v_min: float = 0.0,
    v_max: float = 10.0,
) -> GeneHistogram:
    """Accumulate a streaming per-gene histogram from an iterable of cells.

    Memory-bounded counterpart to ``fit_boundaries_from_cells`` for large samples:
    each cell is CP10k+log1p'd and folded into the fixed grid, holding no raw
    values. Call ``.binner(n_bins)`` (or ``.quantile_boundaries(n_bins)``) after.
    """
    hist = GeneHistogram(n_genes, n_hist_bins=n_hist_bins, v_min=v_min, v_max=v_max)
    for token_ids, counts in cells:
        token_ids = torch.as_tensor(token_ids).long()
        values = cp10k_log1p(torch.as_tensor(counts).float())
        hist.update(token_ids, values)
    return hist


# --------------------------------------------------------------------------- #
# fit_stats: compute the histogram/bins + split ONCE over the whole dataset    #
# --------------------------------------------------------------------------- #
def _shard_token_values(table, keep_lines=None):
    """Vectorized per-shard (token_ids int64, CP10k+log1p values float32, n_cells).

    CLS marker stripped (first element of each cell). If ``keep_lines`` is given,
    only cells whose ``cell_line_id`` is in it contribute. Fully numpy-vectorized
    (no per-cell Python loop) so it can scan the whole dataset for the histogram.
    """
    import numpy as np

    t = table.combine_chunks()
    ga, ea = t.column("genes").chunk(0), t.column("expressions").chunk(0)
    g_off = ga.offsets.to_numpy()
    g_vals = ga.values.to_numpy(zero_copy_only=False)
    e_vals = ea.values.to_numpy(zero_copy_only=False)
    n = len(g_off) - 1
    lengths = (g_off[1:] - g_off[:-1] - 1).clip(min=0)  # per-cell length post-CLS-strip
    keep = np.ones(g_vals.shape[0], dtype=bool)
    keep[g_off[:-1]] = False  # drop the CLS marker (first element of each cell)
    if keep_lines is not None:
        cl = t.column("cell_line_id").to_pylist()
        row_keep = np.fromiter((c in keep_lines for c in cl), dtype=bool, count=n)
        keep &= np.repeat(row_keep, g_off[1:] - g_off[:-1])
        lengths = np.where(row_keep, lengths, 0)
        n_cells = int(row_keep.sum())
    else:
        n_cells = n
    tok = g_vals[keep].astype(np.int64)
    raw = e_vals[keep].astype(np.float64)
    cell_id = np.repeat(np.arange(n), lengths)
    totals = np.zeros(n, dtype=np.float64)
    np.add.at(totals, cell_id, raw)
    np.maximum(totals, 1.0, out=totals)
    vals = np.log1p(raw / totals[cell_id] * 1e4).astype(np.float32)
    return tok, vals, n_cells


def _derive_groups_from_metadata(
    data_dir, split_by, cell_line_metadata, sample_metadata, drug_metadata
):
    """Unique split-group keys from a metadata table (no expression scan)."""
    import pyarrow.parquet as pq

    spec = {
        "cell_line_id": (cell_line_metadata, "Cell_ID_Cellosaur"),
        "sample": (sample_metadata, "sample"),
        "drug": (drug_metadata, "drug"),
    }.get(split_by)
    if spec is None:
        return None
    path, col = spec
    try:
        f = _first_match(data_dir, path)
    except FileNotFoundError:
        return None
    return [
        v
        for v in pq.read_table(f, columns=[col]).column(col).to_pylist()
        if v is not None
    ]


def fit_stats(
    data_dir: str = "/data/tahoe-100m",
    out_dir: str = "/data/tahoe-stats",
    expression_glob: str = "data/*.parquet",
    cell_line_metadata: str = "metadata/cell_line_metadata.parquet",
    sample_metadata: str = "metadata/sample_metadata.parquet",
    gene_metadata: str = "metadata/gene_metadata.parquet",
    drug_metadata: str = "metadata/drug_metadata.parquet",
    liver_only: bool = False,
    val_frac: float = 0.1,
    split_by: str = "cell_line_id",
    n_bins: int = 64,
    n_genes: int = 62713,
    quantile_cells: int = 0,  # 0 = use every scanned cell
    n_hist_bins: int = 4096,
    hist_v_min: float = 0.0,
    hist_v_max: float = 10.0,
    max_shards: int = 0,  # 0 = all shards; else an evenly-spread subset
    seed: int = 0,
):
    """Compute, ONCE over the whole dataset, the reusable stats: per-gene quantile
    histogram + bins, the group-level split, and the metadata maps. Writes them to
    ``out_dir`` (no per-cell cache). ``run(--stats_dir out_dir)`` then reuses these
    for any (subset) cache build without recomputing.

    The histogram is fit fully vectorized over the scanned shards (all by default;
    ``max_shards`` picks an evenly-spread subset — a multi-million-cell spread
    sample yields per-gene quantiles statistically indistinguishable from the full
    dataset). Split groups come from the metadata tables, so enumerating them needs
    no expression scan. Saving the histogram lets bins for any ``n_bins`` be
    re-derived later without re-scanning.
    """
    import pyarrow.parquet as pq

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cl2organ = build_cell_line_to_organ(_first_match(data_dir, cell_line_metadata))
    s2dose = build_sample_to_logconc(_first_match(data_dir, sample_metadata))
    torch.save(
        {"cell_line_to_organ": cl2organ, "sample_to_logconc": s2dose}, out / "maps.pt"
    )
    keep_lines = liver_cell_lines(cl2organ) if liver_only else None

    try:
        toks = (
            pq.read_table(_first_match(data_dir, gene_metadata), columns=["token_id"])
            .column("token_id")
            .to_pylist()
        )
        vocab = max(toks) + 1
    except FileNotFoundError:
        vocab = n_genes

    groups = _derive_groups_from_metadata(
        data_dir, split_by, cell_line_metadata, sample_metadata, drug_metadata
    )
    if groups is None:
        raise ValueError(
            f"fit_stats cannot derive split groups for split_by={split_by!r}; "
            "supported: cell_line_id, sample, drug."
        )
    if keep_lines is not None and split_by == "cell_line_id":
        groups = [g for g in groups if g in keep_lines]
    _train, val_groups = group_level_split(groups, val_frac, seed)
    torch.save(
        {"val_groups": sorted(val_groups), "split_by": split_by}, out / "split.pt"
    )

    files = sorted(glob.glob(os.path.join(data_dir, expression_glob), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No expression parquet under {data_dir}/{expression_glob}"
        )
    if max_shards and max_shards < len(files):
        stride = len(files) // max_shards
        files = files[::stride][:max_shards]

    read_cols = ["genes", "expressions"] + (
        ["cell_line_id"] if keep_lines is not None else []
    )
    hist = GeneHistogram(vocab, n_hist_bins, hist_v_min, hist_v_max)
    cells_used, shards_used = 0, 0
    for f in files:
        t = pq.read_table(f, columns=read_cols)
        tok, vals, n_cells = _shard_token_values(t, keep_lines)
        if tok.size:
            hist.update(torch.from_numpy(tok), torch.from_numpy(vals))
        cells_used += n_cells
        shards_used += 1
        if quantile_cells > 0 and cells_used >= quantile_cells:
            break

    hist.binner(n_bins).save(out / "quantile_bins")
    hist.save(out / "gene_count_histogram")
    torch.save(
        {
            "cells_used": cells_used,
            "shards_used": shards_used,
            "vocab": vocab,
            "n_bins": n_bins,
            "n_hist_bins": n_hist_bins,
            "hist_v_min": hist_v_min,
            "hist_v_max": hist_v_max,
            "split_by": split_by,
            "liver_only": liver_only,
        },
        out / "quantile_stats.pt",
    )
    print(
        f"fit_stats: {cells_used} cells / {shards_used} shards -> {out_dir} "
        f"(n_bins={n_bins}, vocab={vocab}, val_groups={len(val_groups)})"
    )


# --------------------------------------------------------------------------- #
# Streaming cache write (cluster)                                             #
# --------------------------------------------------------------------------- #
def run(
    data_dir: str = "/data/tahoe-100m",
    out_dir: str = "/data/tahoe-cache",
    stats_dir: str = "",  # reuse fit_stats() output (bins+split+maps); skip stats pass
    expression_glob: str = "data/*.parquet",
    cell_line_metadata: str = "metadata/cell_line_metadata.parquet",
    sample_metadata: str = "metadata/sample_metadata.parquet",
    gene_metadata: str = "metadata/gene_metadata.parquet",
    liver_only: bool = False,
    val_frac: float = 0.1,
    split_by: str = "cell_line_id",
    n_bins: int = 64,
    n_genes: int = 62713,  # index space = max Tahoe token_id (62712) + 1, NOT gene count
    quantile_cells: int = 10_000_000,
    n_hist_bins: int = 4096,
    hist_v_min: float = 0.0,
    hist_v_max: float = 10.0,
    subsample: int = 0,
    max_shards: int = 0,
    seed: int = 0,
):
    """Build the cache + split + quantile stats. See module docstring for usage.

    Streams expression_data shards, strips the CLS marker, writes one parquet per
    split storing ``(genes, value)`` with the CP10k+log1p value, and saves the
    metadata maps + quantile boundaries under ``out_dir``.

    Quantile boundaries are fit over ~``quantile_cells`` cells (default 10M; set 0
    to use every cell) sampled across the *whole* dataset via a seeded per-row
    Bernoulli, accumulated into a streaming ``GeneHistogram`` (``n_hist_bins`` grid
    over ``[hist_v_min, hist_v_max]``) so memory is independent of the sample size.
    The histogram, boundaries, and a stats file are all saved.
    """
    import random

    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(os.path.join(data_dir, expression_glob), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No expression parquet under {data_dir}/{expression_glob}"
        )
    # Scan a subset of the (3388) shards spread evenly across the dataset (stride,
    # not the first N) so all cell lines/drugs appear.
    if max_shards and max_shards < len(files):
        stride = len(files) // max_shards
        files = files[::stride][:max_shards]

    if stats_dir:
        # Reuse stats computed ONCE over the whole dataset by fit_stats(): the maps,
        # the group split, and the quantile bins. Skip the histogram/group pass.
        import shutil

        sd = Path(stats_dir)
        cl2organ = torch.load(sd / "maps.pt")["cell_line_to_organ"]
        keep_lines = liver_cell_lines(cl2organ) if liver_only else None
        split = torch.load(sd / "split.pt")
        val_groups, split_by = set(split["val_groups"]), split["split_by"]
        for fn in ("maps.pt", "split.pt", "quantile_bins.npy", "quantile_bins.json"):
            if (sd / fn).exists():
                shutil.copy(sd / fn, out / fn)
    else:
        clm = _first_match(data_dir, cell_line_metadata)
        smd = _first_match(data_dir, sample_metadata)
        cl2organ = build_cell_line_to_organ(clm)
        s2dose = build_sample_to_logconc(smd)
        keep_lines = liver_cell_lines(cl2organ) if liver_only else None
        torch.save(
            {"cell_line_to_organ": cl2organ, "sample_to_logconc": s2dose},
            out / "maps.pt",
        )
        # Index space = max token_id + 1 (token_ids non-contiguous, up to 62712).
        try:
            gmd = _first_match(data_dir, gene_metadata)
            toks = (
                pq.read_table(gmd, columns=["token_id"]).column("token_id").to_pylist()
            )
            vocab = max(toks) + 1
        except FileNotFoundError:
            vocab = n_genes
        total_rows = sum(pq.read_metadata(f).num_rows for f in files)
        keep_p = (
            1.0
            if (liver_only or quantile_cells <= 0)
            else min(1.0, quantile_cells / max(total_rows, 1))
        )
        rng = random.Random(seed)
        # First pass: collect group keys (split) + fit the streaming histogram.
        hist = GeneHistogram(vocab, n_hist_bins, hist_v_min, hist_v_max)
        group_keys, q_used, kept = set(), 0, 0
        # dedupe in case split_by == "cell_line_id" (can't request a column twice)
        pass1_cols = list(
            dict.fromkeys(["genes", "expressions", split_by, "cell_line_id"])
        )
        for f in files:
            t = pq.read_table(f, columns=pass1_cols)
            g_acc, v_acc = [], []
            for i in range(t.num_rows):
                cvcl = t.column("cell_line_id")[i].as_py()
                if keep_lines is not None and cvcl not in keep_lines:
                    continue
                group_keys.add(t.column(split_by)[i].as_py())
                take = (quantile_cells <= 0 or q_used < quantile_cells) and (
                    keep_p >= 1.0 or rng.random() < keep_p
                )
                if take:
                    g = t.column("genes")[i].values.to_numpy(zero_copy_only=False)[1:]
                    e = t.column("expressions")[i].values.to_numpy(
                        zero_copy_only=False
                    )[1:]
                    g_acc.append(torch.from_numpy(g.astype("int64")))
                    v_acc.append(cp10k_log1p(torch.from_numpy(e.copy()).float()))
                    q_used += 1
                kept += 1
                if subsample and kept >= subsample:
                    break
            if g_acc:
                hist.update(torch.cat(g_acc), torch.cat(v_acc))
            if subsample and kept >= subsample:
                break
        _train_groups, val_groups = group_level_split(group_keys, val_frac, seed)
        hist.binner(n_bins).save(out / "quantile_bins")
        hist.save(out / "gene_count_histogram")
        torch.save(
            {
                "quantile_cells_requested": quantile_cells,
                "quantile_cells_used": q_used,
                "keep_p": keep_p,
                "total_rows": total_rows,
                "n_bins": n_bins,
                "n_hist_bins": n_hist_bins,
                "hist_v_min": hist_v_min,
                "hist_v_max": hist_v_max,
            },
            out / "quantile_stats.pt",
        )
        torch.save(
            {"val_groups": sorted(val_groups), "split_by": split_by}, out / "split.pt"
        )

    # Second pass: write per-split parquet shards (CP10k+log1p value, CLS stripped).
    writers = {}
    schema = pa.schema(
        [("genes", pa.list_(pa.int64())), ("value", pa.list_(pa.float32()))]
        + [
            (c, pa.string())
            for c in (
                "drug",
                "sample",
                "cell_line_id",
                "moa-fine",
                "canonical_smiles",
                "plate",
                "BARCODE_SUB_LIB_ID",
            )
        ]
    )
    written = 0
    cols = [
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
    for f in files:
        t = pq.read_table(f, columns=cols)
        rows = {"train": _empty_rows(schema), "val": _empty_rows(schema)}
        for i in range(t.num_rows):
            cvcl = t.column("cell_line_id")[i].as_py()
            if keep_lines is not None and cvcl not in keep_lines:
                continue
            split = assign_split(t.column(split_by)[i].as_py(), val_groups)
            g = t.column("genes")[i].values.to_numpy(zero_copy_only=False)[1:]
            e = t.column("expressions")[i].values.to_numpy(zero_copy_only=False)[1:]
            val = cp10k_log1p(torch.from_numpy(e.copy()).float()).tolist()
            r = rows[split]
            r["genes"].append(g.tolist())
            r["value"].append(val)
            for c in (
                "drug",
                "sample",
                "cell_line_id",
                "moa-fine",
                "canonical_smiles",
                "plate",
                "BARCODE_SUB_LIB_ID",
            ):
                r[c].append(t.column(c)[i].as_py())
            written += 1
            if subsample and written >= subsample:
                break
        for split, r in rows.items():
            if not r["genes"]:
                continue
            (out / split).mkdir(exist_ok=True)
            w = writers.setdefault(
                split,
                pq.ParquetWriter(str(out / split / f"{Path(f).stem}.parquet"), schema),
            )
            w.write_table(pa.table(r, schema=schema))
        if subsample and written >= subsample:
            break
    for w in writers.values():
        w.close()
    print(
        f"Wrote {written} cells to {out_dir} (liver_only={liver_only}, val groups={len(val_groups)})"
    )


def _empty_rows(schema):
    return {name: [] for name in schema.names}


def _first_match(data_dir: str, pattern: str) -> str:
    matches = sorted(glob.glob(os.path.join(data_dir, pattern), recursive=True))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} under {data_dir}")
    return matches[0]


if __name__ == "__main__":
    import fire

    fire.Fire({"fit_stats": fit_stats, "run": run})
