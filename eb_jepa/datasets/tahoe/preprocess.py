"""Tahoe-100M preprocessing & caching (cluster job).

Runs once over ``/data/tahoe-100m`` to produce a training-ready cache:
  1. Build metadata maps: cell_line (CVCL) -> Organ, and sample -> log10 dose.
  2. (Optional) liver filter for the specialize phase.
  3. Group-level train/val split (held-out cell_line / organ / drug — NEVER
     cell-level, to avoid leakage), assigned BEFORE caching.
  4. Stream + subsample expression_data, write per-split parquet shards storing the
     CP10k+log1p value (continuous, mode-agnostic).
  5. Fit per-gene quantile boundaries (mode B) on a sample, stored separately so
     mode/K can change without re-caching.

Pure helpers (maps, split, stats) are unit-tested; the streaming cache write is
a cluster operation. Run on the GPU box (data at /data/tahoe-100m, use uv).

Usage (cluster):
    python -m eb_jepa.datasets.tahoe.preprocess run \
        --data_dir /data/tahoe-100m --out_dir /data/tahoe-cache \
        --liver_only False --val_frac 0.1 --split_by cell_line_id --n_bins 50
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import torch

from eb_jepa.datasets.tahoe.dataset import parse_log_conc
from eb_jepa.datasets.tahoe.normalizer import (
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
def fit_boundaries_from_cells(cells, n_genes: int, n_bins: int = 50) -> QuantileBinner:
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


# --------------------------------------------------------------------------- #
# Streaming cache write (cluster)                                             #
# --------------------------------------------------------------------------- #
def run(
    data_dir: str = "/data/tahoe-100m",
    out_dir: str = "/data/tahoe-cache",
    expression_glob: str = "expression_data/**/*.parquet",
    cell_line_metadata: str = "cell_line_metadata/*.parquet",
    sample_metadata: str = "sample_metadata/*.parquet",
    liver_only: bool = False,
    val_frac: float = 0.1,
    split_by: str = "cell_line_id",
    n_bins: int = 50,
    n_genes: int = 62710,
    subsample: int = 0,
    seed: int = 0,
):
    """Build the cache + split + quantile stats. See module docstring for usage.

    Streams expression_data shards, strips the CLS marker, writes one parquet per
    split storing ``(genes, value)`` with the CP10k+log1p value, and saves the
    metadata maps + quantile boundaries under ``out_dir``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    clm = _first_match(data_dir, cell_line_metadata)
    smd = _first_match(data_dir, sample_metadata)
    cl2organ = build_cell_line_to_organ(clm)
    s2dose = build_sample_to_logconc(smd)
    keep_lines = liver_cell_lines(cl2organ) if liver_only else None

    torch.save(
        {"cell_line_to_organ": cl2organ, "sample_to_logconc": s2dose}, out / "maps.pt"
    )

    files = sorted(glob.glob(os.path.join(data_dir, expression_glob), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No expression parquet under {data_dir}/{expression_glob}"
        )

    # First pass: collect the group keys present (for the split) + a stats sample.
    group_keys, stats_cells, kept = set(), [], 0
    for f in files:
        t = pq.read_table(f, columns=["genes", "expressions", split_by, "cell_line_id"])
        for i in range(t.num_rows):
            cvcl = t.column("cell_line_id")[i].as_py()
            if keep_lines is not None and cvcl not in keep_lines:
                continue
            group_keys.add(t.column(split_by)[i].as_py())
            if len(stats_cells) < 200_000:  # bounded sample for quantiles
                g = t.column("genes")[i].values.to_numpy(zero_copy_only=False)[1:]
                e = t.column("expressions")[i].values.to_numpy(zero_copy_only=False)[1:]
                stats_cells.append((g, e))
            kept += 1
            if subsample and kept >= subsample:
                break
        if subsample and kept >= subsample:
            break

    _train_groups, val_groups = group_level_split(group_keys, val_frac, seed)
    binner = fit_boundaries_from_cells(stats_cells, n_genes=n_genes, n_bins=n_bins)
    binner.save(out / "quantile_bins")
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

    fire.Fire({"run": run})
