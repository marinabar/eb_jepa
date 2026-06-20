"""Precompute the hallmark pathway membership matrix (offline, once — like the
quantile bins). Reads ``gene_metadata.parquet`` (token_id <-> gene_symbol) and the
vendored 50 MSigDB hallmark sets, maps symbols onto Tahoe token_ids, and saves a
dense ``[P, n_genes]`` membership-weight matrix + ordered pathway names.

The training collator loads this (``data.pathway_membership``) and forms a cell's
pathway counts as ``M @ dense(cell counts)`` (CLAUDE.md "Pathways").

Usage (on a Dalia compute node, aarch64 venv):
  python scripts/build_pathway_membership.py \
    --gene-metadata /lustre/work/vivatech-unaite/shared/tahoe-100m/metadata/gene_metadata.parquet \
    --out /lustre/work/vivatech-unaite/shared/tahoe-cache/pathway_membership.pt
"""

from __future__ import annotations

import argparse

import pyarrow.parquet as pq
import torch

from eb_jepa.datasets.tahoe.pathways import membership_matrix, symbol_to_token_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-metadata", required=True, help="gene_metadata.parquet")
    ap.add_argument("--out", required=True, help="output .pt path")
    ap.add_argument("--n-genes", type=int, default=62713, help="token-id index space")
    args = ap.parse_args()

    meta = pq.read_table(args.gene_metadata, columns=["gene_symbol", "token_id"])
    sym2tok = symbol_to_token_map(meta)
    m, names = membership_matrix(sym2tok, n_genes=args.n_genes)

    per_pathway = m.sum(1).long().tolist()
    n_mapped = int((m.sum(0) > 0).sum())
    print(f"pathways={len(names)} n_genes={args.n_genes} genes_in_any_pathway={n_mapped}")
    print(f"members/pathway: min={min(per_pathway)} max={max(per_pathway)}")
    empty = [names[i] for i, c in enumerate(per_pathway) if c == 0]
    if empty:
        print(f"WARNING: {len(empty)} pathways have 0 mapped genes: {empty}")

    torch.save({"M": m, "names": names, "n_genes": args.n_genes}, args.out)
    print(f"saved {args.out}  M={tuple(m.shape)}")


if __name__ == "__main__":
    main()
