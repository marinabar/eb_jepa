"""Hallmark pathway gene sets for pathway probing / pathway tokens (CLAUDE.md "Pathways").

The 50 MSigDB **Hallmark** sets are vendored under ``pathways/hallmark.json`` (fetched
once by ``scripts/fetch_hallmark_pathways.py``). This module loads them and maps the
gene **symbols** onto Tahoe ``token_id``s so a pathway "count" can be formed as the
weighted sum of its member genes' CP10k+log1p counts. Hallmark sets are unweighted, so
membership weights default to ``1.0`` (override later if a weighted variant is adopted).

The symbol -> token_id map is derived from ``gene_metadata`` (columns ``gene_symbol``,
``token_id``); pass it in so this module stays decoupled from where the metadata lives.
"""

from __future__ import annotations

import json
import os

_PATHWAY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pathways")
_HALLMARK_JSON = os.path.join(_PATHWAY_DIR, "hallmark.json")


def load_hallmark(path: str | None = None) -> dict[str, list[str]]:
    """Return ``{pathway_name: [gene_symbol, ...]}`` for the 50 hallmark sets."""
    with open(path or _HALLMARK_JSON, encoding="utf-8") as f:
        payload = json.load(f)
    return payload["sets"]


def hallmark_metadata(path: str | None = None) -> dict:
    """Return the full vendored payload (source URL, version, id_type, sets)."""
    with open(path or _HALLMARK_JSON, encoding="utf-8") as f:
        return json.load(f)


def symbol_to_token_map(gene_metadata) -> dict[str, int]:
    """Build ``{gene_symbol: token_id}`` from a ``gene_metadata`` table.

    Accepts a pandas/pyarrow-like object with ``gene_symbol`` and ``token_id`` columns
    (or a list of ``(symbol, token_id)`` pairs). Symbols are upper-cased to match the
    MSigDB convention; the first occurrence wins on duplicates.
    """
    if hasattr(gene_metadata, "to_pylist"):  # pyarrow Table
        rows = zip(
            gene_metadata.column("gene_symbol").to_pylist(),
            gene_metadata.column("token_id").to_pylist(),
        )
    elif hasattr(gene_metadata, "itertuples"):  # pandas DataFrame
        rows = ((r.gene_symbol, r.token_id) for r in gene_metadata.itertuples())
    else:  # iterable of (symbol, token_id)
        rows = iter(gene_metadata)
    mapping: dict[str, int] = {}
    for symbol, token_id in rows:
        if symbol is None:
            continue
        key = str(symbol).upper()
        if key not in mapping:
            mapping[key] = int(token_id)
    return mapping


def hallmark_to_token_ids(
    symbol_to_token: dict[str, int], path: str | None = None
) -> dict[str, list[int]]:
    """Map each hallmark set to the Tahoe ``token_id``s present in the vocabulary.

    Genes whose symbol is absent from ``symbol_to_token`` are dropped (logged by the
    caller if needed). Symbols are matched case-insensitively.
    """
    upper = {k.upper(): v for k, v in symbol_to_token.items()}
    out: dict[str, list[int]] = {}
    for name, genes in load_hallmark(path).items():
        ids = [upper[g.upper()] for g in genes if g.upper() in upper]
        out[name] = sorted(set(ids))
    return out


def membership_matrix(
    symbol_to_token: dict[str, int],
    n_genes: int = 62713,
    path: str | None = None,
):
    """Dense ``[n_pathways, n_genes]`` membership-weight matrix + ordered names.

    Row ``p``, column ``token_id`` is ``1.0`` iff that gene belongs to pathway ``p``.
    A cell's pathway counts are then ``M @ counts`` where ``counts`` is the dense
    ``[n_genes]`` CP10k+log1p vector (``densify`` output). Returns ``(M, names)``.
    """
    import torch

    mapped = hallmark_to_token_ids(symbol_to_token, path)
    names = sorted(mapped)
    m = torch.zeros(len(names), n_genes, dtype=torch.float32)
    for i, name in enumerate(names):
        ids = torch.tensor(mapped[name], dtype=torch.long)
        if ids.numel():
            m[i, ids] = 1.0
    return m, names
