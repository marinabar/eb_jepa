"""Fetch the MSigDB **Hallmark** gene sets (collection H) and vendor them in-repo.

The 50 hallmark gene sets are the standard, coarse-grained biological pathways used
for the pathway probes and (later) the pathway tokens (CLAUDE.md "Pathways"). We pin
a release and download the canonical symbols ``.gmt`` from the Broad's public host
(no login needed), then write both the raw ``.gmt`` (for provenance) and a parsed
``hallmark.json`` into ``eb_jepa/datasets/tahoe/pathways/`` so the whole team gets
the exact same sets without re-downloading.

Usage:
    python scripts/fetch_hallmark_pathways.py                 # default version
    python scripts/fetch_hallmark_pathways.py --version 2024.1.Hs
"""

from __future__ import annotations

import json
import os
import urllib.request

# Pinned MSigDB release. Symbols collection so it maps onto Tahoe ``gene_symbol``.
DEFAULT_VERSION = "2024.1.Hs"
BROAD_HOST = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release"

# Vendored under the tahoe dataset package so it ships with the source tree.
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "eb_jepa",
    "datasets",
    "tahoe",
    "pathways",
)


def gmt_url(version: str) -> str:
    return f"{BROAD_HOST}/{version}/h.all.v{version}.symbols.gmt"


def parse_gmt(text: str) -> dict[str, dict]:
    """Parse a GMT (one set per line: ``name<TAB>url<TAB>gene1<TAB>gene2 ...``).

    Returns ``{name: {"url": str, "genes": [str, ...]}}``.
    """
    sets: dict[str, dict] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        name, url, genes = parts[0], parts[1], [g for g in parts[2:] if g]
        sets[name] = {"url": url, "genes": genes}
    return sets


def fetch(version: str = DEFAULT_VERSION, out_dir: str = OUT_DIR) -> dict[str, dict]:
    url = gmt_url(version)
    print(f"Downloading hallmark gene sets: {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (pinned host)
        text = resp.read().decode("utf-8")

    sets = parse_gmt(text)
    assert len(sets) == 50, f"expected 50 hallmark sets, got {len(sets)}"

    os.makedirs(out_dir, exist_ok=True)
    gmt_path = os.path.join(out_dir, f"h.all.v{version}.symbols.gmt")
    json_path = os.path.join(out_dir, "hallmark.json")
    with open(gmt_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text if text.endswith("\n") else text + "\n")
    payload = {
        "source": url,
        "version": version,
        "n_sets": len(sets),
        "id_type": "gene_symbol",
        "sets": {name: v["genes"] for name, v in sets.items()},
    }
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    sizes = [len(v["genes"]) for v in sets.values()]
    print(f"Wrote {gmt_path}")
    print(f"Wrote {json_path}")
    print(
        f"{len(sets)} sets · {sum(sizes)} gene memberships · "
        f"sizes min={min(sizes)} median={sorted(sizes)[len(sizes)//2]} max={max(sizes)}"
    )
    return sets


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default=DEFAULT_VERSION, help="MSigDB release tag")
    ap.add_argument("--out-dir", default=OUT_DIR, help="output directory")
    args = ap.parse_args()
    fetch(args.version, args.out_dir)
