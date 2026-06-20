"""Fetch the precomputed gene-embedding cache from MinIO (SSP Cloud).

Pulls the artifacts produced by ``scripts/build_gene_embeddings.py`` and published to

    s3://concordance/hacktheworld/gene_emb_cache/

on the SSP Cloud MinIO (``minio.lab.sspcloud.fr``) into a local directory that
``GeneTokenEmbedding.from_cache`` can load directly (esmc.npy, evo2.npy,
index.parquet, metadata.json).

Access
------
The ``hacktheworld/`` prefix is PUBLIC, so the default mode needs NO credentials —
it downloads over plain HTTPS. Use ``--auth`` only if the prefix is made private
again; then set the standard S3 env vars (Onyxia injects them, or copy from the
MinIO console) and ``uv pip install boto3``:

    export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_SESSION_TOKEN=...
    export AWS_S3_ENDPOINT=minio.lab.sspcloud.fr   # optional; this is the default

Never hard-code credentials here.

Usage
-----
    # public, no credentials (default)
    python scripts/fetch_gene_embeddings.py --out /data/gene_emb_cache
    # quick connectivity test (only the tiny metadata file)
    python scripts/fetch_gene_embeddings.py --out /tmp/gef --files metadata.json
    # authenticated fallback if the prefix is private
    python scripts/fetch_gene_embeddings.py --out ./gene_emb_cache --auth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

DEFAULT_ENDPOINT = os.environ.get("AWS_S3_ENDPOINT", "minio.lab.sspcloud.fr")
DEFAULT_BUCKET = "concordance"
DEFAULT_PREFIX = "hacktheworld/gene_emb_cache"
FILES = ("metadata.json", "index.parquet", "esmc.npy", "evo2.npy")


def _https(endpoint: str) -> str:
    return endpoint if endpoint.startswith("http") else f"https://{endpoint}"


def public_url(endpoint: str, bucket: str, prefix: str, name: str) -> str:
    """Anonymous object URL (works only if the prefix has public-read policy)."""
    return f"{_https(endpoint)}/{bucket}/{prefix}/{name}"


def fetch_public(out: Path, endpoint, bucket, prefix, files=FILES):
    for name in files:
        url = public_url(endpoint, bucket, prefix, name)
        print(f"GET {url}")
        urllib.request.urlretrieve(url, out / name)


def fetch_authenticated(out: Path, endpoint, bucket, prefix, files=FILES):
    try:
        import boto3
    except ImportError:
        sys.exit(
            "boto3 not installed. Run `uv pip install boto3` (or use --public "
            "if the prefix has been made public)."
        )
    s3 = boto3.client(
        "s3",
        endpoint_url=_https(endpoint),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    for name in files:
        key = f"{prefix}/{name}"
        print(f"download s3://{bucket}/{key}")
        s3.download_file(bucket, key, str(out / name))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Fetch the gene-embedding cache from MinIO."
    )
    ap.add_argument(
        "--out", default="/data/gene_emb_cache", help="local destination dir"
    )
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument(
        "--files",
        nargs="+",
        default=list(FILES),
        help="subset of artifacts to fetch (default: all four)",
    )
    ap.add_argument(
        "--auth",
        action="store_true",
        help="authenticated S3 via boto3 + AWS_* env creds (use if prefix is private)",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fetch = fetch_authenticated if args.auth else fetch_public
    fetch(out, args.endpoint, args.bucket, args.prefix, files=tuple(args.files))

    if "metadata.json" in args.files:
        print("metadata:", json.loads((out / "metadata.json").read_text()))
    print("sizes (bytes):", {f: (out / f).stat().st_size for f in args.files})
    print(f"OK -> {out}")


if __name__ == "__main__":
    main()
