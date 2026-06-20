"""Fetch the precomputed gene-embedding cache from MinIO (SSP Cloud).

Pulls the artifacts produced by ``scripts/build_gene_embeddings.py`` and published to

    s3://concordance/hacktheworld/gene_emb_cache/

on the SSP Cloud MinIO (``minio.lab.sspcloud.fr``) into a local directory that
``GeneTokenEmbedding.from_cache`` can load directly (esmc.npy, evo2.npy,
index.parquet, metadata.json).

Credentials
-----------
The prefix is PRIVATE, so set the standard S3 env vars before running. On SSP
Cloud / Onyxia these are injected into every service automatically; otherwise copy
them from the MinIO console ("My Account" -> credentials):

    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_SESSION_TOKEN=...          # STS tokens are temporary — refresh when expired
    export AWS_S3_ENDPOINT=minio.lab.sspcloud.fr   # optional; this is the default

Never hard-code credentials here. ``boto3`` reads them from the environment.

Usage
-----
    uv pip install boto3
    python scripts/fetch_gene_embeddings.py --out /data/gene_emb_cache
    # if the prefix is ever made public, credential-free over plain HTTPS:
    python scripts/fetch_gene_embeddings.py --out ./gene_emb_cache --public
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
        "--public",
        action="store_true",
        help="anonymous HTTPS download (only works if the prefix is public)",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fetch = fetch_public if args.public else fetch_authenticated
    fetch(out, args.endpoint, args.bucket, args.prefix)

    meta = json.loads((out / "metadata.json").read_text())
    sizes = {f: (out / f).stat().st_size for f in FILES}
    print(f"OK -> {out}")
    print("metadata:", meta)
    print("sizes (bytes):", sizes)


if __name__ == "__main__":
    main()
