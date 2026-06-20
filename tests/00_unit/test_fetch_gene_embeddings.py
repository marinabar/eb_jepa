"""URL/key construction for the MinIO gene-embedding fetch script (no network)."""

import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "fge",
    pathlib.Path(__file__).parents[2] / "scripts" / "fetch_gene_embeddings.py",
)
fge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fge)


def test_files_are_the_four_artifacts():
    assert set(fge.FILES) == {"metadata.json", "index.parquet", "esmc.npy", "evo2.npy"}


def test_public_url_from_host():
    assert (
        fge.public_url(
            "minio.lab.sspcloud.fr",
            "concordance",
            "hacktheworld/gene_emb_cache",
            "evo2.npy",
        )
        == "https://minio.lab.sspcloud.fr/concordance/hacktheworld/gene_emb_cache/evo2.npy"
    )


def test_public_url_accepts_full_scheme():
    assert fge.public_url("https://x", "b", "p", "f") == "https://x/b/p/f"


def test_defaults():
    assert fge.DEFAULT_BUCKET == "concordance"
    assert fge.DEFAULT_PREFIX == "hacktheworld/gene_emb_cache"
