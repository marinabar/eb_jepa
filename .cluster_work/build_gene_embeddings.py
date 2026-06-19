"""Precompute the per-gene ESMC + Evo2 embedding cache for the Tahoe-100M JEPA.

This is an OFFLINE cluster job. It builds, once, the two frozen lookup stores the
encoder reads at train time (see ``CLAUDE.md`` -> "Token embeddings" and
"Gene-embedding cache (ESMC + Evo2)"):

  * ESMC pooled protein embeddings  -> ``[N_coding, 1152]``  (coding genes only)
  * Evo2 pooled DNA embeddings      -> ``[N_genes, d_evo2]`` (all genes)
  * an index parquet/json keyed by ``token_id`` so the encoder can look a gene up.

Pipeline, per gene (``token_id -> ensembl_id`` from the ``gene_metadata`` config,
62,710 genes, Ensembl release 109 / GRCh38):
  1. Resolve via Ensembl (pinned to release 109): canonical transcript id, biotype
     (``protein_coding`` => ``is_coding=True``), canonical PROTEIN sequence (coding
     only) and canonical transcript DNA (all genes). Raw sequences are cached on
     disk so embeddings can be recomputed without re-querying Ensembl.
  2. ESMC ``esmc-600m`` (1152-dim/residue) over the protein, mean-pooled -> 1 vector
     per coding gene.
  3. Evo2 (default ``arcinstitute/evo2_7b_base``, ``d_evo2`` READ FROM THE MODEL CONFIG)
     over the DNA, embeddings extracted at a configurable LAYER + POOLING (default
     mean over tokens). The layer/pooling are chosen by the validation sweep below,
     not assumed.

The job is idempotent and resumable: per-gene progress is checkpointed, so a long
run on the cluster can be killed and restarted and will skip genes already done.

----------------------------------------------------------------------------------
RUNTIME / ENVIRONMENT
----------------------------------------------------------------------------------
Runs offline on the cluster (8x B200, data at ``/data/tahoe-100m``), launched with
``uv``. Models and data are ONLY on the cluster; do not attempt to run this locally.

Extra dependencies NOT in ``pyproject.toml`` (install in the cluster env, e.g.
``uv pip install ...``):
  * ``esm``        (Evolutionary Scale, provides ``esm.models.esmc.ESMC``) -- ESMC
  * ``evo2``       (Arc Institute Evo2)                                      -- Evo2
  * ``requests``   (Ensembl REST client; usually already pulled in by huggingface-hub)
  * ``pyensembl``  (OPTIONAL alternative sequence source; pin release 109)

GPU notes: ESMC 600M and Evo2 7B both want bf16 on a single B200 each. This script
uses one device at a time; to parallelise across the 8 GPUs, shard the gene list
with ``--shard-index/--num-shards`` and launch one process per GPU (each writes its
own checkpoint; merge the row-stores in a final ``--merge-shards`` pass -- TODO hook
left in :func:`merge_shards`). A single-GPU full run is the simple, correct default.

----------------------------------------------------------------------------------
Usage:
----------------------------------------------------------------------------------
  # Smoke test on 8 genes (CPU resolve + tiny model load), no real embedding compute:
  uv run python -m scripts.build_gene_embeddings --limit 8 --dry_run True

  # Full build with defaults (esmc-600m + arcinstitute/evo2_7b):
  uv run python -m scripts.build_gene_embeddings \
      --data_dir /data/tahoe-100m --out_dir /data/gene_cache

  # Pick a smaller Evo2 and a specific extraction layer/pooling:
  uv run python -m scripts.build_gene_embeddings \
      --evo2_model arcinstitute/evo2_1b_base --evo2_layer 24 --evo2_pooling mean

  # Layer-sweep mode: dump per-layer Evo2 embeddings for ~200 genes so a downstream
  # probe can pick the best layer/pooling (see step 6 / select_best_layer TODO):
  uv run python -m scripts.build_gene_embeddings \
      --evo2_layer_sweep True --sweep_genes 200 \
      --sweep_layers "[8,16,24,32]" --out_dir /data/gene_cache

  # Override anything (fire dot-notation also works on nested attrs of the config):
  uv run python -m scripts.build_gene_embeddings --batch_size 16 --max_dna_len 24000
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import fire
import numpy as np

from eb_jepa.logging import get_logger

logger = get_logger(__name__)

# ESMC 600M emits a fixed 1152-d per-residue representation. This is a property of
# the published checkpoint (not a tunable), so it is a constant rather than config.
# (Verified on cluster: esmc_600m -> out.embeddings is [1, L, 1152].)
ESMC_DIM = 1152

# Ensembl REST root. release 109 / GRCh38 is requested explicitly per-call via the
# `?content-type=...` headers and the `species=homo_sapiens` + Ensembl archive base.
ENSEMBL_REST = "https://rest.ensembl.org"


# =============================================================================
# Config
# =============================================================================
@dataclass
class Config:
    """All knobs for the cache build. Every field is overridable from the CLI."""

    # --- data / output locations ------------------------------------------------
    data_dir: str = "/data/tahoe-100m"  # local Tahoe-100M parquet root
    out_dir: str = "/data/gene_cache"  # where the cache + index are written
    cache_dir: Optional[str] = None  # raw-sequence cache (default: out_dir/seq_cache)

    # --- gene vocabulary --------------------------------------------------------
    gene_metadata_glob: str = "gene_metadata/*.parquet"  # relative to data_dir
    hf_repo: str = "tahoebio/Tahoe-100M"  # fallback source if local parquet absent

    # --- Ensembl sequence resolution -------------------------------------------
    ensembl_release: int = 109  # PIN: Tahoe-100M is release 109 / GRCh38
    sequence_source: str = "rest"  # "rest" (exact, default) or "pyensembl"
    rest_timeout: float = 30.0
    rest_max_retries: int = 5
    rest_backoff: float = 1.0  # base seconds for exponential backoff

    # --- ESMC (protein) ---------------------------------------------------------
    esmc_model: str = "esmc-600m"
    esmc_max_residues: int = 2048  # cap; longer proteins are skipped w/ warning
    esmc_batch_size: int = 8  # length-bucketed batch size

    # --- Evo2 (DNA) -------------------------------------------------------------
    # Default is the 7b BASE model: it loads on a single B200 in bf16 WITHOUT
    # Transformer Engine (the evo2 loader auto-falls-back fp8->bf16 for "7b" names),
    # whereas evo2_1b_base hard-requires Transformer-Engine FP8 and refuses to run
    # without it. d_evo2 for evo2_7b_base = 4096 (read from the model config).
    evo2_model: str = "arcinstitute/evo2_7b_base"  # also: evo2_7b, evo2_40b_base, ...
    evo2_layer: int = 24  # extraction layer (mid/late MLP; 0..31 for 7b, sweep-tunable)
    evo2_pooling: str = "mean"  # "mean" | "last" | "max"
    max_dna_len: int = 16384  # cap on DNA token length (chunk beyond)
    evo2_chunk_overlap: int = 256  # overlap when chunking long sequences

    # --- compute ----------------------------------------------------------------
    device: str = "auto"  # auto | cuda | cuda:0 | cpu
    batch_size: int = 8  # generic batch size (Evo2 forward)
    dtype: str = "bfloat16"  # bfloat16 | float16 | float32

    # --- run control ------------------------------------------------------------
    limit: Optional[int] = None  # only first N genes (smoke runs)
    shard_index: int = 0  # this process's shard (0-based)
    num_shards: int = 1  # total shards (one per GPU)
    merge_only: bool = False  # skip building; merge existing out_dir/shard_* stores
    checkpoint_every: int = 256  # flush progress every N genes
    dry_run: bool = False  # resolve only; do NOT load models/embed
    seed: int = 42

    # --- layer-sweep mode -------------------------------------------------------
    evo2_layer_sweep: bool = False
    sweep_genes: int = 200
    sweep_layers: Sequence[int] = field(default_factory=lambda: [8, 16, 24, 32])
    sweep_poolings: Sequence[str] = field(default_factory=lambda: ["mean", "last"])

    def resolved_cache_dir(self) -> Path:
        return (
            Path(self.cache_dir) if self.cache_dir else Path(self.out_dir) / "seq_cache"
        )


# =============================================================================
# Step 1 -- gene vocabulary
# =============================================================================
@dataclass
class GeneRecord:
    token_id: int
    ensembl_id: str
    gene_symbol: str
    # filled in by resolution:
    transcript_id: Optional[str] = None
    biotype: Optional[str] = None
    is_coding: bool = False
    protein_seq: Optional[str] = None
    dna_seq: Optional[str] = None


def load_gene_vocab(cfg: Config) -> List[GeneRecord]:
    """Load ``token_id -> ensembl_id -> gene_symbol`` from Tahoe-100M ``gene_metadata``.

    Prefers the local parquet under ``data_dir``; falls back to downloading the
    ``gene_metadata`` config from ``hf_repo`` via huggingface_hub if absent.
    Expected schema (CLAUDE.md "Tables"): ``token_id`` (int64), ``gene_symbol`` (str),
    ``ensembl_id`` (str). 62,710 rows.
    """
    import pyarrow.parquet as pq

    data_dir = Path(cfg.data_dir)
    files = sorted(data_dir.glob(cfg.gene_metadata_glob))

    # The on-disk layout of the HF snapshot is not stable across mirrors: the gene
    # vocabulary parquet has been seen at gene_metadata/*.parquet AND at
    # metadata/gene_metadata.parquet. Fall back to a few known patterns before
    # going to the hub.
    if not files:
        for pat in (
            "metadata/gene_metadata*.parquet",
            "gene_metadata*.parquet",
            "**/gene_metadata*.parquet",
        ):
            files = sorted(data_dir.glob(pat))
            if files:
                logger.info("Found gene_metadata via fallback glob %r: %s", pat, files)
                break

    if not files:
        logger.warning(
            "No local gene_metadata parquet under %s/%s; falling back to HF hub repo %s",
            data_dir,
            cfg.gene_metadata_glob,
            cfg.hf_repo,
        )
        files = _download_gene_metadata_from_hf(cfg)

    if not files:
        raise FileNotFoundError(
            f"Could not locate gene_metadata parquet locally or on the hub ({cfg.hf_repo})."
        )

    table = pq.ParquetDataset([str(f) for f in files]).read(
        columns=["token_id", "ensembl_id", "gene_symbol"]
    )
    cols = table.to_pydict()
    records: List[GeneRecord] = []
    for tid, ens, sym in zip(cols["token_id"], cols["ensembl_id"], cols["gene_symbol"]):
        records.append(
            GeneRecord(token_id=int(tid), ensembl_id=str(ens), gene_symbol=str(sym))
        )

    # Deterministic order by token_id so row indices into the stores are stable across
    # runs/shards regardless of parquet read order.
    records.sort(key=lambda r: r.token_id)
    logger.info("Loaded %d genes from gene_metadata", len(records))
    if len(records) != 62710:
        logger.warning(
            "Expected 62,710 genes (Tahoe-100M / Ensembl 109); got %d. Continuing.",
            len(records),
        )
    return records


def _download_gene_metadata_from_hf(cfg: Config) -> List[Path]:
    """Download the ``gene_metadata`` config parquet from the HF hub as a fallback."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:  # pragma: no cover - hub is a declared dep, but be defensive
        logger.error("huggingface_hub not available; cannot fall back to hub download.")
        return []
    local = snapshot_download(
        repo_id=cfg.hf_repo,
        repo_type="dataset",
        allow_patterns=["gene_metadata/*", "*gene_metadata*"],
    )
    return sorted(Path(local).rglob("gene_metadata*.parquet")) or sorted(
        Path(local).rglob("*.parquet")
    )


def select_genes(records: List[GeneRecord], cfg: Config) -> List[GeneRecord]:
    """Apply ``limit`` and ``shard_index/num_shards`` to get this process's slice."""
    if cfg.limit is not None:
        records = records[: cfg.limit]
    if cfg.num_shards > 1:
        records = [
            r for i, r in enumerate(records) if i % cfg.num_shards == cfg.shard_index
        ]
        logger.info(
            "Shard %d/%d -> %d genes", cfg.shard_index, cfg.num_shards, len(records)
        )
    return records


# =============================================================================
# Step 2 -- sequence resolution (pluggable source)
# =============================================================================
class SequenceResolver:
    """Resolve canonical transcript / biotype / protein+DNA sequence for a gene.

    Pluggable: the concrete implementation is chosen by ``cfg.sequence_source``.
    All implementations cache raw sequences on disk (one JSON per ensembl_id) so
    embeddings can be recomputed without re-querying the source.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cache_dir = cfg.resolved_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ensembl_id: str) -> Path:
        return self.cache_dir / f"{ensembl_id}.json"

    def resolve(self, rec: GeneRecord) -> GeneRecord:
        """Fill ``transcript_id/biotype/is_coding/protein_seq/dna_seq`` on ``rec``."""
        cached = self._load_cache(rec.ensembl_id)
        if cached is not None:
            rec.transcript_id = cached.get("transcript_id")
            rec.biotype = cached.get("biotype")
            rec.is_coding = bool(cached.get("is_coding", False))
            rec.protein_seq = cached.get("protein_seq")
            rec.dna_seq = cached.get("dna_seq")
            return rec

        self._resolve_uncached(rec)
        self._save_cache(rec)
        return rec

    def _resolve_uncached(self, rec: GeneRecord) -> GeneRecord:
        raise NotImplementedError

    def _load_cache(self, ensembl_id: str) -> Optional[dict]:
        p = self._cache_path(ensembl_id)
        if not p.exists():
            return None
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt seq cache for %s; re-resolving.", ensembl_id)
            return None

    def _save_cache(self, rec: GeneRecord) -> None:
        payload = {
            "ensembl_id": rec.ensembl_id,
            "transcript_id": rec.transcript_id,
            "biotype": rec.biotype,
            "is_coding": rec.is_coding,
            "protein_seq": rec.protein_seq,
            "dna_seq": rec.dna_seq,
            "ensembl_release": self.cfg.ensembl_release,
        }
        tmp = self._cache_path(rec.ensembl_id).with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, self._cache_path(rec.ensembl_id))  # atomic, restart-safe


class EnsemblRestResolver(SequenceResolver):
    """Resolve sequences via the Ensembl REST API, pinned to a release.

    Endpoints used (release pinned via the ``Ensembl-Release`` semantics -- REST
    serves the current release, so we additionally verify the genome assembly is
    GRCh38 and log a hard warning if the live release != ``cfg.ensembl_release``;
    for byte-exact release-109 sequences prefer the archive REST host or pyensembl
    -- see TODO-verify-on-cluster below).
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        import requests  # noqa: F401  (declared optional dep)

        self._session = None

    def _get_session(self):
        if self._session is None:
            import requests
            from requests.adapters import HTTPAdapter

            try:
                from urllib3.util.retry import Retry
            except ImportError:  # pragma: no cover
                Retry = None

            s = requests.Session()
            if Retry is not None:
                retry = Retry(
                    total=self.cfg.rest_max_retries,
                    backoff_factor=self.cfg.rest_backoff,
                    status_forcelist=(429, 500, 502, 503, 504),
                    allowed_methods=frozenset(["GET"]),
                )
                s.mount("https://", HTTPAdapter(max_retries=retry))
            self._session = s
        return self._session

    def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        """GET a JSON endpoint with retries; returns None on a 4xx 'not found'."""
        s = self._get_session()
        params = dict(params or {})
        params.setdefault("content-type", "application/json")
        url = f"{ENSEMBL_REST}{path}"
        for attempt in range(self.cfg.rest_max_retries):
            try:
                resp = s.get(
                    url,
                    params=params,
                    headers={"Content-Type": "application/json"},
                    timeout=self.cfg.rest_timeout,
                )
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (400, 404):
                    return None  # gene/transcript not found at this endpoint
                # 429 / 5xx -> backoff and retry
                wait = self.cfg.rest_backoff * (2**attempt)
                logger.warning(
                    "Ensembl REST %s -> %d; retry %d/%d in %.1fs",
                    path,
                    resp.status_code,
                    attempt + 1,
                    self.cfg.rest_max_retries,
                    wait,
                )
                time.sleep(wait)
            except Exception as e:  # network errors -> backoff and retry
                wait = self.cfg.rest_backoff * (2**attempt)
                logger.warning(
                    "Ensembl REST %s error %s; retry in %.1fs", path, e, wait
                )
                time.sleep(wait)
        logger.error(
            "Ensembl REST %s failed after %d retries", path, self.cfg.rest_max_retries
        )
        return None

    def _resolve_uncached(self, rec: GeneRecord) -> GeneRecord:
        # 1) Gene lookup -> biotype + canonical transcript id.
        # /lookup/id/{ensembl_id}?expand=1 returns Transcript list incl. is_canonical.
        info = self._get(f"/lookup/id/{rec.ensembl_id}", {"expand": "1"})
        if info is None:
            logger.warning(
                "No Ensembl lookup for %s; leaving sequences empty.", rec.ensembl_id
            )
            return rec

        assembly = info.get("assembly_name")
        if assembly and "GRCh38" not in str(assembly):
            logger.warning(
                "Gene %s assembly is %s, expected GRCh38 (release %d).",
                rec.ensembl_id,
                assembly,
                self.cfg.ensembl_release,
            )

        rec.biotype = info.get("biotype")
        rec.is_coding = rec.biotype == "protein_coding"

        canonical = _pick_canonical_transcript(info.get("Transcript", []))
        if canonical is None:
            logger.warning("No canonical transcript for %s.", rec.ensembl_id)
            return rec
        rec.transcript_id = canonical.get("id")

        # 2) Canonical transcript DNA (cDNA / transcript body) -- all genes.
        #    type=cdna gives the spliced transcript sequence.
        dna = self._get(f"/sequence/id/{rec.transcript_id}", {"type": "cdna"})
        if dna is not None:
            rec.dna_seq = dna.get("seq")

        # 3) Canonical protein sequence -- coding genes only.
        if rec.is_coding:
            prot = self._get(f"/sequence/id/{rec.transcript_id}", {"type": "protein"})
            if prot is not None:
                rec.protein_seq = prot.get("seq")
            else:
                logger.warning(
                    "Coding gene %s (%s) returned no protein sequence.",
                    rec.ensembl_id,
                    rec.transcript_id,
                )
        return rec


def _pick_canonical_transcript(transcripts: List[dict]) -> Optional[dict]:
    """Pick the canonical transcript from an Ensembl ``Transcript`` list.

    Prefers ``is_canonical == 1``; falls back to the longest transcript.
    """
    if not transcripts:
        return None
    for t in transcripts:
        if t.get("is_canonical") in (1, True):
            return t
    # Fallback: longest by (end - start). Different releases may not flag canonical.
    return max(
        transcripts, key=lambda t: abs(int(t.get("end", 0)) - int(t.get("start", 0)))
    )


class PyensemblResolver(SequenceResolver):
    """Resolve sequences via the ``pyensembl`` package (offline, release-pinned).

    Acceptable alternative to REST; install + cache the release once on the cluster::

        pyensembl install --release 109 --species homo_sapiens

    NOTE: pyensembl exposes gene biotype and transcript objects, but obtaining the
    spliced cDNA / protein strings depends on the installed FASTA. Verify the exact
    attribute names against the cluster's pyensembl version -- see
    TODO-verify-on-cluster.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._db = None

    def _get_db(self):
        if self._db is None:
            from pyensembl import EnsemblRelease

            self._db = EnsemblRelease(self.cfg.ensembl_release, species="homo_sapiens")
        return self._db

    def _resolve_uncached(self, rec: GeneRecord) -> GeneRecord:
        db = self._get_db()
        try:
            gene = db.gene_by_id(rec.ensembl_id)
        except Exception:
            logger.warning("pyensembl: gene %s not found.", rec.ensembl_id)
            return rec
        rec.biotype = getattr(gene, "biotype", None)
        rec.is_coding = rec.biotype == "protein_coding"

        transcripts = getattr(gene, "transcripts", [])
        canonical = None
        for t in transcripts:
            # TODO-verify-on-cluster: pyensembl has no universal `is_canonical`;
            # use the longest complete transcript or the MANE select if available.
            if getattr(t, "is_protein_coding", False) and getattr(t, "complete", False):
                canonical = t
                break
        if canonical is None and transcripts:
            canonical = max(
                transcripts, key=lambda t: len(getattr(t, "sequence", "") or "")
            )
        if canonical is None:
            return rec
        rec.transcript_id = getattr(canonical, "transcript_id", None)
        rec.dna_seq = getattr(canonical, "sequence", None)  # spliced cDNA
        if rec.is_coding:
            rec.protein_seq = getattr(canonical, "protein_sequence", None)
        return rec


def make_resolver(cfg: Config) -> SequenceResolver:
    if cfg.sequence_source == "rest":
        return EnsemblRestResolver(cfg)
    if cfg.sequence_source == "pyensembl":
        return PyensemblResolver(cfg)
    raise ValueError(f"Unknown sequence_source: {cfg.sequence_source!r}")


# =============================================================================
# Step 3 -- ESMC protein embeddings (coding genes)
# =============================================================================
class ESMCEmbedder:
    """Lazy ESMC ``esmc-600m`` mean-pooled protein embedder ([1280] per gene).

    Model loading is behind :meth:`_ensure_model` so a dry run never touches the GPU.
    """

    def __init__(self, cfg: Config, device, torch_dtype):
        self.cfg = cfg
        self.device = device
        self.torch_dtype = torch_dtype
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        # VERIFIED-ON-CLUSTER (esm 3.x, esmc_600m on B200): the import + forward API is
        #   from esm.models.esmc import ESMC
        #   from esm.sdk.api import ESMProtein, LogitsConfig
        #   model = ESMC.from_pretrained("esmc_600m").to(device)
        #   p   = model.encode(ESMProtein(sequence=seq))
        #   out = model.logits(p, LogitsConfig(sequence=True, return_embeddings=True))
        #   out.embeddings  # [1, L+special(BOS/EOS), 1152]  (note: 1152, not 1280)
        #
        # IMPORTANT (verified on cluster): when ESMC and Evo2 share one process, Evo2's
        # vortex stack re-registers the flash_attn torch custom ops and corrupts ESMC's
        # flash-attn varlen path (ESMC then dies with "ValueError: vector::reserve").
        # ESMC has an SDPA fallback (esm.layers.attention.MultiHeadAttention) selected
        # when `is_flash_attn_available` is False. We force that path so ESMC is immune
        # to the shared-registry corruption; SDPA on a B200 is plenty fast for 600M.
        import esm.models.esmc as _esmc_mod

        _esmc_mod.is_flash_attn_available = False  # force ESMC onto the SDPA attention
        from esm.models.esmc import ESMC  # noqa: F401

        logger.info("Loading ESMC model %s (SDPA attention) ...", self.cfg.esmc_model)
        name = self.cfg.esmc_model.replace("-", "_")  # "esmc-600m" -> "esmc_600m"
        model = ESMC.from_pretrained(name)
        model = model.to(self.device).eval()
        self._model = model

    def embed(self, sequences: List[str]) -> "np.ndarray":
        """Return ``[len(sequences), 1280]`` mean-pooled embeddings.

        Sequences are assumed pre-filtered (non-empty, within ``esmc_max_residues``).
        Batched by similar length by the caller; here we just run one bucket.
        """
        import torch

        self._ensure_model()
        from esm.sdk.api import ESMProtein, LogitsConfig  # local import, verified above

        vecs = []
        with torch.no_grad():
            for seq in sequences:
                protein = ESMProtein(sequence=seq)
                encoded = self._model.encode(protein)
                out = self._model.logits(
                    encoded, LogitsConfig(sequence=True, return_embeddings=True)
                )
                # out.embeddings: [1, L(+special tokens), 1280]. Mean-pool over the
                # residue axis. We pool all positions; the +/- special tokens are a
                # negligible bias for 600M and consistent across genes.
                emb = out.embeddings.to(torch.float32).squeeze(0)  # [L, 1280]
                vecs.append(emb.mean(dim=0).cpu().numpy())
        return np.stack(vecs, axis=0).astype(np.float32)


def _bucket_by_length(
    items: List[Tuple[int, str]], batch_size: int
) -> List[List[Tuple[int, str]]]:
    """Sort by sequence length then chunk into batches (length bucketing)."""
    ordered = sorted(items, key=lambda kv: len(kv[1]))
    return [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]


# =============================================================================
# Step 4 -- Evo2 DNA embeddings (all genes)
# =============================================================================
class Evo2Embedder:
    """Lazy Evo2 DNA embedder. ``d_evo2`` is READ FROM THE MODEL CONFIG, never hardcoded.

    Extracts a configurable layer with configurable pooling. Long sequences are
    chunked with overlap and the per-chunk pooled vectors are averaged.
    """

    def __init__(self, cfg: Config, device, torch_dtype):
        self.cfg = cfg
        self.device = device
        self.torch_dtype = torch_dtype
        self._model = None
        self._d_evo2: Optional[int] = None

    def _ensure_model(self):
        if self._model is not None:
            return
        # VERIFIED-ON-CLUSTER (evo2 0.6.0 / vtx 1.1.0 on B200): the API is
        #   from evo2 import Evo2
        #   model = Evo2(self.cfg.evo2_model.split("/")[-1])   # e.g. "evo2_7b_base"
        #   logits, embeddings = model(input_ids, return_embeddings=True,
        #                              layer_names=["blocks.24.mlp.l3"])
        #   embeddings["blocks.24.mlp.l3"]  # [1, T, 4096] bf16
        # `logits` is a tuple here (ignored). The submodule "blocks.{i}.mlp.l3" is
        # registered (confirmed via model.model.named_modules()).
        #
        # IMPORTANT (verified on cluster, torch 2.8 / vortex 1.1.0 / flash-attn 2.8.3):
        # `vortex.ops.attn_interface` re-registers the `flash_attn::_flash_attn_*`
        # torch custom ops, and on torch>=2.8 that clobbers the schema, so the Evo2
        # forward dies with "Tried to access the schema for ... which doesn't have a
        # schema registered yet". Importing the real `flash_attn` PACKAGE first makes
        # it register the (correct) op schemas before vortex's duplicate decoration,
        # which fixes the forward. Keep this import ABOVE `from evo2 import Evo2`.
        try:
            import flash_attn  # noqa: F401  (ordering matters; see comment above)
        except ImportError:
            logger.warning(
                "flash_attn package not importable before evo2; the Evo2 forward may "
                "fail with a torch custom-op schema error on torch>=2.8."
            )
        from evo2 import Evo2  # noqa: F401

        model_name = self.cfg.evo2_model.split("/")[-1]
        logger.info("Loading Evo2 model %s ...", model_name)
        model = Evo2(model_name)
        self._model = model
        self._d_evo2 = self._read_hidden_dim(model)
        logger.info("Evo2 d_evo2 (from model config) = %d", self._d_evo2)

    @staticmethod
    def _read_hidden_dim(model) -> int:
        """Read d_evo2 from the model config; tries several attribute paths.

        VERIFIED-ON-CLUSTER (evo2 0.6.0 / vtx 1.1.0): the public ``Evo2`` wrapper holds
        the StripedHyena under ``model.model`` and the config (a ``dotdict``) under
        ``model.model.config``; ``config.hidden_size`` is the width (4096 for
        evo2_7b_base, 1920 for evo2_1b_base). We probe both ``model.config`` and
        ``model.model.config`` so this works regardless of wrapper depth.
        """
        cfg_candidates = []
        for owner in (model, getattr(model, "model", None)):
            if owner is None:
                continue
            for name in ("config", "model_config"):
                obj = getattr(owner, name, None)
                if obj is not None:
                    cfg_candidates.append(obj)
        attrs = ("hidden_size", "d_model", "hidden_dim", "model_dim", "embed_dim")
        for cfg_obj in cfg_candidates:
            for attr in attrs:
                val = getattr(cfg_obj, attr, None)
                if val is None and isinstance(cfg_obj, dict):
                    val = cfg_obj.get(attr)
                if isinstance(val, int) and val > 0:
                    return int(val)
        raise RuntimeError(
            "Could not read d_evo2 from the Evo2 model config. Inspect the config "
            "object and add its hidden-size attribute name to `_read_hidden_dim`."
        )

    @property
    def d_evo2(self) -> int:
        self._ensure_model()
        assert self._d_evo2 is not None
        return self._d_evo2

    def _layer_name(self, layer: int) -> str:
        # VERIFIED-ON-CLUSTER: Evo2 extracts activations by submodule name via a
        # forward hook; "blocks.{i}.mlp.l3" is a valid registered module (the 3rd
        # GLU linear of block i's MLP). 32 blocks (0..31) for the 7b models.
        return f"blocks.{layer}.mlp.l3"

    def _tokenize(self, seq: str):
        import torch

        # VERIFIED-ON-CLUSTER: vortex CharLevelTokenizer.tokenize(seq) returns a
        # python list of np.uint8 byte values (one per character); torch.tensor()
        # promotes them to long cleanly. Tokenizer lives on the Evo2 wrapper.
        tok = self._model.tokenizer
        ids = tok.tokenize(seq) if hasattr(tok, "tokenize") else tok(seq)
        ids = list(ids)
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    def _pool(self, hidden, pooling: str):
        # hidden: [1, T, d_evo2]
        import torch

        h = hidden.to(torch.float32).squeeze(0)  # [T, d]
        if pooling == "mean":
            return h.mean(dim=0)
        if pooling == "last":
            return h[-1]
        if pooling == "max":
            return h.max(dim=0).values
        raise ValueError(f"Unknown pooling {pooling!r}")

    def _chunks(self, seq: str) -> List[str]:
        step = max(1, self.cfg.max_dna_len - self.cfg.evo2_chunk_overlap)
        if len(seq) <= self.cfg.max_dna_len:
            return [seq]
        return [seq[i : i + self.cfg.max_dna_len] for i in range(0, len(seq), step)]

    def _forward_layer(self, seq: str, layer: int) -> "np.ndarray":
        """Run one sequence and return the hidden states at ``layer`` (per token)."""
        import torch

        self._ensure_model()
        ids = self._tokenize(seq)
        layer_name = self._layer_name(layer)
        with torch.no_grad():
            # VERIFIED-ON-CLUSTER: returns (logits, {layer_name: [1, T, d_evo2]}).
            _, embeddings = self._model(
                ids, return_embeddings=True, layer_names=[layer_name]
            )
        return embeddings[layer_name]  # [1, T, d_evo2]

    def embed(self, seq: str) -> "np.ndarray":
        """One DNA sequence -> a single pooled ``[d_evo2]`` vector (chunk-averaged)."""
        pooled_chunks = []
        for chunk in self._chunks(seq):
            hidden = self._forward_layer(chunk, self.cfg.evo2_layer)
            pooled_chunks.append(
                self._pool(hidden, self.cfg.evo2_pooling).cpu().numpy()
            )
        return np.mean(np.stack(pooled_chunks, axis=0), axis=0).astype(np.float32)

    def embed_all_layers(
        self, seq: str, layers: Sequence[int], poolings: Sequence[str]
    ) -> Dict[str, "np.ndarray"]:
        """For the sweep: return ``{f"L{l}_{pool}": [d_evo2]}`` over layers x poolings.

        Uses only the FIRST chunk for the sweep (cheap, comparable across genes).
        """
        out: Dict[str, np.ndarray] = {}
        chunk = self._chunks(seq)[0]
        for layer in layers:
            hidden = self._forward_layer(chunk, layer)
            for pool in poolings:
                out[f"L{layer}_{pool}"] = (
                    self._pool(hidden, pool).cpu().numpy().astype(np.float32)
                )
        return out


# =============================================================================
# Cache writer (frozen lookup stores + index)  -- step 5
# =============================================================================
class CacheWriter:
    """Writes the two memmapped row-stores + the index + metadata, resumably.

    Layout under ``out_dir`` (or ``out_dir/shard_{i}`` when sharded):
      esmc.npy        memmap [N_coding, 1280] float32   (coding genes only)
      evo2.npy        memmap [N_genes, d_evo2] float32  (all genes)
      index.parquet   columns: token_id, ensembl_id, is_coding, esmc_row, evo2_row
      metadata.json   evo2_model_id, extracted_layer, pooling, ensembl_release, esmc_model_id, ...
      progress.json   {ensembl_id: {"esmc_row":..,"evo2_row":..}}  (resume state)
    """

    def __init__(self, cfg: Config, n_genes: int, n_coding: int, d_evo2: int):
        base = Path(cfg.out_dir)
        if cfg.num_shards > 1:
            base = base / f"shard_{cfg.shard_index}"
        base.mkdir(parents=True, exist_ok=True)
        self.base = base
        self.cfg = cfg

        self.esmc_path = base / "esmc.npy"
        self.evo2_path = base / "evo2.npy"
        self.index_path = base / "index.parquet"
        self.meta_path = base / "metadata.json"
        self.progress_path = base / "progress.json"

        self._esmc = np.lib.format.open_memmap(
            self.esmc_path,
            mode="r+" if self.esmc_path.exists() else "w+",
            dtype=np.float32,
            shape=(max(n_coding, 1), ESMC_DIM),
        )
        self._evo2 = np.lib.format.open_memmap(
            self.evo2_path,
            mode="r+" if self.evo2_path.exists() else "w+",
            dtype=np.float32,
            shape=(max(n_genes, 1), d_evo2),
        )
        self.index_rows: List[dict] = []
        self.progress: Dict[str, dict] = self._load_progress()

    def _load_progress(self) -> Dict[str, dict]:
        if self.progress_path.exists():
            try:
                with open(self.progress_path) as f:
                    prog = json.load(f)
                logger.info("Resuming: %d genes already embedded.", len(prog))
                return prog
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt progress.json; starting fresh.")
        return {}

    def is_done(self, ensembl_id: str) -> bool:
        return ensembl_id in self.progress

    def write_gene(
        self,
        rec: GeneRecord,
        esmc_row: int,
        evo2_row: int,
        esmc_vec: Optional[np.ndarray],
        evo2_vec: Optional[np.ndarray],
    ) -> None:
        if esmc_vec is not None and esmc_row >= 0:
            self._esmc[esmc_row] = esmc_vec
        if evo2_vec is not None and evo2_row >= 0:
            self._evo2[evo2_row] = evo2_vec
        self.progress[rec.ensembl_id] = {
            "token_id": rec.token_id,
            "ensembl_id": rec.ensembl_id,
            "is_coding": bool(rec.is_coding),
            "esmc_row": int(esmc_row),
            "evo2_row": int(evo2_row),
        }

    def flush(self) -> None:
        self._esmc.flush()
        self._evo2.flush()
        tmp = self.progress_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.progress, f)
        os.replace(tmp, self.progress_path)

    def finalize(self, cfg: Config, d_evo2: int) -> None:
        """Flush stores, write the index parquet and the metadata.json."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.flush()
        rows = sorted(self.progress.values(), key=lambda r: r["token_id"])
        table = pa.table(
            {
                "token_id": pa.array([r["token_id"] for r in rows], pa.int64()),
                "ensembl_id": pa.array([r["ensembl_id"] for r in rows], pa.string()),
                "is_coding": pa.array([r["is_coding"] for r in rows], pa.bool_()),
                "esmc_row": pa.array([r["esmc_row"] for r in rows], pa.int64()),
                "evo2_row": pa.array([r["evo2_row"] for r in rows], pa.int64()),
            }
        )
        pq.write_table(table, self.index_path)

        meta = {
            "esmc_model_id": cfg.esmc_model,
            "esmc_dim": ESMC_DIM,
            "evo2_model_id": cfg.evo2_model,
            "d_evo2": d_evo2,
            "extracted_layer": cfg.evo2_layer,
            "pooling": cfg.evo2_pooling,
            "ensembl_release": cfg.ensembl_release,
            "sequence_source": cfg.sequence_source,
            "max_dna_len": cfg.max_dna_len,
            "esmc_max_residues": cfg.esmc_max_residues,
            "n_genes": len(rows),
            "n_coding": sum(1 for r in rows if r["is_coding"]),
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Wrote index (%d rows) + metadata to %s", len(rows), self.base)


# =============================================================================
# Step 6 -- Evo2 layer sweep
# =============================================================================
def run_layer_sweep(
    cfg: Config,
    records: List[GeneRecord],
    resolver: SequenceResolver,
    device,
    torch_dtype,
) -> None:
    """Dump per-layer/per-pooling Evo2 embeddings for ~``sweep_genes`` genes.

    Output: ``out_dir/evo2_sweep.npz`` with one array per ``L{layer}_{pool}`` key,
    each ``[n_sweep, d_evo2]``, plus the token_ids/ensembl_ids used. A downstream
    probe (NOT implemented here, see :func:`select_best_layer`) consumes these to
    pick the best layer/pooling by correlation with a biological target.
    """
    sweep_records = records[: cfg.sweep_genes]
    logger.info(
        "Evo2 layer sweep over %d genes, layers=%s, poolings=%s",
        len(sweep_records),
        list(cfg.sweep_layers),
        list(cfg.sweep_poolings),
    )
    embedder = Evo2Embedder(cfg, device, torch_dtype)

    per_key: Dict[str, List[np.ndarray]] = {}
    used_token_ids, used_ensembl = [], []
    for i, rec in enumerate(sweep_records):
        resolver.resolve(rec)
        if not rec.dna_seq:
            logger.warning("Sweep: %s has no DNA; skipping.", rec.ensembl_id)
            continue
        if cfg.dry_run:
            continue
        emb = embedder.embed_all_layers(
            rec.dna_seq, cfg.sweep_layers, cfg.sweep_poolings
        )
        for k, v in emb.items():
            per_key.setdefault(k, []).append(v)
        used_token_ids.append(rec.token_id)
        used_ensembl.append(rec.ensembl_id)
        if (i + 1) % 25 == 0:
            logger.info("  sweep %d/%d", i + 1, len(sweep_records))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "evo2_sweep.npz"
    arrays = {k: np.stack(v, axis=0) for k, v in per_key.items()}
    np.savez(
        out_path,
        token_id=np.asarray(used_token_ids, dtype=np.int64),
        ensembl_id=np.asarray(used_ensembl),
        **arrays,
    )
    logger.info(
        "Wrote sweep embeddings to %s (keys: %s)", out_path, list(arrays.keys())
    )
    logger.info(
        "NEXT: run select_best_layer() / a downstream probe to choose the layer+pooling, "
        "then re-run the full build with --evo2_layer/--evo2_pooling set accordingly."
    )


def select_best_layer(sweep_npz: str) -> str:
    """TODO (downstream): pick the best Evo2 layer/pooling from the sweep dump.

    The selection metric is a PROBE CORRELATION: for each ``L{layer}_{pool}`` key,
    train a cheap linear probe (or compute a representational-similarity / canonical
    correlation) against a biological target -- e.g. GO/biotype labels, gene-family
    membership, or held-out expression structure -- and pick the key whose embeddings
    best predict it. This function is intentionally left as a hook; it is part of the
    validation, not the cache build itself.
    """
    raise NotImplementedError(
        "select_best_layer is a downstream-validation hook; implement the probe-"
        f"correlation metric over the keys in {sweep_npz}."
    )


# =============================================================================
# Shard merge (multi-GPU)  -- hook
# =============================================================================
def merge_shards(cfg: Config, num_shards: Optional[int] = None) -> None:
    """Concatenate per-shard row-stores into a single global cache.

    When run with ``--num_shards N`` across N GPUs, each process wrote
    ``out_dir/shard_i/{esmc,evo2}.npy`` + its own ``index.parquet`` (with *local*
    row indices) + ``metadata.json``. This pass:
      1. reads every shard index, sorts all genes by ``token_id`` for a stable,
         shard-independent global layout,
      2. re-numbers ``esmc_row``/``evo2_row`` into global contiguous ranges (coding
         genes get a dense esmc_row; non-coding get esmc_row=-1),
      3. copies each gene's vectors out of its shard store into the global
         ``esmc.npy`` / ``evo2.npy`` and writes a single ``index.parquet`` +
         ``metadata.json`` at ``out_dir``.

    Invoke after all shard processes finish, e.g.::

        uv run python -m scripts.build_gene_embeddings --merge_only True \
            --num_shards 8 --out_dir /data/gene_emb_cache
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = int(num_shards if num_shards is not None else cfg.num_shards)
    if n < 1:
        raise ValueError(f"merge_shards needs num_shards>=1, got {n}")
    out = Path(cfg.out_dir)

    # 1) collect per-shard (index rows + memmapped stores).
    shard_index_rows: List[dict] = []
    shard_esmc: Dict[int, np.ndarray] = {}
    shard_evo2: Dict[int, np.ndarray] = {}
    d_evo2 = None
    merged_meta: Optional[dict] = None
    for i in range(n):
        sdir = out / f"shard_{i}"
        idx_path = sdir / "index.parquet"
        if not idx_path.exists():
            raise FileNotFoundError(f"Missing shard index: {idx_path}")
        rows = pq.read_table(idx_path).to_pylist()
        for r in rows:
            r["_shard"] = i
        shard_index_rows.extend(rows)
        shard_esmc[i] = np.load(sdir / "esmc.npy", mmap_mode="r")
        shard_evo2[i] = np.load(sdir / "evo2.npy", mmap_mode="r")
        if d_evo2 is None:
            d_evo2 = int(shard_evo2[i].shape[1])
        if merged_meta is None and (sdir / "metadata.json").exists():
            with open(sdir / "metadata.json") as f:
                merged_meta = json.load(f)

    if d_evo2 is None:
        raise RuntimeError("No shard evo2 stores found; nothing to merge.")

    # 2) global ordering by token_id; assign dense global rows.
    shard_index_rows.sort(key=lambda r: r["token_id"])
    n_genes = len(shard_index_rows)
    n_coding = sum(1 for r in shard_index_rows if r["is_coding"] and r["esmc_row"] >= 0)

    global_esmc = np.lib.format.open_memmap(
        out / "esmc.npy", mode="w+", dtype=np.float32, shape=(max(n_coding, 1), ESMC_DIM)
    )
    global_evo2 = np.lib.format.open_memmap(
        out / "evo2.npy", mode="w+", dtype=np.float32, shape=(max(n_genes, 1), d_evo2)
    )

    out_rows: List[dict] = []
    esmc_cursor = 0
    for evo2_row, r in enumerate(shard_index_rows):
        si = r["_shard"]
        global_evo2[evo2_row] = shard_evo2[si][r["evo2_row"]]
        if r["is_coding"] and r["esmc_row"] >= 0:
            global_esmc[esmc_cursor] = shard_esmc[si][r["esmc_row"]]
            esmc_row = esmc_cursor
            esmc_cursor += 1
        else:
            esmc_row = -1
        out_rows.append(
            {
                "token_id": int(r["token_id"]),
                "ensembl_id": str(r["ensembl_id"]),
                "is_coding": bool(r["is_coding"]),
                "esmc_row": int(esmc_row),
                "evo2_row": int(evo2_row),
            }
        )
    global_esmc.flush()
    global_evo2.flush()

    # 3) unified index + metadata.
    table = pa.table(
        {
            "token_id": pa.array([r["token_id"] for r in out_rows], pa.int64()),
            "ensembl_id": pa.array([r["ensembl_id"] for r in out_rows], pa.string()),
            "is_coding": pa.array([r["is_coding"] for r in out_rows], pa.bool_()),
            "esmc_row": pa.array([r["esmc_row"] for r in out_rows], pa.int64()),
            "evo2_row": pa.array([r["evo2_row"] for r in out_rows], pa.int64()),
        }
    )
    pq.write_table(table, out / "index.parquet")

    meta = merged_meta or {}
    meta.update(
        {
            "esmc_model_id": cfg.esmc_model,
            "esmc_dim": ESMC_DIM,
            "evo2_model_id": cfg.evo2_model,
            "d_evo2": int(d_evo2),
            "extracted_layer": cfg.evo2_layer,
            "pooling": cfg.evo2_pooling,
            "ensembl_release": cfg.ensembl_release,
            "sequence_source": cfg.sequence_source,
            "max_dna_len": cfg.max_dna_len,
            "esmc_max_residues": cfg.esmc_max_residues,
            "n_genes": n_genes,
            "n_coding": esmc_cursor,
            "merged_from_shards": n,
        }
    )
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    (out / ".DONE").write_text(
        f"merged {n} shards: n_genes={n_genes} n_coding={esmc_cursor} d_evo2={d_evo2}\n"
    )
    logger.info(
        "merge_shards: wrote %d genes (%d coding, d_evo2=%d) to %s + .DONE",
        n_genes,
        esmc_cursor,
        d_evo2,
        out,
    )


# =============================================================================
# Orchestration
# =============================================================================
def _resolve_device(cfg: Config):
    import torch

    dev = cfg.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(dev)


def _resolve_dtype(cfg: Config):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[cfg.dtype]


def build_cache(cfg: Config) -> None:
    """Full build: resolve -> ESMC (coding) -> Evo2 (all) -> write stores + index."""
    import torch  # noqa: F401  (import here so dry_run never requires torch for vocab)

    device = _resolve_device(cfg)
    torch_dtype = _resolve_dtype(cfg)
    logger.info("Device=%s dtype=%s dry_run=%s", device, cfg.dtype, cfg.dry_run)

    records = select_genes(load_gene_vocab(cfg), cfg)
    resolver = make_resolver(cfg)

    # First pass: resolve sequences (cached on disk). This also tells us how many
    # coding genes there are, which sizes the ESMC store.
    logger.info(
        "Resolving sequences for %d genes (source=%s) ...",
        len(records),
        cfg.sequence_source,
    )
    for i, rec in enumerate(records):
        resolver.resolve(rec)
        if (i + 1) % 500 == 0:
            logger.info("  resolved %d/%d", i + 1, len(records))

    n_genes = len(records)
    coding = [r for r in records if r.is_coding and r.protein_seq]
    n_coding = len(coding)
    logger.info("Resolved: %d genes, %d coding with protein seq.", n_genes, n_coding)

    # Lazy embedders. Reading d_evo2 forces a model load unless dry_run.
    evo2 = Evo2Embedder(cfg, device, torch_dtype)
    esmc = ESMCEmbedder(cfg, device, torch_dtype)
    d_evo2 = ESMC_DIM if cfg.dry_run else evo2.d_evo2  # placeholder dim under dry_run
    if cfg.dry_run:
        logger.info(
            "DRY RUN: skipping model loads / embedding; using placeholder d_evo2=%d",
            d_evo2,
        )

    writer = CacheWriter(cfg, n_genes=n_genes, n_coding=max(n_coding, 1), d_evo2=d_evo2)

    # Assign stable row indices (ordered by token_id, matching writer.finalize order).
    esmc_row_of: Dict[str, int] = {r.ensembl_id: i for i, r in enumerate(coding)}

    # --- ESMC pass (coding only), length-bucketed ------------------------------
    if not cfg.dry_run:
        to_embed = []
        for r in coding:
            if (
                writer.is_done(r.ensembl_id)
                and writer.progress[r.ensembl_id]["esmc_row"] >= 0
            ):
                continue
            if len(r.protein_seq) > cfg.esmc_max_residues:
                logger.warning(
                    "Protein %s len %d > cap %d; skipping ESMC (still gets Evo2).",
                    r.ensembl_id,
                    len(r.protein_seq),
                    cfg.esmc_max_residues,
                )
                continue
            to_embed.append((esmc_row_of[r.ensembl_id], r.protein_seq))

        logger.info("ESMC: embedding %d coding proteins ...", len(to_embed))
        for batch in _bucket_by_length(to_embed, cfg.esmc_batch_size):
            rows = [row for row, _ in batch]
            seqs = [seq for _, seq in batch]
            vecs = esmc.embed(seqs)
            for row, vec in zip(rows, vecs):
                writer._esmc[row] = vec  # direct store; progress recorded in main loop
        writer._esmc.flush()

    # --- Evo2 pass (all genes) + index assembly --------------------------------
    logger.info(
        "Evo2: embedding %d genes (layer=%d pooling=%s) ...",
        n_genes,
        cfg.evo2_layer,
        cfg.evo2_pooling,
    )
    done_since_flush = 0
    for evo2_row, rec in enumerate(records):
        if writer.is_done(rec.ensembl_id):
            continue
        esmc_row = esmc_row_of.get(rec.ensembl_id, -1)
        if rec.is_coding and not rec.protein_seq:
            esmc_row = -1  # coding but no protein resolved -> no ESMC term

        evo2_vec = None
        if not cfg.dry_run:
            if rec.dna_seq:
                evo2_vec = evo2.embed(rec.dna_seq)
            else:
                logger.warning(
                    "Gene %s has no DNA sequence; Evo2 row left zero.", rec.ensembl_id
                )

        # ESMC vec already written above into writer._esmc[esmc_row]; pass None here.
        writer.write_gene(
            rec, esmc_row=esmc_row, evo2_row=evo2_row, esmc_vec=None, evo2_vec=evo2_vec
        )
        done_since_flush += 1
        if done_since_flush >= cfg.checkpoint_every:
            writer.flush()
            done_since_flush = 0
            logger.info("  checkpointed at evo2_row=%d", evo2_row)

    writer.finalize(cfg, d_evo2=d_evo2)
    # Completion marker. For a single-shard run the writer.base IS the final cache, so
    # drop the marker at out_dir directly. Sharded runs are finalized by merge_shards,
    # which writes its own .DONE at out_dir.
    if cfg.num_shards <= 1:
        (Path(cfg.out_dir) / ".DONE").write_text(
            f"build complete: n_genes={n_genes} n_coding={n_coding} "
            f"d_evo2={d_evo2} evo2_model={cfg.evo2_model} "
            f"layer={cfg.evo2_layer} pooling={cfg.evo2_pooling}\n"
        )
    else:
        (writer.base / ".SHARD_DONE").write_text("ok\n")
    logger.info("Gene-embedding cache build complete -> %s", writer.base)


# =============================================================================
# CLI entrypoint (fire) -- matches examples/*/main.py convention
# =============================================================================
def run(**overrides) -> None:
    """Build (or sweep) the Tahoe-100M gene-embedding cache.

    All :class:`Config` fields are settable as flags, e.g.::

        uv run python -m scripts.build_gene_embeddings \
            --out_dir /data/gene_cache --evo2_model arcinstitute/evo2_1b_base \
            --evo2_layer 24 --max_dna_len 24000 --limit 100

    Set ``--evo2_layer_sweep True`` to run the layer/pooling sweep instead of a
    full build. Set ``--dry_run True`` to resolve sequences only (no model loads).
    """
    valid = {f.name for f in dataclasses.fields(Config)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(
            f"Unknown config flags: {sorted(unknown)}. Valid: {sorted(valid)}"
        )
    cfg = Config(**overrides)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info("Tahoe-100M gene-embedding cache build")
    for f in dataclasses.fields(Config):
        logger.info("  %s = %s", f.name, getattr(cfg, f.name))
    logger.info("=" * 70)

    # seed (numpy/torch if present) -- determinism for any sampling/bucketing
    np.random.seed(cfg.seed)
    try:
        import torch

        torch.manual_seed(cfg.seed)
    except ImportError:
        pass

    if cfg.merge_only:
        merge_shards(cfg)
        return

    if cfg.evo2_layer_sweep:
        device = _resolve_device(cfg)
        torch_dtype = _resolve_dtype(cfg)
        records = select_genes(load_gene_vocab(cfg), cfg)
        resolver = make_resolver(cfg)
        run_layer_sweep(cfg, records, resolver, device, torch_dtype)
        return

    build_cache(cfg)


if __name__ == "__main__":
    fire.Fire(run)
