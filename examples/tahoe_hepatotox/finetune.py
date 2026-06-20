"""Liver-specialize the Subliminal 1.4 small encoder (CLAUDE.md phase 2).

Warm-starts the pretrained sub14 small encoder and continues LeJEPA training on
**liver cell lines only** (Tahoe ``cell_line_id`` whose ``Organ == "Liver"``,
e.g. HepG2 / Huh-7). Everything else — the sub14 recipe (sigmoid attention,
per-cell quantile thermometer counts, pairwise-cosine JEPA + SIGReg,
Muon+AdamW, no scheduler), the streaming dataloader, the eval/probe/t-SNE
harness, DDP, wandb — is **reused verbatim** from ``sub14_main`` by importing its
``train`` entrypoint.

Liver filtering is the only addition: a thin ``LiverTahoeIterableDataset`` subclass
of ``TahoeIterableDataset`` that drops non-hepatic cells while streaming (the cell's
``cell_line_id`` is looked up in the prebuilt ``cell_line_to_organ`` map from
``maps_path``). We inject it by binding it onto ``sub14_main`` before calling
``sub14_main.train`` — no copy of the training loop, no edit to the shared dataset
module. The liver set is small (a handful of lines), so a single GB200 is plenty.

Usage (1 GPU on Dalia):
    /lustre/work/vivatech-unaite/ljung/venv-arm/bin/python -m \
        examples.tahoe_hepatotox.finetune run \
        --config examples/tahoe_hepatotox/cfgs/finetune.yaml
"""
from __future__ import annotations

import os
import time

import torch

from eb_jepa.datasets.tahoe.dataset import TahoeIterableDataset
from eb_jepa.logging import get_logger
from eb_jepa.training_utils import load_config

import examples.tahoe_jepa.sub14_main as sub14_main

logger = get_logger(__name__)


class LiverTahoeIterableDataset(TahoeIterableDataset):
    """Streaming Tahoe reader that yields only hepatic (``Organ == "Liver"``) cells.

    Identical to :class:`TahoeIterableDataset` except ``_read_shard`` skips any cell
    whose ``cell_line_id`` is not in ``liver_cell_lines`` (built once from the
    ``cell_line_to_organ`` map). Filtering at the stream level keeps the change
    surgical and the LeJEPA collate / view logic untouched.
    """

    def __init__(self, *args, liver_cell_lines: set | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if liver_cell_lines is None:
            liver_cell_lines = {
                cvcl for cvcl, organ in (self.cell_line_to_organ or {}).items()
                if organ == "Liver"
            }
        self.liver_cell_lines = set(liver_cell_lines)
        if not self.liver_cell_lines:
            raise ValueError(
                "No liver cell lines resolved — check maps_path / cell_line_to_organ "
                "(need Organ=='Liver' entries)."
            )

    def _read_shard(self, f):
        for item in super()._read_shard(f):
            if item.get("cell_line_id") in self.liver_cell_lines:
                yield item


def _make_liver_dataset_factory(liver_cell_lines: set):
    """Return a TahoeIterableDataset-compatible constructor bound to the liver set."""

    def factory(*args, **kwargs):
        return LiverTahoeIterableDataset(*args, liver_cell_lines=liver_cell_lines, **kwargs)

    return factory


def run(config: str = "examples/tahoe_hepatotox/cfgs/finetune.yaml", **overrides):
    cfg = load_config(config, cli_overrides=overrides or None)
    os.makedirs(cfg.meta.run_dir, exist_ok=True)

    # Resolve the hepatic cell-line set from the prebuilt map (same maps.pt the
    # sub14 loader already reads for organ labels).
    liver = set()
    maps_path = cfg.data.get("maps_path")
    if maps_path and os.path.exists(maps_path):
        cl2organ = torch.load(maps_path).get("cell_line_to_organ", {})
        liver = {cvcl for cvcl, organ in cl2organ.items() if organ == "Liver"}
    if not liver:
        raise FileNotFoundError(
            f"Could not build the liver cell-line set from maps_path={maps_path!r}. "
            "Build maps.pt first (eb_jepa.datasets.tahoe.preprocess) — it must contain "
            "cell_line_to_organ with Organ=='Liver' entries."
        )
    logger.info("liver finetune: %d hepatic cell lines -> %s", len(liver), sorted(liver))

    # Inject the liver-filtering dataset into sub14_main.build_loader without touching
    # the shared module: build_loader references the name TahoeIterableDataset in the
    # sub14_main namespace, so rebinding it here swaps in the filtered reader.
    sub14_main.TahoeIterableDataset = _make_liver_dataset_factory(liver)

    t0 = time.time()
    sub14_main.train(cfg)
    logger.info("Liver finetune done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run})
