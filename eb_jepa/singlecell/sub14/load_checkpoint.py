"""Warm-start a Subliminal 1.4 port from the trained cell-lejepa checkpoint.

The trained checkpoint (e.g. ``subliminal_1_4_F_v4.pt`` — the 4-view round-1
winner) is a plain ``state_dict`` from the original ``Subliminal1_4`` model.
Because this port keeps the *same module names*, the trained weights drop in
key-for-key wherever shapes match. They match only when the port is built at
the trained size (``d_model=512, n_layers=6, n_heads=8, num_bins=16,
proj_dim=128``).

Two groups are deliberately NOT loaded:
- ``probe_heads.*`` — validation-only heads, absent from this port.
- ``gene_embedding.{dna,protein}.{features,in_norm,projection}`` — the input
  adapter for 1.4's DNABERT/ESM features. This port uses *our* Evo2/ESMC
  features (different dims), so the adapter is re-initialised and re-learned.

Everything else — the full sigmoid-attention transformer body, the JEPA
projector, the thermometer count table, the [CELL] token, and
``gene_embedding.out_norm`` — is reused.

The loader is shape-matched: it copies only tensors whose name AND shape
agree with the target model, and returns a structured report so the caller
can log exactly what was reused vs re-initialised.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from eb_jepa.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WarmStartReport:
    loaded: list[str] = field(default_factory=list)         # name in both, shape matched
    skipped_shape: list[str] = field(default_factory=list)  # in both, shape mismatch
    skipped_ckpt_only: list[str] = field(default_factory=list)  # in ckpt, not in model
    missing_in_ckpt: list[str] = field(default_factory=list)    # in model, not in ckpt

    def summary(self) -> str:
        return (
            f"warm-start: {len(self.loaded)} tensors reused, "
            f"{len(self.skipped_shape)} shape-mismatch, "
            f"{len(self.missing_in_ckpt)} fresh-init, "
            f"{len(self.skipped_ckpt_only)} ckpt-only ignored"
        )


def _strip_compile_prefix(sd: dict[str, Any]) -> dict[str, Any]:
    if any(k.startswith("_orig_mod.") for k in sd):
        return {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    return sd


def load_subliminal14_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str,
    *,
    map_location: str = "cpu",
    verbose: bool = True,
) -> WarmStartReport:
    """Shape-matched warm-start of ``model`` from a 1.4 ``state_dict`` file.

    Accepts either a bare ``state_dict`` or a dict wrapping one under
    ``"model"`` / ``"state_dict"`` (so it also reads this port's own
    checkpoints). Returns a :class:`WarmStartReport`.
    """
    raw = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        ckpt = raw["model"]
    elif isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        ckpt = raw["state_dict"]
    else:
        ckpt = raw
    ckpt = _strip_compile_prefix(ckpt)

    model_sd = model.state_dict()
    report = WarmStartReport()
    to_load: dict[str, torch.Tensor] = {}

    for name, tensor in ckpt.items():
        if name not in model_sd:
            report.skipped_ckpt_only.append(name)
            continue
        if tuple(model_sd[name].shape) == tuple(tensor.shape):
            to_load[name] = tensor
            report.loaded.append(name)
        else:
            report.skipped_shape.append(name)

    for name in model_sd:
        if name not in to_load:
            report.missing_in_ckpt.append(name)

    model.load_state_dict(to_load, strict=False)

    if verbose:
        logger.info(report.summary())
        # Group the fresh-init keys by their top module so the front-end
        # re-init (gene_embedding.*) is obvious in the logs.
        fresh_groups: dict[str, int] = {}
        for n in report.missing_in_ckpt:
            top = ".".join(n.split(".")[:2])
            fresh_groups[top] = fresh_groups.get(top, 0) + 1
        if fresh_groups:
            logger.info("fresh-init groups: %s", dict(sorted(fresh_groups.items())))
        if report.skipped_shape:
            logger.info(
                "shape-mismatch (NOT loaded; check model size matches the "
                "trained d_model=512/6L/8h): %s",
                report.skipped_shape[:12],
            )
    return report
