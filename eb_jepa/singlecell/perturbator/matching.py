"""Control matching for the perturbator OT problem (CLAUDE.md "Control matching").

The optimal-transport problem is defined **per stratum** ``(cell_line_id, plate)``:
matching source and target on the same plate *and* line shares the batch/technical
effects, so the perturbator learns the drug effect, not a plate artifact.

- **Source** = the control (``drug == "DMSO_TF"``) cells of that ``(cell_line, plate)``.
- **Target** = the cells treated with a given drug *d* at dose *c* on the *same*
  ``(cell_line, plate)``.

``build_strata`` takes a batch of encoded latents plus the aligned per-cell metadata
lists and returns a flat list of ``Stratum`` records, one per
``(cell_line, plate, drug, dose)`` target group, each carrying its source (control)
latents, target latents and the action ``(smiles, log_conc)``. Strata without
controls or without any treated cells are skipped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

CONTROL_DRUG = "DMSO_TF"


@dataclass
class Stratum:
    """One OT problem: map ``source`` (control) -> ``target`` (treated)."""

    source: torch.Tensor  # [n_control, d]
    target: torch.Tensor  # [n_treated, d]
    smiles: str | None
    log_conc: float
    stratum: tuple  # (cell_line_id, plate)
    drug: str


def _dose_key(log_conc: float) -> float:
    """Group key for a dose: round to stabilize float keys; nan -> sentinel."""
    if log_conc is None or (isinstance(log_conc, float) and math.isnan(log_conc)):
        return float("nan")
    return round(float(log_conc), 6)


def build_strata(
    latents: torch.Tensor,
    cell_line_id: list,
    plate: list,
    drug: list,
    canonical_smiles: list,
    log_conc,
    control_drug: str = CONTROL_DRUG,
) -> list[Stratum]:
    """Group ``latents`` into per-stratum source/target OT problems.

    Args:
        latents: ``[K, d]`` encoded pooled latents, aligned 1:1 with the metadata.
        cell_line_id, plate, drug, canonical_smiles: length-``K`` python lists.
        log_conc: length-``K`` list or 1-D tensor of log10 molar doses (nan = control).
        control_drug: the control drug name (default ``"DMSO_TF"``).
    Returns:
        List of ``Stratum`` (one per ``(cell_line, plate, drug, dose)`` target group).
        Strata with no control source or no treated target are skipped.
    """
    k = latents.shape[0]
    if torch.is_tensor(log_conc):
        log_conc = log_conc.tolist()
    log_conc = list(log_conc)

    # Index cells by stratum -> {"control": [...], "treated": {(drug, dose): [...]}}
    strata: dict[tuple, dict] = {}
    for i in range(k):
        key = (cell_line_id[i], plate[i])
        entry = strata.setdefault(key, {"control": [], "treated": {}})
        if drug[i] == control_drug:
            entry["control"].append(i)
        else:
            tkey = (drug[i], _dose_key(log_conc[i]))
            entry["treated"].setdefault(tkey, []).append(i)

    out: list[Stratum] = []
    for key, entry in strata.items():
        ctrl_idx = entry["control"]
        if not ctrl_idx or not entry["treated"]:
            continue  # need both a source and at least one target group
        source = latents[torch.tensor(ctrl_idx, dtype=torch.long)]
        for (drug_name, dose), tgt_idx in entry["treated"].items():
            if not tgt_idx:
                continue
            target = latents[torch.tensor(tgt_idx, dtype=torch.long)]
            # representative SMILES/dose for the group (all rows share drug+dose)
            rep = tgt_idx[0]
            out.append(
                Stratum(
                    source=source,
                    target=target,
                    smiles=canonical_smiles[rep],
                    log_conc=float(log_conc[rep]) if not _isnan(log_conc[rep]) else float("nan"),
                    stratum=key,
                    drug=drug_name,
                )
            )
    return out


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))
