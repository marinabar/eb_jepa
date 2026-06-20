"""Generate the perturbator dose-shift figure (CLAUDE.md "Perturbator-specific").

Loads the frozen Subliminal-1.4 encoder + a trained perturbator, streams a slice of
Tahoe, encodes control cells per cell line, scans ``(cell_line, drug)`` combos with
multiple doses, ranks them by a monotonicity score, and saves the dose-shift figure
(control cloud + per-drug dose-arrow tracks) to ``visualizations/perturbator/``.

Usage:
    python -m examples.tahoe_perturbator.visualize run \
        --config examples/tahoe_perturbator/cfgs/visualize.yaml
"""

from __future__ import annotations

import ast
import collections
import os

import numpy as np
import torch

from eb_jepa.datasets.tahoe.dataset import TahoeConfig, TahoeIterableDataset
from eb_jepa.logging import get_logger
from eb_jepa.singlecell.perturbator.featurize import DrugFeaturizer
from eb_jepa.singlecell.perturbator.model import Perturbator
from eb_jepa.singlecell.perturbator.visualize import (
    build_dose_track,
    plot_dose_shift,
    rank_combos,
)
from eb_jepa.singlecell.sub14.collator import Sub14Collator
from eb_jepa.singlecell.sub14.features import load_pc_features, random_pc_features
from eb_jepa.training_utils import load_config, setup_seed

logger = get_logger(__name__)

# Reuse the training entrypoint's frozen-encoder / encode helpers.
from examples.tahoe_perturbator.main import build_frozen_encoder, encode_cells  # noqa: E402


def _gather_cells(dataset, collator, n_cells, batch_size, encoder, device, amp):
    """Stream up to ``n_cells`` cells, encode them, return (latents, meta lists)."""
    buf, latents = [], []
    meta = collections.defaultdict(list)
    keys = ("cell_line_id", "drug", "plate", "canonical_smiles")
    seen = 0
    items = []
    for item in dataset:
        items.append(item)
        seen += 1
        if len(items) >= batch_size:
            batch = collator(items)
            z = encode_cells(encoder, batch, device, amp)
            latents.append(z.cpu())
            for k in keys:
                meta[k].extend(batch[k])
            meta["log_conc"].extend(batch["log_conc"].tolist())
            items = []
        if seen >= n_cells:
            break
    if items:
        batch = collator(items)
        z = encode_cells(encoder, batch, device, amp)
        latents.append(z.cpu())
        for k in keys:
            meta[k].extend(batch[k])
        meta["log_conc"].extend(batch["log_conc"].tolist())
    return torch.cat(latents, dim=0), meta


def run(config: str = "examples/tahoe_perturbator/cfgs/visualize.yaml", **overrides):
    cfg = load_config(config, cli_overrides=overrides or None)
    setup_seed(int(cfg.meta.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg.get("amp", True))

    # protein-coding features + frozen encoder
    cache = cfg.encoder.get("gene_emb_cache", "random")
    if cache and cache != "random":
        pc = load_pc_features(cache)
    else:
        pc = random_pc_features(n_pc=int(cfg.encoder.get("smoke_n_pc", 2000)))
        logger.warning("RANDOM PC features (no cache) — smoke/dev only.")
    encoder = build_frozen_encoder(cfg, pc, device)

    # perturbator (+ trained weights)
    featurizer = DrugFeaturizer(
        n_bits=int(cfg.featurizer.get("n_bits", 1024)),
        radius=int(cfg.featurizer.get("radius", 2)),
        use_descriptors=bool(cfg.featurizer.get("use_descriptors", True)),
    )
    objective = str(cfg.loss.get("objective", "flow_matching"))
    perturbator = Perturbator(
        d_model=int(cfg.encoder.d_model),
        action_dim=featurizer.action_dim,
        depth=int(cfg.model.get("depth", 4)),
        d_cond=int(cfg.model.get("d_cond", 256)),
        cond_hidden=cfg.model.get("cond_hidden"),
        time_conditioned=(objective == "flow_matching"),
        n_time_freqs=int(cfg.model.get("n_time_freqs", 64)),
    ).to(device)
    ckpt = cfg.perturbator.get("ckpt", "")
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device, weights_only=False)
        sd = state.get("model_state_dict", state)
        perturbator.load_state_dict(sd, strict=True)
        logger.info(f"loaded perturbator from {ckpt}")
    else:
        logger.warning(f"perturbator ckpt {ckpt!r} not found — using INIT weights.")
    perturbator.eval()

    # data
    data_cfg = TahoeConfig(
        **{k: cfg.data[k] for k in cfg.data if k in TahoeConfig.__dataclass_fields__}
    )
    maps = {}
    if cfg.data.get("maps_path") and os.path.exists(cfg.data.maps_path):
        maps = torch.load(cfg.data.maps_path)
    dataset = TahoeIterableDataset(
        data_cfg, binner=None,
        cell_line_to_organ=maps.get("cell_line_to_organ"),
        sample_to_logconc=maps.get("sample_to_logconc"),
        shuffle=False,
    )
    collator = Sub14Collator(
        token_to_pc_local=pc.token_to_pc_local, n_pc_genes=pc.n_pc_genes,
        num_bins=int(cfg.encoder.get("num_bins", 16)),
        genes_per_bin=int(cfg.encoder.get("genes_per_bin", 32)),
        num_views=1, binomial_subsample=None, seed=int(cfg.meta.seed),
    )

    latents, meta = _gather_cells(
        dataset, collator, int(cfg.viz.n_cells), int(cfg.data.batch_size),
        encoder, device, amp,
    )
    logger.info(f"encoded {latents.shape[0]} cells, d={latents.shape[1]}")

    # index cells by (cell_line, drug) with their per-cell doses
    cl = np.array(meta["cell_line_id"], dtype=object)
    drug = np.array(meta["drug"], dtype=object)
    smiles = np.array(meta["canonical_smiles"], dtype=object)
    log_conc = np.array(meta["log_conc"], dtype=np.float64)
    control_name = str(cfg.viz.get("control_drug", "DMSO_TF"))

    out_dir = cfg.viz.get("out_dir", "visualizations/perturbator")
    os.makedirs(out_dir, exist_ok=True)
    min_cells = int(cfg.viz.get("min_cells_per_group", 30))
    min_doses = int(cfg.viz.get("min_doses", 2))

    target_lines = cfg.viz.get("cell_lines") or sorted(set(cl[cl != None]))  # noqa: E711
    for line in target_lines:
        line_mask = cl == line
        ctrl_mask = line_mask & (drug == control_name)
        if ctrl_mask.sum() < min_cells:
            continue
        control_latents = latents[torch.from_numpy(np.where(ctrl_mask)[0])]

        # build a track per drug that has >= min_doses doses with enough cells
        tracks = []
        drugs_here = [d for d in set(drug[line_mask]) if d and d != control_name]
        for dname in drugs_here:
            dmask = line_mask & (drug == dname)
            doses = sorted(set(round(float(x), 4) for x in log_conc[dmask] if np.isfinite(x)))
            doses = [d for d in doses if (dmask & (np.round(log_conc, 4) == d)).sum() >= min_cells]
            if len(doses) < min_doses:
                continue
            sm = next((s for s in smiles[dmask] if s), None)
            track = build_dose_track(
                perturbator, featurizer, control_latents, dname, sm, doses,
                objective=objective,
                ode_steps=int(cfg.loss.get("ode_steps", 20)),
                ode_method=str(cfg.loss.get("ode_method", "heun")),
            )
            tracks.append(track)
        if not tracks:
            continue
        tracks = rank_combos(tracks)
        top = tracks[: int(cfg.viz.get("top_drugs", 5))]
        safe = str(line).replace("/", "_")
        prefix = os.path.join(out_dir, f"dose_shift_{safe}")
        paths = plot_dose_shift(
            control_latents, top, prefix, cell_line=str(line),
            projector=str(cfg.viz.get("projector", "pca")), seed=int(cfg.meta.seed),
        )
        best = top[0]
        logger.info(
            f"[{line}] {len(top)} drug tracks | best={best.drug} "
            f"score={best.metrics['score']:.3f} -> {paths[0]}"
        )


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run})
