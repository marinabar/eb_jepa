"""Train the perturbator v1 on Tahoe-100M (CLAUDE.md Part II).

The encoder is **frozen**: each batch of cells is encoded (no_grad) to pooled
latents, grouped into per-stratum ``(cell_line_id, plate)`` OT problems by control
matching (source = ``DMSO_TF`` controls, target = treated cells at a given drug+dose
on the same stratum), and the perturbator maps source -> predicted perturbed latent
conditioned on the drug action (SMILES features + dose). The loss is the
sliced-Wasserstein distance between the predicted and the target latent
distributions; only the perturbator is optimized.

Single YAML config drives everything (see cfgs/train.yaml).

Usage:
    python -m examples.tahoe_perturbator.main run \
        --config examples/tahoe_perturbator/cfgs/train.yaml
"""

from __future__ import annotations

import os
import time

import torch

from eb_jepa.datasets.tahoe.dataset import TahoeConfig, init_tahoe_data
from eb_jepa.datasets.tahoe.normalizer import QuantileBinner
from eb_jepa.logging import get_logger
from eb_jepa.schedulers import CosineWithWarmup
from eb_jepa.singlecell.embeddings import GeneTokenEmbedding
from eb_jepa.singlecell.encoder import SingleCellEncoder
from eb_jepa.singlecell.perturbator.featurize import DrugFeaturizer
from eb_jepa.singlecell.perturbator.losses import sliced_wasserstein
from eb_jepa.singlecell.perturbator.matching import build_strata
from eb_jepa.singlecell.perturbator.model import Perturbator
from eb_jepa.training_utils import (
    load_checkpoint,
    load_config,
    save_checkpoint,
    setup_seed,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Frozen encoder                                                              #
# --------------------------------------------------------------------------- #
def build_gene_embedding(cfg) -> GeneTokenEmbedding:
    cache = cfg.encoder.get("gene_emb_cache", "random")
    if cache and cache != "random":
        return GeneTokenEmbedding.from_cache(
            cache, cfg.encoder.d_model, cfg.data.count_mode, cfg.data.n_bins
        )
    logger.warning("Using RANDOM gene embeddings (no cache) — smoke/dev only.")
    return GeneTokenEmbedding.random(
        cfg.data.n_genes,
        cfg.encoder.d_model,
        count_mode=cfg.data.count_mode,
        n_bins=cfg.data.n_bins,
    )


def build_frozen_encoder(cfg, device) -> SingleCellEncoder:
    """Build the encoder from the config and load its frozen checkpoint weights."""
    embed = build_gene_embedding(cfg)
    encoder = SingleCellEncoder(
        embed,
        d_model=cfg.encoder.d_model,
        n_layers=cfg.encoder.n_layers,
        n_heads=cfg.encoder.n_heads,
        n_kv_heads=cfg.encoder.get("n_kv_heads"),
        use_cls=cfg.encoder.use_cls,
        readout=cfg.encoder.readout,
    )
    ckpt = cfg.encoder.get("ckpt", "")
    if ckpt and os.path.exists(ckpt):
        # frozen ESMC/Evo2 buffers are non-persistent -> load non-strictly
        load_checkpoint(ckpt, encoder, device=device, strict=False)
    else:
        logger.warning(f"Encoder checkpoint {ckpt!r} not found — using init weights.")
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


@torch.no_grad()
def encode_cells(encoder: SingleCellEncoder, batch: dict, device, amp: bool) -> torch.Tensor:
    """Encode the single clean view ([1, N, L]) -> pooled latents [N, d_model]."""
    ids = batch["gene_token_ids"][0].to(device)
    pad = batch["pad_mask"][0].to(device)
    cv = batch.get("count_value")
    cb = batch.get("count_bin")
    cv = cv[0].to(device) if cv is not None else None
    cb = cb[0].to(device) if cb is not None else None
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        z = encoder(ids, pad, count_value=cv, count_bin=cb)
    return z.float()


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def perturbator_step(
    perturbator: Perturbator,
    featurizer: DrugFeaturizer,
    latents: torch.Tensor,
    batch: dict,
    cfg,
    device,
) -> tuple[torch.Tensor, dict]:
    """One OT step over all valid strata in the batch. Returns (loss, metrics).

    ``loss`` is the mean sliced-Wasserstein distance over strata; ``metrics`` carries
    the strata count for logging. Returns ``(None, ...)`` if no valid stratum.
    """
    strata = build_strata(
        latents,
        batch["cell_line_id"],
        batch["plate"],
        batch["drug"],
        batch["canonical_smiles"],
        batch["log_conc"],
    )
    min_src = int(cfg.loss.get("min_source", 1))
    min_tgt = int(cfg.loss.get("min_target", 1))
    sw_slices = int(cfg.loss.sw_slices)
    sw_p = int(cfg.loss.get("sw_p", 2))

    losses = []
    for s in strata:
        if s.source.shape[0] < min_src or s.target.shape[0] < min_tgt:
            continue
        action = featurizer.featurize(s.smiles, s.log_conc).to(device)
        pred = perturbator(s.source, action)
        d = sliced_wasserstein(pred, s.target.detach(), n_slices=sw_slices, p=sw_p)
        losses.append(d)
    if not losses:
        return None, {"n_strata": 0}
    loss = torch.stack(losses).mean()
    return loss, {"n_strata": len(losses), "sliced_wasserstein": loss.detach().item()}


def train(cfg, device=None):
    """Run perturbator training from a loaded config. Returns the perturbator."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(cfg.meta.seed)

    # data (single clean view for the latents)
    data_cfg = TahoeConfig(
        **{k: cfg.data[k] for k in cfg.data if k in TahoeConfig.__dataclass_fields__}
    )
    binner = None
    if cfg.data.count_mode == "B" and cfg.data.get("quantile_bins"):
        binner = QuantileBinner.load(cfg.data.quantile_bins)
    maps = {}
    if cfg.data.get("maps_path"):
        maps = torch.load(cfg.data.maps_path)
    train_loader, _ = init_tahoe_data(
        data_cfg,
        binner=binner,
        cell_line_to_organ=maps.get("cell_line_to_organ"),
        sample_to_logconc=maps.get("sample_to_logconc"),
    )

    # frozen encoder + perturbator + featurizer
    encoder = build_frozen_encoder(cfg, device)
    featurizer = DrugFeaturizer(
        n_bits=cfg.featurizer.get("n_bits", 1024),
        radius=cfg.featurizer.get("radius", 2),
        use_descriptors=cfg.featurizer.get("use_descriptors", True),
    )
    if not featurizer.has_rdkit:
        logger.warning("RDKit not available — using deterministic hash featurizer.")
    perturbator = Perturbator(
        d_model=cfg.encoder.d_model,
        action_dim=featurizer.action_dim,
        depth=cfg.model.get("depth", 4),
        d_cond=cfg.model.get("d_cond", 256),
        cond_hidden=cfg.model.get("cond_hidden"),
    ).to(device)

    opt = torch.optim.AdamW(
        perturbator.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=tuple(cfg.optim.get("betas", (0.9, 0.95))),
    )
    max_steps = int(cfg.optim.get("max_steps", 0))
    total_steps = max_steps if max_steps > 0 else cfg.optim.epochs * max(1, len(train_loader))
    sched = CosineWithWarmup(
        opt,
        total_steps,
        warmup_ratio=cfg.optim.get("warmup_ratio", 0.05),
        min_lr=cfg.optim.get("min_lr", 1e-6),
    )

    run = None
    if cfg.wandb.get("enabled", False):
        from eb_jepa.training_utils import setup_wandb

        if cfg.wandb.get("entity"):
            os.environ["WANDB_ENTITY"] = cfg.wandb.entity
        run = setup_wandb(cfg.wandb.project, cfg, cfg.meta.run_dir, enabled=True)

    amp = cfg.training.get("amp", True)
    step = 0
    stop = False
    for epoch in range(cfg.optim.epochs):
        if stop:
            break
        perturbator.train()
        for batch in train_loader:
            latents = encode_cells(encoder, batch, device, amp)  # frozen, no grad
            opt.zero_grad(set_to_none=True)
            loss, metrics = perturbator_step(
                perturbator, featurizer, latents, batch, cfg, device
            )
            if loss is None:
                continue  # no valid stratum in this batch (no control/target pair)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % cfg.training.get("log_every", 20) == 0:
                metrics["loss"] = loss.detach().item()
                metrics["lr"] = sched.get_last_lr()[0]
                logger.info(
                    f"step {step} | "
                    + " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in metrics.items())
                )
                if run is not None:
                    run.log(metrics, step=step)
            if max_steps and step >= max_steps:
                stop = True
                break
        if cfg.training.get("ckpt_every_epoch", True):
            save_checkpoint(
                os.path.join(cfg.meta.run_dir, "perturbator.pt"),
                perturbator,
                opt,
                sched.scheduler,
                epoch=epoch,
                step=step,
            )
    return perturbator


def run(config: str = "examples/tahoe_perturbator/cfgs/train.yaml", **overrides):
    cfg = load_config(config, cli_overrides=overrides or None)
    os.makedirs(cfg.meta.run_dir, exist_ok=True)
    t0 = time.time()
    train(cfg)
    logger.info(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run})
