"""Train the perturbator on Tahoe-100M over a frozen Subliminal-1.4 encoder (CLAUDE.md Part II).

The encoder is **frozen**: each batch of cells is encoded (``no_grad``) to pooled
``[CELL]`` latents, grouped into per-stratum ``(cell_line_id, plate)`` OT problems by
control matching (source = ``DMSO_TF`` controls, target = treated cells at a given
drug+dose on the same stratum), and the perturbator is trained to map the source
control distribution to the target treated distribution, conditioned on the drug
action (SMILES features + dose).

Two selectable objectives (``loss.objective``):

- ``flow_matching`` (default): a FiLM velocity field ``v(x_t, t, action)`` trained by
  rectified conditional flow matching; inference integrates the ODE source->target.
- ``direct``: a one-shot residual map trained by the sliced-Wasserstein OT loss.

Single YAML config drives everything (see cfgs/train.yaml).

Usage:
    python -m examples.tahoe_perturbator.main run \
        --config examples/tahoe_perturbator/cfgs/train.yaml
    torchrun --nproc_per_node=1 -m examples.tahoe_perturbator.main run \
        --config examples/tahoe_perturbator/cfgs/train.yaml
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from eb_jepa.datasets.tahoe.dataset import TahoeConfig, TahoeIterableDataset
from eb_jepa.logging import get_logger
from eb_jepa.schedulers import CosineWithWarmup
from eb_jepa.singlecell.perturbator.featurize import DrugFeaturizer
from eb_jepa.singlecell.perturbator.flow import flow_matching_loss, predict_perturbed
from eb_jepa.singlecell.perturbator.losses import sliced_wasserstein
from eb_jepa.singlecell.perturbator.matching import build_strata
from eb_jepa.singlecell.perturbator.model import Perturbator
from eb_jepa.singlecell.sub14.collator import Sub14Collator
from eb_jepa.singlecell.sub14.features import load_pc_features, random_pc_features
from eb_jepa.singlecell.sub14.model import Subliminal14
from eb_jepa.training_utils import load_config, save_checkpoint, setup_seed

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Frozen Subliminal-1.4 encoder                                               #
# --------------------------------------------------------------------------- #
def build_frozen_encoder(cfg, pc, device):
    """Build the Subliminal-1.4 encoder and load its frozen checkpoint weights.

    The checkpoint is the one saved by ``examples/tahoe_jepa/sub14_main.py``
    (``{"model": state_dict, ...}``); it is loaded shape-matched and the encoder is
    set to ``eval`` with all gradients disabled.
    """
    from eb_jepa.singlecell.sub14.load_checkpoint import load_subliminal14_checkpoint

    enc = cfg.encoder
    num_bins = int(enc.get("num_bins", 16))
    genes_per_bin = int(enc.get("genes_per_bin", 32))
    model = Subliminal14(
        n_pc_genes=pc.n_pc_genes,
        d_model=int(enc.d_model),
        n_heads=int(enc.n_heads),
        n_layers=int(enc.n_layers),
        d_ff=int(enc.get("d_ff", 4 * int(enc.d_model))),
        dropout=float(enc.get("dropout", 0.0)),
        latent_dim=int(enc.get("proj_dim", 128)),
        num_bins=num_bins,
        max_genes_per_cell=num_bins * genes_per_bin,
        dna_features=pc.dna_features,
        protein_features=pc.protein_features,
        freeze_features=True,
        attention_activation=str(enc.get("attention_activation", "sigmoid")),
    )
    ckpt = enc.get("ckpt", "")
    if ckpt and os.path.exists(ckpt):
        load_subliminal14_checkpoint(model, ckpt, map_location="cpu", verbose=True)
    else:
        logger.warning(f"Encoder checkpoint {ckpt!r} not found — using init weights.")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def encode_cells(encoder, batch, device, amp: bool) -> torch.Tensor:
    """Encode the single clean view (Sub14Collator output) -> pooled latents [N, d]."""
    gene_ids = batch["gene_ids"][0].to(device)
    bin_ids = batch["bin_ids"][0].to(device)
    pad = batch["padding_mask"][0].to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        z = encoder.encode(gene_ids, bin_ids, pad)
    return z.float()


# --------------------------------------------------------------------------- #
# Data                                                                         #
# --------------------------------------------------------------------------- #
def build_loader(cfg, pc, rank=0, world=1):
    data_cfg = TahoeConfig(
        **{k: cfg.data[k] for k in cfg.data if k in TahoeConfig.__dataclass_fields__}
    )
    maps = {}
    if cfg.data.get("maps_path") and os.path.exists(cfg.data.maps_path):
        maps = torch.load(cfg.data.maps_path)
    dataset = TahoeIterableDataset(
        data_cfg,
        binner=None,
        cell_line_to_organ=maps.get("cell_line_to_organ"),
        sample_to_logconc=maps.get("sample_to_logconc"),
        rank=rank,
        world_size=world,
        shuffle=(data_cfg.split == "train"),
    )
    # One clean view per cell — we need real (un-augmented) latents, not views.
    collator = Sub14Collator(
        token_to_pc_local=pc.token_to_pc_local,
        n_pc_genes=pc.n_pc_genes,
        num_bins=int(cfg.encoder.get("num_bins", 16)),
        genes_per_bin=int(cfg.encoder.get("genes_per_bin", 32)),
        num_views=1,
        binomial_subsample=None,
        seed=int(cfg.meta.seed) + rank,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.get("num_workers", 0)),
        pin_memory=bool(cfg.data.get("pin_mem", True)),
        drop_last=True,
        collate_fn=collator,
        persistent_workers=int(cfg.data.get("num_workers", 0)) > 0,
    )
    return loader, dataset, collator


# --------------------------------------------------------------------------- #
# Training / eval step                                                         #
# --------------------------------------------------------------------------- #
def _valid_strata(latents, batch, cfg):
    strata = build_strata(
        latents,
        batch["cell_line_id"],
        batch["plate"],
        batch["drug"],
        batch["canonical_smiles"],
        batch["log_conc"],
    )
    min_src = int(cfg.loss.get("min_source", 2))
    min_tgt = int(cfg.loss.get("min_target", 2))
    return [s for s in strata if s.source.shape[0] >= min_src and s.target.shape[0] >= min_tgt]


def perturbator_step(perturbator, featurizer, latents, batch, cfg, device, gen):
    """One step over all valid strata. Returns (loss, metrics) or (None, {...})."""
    objective = str(cfg.loss.get("objective", "flow_matching"))
    strata = _valid_strata(latents, batch, cfg)
    losses, sw_eval = [], []
    sw_slices = int(cfg.loss.get("sw_slices", 256))
    sw_p = int(cfg.loss.get("sw_p", 2))
    for s in strata:
        action = featurizer.featurize(s.smiles, s.log_conc).to(device)
        if objective == "flow_matching":
            loss = flow_matching_loss(
                perturbator, s.source, s.target.detach(), action, generator=gen
            )
            losses.append(loss)
        elif objective == "direct":
            pred = perturbator(s.source, action)
            loss = sliced_wasserstein(pred, s.target.detach(), n_slices=sw_slices, p=sw_p)
            losses.append(loss)
        else:
            raise ValueError(f"unknown objective {objective!r}")
    if not losses:
        return None, {"n_strata": 0}
    loss = torch.stack(losses).mean()
    return loss, {"n_strata": len(losses), f"{objective}_loss": float(loss.detach())}


@torch.no_grad()
def eval_step(perturbator, featurizer, latents, batch, cfg, device):
    """Held-out eval: sliced-W between predicted and target distributions per stratum.

    Reports the OT distance achieved by the (selected-objective) inference path, plus
    the source->target baseline (how far apart control and treated are) so the
    fraction of the gap the perturbator closes is interpretable.
    """
    objective = str(cfg.loss.get("objective", "flow_matching"))
    sw_slices = int(cfg.loss.get("sw_slices", 256))
    ode_steps = int(cfg.loss.get("ode_steps", 20))
    ode_method = str(cfg.loss.get("ode_method", "heun"))
    strata = _valid_strata(latents, batch, cfg)
    pred_sw, base_sw = [], []
    for s in strata:
        action = featurizer.featurize(s.smiles, s.log_conc).to(device)
        pred = predict_perturbed(
            perturbator, s.source, action, objective,
            n_steps=ode_steps, method=ode_method,
        )
        pred_sw.append(float(sliced_wasserstein(pred, s.target, n_slices=sw_slices)))
        base_sw.append(float(sliced_wasserstein(s.source, s.target, n_slices=sw_slices)))
    if not pred_sw:
        return {}
    pred_m, base_m = float(np.mean(pred_sw)), float(np.mean(base_sw))
    closed = 1.0 - pred_m / base_m if base_m > 1e-8 else 0.0
    return {
        "eval/sliced_wasserstein": pred_m,
        "eval/baseline_sliced_wasserstein": base_m,
        "eval/gap_closed_frac": closed,
        "eval/n_strata": len(pred_sw),
    }


def build_eval_batch(dataset, collator_args, cfg, device):
    """Fixed rank-0 eval batch: one deterministic clean view over a held-out sample."""
    n = int(cfg.eval.get("eval_cells", 0))
    if n <= 0:
        return None
    items = dataset.sample_items(n)
    coll = Sub14Collator(**collator_args, num_views=1, binomial_subsample=None,
                         seed=int(cfg.meta.seed))
    return coll(items)


# --------------------------------------------------------------------------- #
# Train                                                                        #
# --------------------------------------------------------------------------- #
def train(cfg, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(int(cfg.meta.seed))
    gen = torch.Generator(device=device).manual_seed(int(cfg.meta.seed))

    # protein-coding features (frozen DNA+protein) for the encoder vocabulary
    cache = cfg.encoder.get("gene_emb_cache", "random")
    if cache and cache != "random":
        pc = load_pc_features(cache)
        logger.info(f"loaded {pc.n_pc_genes} protein-coding genes from {cache}")
    else:
        pc = random_pc_features(n_pc=int(cfg.encoder.get("smoke_n_pc", 2000)))
        logger.warning("RANDOM PC features (no cache) — smoke/dev only.")

    loader, dataset, _ = build_loader(cfg, pc)
    encoder = build_frozen_encoder(cfg, pc, device)

    featurizer = DrugFeaturizer(
        n_bits=int(cfg.featurizer.get("n_bits", 1024)),
        radius=int(cfg.featurizer.get("radius", 2)),
        use_descriptors=bool(cfg.featurizer.get("use_descriptors", True)),
    )
    if not featurizer.has_rdkit:
        logger.warning("RDKit not available — using deterministic hash featurizer.")

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
    n_params = sum(p.numel() for p in perturbator.parameters() if p.requires_grad)
    logger.info(f"perturbator objective={objective} | trainable params: {n_params:,}")

    opt = torch.optim.AdamW(
        perturbator.parameters(),
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
        betas=tuple(cfg.optim.get("betas", (0.9, 0.95))),
    )
    max_steps = int(cfg.optim.get("max_steps", 0))
    total_steps = max_steps if max_steps > 0 else int(cfg.optim.epochs) * 1000
    sched = CosineWithWarmup(
        opt, total_steps,
        warmup_ratio=float(cfg.optim.get("warmup_ratio", 0.05)),
        min_lr=float(cfg.optim.get("min_lr", 1e-6)),
    )

    run = None
    if cfg.wandb.get("enabled", False):
        from eb_jepa.training_utils import setup_wandb

        if cfg.wandb.get("entity"):
            os.environ["WANDB_ENTITY"] = cfg.wandb.entity
        run = setup_wandb(cfg.wandb.project, cfg, cfg.meta.run_dir, enabled=True)

    # fixed eval batch
    collator_args = dict(
        token_to_pc_local=pc.token_to_pc_local,
        n_pc_genes=pc.n_pc_genes,
        num_bins=int(cfg.encoder.get("num_bins", 16)),
        genes_per_bin=int(cfg.encoder.get("genes_per_bin", 32)),
    )
    eval_batch = None
    eval_every = int(cfg.eval.get("eval_every", 0))
    if eval_every > 0:
        eval_batch = build_eval_batch(dataset, collator_args, cfg, device)

    amp = bool(cfg.training.get("amp", True))
    log_every = int(cfg.training.get("log_every", 20))
    max_minutes = float(cfg.training.get("max_minutes", 0))

    def _eval(step):
        if eval_batch is None:
            return
        latents = encode_cells(encoder, eval_batch, device, amp)
        m = eval_step(perturbator, featurizer, latents, eval_batch, cfg, device)
        if m:
            logger.info(
                f"[eval @ {step}] sliced_W={m['eval/sliced_wasserstein']:.4f} "
                f"baseline={m['eval/baseline_sliced_wasserstein']:.4f} "
                f"gap_closed={m['eval/gap_closed_frac']:.3f} (n={m['eval/n_strata']})"
            )
            if run is not None:
                run.log(m, step=step)

    t0 = time.time()
    step = 0
    stop = False
    for epoch in range(int(cfg.optim.epochs)):
        if stop:
            break
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        perturbator.train()
        for batch in loader:
            latents = encode_cells(encoder, batch, device, amp)  # frozen, no grad
            opt.zero_grad(set_to_none=True)
            loss, metrics = perturbator_step(
                perturbator, featurizer, latents, batch, cfg, device, gen
            )
            if loss is None:
                continue  # no valid stratum (no control/target pair) in this batch
            loss.backward()
            if float(cfg.optim.get("max_grad_norm", 0)) > 0:
                torch.nn.utils.clip_grad_norm_(
                    perturbator.parameters(), float(cfg.optim.max_grad_norm)
                )
            opt.step()
            sched.step()
            step += 1

            if step % log_every == 0:
                metrics["loss"] = float(loss.detach())
                metrics["lr"] = float(sched.get_last_lr()[0])
                logger.info(
                    f"step {step} | "
                    + " | ".join(
                        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in metrics.items()
                    )
                )
                if run is not None:
                    run.log(metrics, step=step)
            if eval_every and step % eval_every == 0:
                _eval(step)
            if max_steps and step >= max_steps:
                stop = True
                break
            if max_minutes and (time.time() - t0) / 60.0 >= max_minutes:
                stop = True
                break

        if cfg.training.get("ckpt_every_epoch", True):
            save_checkpoint(
                os.path.join(cfg.meta.run_dir, "perturbator.pt"),
                perturbator, opt, sched.scheduler, epoch=epoch, step=step,
                objective=objective, action_dim=featurizer.action_dim,
                d_model=int(cfg.encoder.d_model),
            )

    _eval(step)
    save_checkpoint(
        os.path.join(cfg.meta.run_dir, "perturbator_final.pt"),
        perturbator, opt, sched.scheduler, step=step,
        objective=objective, action_dim=featurizer.action_dim,
        d_model=int(cfg.encoder.d_model),
    )
    logger.info(f"Done in {time.time() - t0:.1f}s ({step} steps)")
    return perturbator


def run(config: str = "examples/tahoe_perturbator/cfgs/train.yaml", **overrides):
    cfg = load_config(config, cli_overrides=overrides or None)
    os.makedirs(cfg.meta.run_dir, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run})
