"""Train the single-cell LeJEPA encoder on Tahoe-100M.

Single YAML config drives everything (see cfgs/train.yaml). Multi-GPU via
``torchrun`` (DDP + gradient checkpointing first; FSDP later). The SIGReg ECF is
all-reduced across ranks inside the loss, so DDP gives the correct global-batch
Gaussianity test.

Usage:
    # single process (smoke)
    python -m examples.tahoe_jepa.main run --config examples/tahoe_jepa/cfgs/train.yaml
    # 8x B200 (cluster)
    torchrun --nproc_per_node=8 -m examples.tahoe_jepa.main run \
        --config examples/tahoe_jepa/cfgs/train.yaml
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn

from eb_jepa.architectures import Projector
from eb_jepa.datasets.tahoe.dataset import TahoeConfig, init_tahoe_data
from eb_jepa.datasets.tahoe.normalizer import QuantileBinner
from eb_jepa.logging import get_logger
from eb_jepa.losses import LeJEPALoss
from eb_jepa.schedulers import CosineWithWarmup
from eb_jepa.singlecell.embeddings import GeneTokenEmbedding
from eb_jepa.singlecell.encoder import SingleCellEncoder, encode_views
from eb_jepa.training_utils import load_config, save_checkpoint, setup_seed

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Distributed helpers                                                         #
# --------------------------------------------------------------------------- #
def setup_ddp():
    """Init DDP from torchrun env. Returns (is_ddp, rank, world_size, local_rank)."""
    if "RANK" in os.environ and torch.cuda.is_available():
        dist.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    return False, 0, 1, 0


def is_main(rank: int) -> bool:
    return rank == 0


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
class TrainModule(nn.Module):
    """Shared encoder + LeJEPA loss (with its projector). forward(batch) -> loss dict."""

    def __init__(self, encoder: SingleCellEncoder, loss_fn: LeJEPALoss):
        super().__init__()
        self.encoder = encoder
        self.loss_fn = loss_fn

    def forward(self, batch: dict) -> dict:
        return self.loss_fn(encode_views(self.encoder, batch))


def build_gene_embedding(cfg) -> GeneTokenEmbedding:
    cache = cfg.model.get("gene_emb_cache", "random")
    if cache and cache != "random":
        return GeneTokenEmbedding.from_cache(
            cache, cfg.model.d_model, cfg.data.count_mode, cfg.data.n_bins
        )
    logger.warning("Using RANDOM gene embeddings (no cache) — smoke/dev only.")
    return GeneTokenEmbedding.random(
        cfg.data.n_genes,
        cfg.model.d_model,
        count_mode=cfg.data.count_mode,
        n_bins=cfg.data.n_bins,
    )


def build_train_module(cfg) -> TrainModule:
    embed = build_gene_embedding(cfg)
    encoder = SingleCellEncoder(
        embed,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        n_kv_heads=cfg.model.get("n_kv_heads"),
        use_cls=cfg.model.use_cls,
        readout=cfg.model.readout,
        grad_checkpoint=cfg.model.get("grad_checkpoint", False),
    )
    projector = Projector(
        f"{cfg.model.d_model}-{cfg.model.proj_hidden}-{cfg.model.proj_dim}"
    )
    loss_fn = LeJEPALoss(
        projector=projector,
        lamb=cfg.loss.lamb,
        num_slices=cfg.loss.num_slices,
        knots=cfg.loss.get("knots", 17),
        t_max=cfg.loss.get("t_max", 3.0),
    )
    return TrainModule(encoder, loss_fn)


def param_groups(model: nn.Module, weight_decay: float):
    """No weight decay on 1D params (norms, biases, cls token, mask vector)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def move_batch(batch: dict, device) -> dict:
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def train(cfg, device=None):
    """Run training from an already-loaded config (DictConfig). Returns the encoder."""
    is_ddp, rank, world, local_rank = setup_ddp()
    if device is None:
        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
    setup_seed(cfg.meta.seed + rank)

    # data
    data_cfg = TahoeConfig(
        **{k: cfg.data[k] for k in cfg.data if k in TahoeConfig.__dataclass_fields__}
    )
    binner = None
    if cfg.data.count_mode == "B" and cfg.data.get("quantile_bins"):
        binner = QuantileBinner.load(cfg.data.quantile_bins)
    maps = {}
    if cfg.data.get("maps_path"):
        maps = torch.load(cfg.data.maps_path)
    train_loader, dataset = init_tahoe_data(
        data_cfg,
        binner=binner,
        cell_line_to_organ=maps.get("cell_line_to_organ"),
        sample_to_logconc=maps.get("sample_to_logconc"),
    )

    # fixed eval set for t-SNE snapshots along training (rank 0 only)
    eval_batch = eval_labels = tsne_dir = None
    do_tsne = is_main(rank) and bool(cfg.get("eval", {}).get("enabled", False))
    if do_tsne:
        from examples.tahoe_jepa.eval_tsne import build_eval_set

        eval_batch, eval_labels = build_eval_set(
            dataset, data_cfg, int(cfg.eval.get("eval_cells", 2000)), seed=cfg.meta.seed
        )
        tsne_dir = os.path.join(cfg.meta.run_dir, "tsne")
        logger.info(f"t-SNE eval set: {len(eval_labels['organ'])} cells -> {tsne_dir}")

    # model
    model = build_train_module(cfg).to(device)
    if cfg.model.get("compile", False):
        model.encoder = torch.compile(model.encoder)
    if is_ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False
        )

    # optim
    opt = torch.optim.AdamW(
        param_groups(model, cfg.optim.weight_decay),
        lr=cfg.optim.lr,
        betas=tuple(cfg.optim.get("betas", (0.9, 0.95))),
    )
    max_steps = int(cfg.optim.get("max_steps", 0))
    # size the LR schedule to the actual run length (max_steps caps it)
    total_steps = max_steps if max_steps > 0 else cfg.optim.epochs * max(1, len(train_loader))
    sched = CosineWithWarmup(
        opt,
        total_steps,
        warmup_ratio=cfg.optim.get("warmup_ratio", 0.05),
        min_lr=cfg.optim.get("min_lr", 1e-6),
    )

    amp_dtype = torch.bfloat16 if cfg.training.get("amp", True) else torch.float32
    run = None
    if is_main(rank) and cfg.wandb.get("enabled", False):
        from eb_jepa.training_utils import setup_wandb

        if cfg.wandb.get("entity"):
            os.environ["WANDB_ENTITY"] = cfg.wandb.entity  # team; key stays in ~/.netrc
        run = setup_wandb(cfg.wandb.project, cfg, cfg.meta.run_dir, enabled=True)

    def _snapshot(step: int):
        from examples.tahoe_jepa.eval_tsne import tsne_snapshot

        enc = model.module.encoder if is_ddp else model.encoder
        paths = tsne_snapshot(
            enc, eval_batch, eval_labels, tsne_dir, step, device,
            classes=list(cfg.eval.get("classes", ["organ", "cell_line_id", "drug", "moa_fine"])),
            chunk=int(cfg.eval.get("encode_chunk", 64)),
            perplexity=float(cfg.eval.get("perplexity", 30.0)),
            seed=cfg.meta.seed,
            amp=cfg.training.get("amp", True),
        )
        logger.info(f"t-SNE snapshot @ step {step} -> {len(paths)} panels in {tsne_dir}")
        if run is not None:
            import wandb

            run.log(
                {f"tsne/{c}": wandb.Image(p, caption=f"step {step}") for c, p in paths.items()},
                step=step,
            )

    def _probe_eval(step: int):
        from examples.tahoe_jepa.probe_eval import probe_report

        enc = model.module.encoder if is_ddp else model.encoder
        probe_dir = os.path.join(cfg.meta.run_dir, "probes")
        scalars, spectrum_path = probe_report(
            enc, eval_batch, device, probe_dir, step,
            probe_epochs=int(cfg.eval.get("probe_epochs", 150)),
            chunk=int(cfg.eval.get("encode_chunk", 64)),
            amp=cfg.training.get("amp", True),
        )
        logger.info(
            f"probe eval @ step {step} | "
            + " | ".join(f"{k}={v:.4f}" for k, v in sorted(scalars.items()))
        )
        if run is not None:
            import wandb

            run.log(scalars, step=step)
            run.log(
                {"repr/spectrum": wandb.Image(spectrum_path, caption=f"step {step}")},
                step=step,
            )

    tsne_every = int(cfg.get("eval", {}).get("tsne_every", 0)) if do_tsne else 0
    probe_every = int(cfg.get("eval", {}).get("probe_every", 0)) if do_tsne else 0
    if do_tsne:
        _snapshot(0)  # baseline (random init)
    if probe_every:
        _probe_eval(0)  # baseline (random init)

    step = 0
    stop = False
    for epoch in range(cfg.optim.epochs):
        if stop:
            break
        model.train()
        for batch in train_loader:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=cfg.training.get("amp", True),
            ):
                out = model(batch)
            out["loss"].backward()
            opt.step()
            sched.step()
            step += 1
            if is_main(rank) and step % cfg.training.get("log_every", 50) == 0:
                metrics = {k: v.detach().item() for k, v in out.items()}
                metrics["lr"] = sched.get_last_lr()[0]
                logger.info(
                    f"step {step} | "
                    + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                )
                if run is not None:
                    run.log(metrics, step=step)
            if tsne_every and step % tsne_every == 0:
                _snapshot(step)
            if probe_every and step % probe_every == 0:
                _probe_eval(step)
            if max_steps and step >= max_steps:
                stop = True
                break
        if is_main(rank) and cfg.training.get("ckpt_every_epoch", True):
            enc = model.module.encoder if is_ddp else model.encoder
            save_checkpoint(
                os.path.join(cfg.meta.run_dir, "encoder.pt"),
                enc,
                opt,
                sched.scheduler,
                epoch=epoch,
                step=step,
            )
    if is_ddp:
        dist.destroy_process_group()
    return (model.module if is_ddp else model).encoder


def run(config: str = "examples/tahoe_jepa/cfgs/train.yaml", **overrides):
    cfg = load_config(config, cli_overrides=overrides or None)
    os.makedirs(cfg.meta.run_dir, exist_ok=True)
    t0 = time.time()
    train(cfg)
    logger.info(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run})
