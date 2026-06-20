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


def measure_encoder_flops(encoder, batch) -> int:
    """Forward FLOPs of the encoder over the V views of one (local) batch.

    Measured with torch's FlopCounterMode (counts matmuls + SDPA attention) on the
    trained backbone only (the frozen ESMC/Evo2 gather adds none). Training FLOPs per
    step ≈ 3× this (forward + backward) × world_size — the scaling-law "trained
    parts only" budget (CLAUDE.md Objectives).
    """
    from torch.utils.flop_counter import FlopCounterMode

    enc = getattr(encoder, "_orig_mod", encoder)  # unwrap torch.compile if present
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        encode_views(enc, batch)
    return counter.get_total_flops()


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
    collator = train_loader.collate_fn

    # Held-out probe/eval split (cell-level, seeded): these cells are EXCLUDED from
    # SSL training so probe metrics measure generalization, not memorization. The
    # remainder is the SSL training set (a DistributedSampler shards it across ranks).
    from torch.utils.data import DataLoader, Subset

    eval_enabled = bool(cfg.get("eval", {}).get("enabled", False))
    n_eval = int(cfg.get("eval", {}).get("eval_cells", 0)) if eval_enabled else 0
    n_eval = max(0, min(n_eval, len(dataset) - cfg.data.batch_size * max(world, 1)))
    g_split = torch.Generator().manual_seed(cfg.meta.seed)
    perm = torch.randperm(len(dataset), generator=g_split).tolist()
    eval_idx, train_idx = perm[:n_eval], perm[n_eval:]
    train_subset = Subset(dataset, train_idx)

    loader_kwargs = dict(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_mem,
        drop_last=True,
        collate_fn=collator,
    )
    sampler = None
    if is_ddp:
        from torch.utils.data import DistributedSampler

        sampler = DistributedSampler(
            train_subset, num_replicas=world, rank=rank, shuffle=True, drop_last=True
        )
        train_loader = DataLoader(train_subset, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)

    # held-out eval set (rank 0): the excluded cells, one clean full-gene view each
    eval_batch = eval_labels = eval_dir = None
    do_eval = is_main(rank) and eval_enabled and n_eval > 0
    if do_eval:
        from examples.tahoe_jepa.eval_tsne import build_eval_set

        eval_batch, eval_labels = build_eval_set(dataset, data_cfg, idx=eval_idx)
        eval_dir = os.path.join(cfg.meta.run_dir, "eval")
        logger.info(
            f"held-out eval: {len(eval_idx)} cells | SSL train: {len(train_idx)} -> {eval_dir}"
        )

    # model
    model = build_train_module(cfg).to(device)
    if cfg.model.get("compile", False):
        model.encoder = torch.compile(model.encoder)
    if is_ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False
        )
    raw_module = model.module if is_ddp else model
    raw_encoder = getattr(raw_module.encoder, "_orig_mod", raw_module.encoder)

    # optim
    opt = torch.optim.AdamW(
        param_groups(model, cfg.optim.weight_decay),
        lr=cfg.optim.lr,
        betas=tuple(cfg.optim.get("betas", (0.9, 0.95))),
    )
    max_steps = int(cfg.optim.get("max_steps", 0))
    # size the LR schedule to the actual run length (max_steps caps it)
    total_steps = (
        max_steps if max_steps > 0 else cfg.optim.epochs * max(1, len(train_loader))
    )
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

    def _eval(step: int):
        from examples.tahoe_jepa.eval_tsne import periodic_eval

        metrics, path = periodic_eval(
            raw_encoder,
            eval_batch,
            eval_labels,
            eval_dir,
            step,
            device,
            run=run,
            classes=list(
                cfg.eval.get("classes", ["organ", "cell_line_id", "drug", "moa_fine"])
            ),
            chunk=int(cfg.eval.get("encode_chunk", 128)),
            perplexity=float(cfg.eval.get("perplexity", 30.0)),
            seed=cfg.meta.seed,
            amp=cfg.training.get("amp", True),
        )
        key = {
            k: v
            for k, v in metrics.items()
            if k.endswith(("balanced_accuracy", "r2", "effective_rank"))
        }
        logger.info(
            f"[eval @ {step}] "
            + " | ".join(f"{k.split('/', 1)[-1]}={v:.3f}" for k, v in key.items())
            + f" -> {path}"
        )

    # eval cadence is the SAME on every rank (so all reach the barrier together);
    # only rank 0 actually runs the eval.
    eval_every = (
        int(
            cfg.get("eval", {}).get(
                "eval_every", cfg.get("eval", {}).get("tsne_every", 0)
            )
        )
        if eval_enabled
        else 0
    )
    if do_eval:
        _eval(0)  # baseline (random init)
    if is_ddp:
        dist.barrier()  # other ranks wait for rank 0's baseline eval

    # FLOP accounting (trained backbone) + wall-clock budget
    n_params = sum(p.numel() for p in raw_module.parameters() if p.requires_grad)
    max_minutes = float(cfg.training.get("max_minutes", 0))
    flops_per_step = None  # global FLOPs/step (fwd+bwd, all ranks); set on first batch
    cumulative_flops = 0.0
    loop_start = time.time()

    step = 0
    stop = False
    for epoch in range(cfg.optim.epochs):
        if stop:
            break
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            batch = move_batch(batch, device)
            if flops_per_step is None:
                fwd = measure_encoder_flops(raw_encoder, batch)  # per-rank fwd FLOPs
                global_fwd = fwd
                if is_ddp:
                    t = torch.tensor(float(fwd), device=device)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)  # literal sum over GPUs
                    global_fwd = t.item()
                # fwd+bwd (~3x) summed across all GPUs = global training FLOPs/step
                flops_per_step = 3.0 * global_fwd
                if is_main(rank):
                    logger.info(
                        f"encoder fwd FLOPs/rank={fwd:.3e} | "
                        f"train FLOPs/step (global sum)={flops_per_step:.3e} | "
                        f"trainable params={n_params:,}"
                    )
                    if run is not None:
                        run.log(
                            {
                                "flops/fwd_per_rank": fwd,
                                "flops/per_step_global": flops_per_step,
                                "model/trainable_params": n_params,
                            },
                            step=step,
                        )
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
            cumulative_flops += flops_per_step
            if is_main(rank) and step % cfg.training.get("log_every", 50) == 0:
                elapsed = max(time.time() - loop_start, 1e-9)
                loss_m = {k: v.detach().item() for k, v in out.items()}
                lr = sched.get_last_lr()[0]
                cells_seen = step * cfg.data.batch_size * world
                metrics = {
                    **loss_m,
                    "lr": lr,
                    "epoch": epoch,
                    "data/cells_seen": cells_seen,
                    "data/tokens_seen": cells_seen * cfg.data.n_views * cfg.data.L,
                    "flops/cumulative": cumulative_flops,
                    "flops/pflops_cumulative": cumulative_flops / 1e15,
                    "flops/tflops_per_s": (cumulative_flops / 1e12) / elapsed,
                    "throughput/cells_per_s": cells_seen / elapsed,
                }
                logger.info(
                    f"step {step} | loss={loss_m['loss']:.4f} "
                    f"sigreg={loss_m['sigreg_loss']:.3f} inv={loss_m['invariance_loss']:.4f} "
                    f"lr={lr:.2e} | {metrics['flops/pflops_cumulative']:.3f} PFLOP "
                    f"| {metrics['flops/tflops_per_s']:.1f} TFLOP/s "
                    f"| {metrics['throughput/cells_per_s']:.0f} cells/s"
                )
                if run is not None:
                    run.log(metrics, step=step)
            if eval_every and step % eval_every == 0:
                if do_eval:  # rank 0 runs probes + t-SNE on the held-out set
                    _eval(step)
                if is_ddp:
                    dist.barrier()  # other ranks wait while rank 0 evaluates
            if max_steps and step >= max_steps:
                stop = True
                break
            if max_minutes and (time.time() - loop_start) / 60.0 >= max_minutes:
                stop = True
                break
        if is_main(rank) and cfg.training.get("ckpt_every_epoch", True):
            save_checkpoint(
                os.path.join(cfg.meta.run_dir, "encoder.pt"),
                raw_encoder,
                opt,
                sched.scheduler,
                epoch=epoch,
                step=step,
            )
    # always save the final encoder (max_steps/max_minutes stop mid-epoch) so the
    # trained backbone is available for post-hoc probing / t-SNE / scaling laws.
    if is_main(rank):
        save_checkpoint(
            os.path.join(cfg.meta.run_dir, "encoder_final.pt"), raw_encoder, step=step
        )
    if is_ddp:
        dist.destroy_process_group()
    return raw_encoder


def run(config: str = "examples/tahoe_jepa/cfgs/train.yaml", **overrides):
    cfg = load_config(config, cli_overrides=overrides or None)
    os.makedirs(cfg.meta.run_dir, exist_ok=True)
    t0 = time.time()
    train(cfg)
    logger.info(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run})
