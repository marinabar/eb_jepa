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
    embed_norm = cfg.model.get("embed_norm", "none")
    if cache and cache != "random":
        return GeneTokenEmbedding.from_cache(
            cache, cfg.model.d_model, cfg.data.count_mode, cfg.data.n_bins, embed_norm
        )
    logger.warning("Using RANDOM gene embeddings (no cache) — smoke/dev only.")
    return GeneTokenEmbedding.random(
        cfg.data.n_genes,
        cfg.model.d_model,
        count_mode=cfg.data.count_mode,
        n_bins=cfg.data.n_bins,
        embed_norm=embed_norm,
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
        n_pathways=cfg.model.get("n_pathways", 0),
    )
    projector = Projector(
        f"{cfg.model.d_model}-{cfg.model.proj_hidden}-{cfg.model.proj_dim}",
        norm=cfg.model.get("proj_norm", "bn"),
    )
    loss_fn = LeJEPALoss(
        projector=projector,
        lamb=cfg.loss.lamb,
        num_slices=cfg.loss.num_slices,
        knots=cfg.loss.get("knots", 17),
        t_max=cfg.loss.get("t_max", 3.0),
        repr_var_weight=cfg.loss.get("repr_var_weight", 0.0),
        repr_cov_weight=cfg.loss.get("repr_cov_weight", 0.0),
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
        rank=rank,
        world_size=world,
    )
    collator = train_loader.collate_fn

    streaming = bool(getattr(data_cfg, "streaming", False))
    eval_enabled = bool(cfg.get("eval", {}).get("enabled", False))
    n_eval = int(cfg.get("eval", {}).get("eval_cells", 0)) if eval_enabled else 0
    sampler = None
    eval_batch = eval_labels = eval_dir = None
    eval_loss_batch = None  # V-view held-out batch for the eval-set LeJEPA loss

    if streaming:
        # init_tahoe_data already shards the stream across rank/workers (no
        # DistributedSampler). The eval set is a fixed, diverse sample read from the
        # shards (not held out — an infinite stream has no stable index space).
        do_eval = is_main(rank) and eval_enabled and n_eval > 0
        if do_eval:
            from examples.tahoe_jepa.eval_tsne import build_eval_set

            eval_batch, eval_labels = build_eval_set(
                dataset, data_cfg, n_eval, seed=cfg.meta.seed,
                membership=collator.membership,
            )
            eval_dir = os.path.join(cfg.meta.run_dir, "eval")
            logger.info(f"streaming eval set: {n_eval} cells -> {eval_dir}")
    else:
        # Held-out probe/eval split (cell-level, seeded): these cells are EXCLUDED
        # from SSL training so probe metrics measure generalization, not memorization.
        from torch.utils.data import DataLoader, Subset

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
        if is_ddp:
            from torch.utils.data import DistributedSampler

            sampler = DistributedSampler(
                train_subset,
                num_replicas=world,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
            train_loader = DataLoader(train_subset, sampler=sampler, **loader_kwargs)
        else:
            train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)

        # V-view held-out batch for the eval LeJEPA loss — built on ALL ranks, because
        # the loss's SIGReg all-reduces across ranks; a rank-0-only eval-loss would
        # deadlock (other ranks sit at the barrier, never joining the collective).
        # Each rank takes a DISJOINT shard of the (seeded, rank-identical) held-out
        # set so the all-reduced SIGReg sees a true global batch of loss_cells*world
        # distinct cells, not the same loss_cells replicated across ranks.
        if eval_enabled and n_eval > 0:
            loss_shard = eval_idx[rank::world]
            n_loss = min(int(cfg.eval.get("loss_cells", 128)), len(loss_shard))
            eval_loss_batch = move_batch(
                collator([dataset[i] for i in loss_shard[:n_loss]]), device
            )

        do_eval = is_main(rank) and eval_enabled and n_eval > 0
        if do_eval:  # rank 0 owns the probe + t-SNE eval set
            from examples.tahoe_jepa.eval_tsne import build_eval_set

            eval_batch, eval_labels = build_eval_set(
                dataset, data_cfg, idx=eval_idx, membership=collator.membership
            )
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

    # Optional resume / warm-start from a full checkpoint (ckpt_latest.pt). With
    # resume_optim=True the optimizer+scheduler+step are restored (exact continuation);
    # with resume_optim=False only the weights load and a FRESH schedule starts from
    # step 0 — i.e. reuse the checkpoint under any new learning-rate init.
    start_step = 0
    resume_path = cfg.training.get("resume", "")
    if resume_path:
        from eb_jepa.training_utils import load_checkpoint

        resume_optim = bool(cfg.training.get("resume_optim", True))
        info = load_checkpoint(
            resume_path,
            raw_module,
            optimizer=opt if resume_optim else None,
            scheduler=sched.scheduler if resume_optim else None,
            device=device,
            strict=False,
        )
        start_step = int(info.get("step", 0)) if resume_optim else 0
        logger.info(
            f"resumed from {resume_path} @ saved step {info.get('step', 0)} "
            f"(optimizer/scheduler {'restored' if resume_optim else 'fresh — new LR'})"
        )

    amp_dtype = torch.bfloat16 if cfg.training.get("amp", True) else torch.float32
    run = None
    if is_main(rank) and cfg.wandb.get("enabled", False):
        from eb_jepa.training_utils import setup_wandb

        if cfg.wandb.get("entity"):
            os.environ["WANDB_ENTITY"] = cfg.wandb.entity  # team; key stays in ~/.netrc
        run = setup_wandb(cfg.wandb.project, cfg, cfg.meta.run_dir, enabled=True)

    # best-checkpoint trackers: save the best encoder for the global (train) loss and
    # for the held-out eval loss, refreshed at every eval.
    best = {"train": float("inf"), "eval": float("inf")}

    def _eval_loss():
        """LeJEPA loss on the held-out V-view batch (same loss as training).

        Called by ALL ranks (the loss's SIGReg all-reduces -> every rank must join).
        Save/restore SIGReg.step so the eval doesn't perturb the training projection
        sequence (all ranks advance+restore identically -> stay lock-step).
        """
        if eval_loss_batch is None:
            return None
        sig = raw_module.loss_fn.sigreg
        saved_step = sig.step
        was_training = raw_encoder.training
        raw_encoder.eval()
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=cfg.training.get("amp", True),
            ),
        ):
            out = raw_module.loss_fn(encode_views(raw_encoder, eval_loss_batch))
        if was_training:
            raw_encoder.train()
        sig.step = saved_step  # keep DDP lock-step
        return {f"eval/{k}": float(v) for k, v in out.items()}

    def _save_best(tag: str, value: float, step: int):
        save_checkpoint(
            os.path.join(cfg.meta.run_dir, f"best_{tag}.pt"), raw_encoder, step=step
        )
        logger.info(
            f"  ↳ new best {tag} loss={value:.4f} @ step {step} (best_{tag}.pt)"
        )

    def _eval(step: int, train_loss: float | None, eloss: dict | None):
        """Rank-0 probes + t-SNE + best-checkpoint logging. ``eloss`` is the global
        held-out LeJEPA loss already computed collectively by all ranks."""
        from examples.tahoe_jepa.eval_tsne import periodic_eval

        metrics, paths = periodic_eval(
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
        if eloss is not None:
            metrics.update(eloss)
            if run is not None:
                run.log(eloss, step=step)
        key = {
            k: v
            for k, v in metrics.items()
            if k.endswith(("balanced_accuracy", "r2", "effective_rank"))
            or k == "eval/loss"
        }
        logger.info(
            f"[eval @ {step}] "
            + " | ".join(f"{k.split('/', 1)[-1]}={v:.3f}" for k, v in key.items())
            + f" -> {len(paths)} t-SNE panels in {eval_dir}"
        )
        # best-of checkpoints (held-out eval loss + global train loss)
        if eloss is not None and eloss["eval/loss"] < best["eval"]:
            best["eval"] = eloss["eval/loss"]
            _save_best("eval", best["eval"], step)
        if train_loss is not None and train_loss < best["train"]:
            best["train"] = train_loss
            _save_best("train", best["train"], step)

    def _run_eval(step: int, train_loss: float | None = None):
        # eval loss runs on ALL ranks (collective SIGReg all-reduce -> no deadlock);
        # probes + t-SNE + checkpoints on rank 0; barrier so the others wait for it.
        eloss = _eval_loss()
        if do_eval:
            _eval(step, train_loss, eloss)
        if is_ddp:
            dist.barrier()

    # eval cadence is the SAME on every rank (so all reach the collective together)
    eval_every = (
        int(
            cfg.get("eval", {}).get(
                "eval_every", cfg.get("eval", {}).get("tsne_every", 0)
            )
        )
        if eval_enabled
        else 0
    )
    _run_eval(0)  # baseline (random init)

    # FLOP accounting (trained backbone) + wall-clock budget
    n_params = sum(p.numel() for p in raw_module.parameters() if p.requires_grad)
    max_minutes = float(cfg.training.get("max_minutes", 0))
    flops_per_step = None  # global FLOPs/step (fwd+bwd, all ranks); set on first batch
    cumulative_flops = 0.0
    loop_start = time.time()

    # Periodic FULL checkpoint (model + optimizer + scheduler + step) every N steps ->
    # rolling ckpt_latest.pt, so the run is reusable/resumable at any time regardless
    # of the LR schedule. 0 disables. Frozen ESMC/Evo2 tables are persistent=False, so
    # these stay small (learned params + AdamW state only).
    ckpt_every = int(cfg.training.get("ckpt_every", 0))

    def _save_full(step: int, epoch: int):
        save_checkpoint(
            os.path.join(cfg.meta.run_dir, "ckpt_latest.pt"),
            raw_module,
            opt,
            sched.scheduler,
            epoch=epoch,
            step=step,
        )

    step = start_step
    stop = False
    for epoch in range(cfg.optim.epochs):
        if stop:
            break
        if sampler is not None:
            sampler.set_epoch(epoch)
        elif streaming and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)  # reshuffle shard order per epoch
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
                _run_eval(step, train_loss=out["loss"].item())
            if is_main(rank) and ckpt_every and step % ckpt_every == 0:
                _save_full(step, epoch)
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
    # guaranteed final eval (a clean last scaling-law point) before saving
    if step > 0:
        _run_eval(step, train_loss=out["loss"].item())
    # always save the final encoder (max_steps/max_minutes stop mid-epoch) so the
    # trained backbone is available for post-hoc probing / t-SNE / scaling laws, plus a
    # final full checkpoint (resumable) for continuing the run later.
    if is_main(rank):
        save_checkpoint(
            os.path.join(cfg.meta.run_dir, "encoder_final.pt"), raw_encoder, step=step
        )
        if ckpt_every:
            _save_full(step, cfg.optim.epochs)
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
