"""Train the Subliminal 1.4 (faithful port) cell encoder on Tahoe-100M.

Single YAML config drives everything (see cfgs/sub14.yaml). Multi-GPU via
torchrun (DDP). The SIGReg ECF is all-reduced across ranks inside the
loss, so DDP gives the correct global-batch Gaussianity test.

Recipe (all from the tuned 1.4 reference except model scale):
    - V views per cell via per-cell quantile-thermometer sampling +
      binomial count subsample,
    - sigmoid-attention encoder over protein-coding genes,
    - loss = pairwise-cosine JEPA invariance + sigreg_weight * SIGReg,
    - Muon (2-D body) + AdamW (rest), NO LR scheduler, bf16 throughout.

Usage:
    python -m examples.tahoe_jepa.sub14_main run --config examples/tahoe_jepa/cfgs/sub14.yaml
    torchrun --nproc_per_node=4 -m examples.tahoe_jepa.sub14_main run \
        --config examples/tahoe_jepa/cfgs/sub14.yaml
"""
from __future__ import annotations

import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eb_jepa.datasets.tahoe.dataset import TahoeConfig, TahoeIterableDataset
from eb_jepa.logging import get_logger
from eb_jepa.singlecell.sub14.collator import Sub14Collator
from eb_jepa.singlecell.sub14.features import load_pc_features, random_pc_features
from eb_jepa.singlecell.sub14.model import Subliminal14
from eb_jepa.singlecell.sub14.optim import build_muon_adamw_optimizer
from eb_jepa.singlecell.sub14.sigreg import SIGReg
from eb_jepa.training_utils import load_config, setup_seed

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Distributed                                                                 #
# --------------------------------------------------------------------------- #
def setup_ddp():
    if "RANK" in os.environ and torch.cuda.is_available():
        dist.init_process_group("nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    return False, 0, 1, 0


def is_main(rank: int) -> bool:
    return rank == 0


# --------------------------------------------------------------------------- #
# Loss                                                                        #
# --------------------------------------------------------------------------- #
def jepa_pair_loss(view_projections: list[torch.Tensor], kind: str = "cosine") -> torch.Tensor:
    """Average ordered-pair JEPA loss across views (1.4 reference form)."""
    pairs: list[torch.Tensor] = []
    for i in range(len(view_projections)):
        for j in range(len(view_projections)):
            if i == j:
                continue
            a, b = view_projections[i], view_projections[j]
            if kind == "cosine":
                pairs.append(1.0 - F.cosine_similarity(a, b, dim=-1).mean())
            elif kind == "mse":
                pairs.append(F.mse_loss(a, b))
            else:
                raise ValueError(f"Unknown jepa_loss kind: {kind}")
    if not pairs:
        raise ValueError("Need at least 2 views for JEPA loss")
    return torch.stack(pairs).mean()


# --------------------------------------------------------------------------- #
# Build                                                                       #
# --------------------------------------------------------------------------- #
def build_model(cfg, pc, train_dtype, device) -> Subliminal14:
    num_bins = int(cfg.data.get("num_bins", 16))
    genes_per_bin = int(cfg.data.get("genes_per_bin", 32))
    model = Subliminal14(
        n_pc_genes=pc.n_pc_genes,
        d_model=int(cfg.model.d_model),
        n_heads=int(cfg.model.n_heads),
        n_layers=int(cfg.model.n_layers),
        d_ff=int(cfg.model.d_ff),
        dropout=float(cfg.model.get("dropout", 0.1)),
        latent_dim=int(cfg.model.get("proj_dim", 128)),
        num_bins=num_bins,
        max_genes_per_cell=num_bins * genes_per_bin,
        dna_features=pc.dna_features,
        protein_features=pc.protein_features,
        freeze_features=bool(cfg.model.get("freeze_features", True)),
        attention_activation=str(cfg.model.get("attention_activation", "sigmoid")),
        grad_checkpoint=bool(cfg.model.get("grad_checkpoint", False)),
    )
    return model.to(device=device, dtype=train_dtype)


def build_loader(cfg, pc, rank, world):
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
    collator = Sub14Collator(
        token_to_pc_local=pc.token_to_pc_local,
        n_pc_genes=pc.n_pc_genes,
        num_bins=int(cfg.data.get("num_bins", 16)),
        genes_per_bin=int(cfg.data.get("genes_per_bin", 32)),
        num_views=int(cfg.data.get("n_views", 4)),
        binomial_subsample=cfg.data.get("binomial_subsample", None),
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
# Eval (rank-0 only; no collectives — uses model.encode, not the loss)        #
# --------------------------------------------------------------------------- #
def _encode_eval(model, eval_views: dict, device, train_dtype, chunk: int) -> torch.Tensor:
    """Encode the (single-view) eval batch -> pre-projection reps [N, d]."""
    gene_ids = eval_views["gene_ids"][0]
    bin_ids = eval_views["bin_ids"][0]
    pad = eval_views["padding_mask"][0]
    reps = []
    model.eval()
    with torch.no_grad():
        for s in range(0, gene_ids.size(0), chunk):
            sl = slice(s, s + chunk)
            r = model.encode(
                gene_ids[sl].to(device),
                bin_ids[sl].to(device),
                pad[sl].to(device),
            )
            reps.append(r.float().cpu())
    model.train()
    return torch.cat(reps, dim=0)


def _eval_loss(model, eval_multi, sigreg, sigreg_weight, jepa_kind, device, loss_cells):
    """Held-out LeJEPA loss on a small multi-view batch (same recipe as training).

    Mirrors the eb_jepa runs' ``eval/loss`` / ``eval/invariance_loss`` /
    ``eval/sigreg_loss`` keys so sub14 sits in the same dashboards. SIGReg's
    lock-step RNG counter is saved/restored so the eval doesn't perturb the
    training projection sequence.
    """
    if eval_multi is None:
        return {}
    saved_step = sigreg.step
    g, b, p = eval_multi["gene_ids"], eval_multi["bin_ids"], eval_multi["padding_mask"]
    nv = int(eval_multi["n_views"])
    n = min(int(loss_cells), g.size(1))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        projs, sig = [], torch.zeros((), device=device, dtype=torch.float32)
        for vw in range(nv):
            out = model(g[vw, :n].to(device), b[vw, :n].to(device), p[vw, :n].to(device))
            projs.append(out.cell_projection)
            sig = sig + sigreg(out.cell_projection).float()
        jepa = jepa_pair_loss(projs, kind=jepa_kind).float()
        siga = sig / float(nv)
        loss = jepa + sigreg_weight * siga
    if was_training:
        model.train()
    sigreg.step = saved_step
    return {
        "eval/loss": float(loss),
        "eval/invariance_loss": float(jepa),
        "eval/sigreg_loss": float(siga),
    }


def run_eval(model, eval_single, eval_meta, eval_multi, sigreg, sigreg_weight, jepa_kind,
             eval_dir, step, device, train_dtype, run, cfg):
    """Detached probes + per-class t-SNE + held-out loss, logged with the SAME
    wandb keys as the eb_jepa LeJEPA runs (probe/<key>/<metric>, repr/effective_rank,
    tsne/<class>, eval/loss) so sub14 lands in the shared dashboards."""
    from eb_jepa.singlecell.probes import run_probe_suite
    from eb_jepa.singlecell.visualize import effective_rank, plot_tsne_single, tsne_embed

    reps = _encode_eval(model, eval_single, device, train_dtype, int(cfg.eval.get("encode_chunk", 256)))
    metrics: dict = {}
    try:
        suite = run_probe_suite(reps, dict(eval_meta))  # {"clf/organ": {...}, "reg/...": {...}}
        for key, m in suite.items():
            for mk, mv in m.items():
                metrics[f"probe/{key}/{mk}"] = float(mv)
    except Exception:
        logger.warning("probe suite failed at step %d", step, exc_info=True)
    metrics["repr/effective_rank"] = float(effective_rank(reps))
    metrics.update(_eval_loss(model, eval_multi, sigreg, sigreg_weight, jepa_kind, device,
                              int(cfg.eval.get("loss_cells", 128))))

    # per-class t-SNE panels (same figures + keys as eb_jepa periodic_eval)
    paths: dict = {}
    try:
        os.makedirs(eval_dir, exist_ok=True)
        emb = tsne_embed(reps, seed=int(cfg.meta.seed), perplexity=float(cfg.eval.get("perplexity", 30.0)))
        for c in list(cfg.eval.get("classes", ["organ", "cell_line_id", "drug", "moa_fine"])):
            if c in eval_meta:
                p = os.path.join(eval_dir, f"tsne_{c}_step{step:06d}.png")
                plot_tsne_single(emb, eval_meta[c], p, name=c, step=step)
                paths[c] = p
    except Exception:
        logger.warning("t-SNE snapshot failed at step %d", step, exc_info=True)

    if run is not None:
        log = dict(metrics)
        try:
            import wandb

            for c, p in paths.items():
                log[f"tsne/{c}"] = wandb.Image(p, caption=f"step {step}")
        except Exception:
            pass
        run.log(log, step=step)

    logger.info(
        f"[eval @ {step}] effective_rank={metrics['repr/effective_rank']:.2f}"
        + (f" | eval/loss={metrics['eval/loss']:.3f}" if "eval/loss" in metrics else "")
        + "".join(
            f" | {k.split('/', 1)[-1]}={v:.3f}"
            for k, v in metrics.items()
            if k.startswith("probe/") and k.endswith("balanced_accuracy")
        )
    )


def build_eval_set(dataset, collator_cls_args, cfg):
    """Rank-0 fixed eval set. ``single`` = one deterministic full view (probes /
    t-SNE on the pre-projection rep); ``multi`` = a small V-view batch for the
    held-out LeJEPA loss (same view recipe as training)."""
    n_eval = int(cfg.eval.get("eval_cells", 0))
    items = dataset.sample_items(n_eval)
    single = Sub14Collator(
        **collator_cls_args, num_views=1, binomial_subsample=None, seed=int(cfg.meta.seed),
    )(items)
    meta = {k: single[k] for k in ("organ", "cell_line_id", "drug", "moa_fine", "sample", "plate") if k in single}
    n_loss = max(128, int(cfg.eval.get("loss_cells", 128)))
    multi = Sub14Collator(
        **collator_cls_args,
        num_views=int(cfg.data.get("n_views", 4)),
        binomial_subsample=cfg.data.get("binomial_subsample", None),
        seed=int(cfg.meta.seed),
    )(items[:n_loss])
    return single, meta, multi


# --------------------------------------------------------------------------- #
# Train                                                                       #
# --------------------------------------------------------------------------- #
def train(cfg, device=None):
    is_ddp, rank, world, local_rank = setup_ddp()
    if device is None:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    setup_seed(int(cfg.meta.seed) + rank)

    native_bf16 = bool(cfg.training.get("native_bf16", False)) and device.type == "cuda"
    train_dtype = torch.bfloat16 if native_bf16 else torch.float32

    # PC features (frozen DNA+protein, protein-coding subset)
    cache = cfg.model.get("gene_emb_cache", "random")
    if cache and cache != "random":
        pc = load_pc_features(cache)
        if is_main(rank):
            logger.info(f"loaded {pc.n_pc_genes} protein-coding genes from {cache}")
    else:
        pc = random_pc_features(n_pc=int(cfg.model.get("smoke_n_pc", 2000)))
        if is_main(rank):
            logger.warning("RANDOM PC features (no cache) — smoke/dev only.")

    loader, dataset, collator = build_loader(cfg, pc, rank, world)

    model = build_model(cfg, pc, train_dtype, device)

    # Optional warm-start from a trained Subliminal 1.4 checkpoint (shape-matched:
    # reuses the transformer body + projector + count table + [CELL] token when the
    # model is at the trained size; re-inits the Evo2/ESMC input adapter).
    warm = cfg.model.get("warmstart_checkpoint", "")
    if warm:
        if not os.path.exists(warm):
            raise FileNotFoundError(
                f"warmstart_checkpoint not found: {warm}. Stage the trained 1.4 "
                "checkpoint there first (see scripts/dalia_sub14.sbatch header)."
            )
        from eb_jepa.singlecell.sub14.load_checkpoint import load_subliminal14_checkpoint

        load_subliminal14_checkpoint(model, warm, map_location="cpu", verbose=is_main(rank))
        model = model.to(device=device, dtype=train_dtype)

    if bool(cfg.model.get("compile", False)):
        model = torch.compile(model)
    if is_ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False
        )
    raw_module = model.module if is_ddp else model
    raw_module = getattr(raw_module, "_orig_mod", raw_module)

    sigreg = SIGReg(
        num_slices=int(cfg.loss.get("sigreg_slices", 256)),
        knots=int(cfg.loss.get("sigreg_knots", 17)),
        t_max=float(cfg.loss.get("sigreg_t_max", 3.0)),
    ).to(device=device, dtype=train_dtype)

    opt = build_muon_adamw_optimizer(
        raw_module,
        muon_lr=float(cfg.optim.get("muon_lr", 2e-4)),
        adamw_lr=float(cfg.optim.get("adamw_lr", 2e-4)),
        muon_momentum=float(cfg.optim.get("muon_momentum", 0.95)),
        muon_weight_decay=float(cfg.optim.get("muon_weight_decay", 0.1)),
        muon_ns_steps=int(cfg.optim.get("muon_ns_steps", 5)),
        adamw_betas=(float(cfg.optim.get("adamw_beta1", 0.9)), float(cfg.optim.get("adamw_beta2", 0.95))),
        adamw_eps=float(cfg.optim.get("adamw_eps", 1e-8)),
        adamw_weight_decay=float(cfg.optim.get("adamw_weight_decay", 0.0)),
    )

    sigreg_weight = float(cfg.loss.get("sigreg_weight", 0.4))
    jepa_kind = str(cfg.loss.get("jepa_loss", "cosine"))
    max_grad_norm = float(cfg.optim.get("max_grad_norm", 1.0))
    max_steps = int(cfg.optim.get("max_steps", 0))
    max_minutes = float(cfg.training.get("max_minutes", 0))
    log_every = int(cfg.training.get("log_every", 25))
    ckpt_every = int(cfg.training.get("ckpt_every_steps", 1000))
    eval_enabled = bool(cfg.eval.get("enabled", False))
    eval_every = int(cfg.eval.get("eval_every", 0)) if eval_enabled else 0

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (device.type == "cuda" and not native_bf16 and cfg.training.get("amp", False))
        else nullcontext()
    )

    # wandb + eval set (rank 0)
    run = None
    eval_single = eval_meta = eval_multi = eval_dir = None
    if is_main(rank):
        if cfg.wandb.get("enabled", False):
            from eb_jepa.training_utils import setup_wandb

            if cfg.wandb.get("entity"):
                os.environ["WANDB_ENTITY"] = cfg.wandb.entity
            run = setup_wandb(cfg.wandb.project, cfg, cfg.meta.run_dir, enabled=True)
        if eval_enabled and int(cfg.eval.get("eval_cells", 0)) > 0:
            eval_dir = os.path.join(cfg.meta.run_dir, "eval")
            collator_args = dict(
                token_to_pc_local=pc.token_to_pc_local,
                n_pc_genes=pc.n_pc_genes,
                num_bins=int(cfg.data.get("num_bins", 16)),
                genes_per_bin=int(cfg.data.get("genes_per_bin", 32)),
            )
            eval_single, eval_meta, eval_multi = build_eval_set(dataset, collator_args, cfg)
            logger.info(f"eval set: {eval_single['batch_size']} cells -> {eval_dir}")

    n_params = sum(p.numel() for p in raw_module.parameters() if p.requires_grad)
    if is_main(rank):
        logger.info(f"trainable params: {n_params:,} | n_pc_genes={pc.n_pc_genes} | dtype={train_dtype}")

    def _eval(step):
        if is_main(rank) and eval_single is not None:
            run_eval(raw_module, eval_single, eval_meta, eval_multi, sigreg, sigreg_weight,
                     jepa_kind, eval_dir, step, device, train_dtype, run, cfg)
        if is_ddp:
            dist.barrier()

    def _save(step, tag="encoder"):
        if not is_main(rank):
            return
        os.makedirs(cfg.meta.run_dir, exist_ok=True)
        state = {k.replace("_orig_mod.", "", 1): v for k, v in raw_module.state_dict().items()}
        torch.save(
            {"model": state, "optimizer": opt.state_dict(), "step": step, "n_pc_genes": pc.n_pc_genes},
            os.path.join(cfg.meta.run_dir, f"{tag}.pt"),
        )
        logger.info(f"  -> saved {tag}.pt @ step {step}")

    _eval(0)  # baseline (random init)

    loop_start = time.time()
    step = 0
    stop = False
    for epoch in range(int(cfg.optim.epochs)):
        if stop:
            break
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        model.train()
        for batch in loader:
            gene_ids = batch["gene_ids"].to(device, non_blocking=True)
            bin_ids = batch["bin_ids"].to(device, non_blocking=True)
            pad = batch["padding_mask"].to(device, non_blocking=True)
            n_views = int(batch["n_views"])

            opt.zero_grad(set_to_none=True)
            with autocast:
                projs: list[torch.Tensor] = []
                sigreg_total = torch.zeros((), device=device, dtype=train_dtype)
                for vw in range(n_views):
                    out = model(gene_ids[vw], bin_ids[vw], pad[vw])
                    projs.append(out.cell_projection)
                    sigreg_total = sigreg_total + sigreg(out.cell_projection)
                jepa = jepa_pair_loss(projs, kind=jepa_kind)
                sigreg_avg = sigreg_total / float(n_views)
                loss = jepa + sigreg_weight * sigreg_avg

            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(raw_module.parameters(), max_grad_norm)
            opt.step()
            step += 1

            if is_main(rank) and step % log_every == 0:
                elapsed = max(time.time() - loop_start, 1e-9)
                cells = step * int(cfg.data.batch_size) * world
                g_per_view = int(cfg.data.get("num_bins", 16)) * int(cfg.data.get("genes_per_bin", 32))
                # Same metric keys as the eb_jepa LeJEPA runs so sub14 shares the
                # dashboards: loss / invariance_loss / sigreg_loss / lr / data,throughput.
                m = {
                    "loss": float(loss.detach().item()),
                    "invariance_loss": float(jepa.detach().item()),
                    "sigreg_loss": float(sigreg_avg.detach().item()),
                    "lr": float(cfg.optim.get("adamw_lr", 2e-4)),  # constant (no scheduler)
                    "epoch": epoch,
                    "data/cells_seen": cells,
                    "data/tokens_seen": cells * n_views * g_per_view,
                    "throughput/cells_per_s": cells / elapsed,
                }
                logger.info(
                    f"step {step} | loss={m['loss']:.4f} inv={m['invariance_loss']:.4f} "
                    f"sigreg={m['sigreg_loss']:.3f} | {m['throughput/cells_per_s']:.0f} cells/s"
                )
                if run is not None:
                    run.log(m, step=step)

            if eval_every and step % eval_every == 0:
                _eval(step)
            if ckpt_every and step % ckpt_every == 0:
                _save(step)
            if max_steps and step >= max_steps:
                stop = True
                break
            if max_minutes and (time.time() - loop_start) / 60.0 >= max_minutes:
                stop = True
                break

    if step > 0:
        _eval(step)
    _save(step, tag="encoder_final")
    if is_ddp:
        dist.destroy_process_group()
    return raw_module


def run(config: str = "examples/tahoe_jepa/cfgs/sub14.yaml", **overrides):
    cfg = load_config(config, cli_overrides=overrides or None)
    os.makedirs(cfg.meta.run_dir, exist_ok=True)
    t0 = time.time()
    train(cfg)
    logger.info(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run})
