"""Intuitive physics — video-JEPA training entrypoint.

The DATA (procedural bouncing MNIST, in ``stimuli.py``) and the TRAINING LOOP are
provided, reusing the eb_jepa core. The scientific exercise is the
violation-of-expectation PROBE in ``eval.py`` (a ``# TODO``): compute the per-clip
latent prediction energy (``predcost``) and compare it on matched plausible vs
impossible clips.

Run:  python -m examples.intuitive_physics.main --fname examples/intuitive_physics/cfgs/train.yaml
"""
from pathlib import Path

import fire
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.architectures import Projector, ResNet5, ResUNet, StateOnlyPredictor
from eb_jepa.jepa import JEPA
from eb_jepa.logging import get_logger
from eb_jepa.losses import SquareLossSeq, VCLoss
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_exp_name,
    get_unified_experiment_dir,
    load_config,
    log_config,
    log_epoch,
    save_checkpoint,
    setup_device,
    setup_seed,
    setup_wandb,
)
from examples.intuitive_physics.stimuli import ProceduralBouncingMNIST

logger = get_logger(__name__)


def build_jepa(cfg, device):
    """Video-JEPA: ResNet5 encoder + ResUNet predictor + VICReg + SquareLossSeq."""
    encoder = ResNet5(cfg.model.dobs, cfg.model.henc, cfg.model.dstc)
    predictor = StateOnlyPredictor(
        ResUNet(2 * cfg.model.dstc, cfg.model.hpre, cfg.model.dstc), context_length=2)
    projector = Projector(f"{cfg.model.dstc}-{cfg.model.dstc * 4}-{cfg.model.dstc * 4}")
    regularizer = VCLoss(cfg.loss.std_coeff, cfg.loss.cov_coeff, proj=projector)
    return JEPA(encoder, encoder, predictor, regularizer, SquareLossSeq(projector)).to(device)


def run(fname="examples/intuitive_physics/cfgs/train.yaml", cfg=None, folder=None, **overrides):
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)

    if folder is not None:
        exp_dir = Path(folder)
    elif cfg.meta.get("model_folder"):
        exp_dir = Path(cfg.meta.model_folder)
    else:
        exp_dir = get_unified_experiment_dir(
            example_name="intuitive_physics",
            sweep_name=get_default_dev_name(),
            exp_name=get_exp_name("intuitive_physics", cfg),
            seed=cfg.meta.seed,
        )
    exp_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = setup_wandb(
        project="eb_jepa",
        config={"example": "intuitive_physics", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir=exp_dir,
        run_name=exp_dir.name,
        tags=["intuitive_physics", f"seed_{cfg.meta.seed}"],
        enabled=cfg.logging.log_wandb,
    )

    train_set = ProceduralBouncingMNIST(split="train", n_samples=cfg.data.n_train,
                                        T=cfg.data.T, seed=cfg.meta.seed)
    loader = DataLoader(train_set, batch_size=cfg.data.batch_size, shuffle=True,
                        num_workers=cfg.data.num_workers, drop_last=True)

    jepa = build_jepa(cfg, device)
    optimizer = Adam(jepa.parameters(), lr=cfg.optim.lr)
    log_config(cfg)
    logger.info(f"Training {cfg.optim.epochs} epochs on {len(train_set)} clips -> {exp_dir}")

    global_step = 0
    for epoch in range(cfg.optim.epochs):
        jepa.train()
        last = {}
        for batch in tqdm(loader, desc=f"epoch {epoch}",
                          disable=cfg.logging.get("tqdm_silent", True)):
            x = batch["video"].to(device)
            optimizer.zero_grad()
            _, (loss, regl, _, regd, pl) = jepa.unroll(
                x, actions=None, nsteps=cfg.model.steps, unroll_mode="parallel",
                compute_loss=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            last = {"loss": loss.item(), "vc": regl.item(), "pred": pl.item(), **regd}

        # Watch the VICReg std term: if it stays high while pred -> 0, the encoder
        # is collapsing and the energy gap becomes meaningless.
        metrics = {"train/loss": last["loss"], "train/vc_loss": last["vc"],
                   "train/pred_loss": last["pred"], "train/std_loss": last["std_loss"],
                   "train/cov_loss": last["cov_loss"]}
        if wandb_run:
            import wandb
            wandb.log(metrics, step=global_step)
        log_epoch(epoch, {"loss": last["loss"], "vc": last["vc"], "pred": last["pred"]},
                  total_epochs=cfg.optim.epochs)
        save_checkpoint(exp_dir / "latest.pth.tar", model=jepa, optimizer=optimizer,
                        epoch=epoch, step=global_step)

    if wandb_run:
        import wandb
        wandb.finish()
    logger.info(f"done -> {exp_dir}/latest.pth.tar  (now run eval.py to probe the energy gap)")


if __name__ == "__main__":
    fire.Fire(run)
