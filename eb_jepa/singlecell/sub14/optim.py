"""Muon + AdamW hybrid optimizer (faithful port of Subliminal 1.4).

Routes 2-D body weights to :class:`torch.optim.Muon` (PyTorch >= 2.9,
``adjust_lr_fn="match_rms_adamw"`` so muon_lr can equal adamw_lr per the
Moonshot/Kimi recommendation) and everything else (1-D norms/biases,
``nn.Embedding`` tables, scalars) to AdamW. A :class:`CompositeOptimizer`
drives both through one ``Optimizer``-shaped object.

If ``torch.optim.Muon`` is unavailable (torch < 2.9, e.g. a local dev
box), :func:`build_muon_adamw_optimizer` falls back to a single AdamW
over all params and logs a warning — the Dalia venv (torch 2.11) uses
real Muon.

References:
    - Liu et al., "Muon is Scalable for LLM Training", arXiv:2502.16982.
    - https://kellerjordan.github.io/posts/muon/
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from eb_jepa.logging import get_logger

logger = get_logger(__name__)


def partition_params_for_muon(
    model: nn.Module, *, extra_adamw_param_names: Iterable[str] = ()
) -> tuple[list[Tensor], list[Tensor]]:
    """Split params into ``(muon_params, adamw_params)``.

    Muon: every 2-D weight not in an ``nn.Embedding`` and not excluded by
    name. AdamW: 1-D tensors, embeddings, scalars, and named exclusions.
    """
    extra_excluded = set(extra_adamw_param_names)
    embedding_param_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            embedding_param_ids.update(id(p) for p in module.parameters())

    muon_params: list[Tensor] = []
    adamw_params: list[Tensor] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name in extra_excluded:
            adamw_params.append(p)
        elif id(p) in embedding_param_ids:
            adamw_params.append(p)
        elif p.ndim == 2:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


class CompositeOptimizer(torch.optim.Optimizer):
    """Drives multiple inner optimisers through one ``Optimizer``-like API."""

    def __init__(self, optimizers: list[torch.optim.Optimizer], names: list[str]) -> None:
        if len(optimizers) != len(names):
            raise ValueError("optimizers and names must have the same length")
        if not optimizers:
            raise ValueError("CompositeOptimizer requires at least one inner optimizer")
        self._optimizers = optimizers
        self._names = names
        for opt, name in zip(optimizers, names, strict=True):
            for group in opt.param_groups:
                group.setdefault("_optim_kind", name)

    @property
    def optimizers(self) -> list[torch.optim.Optimizer]:
        return list(self._optimizers)

    @property
    def names(self) -> list[str]:
        return list(self._names)

    @property
    def param_groups(self) -> list[dict[str, Any]]:  # type: ignore[override]
        groups: list[dict[str, Any]] = []
        for opt in self._optimizers:
            groups.extend(opt.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        for opt in self._optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any | None = None) -> Any:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self._optimizers:
            opt.step()
        return loss

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        return {
            "names": list(self._names),
            "states": [opt.state_dict() for opt in self._optimizers],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # type: ignore[override]
        names = state_dict.get("names", self._names)
        states = state_dict.get("states", [])
        if len(states) != len(self._optimizers):
            raise ValueError(
                f"state_dict has {len(states)} sub-states but optimizer has "
                f"{len(self._optimizers)} sub-optimizers"
            )
        for opt, sub_state, expected_name, actual_name in zip(
            self._optimizers, states, self._names, names, strict=True
        ):
            if expected_name != actual_name:
                raise ValueError(
                    f"name mismatch on resume: expected {expected_name!r}, got {actual_name!r}"
                )
            opt.load_state_dict(sub_state)


def build_muon_adamw_optimizer(
    model: nn.Module,
    *,
    muon_lr: float,
    adamw_lr: float,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.1,
    muon_ns_steps: int = 5,
    muon_adjust_lr_fn: str = "match_rms_adamw",
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    adamw_eps: float = 1e-8,
    adamw_weight_decay: float = 0.0,
    extra_adamw_param_names: Iterable[str] = (),
) -> CompositeOptimizer:
    """Build a Muon + AdamW composite optimizer for ``model``."""
    muon_cls = getattr(torch.optim, "Muon", None)
    if muon_cls is None:
        logger.warning(
            "torch.optim.Muon unavailable (torch %s < 2.9) — falling back to "
            "AdamW over all params. Use torch >= 2.9 (Dalia venv) for the "
            "real Muon body optimiser.",
            torch.__version__,
        )
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=adamw_lr,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=adamw_weight_decay,
        )
        return CompositeOptimizer([opt], ["adamw"])

    muon_params, adamw_params = partition_params_for_muon(
        model, extra_adamw_param_names=extra_adamw_param_names
    )
    optimizers: list[torch.optim.Optimizer] = []
    names: list[str] = []
    if muon_params:
        optimizers.append(
            muon_cls(
                muon_params,
                lr=muon_lr,
                momentum=muon_momentum,
                weight_decay=muon_weight_decay,
                ns_steps=muon_ns_steps,
                adjust_lr_fn=muon_adjust_lr_fn,
            )
        )
        names.append("muon")
    if adamw_params:
        optimizers.append(
            torch.optim.AdamW(
                adamw_params,
                lr=adamw_lr,
                betas=adamw_betas,
                eps=adamw_eps,
                weight_decay=adamw_weight_decay,
            )
        )
        names.append("adamw")
    if not optimizers:
        raise ValueError("Model has no trainable parameters")
    return CompositeOptimizer(optimizers, names)
