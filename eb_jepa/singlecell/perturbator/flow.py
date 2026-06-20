"""Conditional / rectified flow matching for the perturbator (CLAUDE.md Part II).

We have no paired control/perturbed cells, only the two distributions per stratum
``(cell_line, plate, drug, dose)``. Rectified flow matching learns a velocity field
``v_theta(x_t, t, action)`` that transports the source (control) latent distribution
to the target (treated) distribution:

1. draw a source latent ``x0`` (a control cell of the stratum) and an *independent*
   target latent ``x1`` (a treated cell of the stratum) — an unpaired random
   coupling, the rectified-flow choice;
2. sample a time ``t ~ U(0, 1)`` and form the linear interpolant
   ``x_t = (1 - t) * x0 + t * x1``;
3. regress the velocity to the constant displacement of that straight path:
   ``loss = || v_theta(x_t, t, action) - (x1 - x0) ||^2``.

At inference the perturbed latent is obtained by integrating the learned ODE
``dx/dt = v_theta(x, t, action)`` from ``t=0`` (source latent) to ``t=1`` with a
rigorous fixed-step solver (Euler or Heun / explicit midpoint). With the zero-init
head the velocity is zero at init, so the ODE is the identity map at init.
"""

from __future__ import annotations

import torch

from eb_jepa.singlecell.perturbator.model import Perturbator


def _couple_source_target(
    source: torch.Tensor, target: torch.Tensor, generator: torch.Generator | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a random unpaired coupling of equal length ``n = max(|src|, |tgt|)``.

    Rectified flow uses an arbitrary (independent) coupling of the two marginals.
    The smaller cloud is resampled **with replacement** to ``n`` so every step sees a
    balanced batch; both are shuffled independently so the pairing is random.
    """
    ns, nt = source.shape[0], target.shape[0]
    n = max(ns, nt)
    idx_s = torch.randint(ns, (n,), generator=generator, device=source.device)
    idx_t = torch.randint(nt, (n,), generator=generator, device=target.device)
    return source[idx_s], target[idx_t]


def flow_matching_loss(
    model: Perturbator,
    source: torch.Tensor,
    target: torch.Tensor,
    action: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Rectified conditional-flow-matching loss for one stratum.

    Args:
        model: a ``Perturbator`` built with ``time_conditioned=True``.
        source: ``[ns, d]`` source (control) latents.
        target: ``[nt, d]`` target (treated) latents (detached — no grad to encoder).
        action: ``[action_dim]`` (broadcast) or ``[n, action_dim]`` action vector.
        generator: optional RNG for the coupling / time draw (reproducible).
    Returns:
        Scalar MSE between the predicted velocity and the straight-path displacement.
    """
    x0, x1 = _couple_source_target(source, target.detach(), generator)
    n = x0.shape[0]
    t = torch.rand(n, generator=generator, device=x0.device, dtype=x0.dtype)
    x_t = (1.0 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * x1
    v_target = x1 - x0
    v_pred = model.velocity(x_t, t, action)
    return (v_pred - v_target).square().mean()


@torch.no_grad()
def ode_sample(
    model: Perturbator,
    source: torch.Tensor,
    action: torch.Tensor,
    n_steps: int = 20,
    method: str = "heun",
    t0: float = 0.0,
    t1: float = 1.0,
    return_path: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Integrate ``dx/dt = v_theta(x, t, action)`` from ``t0`` to ``t1``.

    Args:
        model: a flow-matching ``Perturbator`` (``time_conditioned=True``).
        source: ``[N, d]`` source (control) latents = the initial state at ``t0``.
        action: ``[action_dim]`` or ``[N, action_dim]`` action vector.
        n_steps: number of fixed integration steps.
        method: ``"euler"`` (1st order) or ``"heun"`` / ``"midpoint"`` (2nd order).
        t0, t1: integration interval (default the full path ``[0, 1]``).
        return_path: if True, also return the full integrated path
            ``[n_steps + 1, N, d]`` — the state at every integration time
            ``t0 .. t1`` (``path[0] == source``, ``path[-1] == final state``).
    Returns:
        ``[N, d]`` predicted perturbed latents (the state at ``t1``); or, when
        ``return_path``, the tuple ``(final [N, d], path [n_steps + 1, N, d])``.

    With ``method="euler"`` and ``n_steps=1`` over ``[0, 1]`` this reduces to a single
    Euler step ``source + v(source, 0)`` — the direct one-shot prediction.
    """
    method = method.lower()
    if method not in ("euler", "heun", "midpoint"):
        raise ValueError(f"unknown ODE method {method!r}")
    n_steps = max(1, int(n_steps))
    dt = (t1 - t0) / n_steps
    x = source
    path = [x] if return_path else None
    for i in range(n_steps):
        t = t0 + i * dt
        t_vec = torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
        v = model.velocity(x, t_vec, action)
        if method == "euler":
            x = x + dt * v
        elif method == "heun":  # Heun (predictor-corrector / trapezoidal)
            x_pred = x + dt * v
            t_next = torch.full(
                (x.shape[0],), float(t + dt), device=x.device, dtype=x.dtype
            )
            v_next = model.velocity(x_pred, t_next, action)
            x = x + 0.5 * dt * (v + v_next)
        else:  # explicit midpoint
            t_mid = torch.full(
                (x.shape[0],), float(t + 0.5 * dt), device=x.device, dtype=x.dtype
            )
            x_mid = x + 0.5 * dt * v
            x = x + dt * model.velocity(x_mid, t_mid, action)
        if return_path:
            path.append(x)
    if return_path:
        return x, torch.stack(path, dim=0)  # [n_steps + 1, N, d]
    return x


@torch.no_grad()
def predict_perturbed(
    model: Perturbator,
    source: torch.Tensor,
    action: torch.Tensor,
    objective: str,
    n_steps: int = 20,
    method: str = "heun",
) -> torch.Tensor:
    """Predict perturbed latents under either objective (unified inference helper).

    ``objective == "flow_matching"`` integrates the ODE; ``"direct"`` applies the
    one-shot residual map. Used by training eval and the dose-shift visualizer so
    both paths share a single inference entry point.
    """
    if objective == "flow_matching":
        return ode_sample(model, source, action, n_steps=n_steps, method=method)
    if objective == "direct":
        return model(source, action)
    raise ValueError(f"unknown objective {objective!r}")
