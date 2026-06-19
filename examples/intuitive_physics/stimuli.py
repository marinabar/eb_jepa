"""Procedural bouncing-MNIST stimuli for the intuitive-physics probe.

A single MNIST digit moves on a 64x64 canvas with constant velocity and bounces
elastically off the walls. For each plausible base trajectory we build a
*matched* impossible twin that is pixel-identical until a known violation frame
``t_v`` and then breaks one physical law:

- ``teleport``    : position jumps to a far on-canvas location (continuity),
- ``reversal``    : velocity instantly negates in free flight (no force),
- ``passthrough`` : the digit passes through a wall and wraps out the opposite
                    side (impenetrability).

``video`` is ``[1, T, 64, 64]`` in ``[0, 1]`` — the video-JEPA encoder's format.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

CANVAS = 64
DIGIT = 28
MAXPOS = CANVAS - DIGIT  # 36: top-left coord range is [0, MAXPOS]
T_DEFAULT = 10
VIOLATIONS = ("teleport", "reversal", "passthrough")


def load_mnist_digits(split: str = "train") -> np.ndarray:
    """MNIST glyphs as ``[N, 28, 28]`` in ``[0, 1]``, cached under ``$EBJEPA_DSETS``."""
    root = os.environ.get("EBJEPA_DSETS", os.path.join(os.getcwd(), "data"))
    cache = os.path.join(root, f"mnist_glyphs_{split}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    from torchvision.datasets import MNIST

    arr = MNIST(root=root, train=(split == "train"), download=True).data.numpy()
    arr = arr.astype(np.float32) / 255.0
    os.makedirs(root, exist_ok=True)
    np.save(cache, arr)
    return arr


def simulate(start, vel, T, event=None, t_v=None, teleport_target=None):
    """Top-left positions ``[T, 2]`` (row, col) with elastic bounces; ``event``
    injects a violation at ``t_v``. Returns ``(positions, bounced)``."""
    p = start.astype(np.float64).copy()
    v = vel.astype(np.float64).copy()
    pos = np.zeros((T, 2))
    pos[0] = p
    bounced = [False] * T
    for t in range(1, T):
        if t == t_v and event == "reversal":
            v = -v
        if t == t_v and event == "teleport":
            p = teleport_target.astype(np.float64).copy()
            pos[t] = p
            continue
        p = p + v
        if event == "passthrough" and t >= t_v:
            p = np.mod(p, MAXPOS)  # wrap through the wall, stay on-canvas
        else:
            for d in range(2):  # elastic reflection
                if p[d] < 0:
                    p[d] = -p[d]; v[d] = -v[d]; bounced[t] = True
                elif p[d] > MAXPOS:
                    p[d] = 2 * MAXPOS - p[d]; v[d] = -v[d]; bounced[t] = True
        pos[t] = p
    return pos, bounced


def render_clip(positions: np.ndarray, digit: np.ndarray) -> np.ndarray:
    """Render ``[T, 2]`` top-left positions of a 28x28 glyph to ``[1, T, 64, 64]``."""
    T = positions.shape[0]
    video = np.zeros((1, T, CANVAS, CANVAS), dtype=np.float32)
    for t in range(T):
        r0, c0 = int(round(positions[t, 0])), int(round(positions[t, 1]))
        sr0, sc0 = max(0, -r0), max(0, -c0)
        dr0, dc0 = max(0, r0), max(0, c0)
        dr1, dc1 = min(CANVAS, r0 + DIGIT), min(CANVAS, c0 + DIGIT)
        if dr1 <= dr0 or dc1 <= dc0:
            continue
        sr1, sc1 = sr0 + (dr1 - dr0), sc0 + (dc1 - dc0)
        video[0, t, dr0:dr1, dc0:dc1] = np.maximum(
            video[0, t, dr0:dr1, dc0:dc1], digit[sr0:sr1, sc0:sc1])
    return video


def heatmap_from_positions(positions: np.ndarray, map_size: int = 8) -> np.ndarray:
    """Exact digit-center heatmap ``[T, map_size, map_size]`` (one hot cell/frame)."""
    T = positions.shape[0]
    hm = np.zeros((T, map_size, map_size), dtype=np.float32)
    centers = positions + DIGIT / 2.0
    for t in range(T):
        cy = int(centers[t, 0] / CANVAS * map_size)
        cx = int(centers[t, 1] / CANVAS * map_size)
        if 0 <= cy < map_size and 0 <= cx < map_size:
            hm[t, cy, cx] = 1.0
    return hm


def _sample_start_vel(rng, speed=(3.0, 6.0)):
    start = rng.uniform(0, MAXPOS, size=2)
    angle = rng.uniform(0, 2 * np.pi)
    s = rng.uniform(*speed)
    return start, np.array([np.sin(angle), np.cos(angle)]) * s


def sample_plausible(rng, T):
    start, vel = _sample_start_vel(rng)
    pos, _ = simulate(start, vel, T)
    return pos


def sample_pair(rng, violation, T, max_tries=200):
    """``(plaus_pos, imposs_pos, t_v)`` sharing frames ``0..t_v-1`` exactly."""
    lo, hi = 3, T - 3
    for _ in range(max_tries):
        start, vel = _sample_start_vel(rng)
        plaus, bounced = simulate(start, vel, T)
        if violation == "passthrough":
            cand = [t for t in range(lo, hi + 1) if bounced[t]]
            if not cand:
                continue
            t_v = cand[0]
            imposs, _ = simulate(start, vel, T, event="passthrough", t_v=t_v)
            return plaus, imposs, t_v
        free = [t for t in range(lo, hi + 1)
                if not any(bounced[max(0, t - 1):t + 2])
                and 5 <= plaus[t, 0] <= MAXPOS - 5 and 5 <= plaus[t, 1] <= MAXPOS - 5]
        if not free:
            continue
        t_v = free[len(free) // 2]
        if violation == "reversal":
            imposs, _ = simulate(start, vel, T, event="reversal", t_v=t_v)
            return plaus, imposs, t_v
        if violation == "teleport":
            for _ in range(80):
                tgt = rng.uniform(0, MAXPOS, size=2)
                if np.linalg.norm(tgt - plaus[t_v]) >= 18:
                    imposs, _ = simulate(start, vel, T, event="teleport", t_v=t_v,
                                         teleport_target=tgt)
                    return plaus, imposs, t_v
    raise RuntimeError(f"could not sample a clean '{violation}' pair in {max_tries} tries")


class ProceduralBouncingMNIST(Dataset):
    """Plausible single-digit bouncing clips for training the video-JEPA.

    Returns ``{"video": [1, T, 64, 64], "digit_location": [T, 8, 8]}``.
    """

    def __init__(self, split="train", n_samples=9000, T=T_DEFAULT, seed=2025, map_size=8):
        self.T, self.map_size = T, map_size
        self.digits = load_mnist_digits("train" if split == "train" else "test")
        base = {"train": 0, "val": 7_000_000}[split]
        self.seeds = [seed + base + i for i in range(n_samples)]

    def __len__(self):
        return len(self.seeds)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        rng = np.random.RandomState(self.seeds[idx])
        digit = self.digits[rng.randint(len(self.digits))]
        pos = sample_plausible(rng, self.T)
        return {"video": torch.from_numpy(render_clip(pos, digit)),
                "digit_location": torch.from_numpy(heatmap_from_positions(pos, self.map_size))}


def build_probe_pairs(n_pairs=200, T=T_DEFAULT, seed=12345,
                      violations: Tuple[str, ...] = VIOLATIONS, digit_split="test"
                      ) -> Dict[str, Dict[str, torch.Tensor]]:
    """Held-out matched pairs per violation type: ``{viol: {plausible, impossible, t_v}}``."""
    digits = load_mnist_digits(digit_split)
    rng = np.random.RandomState(seed)
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for viol in violations:
        plaus, imposs, tvs = [], [], []
        for _ in range(n_pairs):
            digit = digits[rng.randint(len(digits))]
            pp, ip, t_v = sample_pair(rng, viol, T)
            plaus.append(render_clip(pp, digit)); imposs.append(render_clip(ip, digit)); tvs.append(t_v)
        out[viol] = {"plausible": torch.from_numpy(np.stack(plaus)),
                     "impossible": torch.from_numpy(np.stack(imposs)),
                     "t_v": torch.tensor(tvs)}
    return out
