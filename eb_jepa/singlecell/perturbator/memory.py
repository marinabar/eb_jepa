"""Per-stratum latent memory bank for the perturbator (CLAUDE.md "Target embeddings
are kept in a memory of arrival-distribution embeddings").

The perturbator trains per stratum ``(cell_line, plate, drug, dose)`` on a source
(control) latent cloud and a target (treated) latent cloud that must co-occur in one
streamed batch. With few cell lines (e.g. liver = 3 lines) a single batch rarely
contains both control AND treated cells of the same stratum, so very few strata yield
a loss per step and the signal is noisy.

Because the encoder is **frozen**, its latents are stationary: detached latents
accumulated across past steps are exact extra samples of each stratum's source/target
distribution. This bank stores those past latents in per-key FIFO ring buffers so each
step can augment its (small) per-batch clouds before computing the OT / flow-matching
loss. The bank holds **detached, no-grad** tensors — there is no gradient through it and
no double counting (augment for the loss, then update with the current batch).

Two keyed stores:
- **SOURCE** keyed by ``(cell_line, plate)`` — the control distribution of a stratum.
- **TARGET** keyed by ``(cell_line, plate, drug, round(log_conc, 4))`` — the treated
  distribution at a given drug+dose.
"""

from __future__ import annotations

from collections import OrderedDict

import torch


class _RingStore:
    """A keyed collection of per-key FIFO ring buffers of detached latents.

    Each key maps to a list of ``[n_i, d]`` tensor chunks; the concatenation is the
    cloud for that key, capped to ``capacity_per_key`` most-recent rows (FIFO). The
    number of keys (and total rows) is bounded by evicting the least-recently-touched
    key, so the bank stays memory-bounded.
    """

    def __init__(
        self,
        capacity_per_key: int = 1024,
        device: str | torch.device = "cpu",
        max_keys: int | None = None,
        max_entries: int | None = None,
    ) -> None:
        self.capacity_per_key = int(capacity_per_key)
        self.device = torch.device(device)
        self.max_keys = max_keys
        self.max_entries = max_entries
        # OrderedDict preserves recency: move_to_end on touch, popitem(last=False) evicts oldest.
        self._store: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()

    def update(self, key: tuple, latents: torch.Tensor) -> None:
        """Append ``latents`` (detached) to ``key``'s buffer, FIFO-capped per key."""
        if latents is None or latents.numel() == 0:
            return
        chunk = latents.detach().to(self.device).contiguous()
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)
        cur = self._store.get(key)
        buf = chunk if cur is None else torch.cat([cur, chunk], dim=0)
        if buf.shape[0] > self.capacity_per_key:
            buf = buf[-self.capacity_per_key :].contiguous()  # FIFO: drop the oldest rows
        self._store[key] = buf
        self._store.move_to_end(key)  # most-recently touched
        self._evict()

    def get(self, key: tuple) -> torch.Tensor | None:
        """Return the cloud for ``key`` (``[n, d]``) or ``None`` if empty."""
        buf = self._store.get(key)
        if buf is None or buf.shape[0] == 0:
            return None
        return buf

    def _evict(self) -> None:
        """Bound the store by key count and total rows, evicting oldest-touched keys."""
        if self.max_keys is not None:
            while len(self._store) > self.max_keys:
                self._store.popitem(last=False)
        if self.max_entries is not None:
            while self.total_entries() > self.max_entries and len(self._store) > 0:
                self._store.popitem(last=False)

    def n_keys(self) -> int:
        return len(self._store)

    def total_entries(self) -> int:
        return int(sum(b.shape[0] for b in self._store.values()))


class LatentMemoryBank:
    """Source/target latent memory bank keyed by stratum (frozen-encoder latents).

    Args:
        capacity_per_key: max rows kept per key (FIFO ring buffer); default 1024.
        device: where the buffers live (default ``"cpu"`` to spare GPU memory); the
            caller moves returned tensors to the compute device/dtype for the loss.
        max_keys: optional cap on the number of distinct keys per store.
        max_entries: optional cap on total rows per store.
    """

    def __init__(
        self,
        capacity_per_key: int = 1024,
        device: str | torch.device = "cpu",
        max_keys: int | None = None,
        max_entries: int | None = None,
    ) -> None:
        self.source = _RingStore(capacity_per_key, device, max_keys, max_entries)
        self.target = _RingStore(capacity_per_key, device, max_keys, max_entries)

    # --- update (after computing the step's losses) ---------------------------- #
    def update_source(self, key: tuple, latents: torch.Tensor) -> None:
        self.source.update(key, latents)

    def update_target(self, key: tuple, latents: torch.Tensor) -> None:
        self.target.update(key, latents)

    # --- retrieve (to augment the step's clouds) ------------------------------- #
    def get_source(self, key: tuple) -> torch.Tensor | None:
        return self.source.get(key)

    def get_target(self, key: tuple) -> torch.Tensor | None:
        return self.target.get(key)

    # --- logging / introspection ---------------------------------------------- #
    def n_source_keys(self) -> int:
        return self.source.n_keys()

    def n_target_keys(self) -> int:
        return self.target.n_keys()

    def total_entries(self) -> int:
        return self.source.total_entries() + self.target.total_entries()

    def __len__(self) -> int:
        return self.source.n_keys() + self.target.n_keys()


def make_source_key(cell_line, plate) -> tuple:
    """Source key for a stratum: ``(cell_line, plate)``."""
    return (cell_line, plate)


def make_target_key(cell_line, plate, drug, log_conc) -> tuple:
    """Target key for a stratum: ``(cell_line, plate, drug, round(log_conc, 4))``.

    ``log_conc`` may be ``nan`` (rounding ``nan`` is a stable sentinel key).
    """
    return (cell_line, plate, drug, round(float(log_conc), 4))
