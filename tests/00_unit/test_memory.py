"""Unit tests for the per-stratum latent memory bank (CPU, tiny dims)."""

import torch

from eb_jepa.singlecell.perturbator.flow import flow_matching_loss
from eb_jepa.singlecell.perturbator.featurize import DrugFeaturizer
from eb_jepa.singlecell.perturbator.memory import (
    LatentMemoryBank,
    make_source_key,
    make_target_key,
)
from eb_jepa.singlecell.perturbator.model import Perturbator


class TestRingBuffer:
    def test_get_none_on_empty(self):
        bank = LatentMemoryBank(capacity_per_key=8)
        assert bank.get_source(("L1", "p1")) is None
        assert bank.get_target(("L1", "p1", "drugX", 0.0)) is None
        assert len(bank) == 0
        assert bank.total_entries() == 0

    def test_fifo_eviction_and_capacity(self):
        bank = LatentMemoryBank(capacity_per_key=10)
        key = ("L1", "p1")
        # push 6 then 7 rows (13 total) -> capped to the 10 most recent (FIFO drops oldest)
        first = torch.arange(6, dtype=torch.float32).reshape(6, 1)
        second = torch.arange(100, 107, dtype=torch.float32).reshape(7, 1)
        bank.update_source(key, first)
        bank.update_source(key, second)
        cloud = bank.get_source(key)
        assert cloud.shape == (10, 1)
        # the 3 oldest rows of `first` (0,1,2) were evicted; 3,4,5 then all of `second`
        expected = torch.cat([first[3:], second], 0)
        assert torch.allclose(cloud, expected)

    def test_detached_no_grad(self):
        bank = LatentMemoryBank(capacity_per_key=4)
        x = torch.randn(3, 2, requires_grad=True)
        bank.update_source(("L1", "p1"), x)
        stored = bank.get_source(("L1", "p1"))
        assert not stored.requires_grad
        assert stored.is_contiguous()

    def test_source_target_keying_independent(self):
        bank = LatentMemoryBank(capacity_per_key=16)
        skey = make_source_key("L1", "p1")
        tkey = make_target_key("L1", "p1", "drugX", -6.0)
        bank.update_source(skey, torch.randn(4, 3))
        bank.update_target(tkey, torch.randn(5, 3))
        assert bank.get_source(skey).shape[0] == 4
        assert bank.get_target(tkey).shape[0] == 5
        # source key is NOT a target key and vice versa
        assert bank.get_target(skey) is None
        assert bank.get_source(tkey) is None
        assert bank.n_source_keys() == 1
        assert bank.n_target_keys() == 1
        assert bank.total_entries() == 9

    def test_target_key_dose_rounding(self):
        # doses that round to the same 4-decimal key collapse to one buffer.
        bank = LatentMemoryBank(capacity_per_key=16)
        k1 = make_target_key("L1", "p1", "drugX", -6.000001)
        k2 = make_target_key("L1", "p1", "drugX", -6.000002)
        assert k1 == k2
        bank.update_target(k1, torch.randn(2, 3))
        bank.update_target(k2, torch.randn(2, 3))
        assert bank.n_target_keys() == 1
        assert bank.get_target(k1).shape[0] == 4

    def test_max_keys_eviction(self):
        bank = LatentMemoryBank(capacity_per_key=8, max_keys=2)
        bank.update_source(("L1", "p1"), torch.randn(2, 2))
        bank.update_source(("L2", "p1"), torch.randn(2, 2))
        bank.update_source(("L3", "p1"), torch.randn(2, 2))  # evicts oldest (L1,p1)
        assert bank.n_source_keys() == 2
        assert bank.get_source(("L1", "p1")) is None
        assert bank.get_source(("L3", "p1")) is not None


class TestMemoryAugmentation:
    def test_augmented_cloud_is_larger_and_grad_flows_to_model_only(self):
        torch.manual_seed(0)
        feat = DrugFeaturizer(n_bits=16)
        model = Perturbator(
            d_model=8, action_dim=feat.action_dim, depth=2, d_cond=16,
            time_conditioned=True,
        )
        bank = LatentMemoryBank(capacity_per_key=64)
        skey = make_source_key("L1", "p1")
        tkey = make_target_key("L1", "p1", "drugX", -6.0)

        # seed the bank with past frozen-encoder latents (detached)
        bank.update_source(skey, torch.randn(20, 8))
        bank.update_target(tkey, torch.randn(25, 8))

        # current (small) batch clouds — these DO carry grad to the model via the loss
        batch_source = torch.randn(3, 8)
        batch_target = torch.randn(4, 8)

        bank_src = bank.get_source(skey)
        bank_tgt = bank.get_target(tkey)
        source_cloud = torch.cat([batch_source, bank_src], 0)
        target_cloud = torch.cat([batch_target, bank_tgt], 0)
        # augmentation grows the clouds well beyond the per-batch size
        assert source_cloud.shape[0] == 23 and target_cloud.shape[0] == 29

        action = feat.featurize("CCO", -6.0)
        gen = torch.Generator().manual_seed(1)
        loss = flow_matching_loss(
            model, source_cloud, target_cloud.detach(), action, generator=gen
        )
        assert torch.isfinite(loss) and loss.item() > 0
        loss.backward()
        # grad flows to the perturbator
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)
        # stored latents never require grad (no gradient into the bank)
        assert not bank_src.requires_grad and not bank_tgt.requires_grad
        assert bank_src.grad is None and bank_tgt.grad is None
