"""Periodic probe-eval report: detached probes + collapse diagnostics on a fixed
single-view eval batch (mirrors the wandb-logged eval in main.py). CPU-only,
tiny tensors, no wandb."""

import os

import torch

from eb_jepa.datasets.tahoe.dataset import TahoeCollator, TahoeConfig
from eb_jepa.singlecell.embeddings import GeneTokenEmbedding
from eb_jepa.singlecell.encoder import SingleCellEncoder

from examples.tahoe_jepa.probe_eval import probe_report


def _cell(i):
    gen = torch.Generator().manual_seed(i)
    g = 6 + i % 4
    return {
        "gene_token_ids": torch.randperm(48, generator=gen)[:g] + 2,
        "values": torch.rand(g, generator=gen),
        "drug": "a" if i % 2 else "b",
        "sample": "s1" if i % 2 else "s2",
        "cell_line_id": "CVCL_0001" if i % 2 else "CVCL_0002",
        "organ": "Liver" if i % 2 else "Lung",
        "moa_fine": "x" if i % 3 else "y",
        "plate": "p1" if i % 2 else "p2",
        "canonical_smiles": "CCO",
        "log_conc": -7.0,
    }


def _eval_batch(n=24):
    # single clean full-gene view, exactly like eval_tsne.build_eval_set
    cfg = TahoeConfig(
        data_dir="",
        L=16,
        n_views=1,
        n_genes=50,
        view_mode="drop",
        gene_keep_frac=1.0,
        gene_mask_frac=0.0,
    )
    coll = TahoeCollator(cfg)
    return coll([_cell(i) for i in range(n)])


def test_probe_report_scalars_and_spectrum(tmp_path):
    embed = GeneTokenEmbedding.random(50, 32, d_esmc=16, d_evo2=12)
    enc = SingleCellEncoder(embed, d_model=32, n_layers=2, n_heads=4)
    eval_batch = _eval_batch()
    device = torch.device("cpu")

    scalars, spectrum_path = probe_report(
        enc,
        eval_batch,
        device,
        str(tmp_path),
        step=0,
        probe_epochs=30,
        chunk=8,
        amp=False,
    )

    expected = [
        "probe/clf/organ/balanced_accuracy",
        "probe/clf/organ/above_chance",
        "repr/effective_rank",
        "repr/effrank_ratio",
    ]
    for k in expected:
        assert k in scalars, f"missing scalar {k}"
    for k, v in scalars.items():
        assert isinstance(v, float)
        assert v == v and abs(v) != float("inf"), f"non-finite scalar {k}={v}"

    # interpretable derived metrics are consistent
    assert scalars["probe/clf/organ/above_chance"] == (
        scalars["probe/clf/organ/balanced_accuracy"]
        - 1.0 / scalars["probe/clf/organ/n_classes"]
    )
    assert 0.0 <= scalars["repr/effrank_ratio"] <= 1.0 + 1e-6

    assert os.path.exists(spectrum_path)
    assert spectrum_path.endswith("spectrum_step000000.png")
