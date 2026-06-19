"""CPU smoke test: the JEPA training loop runs end-to-end on synthetic data with
random (stub) gene embeddings — exercises dataset -> collate -> encoder -> LeJEPA
loss -> optimizer step without a GPU or the real embedding cache.
"""

import numpy as np
import torch
from omegaconf import OmegaConf

from examples.tahoe_jepa.main import build_train_module, train

N_GENES = 50


def _write_parquet(path, n_cells=16, seed=0):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    cols = {
        k: []
        for k in (
            "genes",
            "expressions",
            "drug",
            "sample",
            "cell_line_id",
            "moa-fine",
            "canonical_smiles",
            "plate",
            "BARCODE_SUB_LIB_ID",
        )
    }
    for c in range(n_cells):
        g = int(rng.integers(5, 30))
        toks = rng.choice(np.arange(2, N_GENES), size=g, replace=False).astype(np.int64)
        counts = rng.integers(1, 40, size=g).astype(np.float32)
        cols["genes"].append([1] + toks.tolist())
        cols["expressions"].append([-2.0] + counts.tolist())
        cols["drug"].append("DMSO_TF" if c % 3 == 0 else "drug_a")
        cols["sample"].append(f"smp_{c % 4}")
        cols["cell_line_id"].append("CVCL_0001")
        cols["moa-fine"].append("unclear")
        cols["canonical_smiles"].append("CCO")
        cols["plate"].append("plate1")
        cols["BARCODE_SUB_LIB_ID"].append(f"bc_{c}")
    pq.write_table(pa.table(cols), str(path / "shard0.parquet"))


def _cfg(data_dir):
    return OmegaConf.create(
        {
            "meta": {"seed": 0, "run_dir": str(data_dir / "run")},
            "data": {
                "data_dir": data_dir if isinstance(data_dir, str) else str(data_dir),
                "split": "train",
                "L": 16,
                "n_views": 2,
                "view_mode": "drop",
                "gene_keep_frac": 0.7,
                "gene_mask_frac": 0.3,
                "count_mode": "A",
                "n_bins": 10,
                "n_genes": N_GENES,
                "batch_size": 8,
                "num_workers": 0,
                "pin_mem": False,
                "quantile_bins": "",
                "maps_path": "",
            },
            "model": {
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 4,
                "n_kv_heads": 1,
                "use_cls": False,
                "readout": "meanpool",
                "proj_hidden": 64,
                "proj_dim": 16,
                "grad_checkpoint": False,
                "compile": False,
                "gene_emb_cache": "random",
            },
            "loss": {"lamb": 0.05, "num_slices": 32, "knots": 17, "t_max": 3.0},
            "optim": {
                "epochs": 1,
                "lr": 1e-3,
                "weight_decay": 0.05,
                "warmup_ratio": 0.1,
                "min_lr": 1e-6,
                "betas": [0.9, 0.95],
            },
            "training": {"amp": False, "log_every": 1, "ckpt_every_epoch": False},
            "wandb": {"enabled": False, "project": "test"},
        }
    )


def test_build_train_module_forward(tmp_path):
    _write_parquet(tmp_path)
    cfg = _cfg(tmp_path)
    cfg.data.data_dir = str(tmp_path)
    tm = build_train_module(cfg)
    # a forward over a synthetic batch returns a finite loss dict
    from eb_jepa.datasets.tahoe.dataset import TahoeCollator, TahoeConfig, TahoeDataset

    dcfg = TahoeConfig(data_dir=str(tmp_path), L=16, n_views=2, n_genes=N_GENES)
    ds = TahoeDataset(dcfg)
    batch = TahoeCollator(dcfg)([ds[i] for i in range(8)])
    out = tm(batch)
    assert torch.isfinite(out["loss"]) and "sigreg_loss" in out


def test_train_smoke_runs(tmp_path):
    _write_parquet(tmp_path)
    cfg = _cfg(tmp_path)
    cfg.data.data_dir = str(tmp_path)
    encoder = train(cfg, device=torch.device("cpu"))
    assert encoder is not None
    # encoder is usable after training
    from eb_jepa.datasets.tahoe.dataset import TahoeCollator, TahoeConfig, TahoeDataset
    from eb_jepa.singlecell.encoder import encode_views

    dcfg = TahoeConfig(data_dir=str(tmp_path), L=16, n_views=1, n_genes=N_GENES)
    ds = TahoeDataset(dcfg)
    batch = TahoeCollator(dcfg)([ds[i] for i in range(4)])
    with torch.no_grad():
        reps = encode_views(encoder, batch)
    assert reps.shape == (1, 4, 32) and torch.isfinite(reps).all()
