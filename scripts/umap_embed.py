"""Encode the whole cached dataset with a trained sub14 encoder and draw UMAPs.

Loads a sub14 checkpoint, encodes every cached cell (pre-projection representation),
runs UMAP, and saves one figure per metadata colouring (organ, cell line, MoA,
control-vs-perturbed, plate, drug). House-style scatter.

    PYTHONPATH=/data/eb_jepa .venv/bin/python scripts/umap_embed.py \
        --ckpt /data/runs/sub14_law/xd_d512_l8/encoder_final.pt --out /data/runs/sub14_law/umap/xd
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from eb_jepa.datasets.tahoe.dataset import TahoeConfig, TahoeIterableDataset
from eb_jepa.singlecell.sub14.collator import Sub14Collator
from eb_jepa.singlecell.sub14.features import load_pc_features
from eb_jepa.singlecell.sub14.model import Subliminal14
from eb_jepa.training_utils import load_config

INK, MUTED = "#1d2433", "#7a8699"


def plot_umap(emb, labels, path, name):
    labels = [str(x) for x in labels]
    cats = sorted(set(labels))
    cmap = plt.get_cmap("tab20" if len(cats) <= 20 else "gist_ncar", max(len(cats), 1))
    idx = {c: i for i, c in enumerate(cats)}
    cols = np.array([cmap(idx[l]) for l in labels])
    order = np.random.permutation(len(labels))  # avoid one class painting over others
    fig, ax = plt.subplots(figsize=(9.5, 8))
    ax.scatter(emb[order, 0], emb[order, 1], c=cols[order], s=2.5, alpha=0.6, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"UMAP — coloured by {name}", color=INK, fontweight="bold", loc="left")
    ax.text(0, 1.01, f"xd_d512_l8 (37M) embeddings · {len(labels):,} cells · {len(cats)} {name}",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    if len(cats) <= 50:
        h = [plt.Line2D([], [], marker="o", ls="", color=cmap(idx[c]), label=c, ms=6) for c in cats]
        ax.legend(handles=h, frameon=False, fontsize=6.5, loc="center left",
                  bbox_to_anchor=(1.0, 0.5), ncol=1 if len(cats) <= 16 else 2)
    fig.tight_layout()
    fig.savefig(f"{path}_{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{path}_{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", f"{path}_{name}.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/data/runs/sub14_law/xd_d512_l8/encoder_final.pt")
    ap.add_argument("--cfg", default="examples/tahoe_jepa/cfgs/sub14_scaling.yaml")
    ap.add_argument("--out", default="/data/runs/sub14_law/umap/xd")
    ap.add_argument("--n", type=int, default=0)  # 0 = all cached cells
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=2048)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dev = torch.device("cuda")

    cfg = load_config(args.cfg)
    pc = load_pc_features(cfg.model.gene_emb_cache)
    maps = torch.load(cfg.data.maps_path)
    dcfg = TahoeConfig(**{k: cfg.data[k] for k in cfg.data if k in TahoeConfig.__dataclass_fields__})
    ds = TahoeIterableDataset(dcfg, binner=None, cell_line_to_organ=maps.get("cell_line_to_organ"),
                              sample_to_logconc=maps.get("sample_to_logconc"), rank=0, world_size=1, shuffle=False)
    items = ds.sample_items(args.n or 1_000_000)
    print(f"loaded {len(items)} cells", flush=True)

    nb, gpb = int(cfg.data.get("num_bins", 16)), int(cfg.data.get("genes_per_bin", 32))
    model = Subliminal14(
        n_pc_genes=pc.n_pc_genes, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff, dropout=0.0,
        latent_dim=int(cfg.model.get("proj_dim", 128)), num_bins=nb,
        max_genes_per_cell=nb * gpb, dna_features=pc.dna_features,
        protein_features=pc.protein_features, freeze_features=True,
        attention_activation=str(cfg.model.get("attention_activation", "sigmoid")),
    )
    sd = torch.load(args.ckpt, map_location="cpu")["model"]
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"loaded ckpt; missing={len(miss)} unexpected={len(unexp)}", flush=True)
    model = model.to(dev, torch.bfloat16).eval()

    col = Sub14Collator(token_to_pc_local=pc.token_to_pc_local, n_pc_genes=pc.n_pc_genes,
                        num_bins=nb, genes_per_bin=gpb, num_views=1, binomial_subsample=None, seed=0)
    KEYS = ("organ", "cell_line_id", "drug", "moa_fine", "sample", "plate")
    meta = {k: [] for k in KEYS}
    reps, BS = [], 4096
    with torch.no_grad():
        for i in range(0, len(items), BS):
            b = col(items[i:i + BS])
            r = model.encode(b["gene_ids"][0].to(dev), b["bin_ids"][0].to(dev),
                             b["padding_mask"][0].to(dev))
            reps.append(r.float().cpu())
            for k in KEYS:
                if k in b:
                    meta[k] += list(b[k])
            print(f"encoded {min(i + BS, len(items))}/{len(items)}", flush=True)
    reps = torch.cat(reps).numpy()
    print(f"reps {reps.shape}", flush=True)

    import umap

    emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42, verbose=True).fit_transform(reps)
    np.save(f"{args.out}_emb.npy", emb)
    meta["treatment"] = ["control (DMSO)" if str(d) == "DMSO_TF" else "perturbed"
                         for d in meta.get("drug", [])]

    for c in ("organ", "cell_line_id", "moa_fine", "treatment", "plate", "drug"):
        if meta.get(c) and len(meta[c]) == len(emb):
            plot_umap(emb, meta[c], args.out, c)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
