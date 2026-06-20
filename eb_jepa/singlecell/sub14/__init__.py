"""Subliminal 1.4 — faithful port of the tuned cell-LeJEPA encoder.

This package ports the working Subliminal 1.4 architecture (from the
``cell-lejepa`` repo) into eb_jepa with its tuned hyperparameters intact:

- protein-coding-gene-only vocabulary, gene identity = frozen DNA (Evo2)
  + protein (ESMC) features through learned projections (no learned
  per-gene table),
- per-cell equal-frequency quantile binning of counts with a
  *thermometer* (cumulative) bin embedding,
- sigmoid-gated self-attention (arXiv:2409.04431) instead of softmax,
- pre-norm RMSNorm + SwiGLU transformer with a learnable ``[CELL]`` token,
- multi-view JEPA: pairwise-cosine invariance + SIGReg regularisation,
- Muon (2-D body weights) + AdamW (embeddings / norms / scalars), no LR
  scheduler.

The only thing deliberately changed from the 1.4 reference is the model
*scale* (``d_model`` / ``n_layers`` / ``n_heads`` are larger here); every
other knob mirrors the tuned recipe.
"""

from eb_jepa.singlecell.sub14.embeddings import (
    ProteinCodingGeneEmbeddings,
    QuantileThermometerCountEmbedding,
)
from eb_jepa.singlecell.sub14.encoder import EncoderSub14
from eb_jepa.singlecell.sub14.hierarchical import (
    HierarchicalEncoderSub14,
    HierarchicalSubliminal14,
    HierCrossAttention,
    PathwayHierarchyBlock,
)
from eb_jepa.singlecell.sub14.model import Subliminal14, Subliminal14Output
from eb_jepa.singlecell.sub14.sigreg import SIGReg
from eb_jepa.singlecell.sub14.sigmoid_attention import SigmoidAttention
from eb_jepa.singlecell.sub14.swiglu import SwiGLUFFN

__all__ = [
    "ProteinCodingGeneEmbeddings",
    "QuantileThermometerCountEmbedding",
    "EncoderSub14",
    "Subliminal14",
    "Subliminal14Output",
    "HierarchicalSubliminal14",
    "HierarchicalEncoderSub14",
    "PathwayHierarchyBlock",
    "HierCrossAttention",
    "SIGReg",
    "SigmoidAttention",
    "SwiGLUFFN",
]
