"""Perturbator (CLAUDE.md Part II): predict the latent of a drug-perturbed cell.

v1 keeps the encoder **frozen** and operates on its pooled latents. A small
FiLM-conditioned transformer maps a *source* (control) latent distribution to the
*target* (treated) latent distribution, conditioned on the drug action (SMILES
features + dose). Training is unpaired optimal transport: a sliced-Wasserstein
distance between the predicted and the target latent distributions, with control
matching per stratum ``(cell_line_id, plate)``.
"""

from eb_jepa.singlecell.perturbator.featurize import DrugFeaturizer
from eb_jepa.singlecell.perturbator.losses import sliced_wasserstein
from eb_jepa.singlecell.perturbator.matching import build_strata, Stratum
from eb_jepa.singlecell.perturbator.model import Perturbator

__all__ = [
    "DrugFeaturizer",
    "sliced_wasserstein",
    "build_strata",
    "Stratum",
    "Perturbator",
]
