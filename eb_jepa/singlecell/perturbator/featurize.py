"""Drug + dose featurization for the perturbator action (CLAUDE.md "Drug featurization").

A drug is featurized from its ``canonical_smiles`` into a fixed-dim float vector:
a Morgan fingerprint (configurable ``radius`` / ``n_bits``) concatenated with a
handful of RDKit physico-chemical descriptors (MolWt, MolLogP, TPSA, NumHAcceptors,
NumHDonors, NumRotatableBonds). RDKit is imported **lazily**; when it is not
installed the featurizer falls back to a *deterministic* hash of the SMILES string
producing a vector of the same dimension, so the whole pipeline (and the tests)
runs without RDKit.

The dose is appended as two extra channels: ``[validity_flag, log_conc]``. Controls
carry ``nan`` ``log_conc`` (see ``parse_log_conc``); they are encoded as the
sentinel ``[0.0, 0.0]`` ("no dose"). The total ``action_dim`` is therefore
``n_bits + n_descriptors + 2``.
"""

from __future__ import annotations

import hashlib
import math

import torch

# RDKit descriptor names, in a fixed order (kept stable for cache/embedding layout).
_DESCRIPTORS = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
)


def _try_import_rdkit():
    """Return the RDKit handles we need, or ``None`` if RDKit is unavailable."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
    except ImportError:
        return None
    return {
        "Chem": Chem,
        "AllChem": AllChem,
        "Crippen": Crippen,
        "Descriptors": Descriptors,
        "Lipinski": Lipinski,
        "rdMolDescriptors": rdMolDescriptors,
    }


def _hash_vector(text: str, dim: int) -> torch.Tensor:
    """Deterministic float vector in [0, 1) of length ``dim`` from a string.

    Used as the RDKit-free fallback. SHA-256 is expanded into enough bytes by
    hashing ``"{i}:{text}"`` per block; each byte -> one channel scaled to [0, 1).
    """
    out = torch.empty(dim, dtype=torch.float32)
    filled = 0
    block = 0
    while filled < dim:
        digest = hashlib.sha256(f"{block}:{text}".encode("utf-8")).digest()
        for byte in digest:
            if filled >= dim:
                break
            out[filled] = byte / 255.0
            filled += 1
        block += 1
    return out


class DrugFeaturizer:
    """SMILES (+ dose) -> fixed-dim action vector, with an RDKit-free fallback.

    Args:
        n_bits: Morgan fingerprint length (also the hash-fallback fingerprint length).
        radius: Morgan fingerprint radius.
        use_descriptors: append the RDKit physico-chemical descriptors.

    The base drug features (fingerprint [+ descriptors]) are cached by SMILES string.
    """

    def __init__(self, n_bits: int = 1024, radius: int = 2, use_descriptors: bool = True):
        self.n_bits = int(n_bits)
        self.radius = int(radius)
        self.use_descriptors = bool(use_descriptors)
        self._rdkit = _try_import_rdkit()
        self._cache: dict[str, torch.Tensor] = {}
        self.has_rdkit = self._rdkit is not None

    @property
    def n_descriptors(self) -> int:
        return len(_DESCRIPTORS) if self.use_descriptors else 0

    @property
    def drug_dim(self) -> int:
        """Dimension of the SMILES-only features (fingerprint [+ descriptors])."""
        return self.n_bits + self.n_descriptors

    @property
    def action_dim(self) -> int:
        """Full action dimension: drug features + [dose_validity, log_conc]."""
        return self.drug_dim + 2

    # ------------------------------------------------------------------ #
    # Drug (SMILES) features                                             #
    # ------------------------------------------------------------------ #
    def _rdkit_drug_features(self, smiles: str) -> torch.Tensor | None:
        rk = self._rdkit
        mol = rk["Chem"].MolFromSmiles(smiles)
        if mol is None:  # unparseable SMILES -> fall back to the hash path
            return None
        fp = rk["rdMolDescriptors"].GetMorganFingerprintAsBitVect(
            mol, self.radius, nBits=self.n_bits
        )
        fp_t = torch.zeros(self.n_bits, dtype=torch.float32)
        for bit in fp.GetOnBits():
            fp_t[bit] = 1.0
        if not self.use_descriptors:
            return fp_t
        d = rk["Descriptors"]
        crippen = rk["Crippen"]
        lip = rk["Lipinski"]
        vals = {
            "MolWt": d.MolWt(mol),
            "MolLogP": crippen.MolLogP(mol),
            "TPSA": rk["rdMolDescriptors"].CalcTPSA(mol),
            "NumHAcceptors": lip.NumHAcceptors(mol),
            "NumHDonors": lip.NumHDonors(mol),
            "NumRotatableBonds": lip.NumRotatableBonds(mol),
        }
        desc = torch.tensor([float(vals[name]) for name in _DESCRIPTORS])
        return torch.cat([fp_t, desc])

    def _fallback_drug_features(self, smiles: str) -> torch.Tensor:
        """Deterministic RDKit-free features of the same dimension as the RDKit path.

        The fingerprint block is a hashed binary vector; the descriptor block (if
        enabled) is a hashed real vector — both depend only on the SMILES string.
        """
        fp = (_hash_vector(f"fp:{smiles}", self.n_bits) > 0.5).float()
        if not self.use_descriptors:
            return fp
        desc = _hash_vector(f"desc:{smiles}", self.n_descriptors)
        return torch.cat([fp, desc])

    def drug_features(self, smiles: str | None) -> torch.Tensor:
        """Cached SMILES-only features ``[drug_dim]``. ``None`` -> zeros (no drug)."""
        if not smiles:
            return torch.zeros(self.drug_dim, dtype=torch.float32)
        cached = self._cache.get(smiles)
        if cached is not None:
            return cached
        feat = None
        if self._rdkit is not None:
            feat = self._rdkit_drug_features(smiles)
        if feat is None:
            feat = self._fallback_drug_features(smiles)
        self._cache[smiles] = feat
        return feat

    # ------------------------------------------------------------------ #
    # Action (drug + dose)                                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _dose_channels(log_conc: float | None) -> torch.Tensor:
        """``[validity_flag, value]``: ``[0, 0]`` for controls / missing dose."""
        if log_conc is None or (isinstance(log_conc, float) and math.isnan(log_conc)):
            return torch.zeros(2, dtype=torch.float32)
        return torch.tensor([1.0, float(log_conc)], dtype=torch.float32)

    def featurize(self, smiles: str | None, log_conc: float | None = None) -> torch.Tensor:
        """One action vector ``[action_dim]`` = drug features ++ dose channels."""
        return torch.cat([self.drug_features(smiles), self._dose_channels(log_conc)])

    def featurize_batch(
        self, smiles: list[str | None], log_conc=None
    ) -> torch.Tensor:
        """Stack ``featurize`` over a batch -> ``[B, action_dim]``.

        ``log_conc`` may be ``None`` (all controls), a python list, or a 1-D tensor.
        """
        n = len(smiles)
        if log_conc is None:
            doses: list = [None] * n
        elif torch.is_tensor(log_conc):
            doses = log_conc.tolist()
        else:
            doses = list(log_conc)
        return torch.stack([self.featurize(smiles[i], doses[i]) for i in range(n)])

    __call__ = featurize
