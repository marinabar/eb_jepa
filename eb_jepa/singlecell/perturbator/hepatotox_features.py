"""Hepatotoxicity-focused "virtual pathways" drug featurization (CLAUDE.md II.1).

A drug SMILES is mapped to a fixed-dim vector of scores over **hepatotoxicity-relevant
virtual pathways** — the chemistry "minimum information entities" (MIEs) that drive
Drug-Induced Liver Injury (DILI). This mirrors the *Virtual Pathways* approach
(``github.com/Kharoh/virtual-pathway-cyp``: ``vp.dili.mechanistic.mechanistic_matrix``),
which composes one descriptor block per pathway:

    block      cols  meaning
    --------   ----  -----------------------------------------------------------
    cyp        5     CYP-isoform inhibition propensity (CYP1A2/2C9/2C19/2D6/3A4)
    reactive   4     reactive-metabolite / bioactivation structural alerts
    nrf2       2     NRF2/KEAP1 oxidative-stress: P(ARE) proxy + Michael-acceptor count
    mito       2     mitochondrial OXPHOS liability: protonophore uncoupler + lipophilicity
    bsep       1     BSEP/ABCB11 (bile-salt-export-pump) inhibition propensity
    desc       8     coarse physico-chemical descriptors (MW, logP, TPSA, HBD, HBA, ...)

In the reference repo the cyp / nrf2 / mito / bsep **probability** columns come from
XGBoost classifiers trained on public bioassay data (PubChem qHTS, ChEMBL, Tox21
SR-ARE / mito-membrane-potential, BSEP IC50 sets). Those trained boosters are **not
shippable offline**, so here every probability column is a **transparent RDKit
surrogate** — a documented structural-alert / physchem heuristic derived from the
SAME chemistry the reference models key on (the reference repo's own SMARTS alert
lists for Michael acceptors, uncouplers, and reactive metabolites are reused
verbatim). Surrogate columns are flagged ``is_surrogate=True`` so a trained
predictor can be dropped in later without changing the feature contract: register a
callable via :meth:`HepatotoxPathwayFeaturizer.set_predictor` keyed by feature name.

RDKit is imported **lazily** (like ``DrugFeaturizer``); without it the featurizer
falls back to a deterministic SMILES hash of the same dimension so the pipeline and
tests still run. The two surrogate scoring rules ("more matching alerts -> higher
score") are calibrated by a saturating ``1 - exp(-k * hits)`` map so a score lands
in ``[0, 1]`` and reads like a (rough) probability.
"""

from __future__ import annotations

import hashlib
import math

import torch

from eb_jepa.singlecell.perturbator.featurize import DrugFeaturizer

# =========================================================================== #
# DILIrank / LiverTox reference label vocabulary (drug-name -> DILI concern).  #
# =========================================================================== #
# Primary label source for the perturbator hepatotoxicity validation. Drug
# names are matched case-insensitively against Tahoe's ``drug`` column.
#
# Sources (cited, editable — extend freely):
#   - FDA DILIrank dataset (Chen et al., Drug Discov Today 2016): "vMost-DILI-
#     Concern" -> HEPATOTOX_DRUGS (positive), "vNo-DILI-Concern" -> LOW_DILI_DRUGS
#     (negative). https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-dataset
#   - NIH LiverTox (https://www.ncbi.nlm.nih.gov/books/NBK547852/): well-described
#     clinical hepatotoxins / well-tolerated agents.
# These are clinical DILI labels by parent compound; they intentionally span well
# beyond Tahoe so the by-name intersection with the liver subset is maximised. The
# perturbator never sees the label — it predicts a latent shift; the label only
# scores that shift. Names are lower-cased; INN spellings + common synonyms both
# listed so the case-insensitive match catches Tahoe naming variants.
HEPATOTOX_DRUGS: set[str] = {
    # --- analgesics / NSAIDs (most-DILI-concern) ---
    "acetaminophen", "paracetamol", "diclofenac", "nimesulide", "bromfenac",
    "sulindac", "naproxen", "indomethacin", "ibufenac", "benoxaprofen",
    # --- antibiotics / antifungals / antivirals ---
    "isoniazid", "rifampicin", "rifampin", "pyrazinamide", "ketoconazole",
    "trovafloxacin", "telithromycin", "erythromycin", "nitrofurantoin",
    "flucloxacillin", "minocycline", "nevirapine", "stavudine", "didanosine",
    "zidovudine", "ritonavir", "tipranavir", "fialuridine", "amoxicillin",
    "clarithromycin", "azithromycin", "sulfamethoxazole", "voriconazole",
    "itraconazole", "fluconazole", "terbinafine", "dapsone", "ethionamide",
    # --- thiazolidinediones / antidiabetics ---
    "troglitazone", "pioglitazone", "rosiglitazone", "acarbose",
    # --- CNS / antiepileptics / antidepressants ---
    "valproic acid", "valproate", "carbamazepine", "phenytoin", "felbamate",
    "nefazodone", "tolcapone", "pemoline", "duloxetine", "bupropion",
    "lamotrigine", "phenobarbital", "chlorpromazine", "imipramine",
    "amineptine", "agomelatine", "disulfiram",
    # --- cardiovascular ---
    "amiodarone", "bosentan", "labetalol", "ticlopidine", "hydralazine",
    "methyldopa", "perhexiline", "dronedarone", "fenofibrate",
    # --- oncology / immunomodulators ---
    "tamoxifen", "methotrexate", "flutamide", "leflunomide", "azathioprine",
    "mercaptopurine", "cytarabine", "dacarbazine", "asparaginase", "imatinib",
    "lapatinib", "pazopanib", "sunitinib", "sorafenib", "regorafenib",
    "gefitinib", "erlotinib", "nilotinib", "ponatinib", "bortezomib",
    "trabectedin", "idelalisib", "crizotinib", "ceritinib", "dasatinib",
    "axitinib", "vandetanib", "cabozantinib",
    # --- misc well-known hepatotoxins ---
    "dantrolene", "halothane", "allopurinol", "interferon", "niacin",
    "ketoprofen", "tacrine", "zileuton", "etretinate", "acitretin",
    "danazol", "stanozolol", "methyltestosterone", "cyclophosphamide",
    "busulfan", "thioguanine", "gemtuzumab",
}

# vNo-DILI-Concern / well-tolerated agents (negatives).
LOW_DILI_DRUGS: set[str] = {
    "aspirin", "ibuprofen", "celecoxib", "metformin", "atenolol", "famotidine",
    "loratadine", "cetirizine", "fexofenadine", "ranitidine", "lisinopril",
    "enalapril", "amlodipine", "nifedipine", "omeprazole", "pantoprazole",
    "lansoprazole", "simvastatin", "pravastatin", "rosuvastatin", "warfarin",
    "furosemide", "hydrochlorothiazide", "spironolactone", "metoprolol",
    "propranolol", "losartan", "valsartan", "clopidogrel", "digoxin",
    "levothyroxine", "gabapentin", "pregabalin", "sertraline", "fluoxetine",
    "citalopram", "escitalopram", "venlafaxine", "diphenhydramine",
    "cefalexin", "cephalexin", "penicillin", "doxycycline", "ciprofloxacin",
    "levofloxacin", "metronidazole", "acyclovir", "oseltamivir", "ondansetron",
    "metoclopramide", "prednisone", "prednisolone", "dexamethasone",
    "hydroxychloroquine", "colchicine", "montelukast",
    "salbutamol", "albuterol", "budesonide", "fluticasone", "insulin",
    "glipizide", "glyburide", "sitagliptin", "rivaroxaban", "apixaban",
    "dabigatran", "ezetimibe", "tamsulosin", "finasteride", "sildenafil",
    "tadalafil", "naratriptan", "sumatriptan", "lorazepam", "diazepam",
    "zolpidem", "buspirone", "mirtazapine", "tramadol", "morphine",
    "oxycodone", "codeine", "cyclobenzaprine", "baclofen",
}

# --------------------------------------------------------------------------- #
# Weak hepatotoxicity label from drug_metadata MoA / target text (SECONDARY    #
# label source). Used only when DILIrank-by-name coverage is too thin; reported #
# separately. Keyed on substrings found in ``moa-broad`` / ``moa-fine`` /       #
# ``targets`` that map to canonical DILI mechanisms (mitochondrial toxicity,    #
# reactive-metabolite / oxidative stress, BSEP-mediated cholestasis).           #
# --------------------------------------------------------------------------- #
HEPATOTOX_MOA_KEYWORDS: tuple[str, ...] = (
    "topoisomerase", "dna synthesis", "dna damage", "alkylating",
    "tyrosine kinase", "kinase inhibitor", "mtor", "proteasome",
    "cyp", "cytochrome", "mitochond", "oxidative phosphorylation",
    "hdac", "antimetabolite", "antifolate", "estrogen receptor",
    "androgen receptor", "retinoic", "statin", "hmg-coa",
)
LOW_DILI_MOA_KEYWORDS: tuple[str, ...] = (
    "antihistamine", "histamine receptor", "beta blocker", "beta-adrenergic",
    "calcium channel", "ace inhibitor", "angiotensin", "proton pump",
    "diuretic", "glp-1", "dpp-4",
)


def dili_label_by_name(
    drug: str | None,
    hepatotox: set[str] | None = None,
    low: set[str] | None = None,
) -> str | None:
    """Map a Tahoe ``drug`` name to ``"hepatotoxic"`` / ``"low_concern"`` / ``None``.

    Case-insensitive exact-name match against the DILIrank/LiverTox sets above
    (the primary label source). ``None`` (unknown) drugs are excluded from the
    DILI probe rather than assumed safe.
    """
    if not drug:
        return None
    h = hepatotox if hepatotox is not None else HEPATOTOX_DRUGS
    lo = low if low is not None else LOW_DILI_DRUGS
    d = drug.strip().lower()
    if d in h:
        return "hepatotoxic"
    if d in lo:
        return "low_concern"
    return None


def weak_dili_label_from_moa(text: str | None) -> str | None:
    """Weak DILI label from free-text MoA / target metadata (secondary source).

    Returns ``"hepatotoxic"`` if any high-DILI MoA keyword matches, ``"low_concern"``
    if a low-DILI keyword matches (and no high-DILI keyword), else ``None``. This is a
    coarse mechanistic prior, NOT a clinical label — reported as a separate source.
    """
    if not text:
        return None
    t = str(text).lower()
    if any(k in t for k in HEPATOTOX_MOA_KEYWORDS):
        return "hepatotoxic"
    if any(k in t for k in LOW_DILI_MOA_KEYWORDS):
        return "low_concern"
    return None

# --------------------------------------------------------------------------- #
# Structural-alert SMARTS libraries.                                          #
# Reused verbatim from virtual-pathway-cyp:                                   #
#   - REACTIVE_SMARTS  <- vp.dili.alerts.CUSTOM_REACTIVE_SMARTS               #
#   - MICHAEL_SMARTS   <- vp.nrf2.features.MICHAEL_ACCEPTOR_SMARTS            #
#   - UNCOUPLER_SMARTS <- vp.oxphos.features.UNCOUPLER_SMARTS                 #
# (the reference repo's trained QSAR boosters are not available offline, so   #
#  these training-free alert lists are the reusable, transparent core).       #
# --------------------------------------------------------------------------- #
REACTIVE_SMARTS: list[tuple[str, str]] = [
    ("acyl_glucuronide_precursor", "[c,$([#6]=[#6])][CX4H1,CX4H0]([#6])[CX3](=O)[OX2H1]"),
    ("furan_ring", "c1ccoc1"),
    ("thiophene_2_aryl", "c1cc([cH0,cH1]2sccc2)c[cH0,cH1][cH0,cH1]1"),
    ("anilide_NH", "[c;$(c1ccccc1)]N([H])C(=O)[#6]"),
    ("hydrazone", "[#6]=N-N([H])[#6,H]"),
    ("nitroaromatic", "[c]N(=O)=O"),
    ("para_phenol_NH", "[OH][c]1[cH][cH][c]([N;$([NX3H2]),$([NX3H1]C)])[cH][cH]1"),
]

MICHAEL_SMARTS: list[tuple[str, str]] = [
    ("enone_michael", "[CX3]=[CX3]-[CX3]=[OX1]"),
    ("acrylate_amide", "[CX3]=[CX3]-[CX3](=[OX1])[#7,#8]"),
    ("vinyl_sulfone", "[CX3]=[CX3]-[SX4](=[OX1])(=[OX1])"),
    ("vinyl_nitrile", "[CX3]=[CX3]-[CX2]#[NX1]"),
    ("quinone", "O=C1C=CC(=O)C=C1"),
    ("quinone_ortho", "O=C1C=CC=CC1=O"),
    ("maleimide", "O=C1C=CC(=O)N1"),
    ("alpha_halo_carbonyl", "[CX3](=[OX1])[CX4][F,Cl,Br,I]"),
    ("epoxide", "[OX2r3]1[#6r3][#6r3]1"),
    ("aziridine", "[NX3r3]1[#6r3][#6r3]1"),
    ("isothiocyanate", "[NX2]=[CX2]=[SX1]"),
    ("beta_lactam", "O=C1CCN1"),
]

UNCOUPLER_SMARTS: list[tuple[str, str]] = [
    ("phenol", "[OX2H][c]"),
    ("aromatic_carboxyl", "[cX3]-[CX3](=O)[OX2H1]"),
    ("nitroaromatic", "[c][$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    ("acyl_sulfonamide", "[SX4](=O)(=O)[NX3H1][CX3]=O"),
    ("aryl_sulfonamide_NH", "[c][SX4](=O)(=O)[NX3H1,NX3H2]"),
    ("tetrazole", "c1nnn[nH]1"),
    ("perhalo_phenol", "[OX2H][c]([F,Cl,Br,I])[c][c]([F,Cl,Br,I])"),
]

# CYP isoforms in the reference repo's fixed order (vp.cyp.inhibitor.model.INHIBITOR_ISOFORMS).
CYP_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]

# Coarse physchem descriptor keys (vp.dili.features.coarse_descriptors), fixed order.
_DESC_KEYS = ["mw", "logp", "tpsa", "hbd", "hba", "n_rot", "n_aromatic_rings", "n_heavy"]

# Scale factors for descriptor channels so the raw values land in a ~[0,1]-ish range
# (purely for numerical conditioning of a downstream linear layer; documented + fixed).
_DESC_SCALE = {
    "mw": 500.0, "logp": 5.0, "tpsa": 140.0, "hbd": 5.0,
    "hba": 10.0, "n_rot": 10.0, "n_aromatic_rings": 4.0, "n_heavy": 40.0,
}


def _saturating(hits: float, k: float = 0.8) -> float:
    """Map a non-negative alert count to ``[0, 1)`` via ``1 - exp(-k*hits)``.

    Monotone, 0 at no hits, saturating — turns a discrete substructure count into a
    smooth pseudo-probability for the surrogate columns.
    """
    return float(1.0 - math.exp(-k * max(0.0, hits)))


def _try_import_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    except ImportError:
        return None
    return {
        "Chem": Chem,
        "Crippen": Crippen,
        "Descriptors": Descriptors,
        "Lipinski": Lipinski,
        "rdMolDescriptors": rdMolDescriptors,
    }


def _hash_unit(text: str) -> float:
    """Deterministic float in [0, 1) from a string (RDKit-free fallback channel)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


class HepatotoxPathwayFeaturizer:
    """SMILES -> hepatotoxicity virtual-pathway score vector.

    Args:
        cyp_use_alerts: include the 5 CYP-inhibition surrogate columns.
        include_bsep: include the BSEP-inhibition surrogate column.
        include_descriptors: include the 8 coarse physchem descriptor columns.
        sat_k: steepness of the count -> pseudo-prob saturation.

    The feature vector is the concatenation of, in order: CYP (5), reactive (4),
    NRF2 (2), mito (2), [BSEP (1)], [desc (8)]. ``feature_names`` is the column
    contract; ``surrogate_mask`` flags which columns are heuristic surrogates.

    A trained predictor can be injected per-column with :meth:`set_predictor`;
    registered columns then come from the model and lose their surrogate flag.
    """

    def __init__(
        self,
        cyp_use_alerts: bool = True,
        include_bsep: bool = True,
        include_descriptors: bool = True,
        sat_k: float = 0.8,
    ):
        self.cyp_use_alerts = bool(cyp_use_alerts)
        self.include_bsep = bool(include_bsep)
        self.include_descriptors = bool(include_descriptors)
        self.sat_k = float(sat_k)
        self._rdkit = _try_import_rdkit()
        self.has_rdkit = self._rdkit is not None
        self._cache: dict[str, torch.Tensor] = {}
        self._patterns: dict | None = None
        # name -> callable(smiles)->float, overrides the surrogate for that column.
        self._predictors: dict[str, object] = {}

        # Build the (name, kind, is_surrogate) column contract.
        cols: list[tuple[str, str]] = []
        if self.cyp_use_alerts:
            cols += [(f"cyp_inh_{iso}", "surrogate") for iso in CYP_ISOFORMS]
        cols += [
            ("reactive_parent_alerts", "surrogate"),
            ("reactive_n_distinct", "surrogate"),
            ("reactive_has_furan_or_nitro", "surrogate"),
            ("reactive_aromatic_amine", "surrogate"),
        ]
        cols += [
            ("nrf2_are_prob", "surrogate"),
            ("nrf2_michael_count", "surrogate"),
        ]
        cols += [
            ("mito_uncoupler_prob", "surrogate"),
            ("mito_lipophilic_acid", "surrogate"),
        ]
        if self.include_bsep:
            cols += [("bsep_inhib_prob", "surrogate")]
        if self.include_descriptors:
            cols += [(f"desc_{k}", "descriptor") for k in _DESC_KEYS]
        self._cols = cols
        self.feature_names = [c[0] for c in cols]
        self.surrogate_mask = [c[1] == "surrogate" for c in cols]

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    # ------------------------------------------------------------------ #
    # Trained-predictor injection                                        #
    # ------------------------------------------------------------------ #
    def set_predictor(self, feature_name: str, fn) -> None:
        """Override a column with a trained predictor ``fn(smiles)->float``.

        Clears the surrogate flag for that column. ``feature_name`` must be one of
        :attr:`feature_names`. Invalidates the per-SMILES cache.
        """
        if feature_name not in self.feature_names:
            raise KeyError(f"unknown feature {feature_name!r}; valid: {self.feature_names}")
        self._predictors[feature_name] = fn
        idx = self.feature_names.index(feature_name)
        self.surrogate_mask[idx] = False
        self._cache.clear()

    # ------------------------------------------------------------------ #
    # RDKit pattern lazy-compile                                         #
    # ------------------------------------------------------------------ #
    def _get_patterns(self) -> dict:
        if self._patterns is not None:
            return self._patterns
        Chem = self._rdkit["Chem"]

        def compile_list(lst):
            out = []
            for name, smarts in lst:
                patt = Chem.MolFromSmarts(smarts)
                if patt is not None:
                    out.append((name, patt))
            return out

        self._patterns = {
            "reactive": compile_list(REACTIVE_SMARTS),
            "michael": compile_list(MICHAEL_SMARTS),
            "uncoupler": compile_list(UNCOUPLER_SMARTS),
            "aromatic_amine": Chem.MolFromSmarts("[c][NX3;H2,H1;!$(NC=O)]"),
        }
        return self._patterns

    # ------------------------------------------------------------------ #
    # Surrogate scoring (RDKit available)                                #
    # ------------------------------------------------------------------ #
    def _rdkit_features(self, smiles: str) -> torch.Tensor | None:
        rk = self._rdkit
        mol = rk["Chem"].MolFromSmiles(smiles)
        if mol is None:
            return None
        pat = self._get_patterns()

        def count_hits(patterns) -> int:
            return sum(1 for _name, p in patterns if mol.HasSubstructMatch(p))

        reactive_hits = [n for n, p in pat["reactive"] if mol.HasSubstructMatch(p)]
        michael_n = count_hits(pat["michael"])
        uncoupler_n = count_hits(pat["uncoupler"])

        d = rk["Descriptors"]
        crippen = rk["Crippen"]
        lip = rk["Lipinski"]
        logp = float(crippen.MolLogP(mol))
        mw = float(d.MolWt(mol))
        tpsa = float(rk["rdMolDescriptors"].CalcTPSA(mol))
        desc_vals = {
            "mw": mw, "logp": logp, "tpsa": tpsa,
            "hbd": float(lip.NumHDonors(mol)), "hba": float(lip.NumHAcceptors(mol)),
            "n_rot": float(lip.NumRotatableBonds(mol)),
            "n_aromatic_rings": float(lip.NumAromaticRings(mol)),
            "n_heavy": float(mol.GetNumHeavyAtoms()),
        }

        has_aromatic_amine = (
            pat["aromatic_amine"] is not None
            and mol.HasSubstructMatch(pat["aromatic_amine"])
        )
        has_furan_or_nitro = any(
            n in ("furan_ring", "nitroaromatic") for n in reactive_hits
        )
        # Acidic group present (uncoupler precursor) AND lipophilic -> protonophore chemotype.
        lipophilic_acid = float(uncoupler_n > 0 and logp >= 2.0)

        # CYP surrogate: inhibition propensity rises with lipophilicity, size and
        # aromatic ring count (the established physchem correlates of broad CYP
        # inhibition). One mild per-isoform offset so the 5 columns are not identical
        # (a placeholder for the trained per-isoform booster). All in [0, 1].
        cyp_base = 1.0 / (1.0 + math.exp(-(0.6 * (logp - 2.0) + 0.5 * (desc_vals["n_aromatic_rings"] - 1.0))))
        cyp_offsets = {"CYP1A2": 0.05, "CYP2C9": 0.0, "CYP2C19": -0.02, "CYP2D6": 0.03, "CYP3A4": 0.08}

        values: dict[str, float] = {}
        if self.cyp_use_alerts:
            for iso in CYP_ISOFORMS:
                values[f"cyp_inh_{iso}"] = float(min(1.0, max(0.0, cyp_base + cyp_offsets[iso])))
        values["reactive_parent_alerts"] = float(len(reactive_hits))
        values["reactive_n_distinct"] = float(len(set(reactive_hits)))
        values["reactive_has_furan_or_nitro"] = float(has_furan_or_nitro)
        values["reactive_aromatic_amine"] = float(has_aromatic_amine)
        values["nrf2_are_prob"] = _saturating(michael_n, self.sat_k)
        values["nrf2_michael_count"] = float(michael_n)
        values["mito_uncoupler_prob"] = _saturating(uncoupler_n, self.sat_k)
        values["mito_lipophilic_acid"] = lipophilic_acid
        if self.include_bsep:
            # BSEP inhibition: large, lipophilic, amphipathic molecules (the canonical
            # cholestatic chemotype, e.g. cyclosporine/troglitazone). Surrogate from MW+logP.
            bsep = 1.0 / (1.0 + math.exp(-(0.004 * (mw - 400.0) + 0.5 * (logp - 3.0))))
            values["bsep_inhib_prob"] = float(bsep)
        if self.include_descriptors:
            for k in _DESC_KEYS:
                values[f"desc_{k}"] = desc_vals[k] / _DESC_SCALE[k]

        return self._assemble(values, smiles)

    def _assemble(self, values: dict[str, float], smiles: str) -> torch.Tensor:
        """Order ``values`` per the column contract, applying any trained overrides."""
        out = torch.empty(self.feature_dim, dtype=torch.float32)
        for i, name in enumerate(self.feature_names):
            if name in self._predictors:
                try:
                    out[i] = float(self._predictors[name](smiles))
                    continue
                except Exception:
                    pass  # fall through to surrogate value
            out[i] = float(values.get(name, 0.0))
        return out

    def _fallback_features(self, smiles: str) -> torch.Tensor:
        """Deterministic RDKit-free vector of the right dimension (hash surrogate)."""
        out = torch.empty(self.feature_dim, dtype=torch.float32)
        for i, name in enumerate(self.feature_names):
            if name in self._predictors:
                try:
                    out[i] = float(self._predictors[name](smiles))
                    continue
                except Exception:
                    pass
            out[i] = _hash_unit(f"{name}:{smiles}")
        return out

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def featurize(self, smiles: str | None) -> torch.Tensor:
        """One hepatotox feature vector ``[feature_dim]``. ``None`` -> zeros."""
        if not smiles:
            return torch.zeros(self.feature_dim, dtype=torch.float32)
        cached = self._cache.get(smiles)
        if cached is not None:
            return cached
        feat = None
        if self._rdkit is not None:
            feat = self._rdkit_features(smiles)
        if feat is None:
            feat = self._fallback_features(smiles)
        self._cache[smiles] = feat
        return feat

    def featurize_batch(self, smiles: list[str | None]) -> torch.Tensor:
        """Stack :meth:`featurize` over a batch -> ``[B, feature_dim]``."""
        return torch.stack([self.featurize(s) for s in smiles])

    def feature_dict(self, smiles: str | None) -> dict[str, float]:
        """Named feature dict (handy for reports / debugging)."""
        vec = self.featurize(smiles)
        return {n: float(v) for n, v in zip(self.feature_names, vec.tolist())}

    __call__ = featurize


class HepatotoxActionFeaturizer:
    """Dose-aware action featurizer mirroring :class:`DrugFeaturizer`'s interface.

    Wraps a :class:`HepatotoxPathwayFeaturizer` so the perturbator can consume the
    hepatotoxicity virtual-pathway scores as its **action** vector while still
    seeing the dose. The action vector is the pathway feature block followed by the
    **same two dose channels** ``[validity_flag, log_conc]`` that ``DrugFeaturizer``
    appends (controls / missing dose -> sentinel ``[0, 0]``), so
    ``examples/tahoe_perturbator/main.py`` can treat both featurizers uniformly via
    the shared ``.action_dim`` / ``.featurize(smiles, log_conc)`` /
    ``.featurize_batch(smiles_list, log_conc)`` contract.

    Args:
        pathway_featurizer: an existing :class:`HepatotoxPathwayFeaturizer`, or
            ``None`` to build a default one.
        **pathway_kwargs: forwarded to :class:`HepatotoxPathwayFeaturizer` when
            ``pathway_featurizer`` is not given.
    """

    def __init__(
        self,
        pathway_featurizer: "HepatotoxPathwayFeaturizer | None" = None,
        **pathway_kwargs,
    ):
        self.pathway = pathway_featurizer or HepatotoxPathwayFeaturizer(**pathway_kwargs)
        # expose the underlying chemistry contract for downstream attribution
        self.feature_names = self.pathway.feature_names
        self.surrogate_mask = self.pathway.surrogate_mask
        self.has_rdkit = self.pathway.has_rdkit

    @property
    def drug_dim(self) -> int:
        """Dimension of the SMILES-only (pathway) feature block."""
        return self.pathway.feature_dim

    @property
    def action_dim(self) -> int:
        """Full action dimension: pathway features + [dose_validity, log_conc]."""
        return self.pathway.feature_dim + 2

    def drug_features(self, smiles: str | None) -> torch.Tensor:
        """Cached SMILES-only pathway features ``[drug_dim]``."""
        return self.pathway.featurize(smiles)

    def featurize(self, smiles: str | None, log_conc: float | None = None) -> torch.Tensor:
        """One action vector ``[action_dim]`` = pathway features ++ dose channels."""
        dose = DrugFeaturizer._dose_channels(log_conc)
        return torch.cat([self.pathway.featurize(smiles), dose])

    def featurize_batch(self, smiles: list[str | None], log_conc=None) -> torch.Tensor:
        """Stack :meth:`featurize` over a batch -> ``[B, action_dim]``.

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

    def set_predictor(self, feature_name: str, fn) -> None:
        """Forward to the underlying pathway featurizer (trained-predictor injection)."""
        self.pathway.set_predictor(feature_name, fn)
        self.surrogate_mask = self.pathway.surrogate_mask

    __call__ = featurize
