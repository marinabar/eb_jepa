"""Unit tests for the hepatotoxicity virtual-pathway featurizer (CPU, no heavy deps).

Tests run with or without RDKit: shape/determinism/contract checks are RDKit-free;
the surrogate-sanity checks (known hepatotoxins score higher on the relevant pathway
columns) are skipped if RDKit is unavailable.
"""

import math

import pytest
import torch

from eb_jepa.singlecell.perturbator.hepatotox_features import (
    CYP_ISOFORMS,
    HepatotoxPathwayFeaturizer,
    _saturating,
)


class TestContract:
    def test_dim_and_names_consistent(self):
        feat = HepatotoxPathwayFeaturizer()
        assert feat.feature_dim == len(feat.feature_names)
        assert feat.feature_dim == len(feat.surrogate_mask)
        # blocks: 5 cyp + 4 reactive + 2 nrf2 + 2 mito + 1 bsep + 8 desc = 22
        assert feat.feature_dim == 22
        for iso in CYP_ISOFORMS:
            assert f"cyp_inh_{iso}" in feat.feature_names

    def test_toggle_blocks(self):
        feat = HepatotoxPathwayFeaturizer(
            cyp_use_alerts=False, include_bsep=False, include_descriptors=False
        )
        # 4 reactive + 2 nrf2 + 2 mito = 8
        assert feat.feature_dim == 8
        assert not any(n.startswith("cyp_inh_") for n in feat.feature_names)
        assert not any(n.startswith("desc_") for n in feat.feature_names)

    def test_shape_and_determinism(self):
        feat = HepatotoxPathwayFeaturizer()
        a = feat.featurize("CCO")
        b = feat.featurize("CCO")
        assert a.shape == (feat.feature_dim,)
        assert torch.allclose(a, b)  # cached + deterministic
        # distinct molecules -> distinct vectors
        c = feat.featurize("O=C(C)Oc1ccccc1C(=O)O")  # aspirin
        assert not torch.allclose(a, c)

    def test_none_is_zero(self):
        feat = HepatotoxPathwayFeaturizer()
        z = feat.featurize(None)
        assert z.shape == (feat.feature_dim,)
        assert torch.count_nonzero(z) == 0

    def test_batch(self):
        feat = HepatotoxPathwayFeaturizer()
        X = feat.featurize_batch(["CCO", None, "c1ccccc1"])
        assert X.shape == (3, feat.feature_dim)
        assert torch.count_nonzero(X[1]) == 0  # None row is zeros

    def test_feature_dict(self):
        feat = HepatotoxPathwayFeaturizer()
        d = feat.feature_dict("CCO")
        assert set(d) == set(feat.feature_names)
        assert all(isinstance(v, float) for v in d.values())

    def test_finite(self):
        feat = HepatotoxPathwayFeaturizer()
        for smi in ("CCO", "c1ccccc1", "O=C(C)Oc1ccccc1C(=O)O", "garbage_not_smiles"):
            assert torch.isfinite(feat.featurize(smi)).all()


class TestPredictorInjection:
    def test_set_predictor_overrides_and_clears_surrogate(self):
        feat = HepatotoxPathwayFeaturizer()
        name = "cyp_inh_CYP3A4"
        idx = feat.feature_names.index(name)
        assert feat.surrogate_mask[idx] is True
        feat.set_predictor(name, lambda smi: 0.123)
        assert feat.surrogate_mask[idx] is False
        v = feat.featurize("CCO")
        assert abs(float(v[idx]) - 0.123) < 1e-5

    def test_unknown_feature_raises(self):
        feat = HepatotoxPathwayFeaturizer()
        with pytest.raises(KeyError):
            feat.set_predictor("not_a_feature", lambda s: 0.0)


class TestSaturating:
    def test_monotone_bounded(self):
        assert _saturating(0) == 0.0
        assert 0.0 < _saturating(1) < _saturating(3) < 1.0


_RDKIT = HepatotoxPathwayFeaturizer().has_rdkit


@pytest.mark.skipif(not _RDKIT, reason="RDKit not installed; surrogate sanity skipped")
class TestSurrogateSanity:
    def test_michael_acceptor_raises_nrf2(self):
        feat = HepatotoxPathwayFeaturizer()
        i_nrf2 = feat.feature_names.index("nrf2_michael_count")
        # acrolein (a Michael acceptor) vs ethanol (none)
        acrolein = feat.featurize("C=CC=O")
        ethanol = feat.featurize("CCO")
        assert acrolein[i_nrf2] >= 1.0
        assert ethanol[i_nrf2] == 0.0

    def test_nitrophenol_flags_uncoupler(self):
        feat = HepatotoxPathwayFeaturizer()
        i = feat.feature_names.index("mito_uncoupler_prob")
        # 2,4-dinitrophenol is the canonical mitochondrial uncoupler
        dnp = feat.featurize("Oc1ccc(cc1[N+](=O)[O-])[N+](=O)[O-]")
        ethanol = feat.featurize("CCO")
        assert dnp[i] > ethanol[i]

    def test_acetaminophen_has_reactive_alert(self):
        feat = HepatotoxPathwayFeaturizer()
        i = feat.feature_names.index("reactive_parent_alerts")
        # paracetamol: para-aminophenol amide -> known reactive (NAPQI) precursor
        apap = feat.featurize("CC(=O)Nc1ccc(O)cc1")
        assert apap[i] >= 1.0

    def test_descriptors_scaled_reasonably(self):
        feat = HepatotoxPathwayFeaturizer()
        i_mw = feat.feature_names.index("desc_mw")
        v = feat.featurize("CCO")  # MW ~46 -> /500 ~ 0.09
        assert 0.0 < float(v[i_mw]) < 0.5
