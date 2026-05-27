import numpy as np
import pytest

from kmcpy.models.base import BaseModel
from kmcpy.models.composite_lce_model import CompositeLCEModel
from kmcpy.models.fitting.fitter import LCEFitter
from kmcpy.models.fitting.registry import get_fitter_for_model, register_fitter
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion


def test_composite_lce_model_fit_is_not_a_combined_submodel_fitter():
    with pytest.raises(NotImplementedError, match="Fit LocalClusterExpansion models separately"):
        CompositeLCEModel().fit()


def test_composite_lce_model_build_is_not_a_combined_submodel_builder():
    with pytest.raises(NotImplementedError, match="Build LocalClusterExpansion models separately"):
        CompositeLCEModel().build()


def test_lce_fitter_default_normalization_matches_legacy_values(tmp_path):
    correlation_matrix = np.array([
        [1.0, 0.0, 2.0],
        [2.0, 1.0, 0.0],
        [3.0, 1.0, 1.0],
        [4.0, 0.0, 3.0],
        [5.0, 2.0, 2.0],
    ])
    target = np.array([10.0, 12.0, 13.0, 16.0, 18.0])
    weight = np.array([1.0, 2.0, 1.5, 0.5, 3.0])

    corr_file = tmp_path / "correlation_matrix.txt"
    target_file = tmp_path / "e_kra.txt"
    weight_file = tmp_path / "weight.txt"
    keci_file = tmp_path / "keci.txt"
    np.savetxt(corr_file, correlation_matrix)
    np.savetxt(target_file, target)
    np.savetxt(weight_file, weight)

    params, y_pred, y_true = LCEFitter().fit(
        alpha=0.05,
        ekra_fname=str(target_file),
        weight_fname=str(weight_file),
        corr_fname=str(corr_file),
        keci_fname=str(keci_file),
        lce_params_fname=None,
        lce_params_history_fname=None,
    )

    np.testing.assert_allclose(
        params.keci,
        [1.9533446410834194, 0.0, 0.0],
        atol=1e-12,
    )
    assert params.empty_cluster == pytest.approx(7.967045876411174)
    np.testing.assert_allclose(
        y_pred,
        [
            9.920390517494592,
            11.873735158578013,
            13.827079799661432,
            15.780424440744852,
            17.73376908182827,
        ],
        atol=1e-12,
    )
    np.testing.assert_allclose(y_true, target)
    assert params.rmse == pytest.approx(0.40630870108942163)
    assert params.normalize is True

    plain_params, _, _ = LCEFitter().fit(
        alpha=0.05,
        ekra_fname=str(target_file),
        weight_fname=str(weight_file),
        corr_fname=str(corr_file),
        keci_fname=str(tmp_path / "plain_keci.txt"),
        lce_params_fname=None,
        lce_params_history_fname=None,
        normalize=False,
    )
    assert plain_params.normalize is False
    assert not np.allclose(plain_params.keci, params.keci)


def test_fitter_registry_resolves_lce_fitter_through_mro():
    class DerivedLocalClusterExpansion(LocalClusterExpansion):
        pass

    assert get_fitter_for_model(LocalClusterExpansion) is LCEFitter
    assert get_fitter_for_model(DerivedLocalClusterExpansion) is LCEFitter


def test_base_model_fit_dispatches_via_registry():
    class DummyModel(BaseModel):
        def as_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

    class DummyFitter:
        def fit(self, *args, **kwargs):
            return kwargs["alpha"]

    register_fitter(DummyModel, DummyFitter)

    assert DummyModel().fit(alpha=3.0) == 3.0


def test_base_model_fit_raises_without_registered_fitter():
    class UnregisteredModel(BaseModel):
        def as_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

    with pytest.raises(NotImplementedError, match="has no fitter registered"):
        UnregisteredModel().fit(alpha=1.0)
