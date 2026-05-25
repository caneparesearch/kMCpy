import numpy as np
import pytest

from kmcpy.models.base import BaseModel
from kmcpy.models.composite_lce_model import CompositeLCEModel
from kmcpy.models.fitting.fitter import LCEFitter
from kmcpy.models.fitting.registry import get_fitter_for_model, register_fitter
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion


def test_composite_lce_model_fit_dispatches_to_lce_fitters(monkeypatch):
    calls = []

    def fake_fit(self, **kwargs):
        calls.append(kwargs)
        return {"keci": [1.0], "empty_cluster": 0.0}, np.array([1.0]), np.array([1.0])

    monkeypatch.setattr(LocalClusterExpansion, "fit", fake_fit)

    results = CompositeLCEModel().fit(
        kra_fit_kwargs={"alpha": 1.0},
        site_fit_kwargs={"alpha": 2.0},
    )

    assert "kra" in results
    assert "site" in results
    assert calls == [{"alpha": 1.0}, {"alpha": 2.0}]


def test_composite_lce_model_instance_fit_updates_submodel_parameters(monkeypatch):
    def fake_fit(self, **kwargs):
        alpha = kwargs["alpha"]
        return {
            "keci": [alpha],
            "empty_cluster": alpha + 1.0,
        }, np.array([alpha]), np.array([alpha])

    monkeypatch.setattr(LocalClusterExpansion, "fit", fake_fit)

    composite = CompositeLCEModel(
        site_model=LocalClusterExpansion(),
        kra_model=LocalClusterExpansion(),
    )
    results = composite.fit(
        kra_fit_kwargs={"alpha": 1.5},
        site_fit_kwargs={"alpha": 0.5},
    )

    assert np.allclose(results["kra"][1], np.array([1.5]))
    assert np.allclose(results["site"][1], np.array([0.5]))
    assert composite.kra_model.keci == [1.5]
    assert composite.kra_model.empty_cluster == 2.5
    assert composite.site_model.keci == [0.5]
    assert composite.site_model.empty_cluster == 1.5


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
        def __str__(self):
            return "DummyModel"

        def __repr__(self):
            return "DummyModel()"

        def compute(self, *args, **kwargs):
            return 0.0

        def compute_probability(self, *args, **kwargs):
            return 0.0

        def build(self, *args, **kwargs):
            return None

        def as_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        @classmethod
        def from_file(cls, filename):
            return cls()

    class DummyFitter:
        def fit(self, *args, **kwargs):
            return kwargs["alpha"]

    register_fitter(DummyModel, DummyFitter)

    assert DummyModel().fit(alpha=3.0) == 3.0


def test_base_model_fit_raises_without_registered_fitter():
    class UnregisteredModel(BaseModel):
        def __str__(self):
            return "UnregisteredModel"

        def __repr__(self):
            return "UnregisteredModel()"

        def compute(self, *args, **kwargs):
            return 0.0

        def compute_probability(self, *args, **kwargs):
            return 0.0

        def build(self, *args, **kwargs):
            return None

        def as_dict(self):
            return {}

        @classmethod
        def from_dict(cls, data):
            return cls()

        @classmethod
        def from_file(cls, filename):
            return cls()

    with pytest.raises(NotImplementedError, match="has no fitter registered"):
        UnregisteredModel().fit(alpha=1.0)
