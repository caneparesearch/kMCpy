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
