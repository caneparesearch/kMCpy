import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kmcpy.models.base import BaseModel
from kmcpy.models.composite_lce_model import CompositeLCEModel
from kmcpy.models.fitting.fitter import LCEFitter
from kmcpy.models.fitting.registry import get_fitter_for_model, register_fitter
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion


def test_lcefitter_from_json_supports_direct_payload(tmp_path):
    payload = {
        "time_stamp": 10.0,
        "time": "01/01/2024, 00:00:00",
        "keci": [1.0, 2.0],
        "empty_cluster": 3.0,
        "weight": [1.0, 1.0],
        "alpha": 4.0,
        "rmse": 5.0,
        "loocv": 6.0,
    }
    path = tmp_path / "lce_fit.json"
    path.write_text(json.dumps(payload))

    fitter = LCEFitter.from_json(str(path))

    assert fitter.model_parameters.keci == [1.0, 2.0]
    assert fitter.model_parameters.alpha == 4.0
    assert fitter.model_parameters.time == "01/01/2024, 00:00:00"


def test_lcefitter_from_json_supports_legacy_history_payload():
    fit_results = Path(__file__).parent / "files" / "input" / "fitting_results.json"

    fitter = LCEFitter.from_json(str(fit_results))

    assert fitter.model_parameters.alpha == 1.5
    assert fitter.model_parameters.keci[0] == 10.5029085379
    assert fitter.model_parameters.time == "01/21/2022, 16:49:17"


def test_local_cluster_expansion_fit_writes_legacy_results_history(tmp_path):
    file_path = Path(__file__).parent / "files"
    fit_results_fname = tmp_path / "fitting_results.json"

    _, y_pred, y_true = LocalClusterExpansion().fit(
        alpha=1.5,
        max_iter=1000000,
        ekra_fname=str(file_path / "fitting" / "local_cluster_expansion" / "e_kra.txt"),
        keci_fname=str(tmp_path / "keci.txt"),
        weight_fname=str(file_path / "fitting" / "local_cluster_expansion" / "weight.txt"),
        corr_fname=str(
            file_path / "fitting" / "local_cluster_expansion" / "correlation_matrix.txt"
        ),
        fit_results_fname=str(fit_results_fname),
        lce_params_fname=None,
        lce_params_history_fname=None,
    )

    assert np.allclose(y_pred, y_true, rtol=0.3, atol=10.0)
    assert fit_results_fname.exists()

    df = pd.read_json(fit_results_fname, orient="index")
    assert len(df) == 1
    assert df.iloc[0]["alpha"] == 1.5


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
        def from_json(cls, fname):
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
        def from_json(cls, fname):
            return cls()

    with pytest.raises(NotImplementedError, match="has no fitter registered"):
        UnregisteredModel().fit(alpha=1.0)
