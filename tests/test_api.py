import pytest

from kmcpy.api import run
from kmcpy.simulator.kmc import KMC
from tests.test_utils import create_test_config


def test_run_wrapper_uses_kmc_from_config_and_returns_tracker(monkeypatch):
    config = create_test_config(name="API_Run_Test")
    expected_tracker = object()
    observed = {}

    class DummyKMC:
        def run(self, label=None):
            observed["run_label"] = label
            return expected_tracker

    def fake_from_config(cls, config):
        observed["from_config"] = config
        return DummyKMC()

    monkeypatch.setattr(KMC, "from_config", classmethod(fake_from_config))

    tracker = run(config, label="unit")

    assert tracker is expected_tracker
    assert observed["from_config"] is config
    assert observed["run_label"] == "unit"


def test_run_wrapper_passes_none_label_by_default(monkeypatch):
    config = create_test_config(name="Default_Label_Test")
    observed = {}

    class DummyKMC:
        def run(self, label=None):
            observed["run_label"] = label
            return object()

    def fake_from_config(cls, config):
        return DummyKMC()

    monkeypatch.setattr(KMC, "from_config", classmethod(fake_from_config))

    run(config)

    assert observed["run_label"] is None


def test_run_wrapper_validates_config_type():
    with pytest.raises(TypeError, match="Configuration"):
        run(config={})
