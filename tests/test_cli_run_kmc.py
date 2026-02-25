import argparse

import pytest

from kmcpy.cli import run_kmc as run_kmc_module


@pytest.mark.unit
def test_run_kmc_direct_args_accept_model_file(monkeypatch):
    class DummyKMC:
        called = False

        @classmethod
        def from_config(cls, config):
            cls.called = True
            return cls()

        def run(self, config):
            return {"ok": True, "name": config.name}

    monkeypatch.setattr(run_kmc_module, "KMC", DummyKMC)

    args = argparse.Namespace(
        input=None,
        supercell_shape=(1, 1, 1),
        model_file="fake_model.json",
        structure_file="fake_structure.cif",
        event_file="fake_events.json",
        attempt_frequency=1e13,
        temperature=300.0,
        convert_to_primitive_cell=False,
        immutable_sites=[],
        initial_occupations=[1, -1],
    )

    run_kmc_module.run_kmc(args)
    assert DummyKMC.called is True


@pytest.mark.unit
def test_run_kmc_cli_rejects_removed_legacy_flags(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_kmc",
            "--cluster_expansion_file",
            "legacy.json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        run_kmc_module.main()

    assert exc_info.value.code == 2
