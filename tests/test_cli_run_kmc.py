import argparse

import pytest

from kmcpy.cli import run_kmc as run_kmc_module
from kmcpy.cli.main import main as kmcpy_main


@pytest.mark.unit
def test_run_kmc_direct_args_accept_model_file(monkeypatch):
    class DummyKMC:
        called = False

        def __init__(self, config):
            self.config = config

        @classmethod
        def from_config(cls, config):
            cls.called = True
            return cls(config)

        def run(self):
            return {"ok": True, "name": self.config.name}

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
        site_mapping={"Na": ["Na", "X"]},
        initial_occupations=[0, 1],
    )

    run_kmc_module.run_kmc(args)
    assert DummyKMC.called is True


@pytest.mark.unit
def test_run_kmc_direct_args_parse_list_like_strings(monkeypatch):
    captured = {}

    class DummyKMC:
        @classmethod
        def from_config(cls, config):
            captured["supercell_shape"] = config.system_config.supercell_shape
            captured["site_mapping"] = config.system_config.site_mapping
            return cls()

        def run(self):
            return {"ok": True}

    monkeypatch.setattr(run_kmc_module, "KMC", DummyKMC)

    args = argparse.Namespace(
        input=None,
        supercell_shape="[2, 1, 1]",
        model_file="fake_model.json",
        structure_file="fake_structure.cif",
        event_file="fake_events.json",
        attempt_frequency=1e13,
        temperature=300.0,
        convert_to_primitive_cell=False,
        site_mapping='{ "Na": ["Na", "X"], "O": "O" }',
    )

    run_kmc_module.run_kmc(args)

    assert captured["supercell_shape"] == (2, 1, 1)
    assert captured["site_mapping"] == {"Na": ["Na", "X"], "O": "O"}


@pytest.mark.unit
def test_run_kmc_help_mentions_modern_workflow(capsys):
    with pytest.raises(SystemExit) as exc_info:
        run_kmc_module.main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "kmcpy init --output input_template.yaml" in output
    assert "kmcpy sample all --output-dir kmcpy_sample" in output
    assert "--initial_state_file" in output
    assert "--initial_occupations" in output
    assert "--kmc_passes" in output
    assert "property callbacks" in output


@pytest.mark.unit
def test_kmcpy_run_alias_dispatches_to_run_kmc(monkeypatch):
    captured = {}

    def fake_run_kmc(args):
        captured["input"] = args.input
        captured["command"] = args.command

    monkeypatch.setattr("kmcpy.cli.main.run_kmc", fake_run_kmc)

    exit_code = kmcpy_main(["run", "--input", "input.yaml"])

    assert exit_code == 0
    assert captured == {"input": "input.yaml", "command": "run"}


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
