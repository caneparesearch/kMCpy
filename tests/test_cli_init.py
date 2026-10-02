from pathlib import Path

import pytest

from kmcpy.cli.init import write_template
from kmcpy.cli.main import main as kmcpy_main
from kmcpy.io.files import load_raw_data
from kmcpy.simulation import validate_simulation_input
from kmcpy.simulator.config import Configuration


@pytest.mark.unit
def test_write_template_creates_valid_simulation_input(tmp_path: Path):
    output_file = tmp_path / "input_template.yaml"
    written = write_template(output_file)

    assert written == output_file
    template_text = output_file.read_text(encoding="utf-8")
    for section in ("lattice_structure:", "events:", "model:", "state:", "run:"):
        assert section in template_text
    assert "property_callbacks" in template_text

    data = load_raw_data(output_file)
    validate_simulation_input(data)
    assert data["lattice_structure"]["structure"] == "path/to/structure.cif"
    assert data["run"] == {
        "temperature": 300.0,
        "kmc_passes": 10000,
        "random_seed": None,
        "output_dir": "results",
    }


@pytest.mark.unit
def test_configuration_template_is_still_available(tmp_path: Path):
    output_file = tmp_path / "config.yaml"
    write_template(output_file, template_format="configuration")

    template_text = output_file.read_text(encoding="utf-8")
    assert "# ----- Runtime fields -----" in template_text
    config = Configuration.from_file(str(output_file))
    assert config.structure_file == "path/to/structure.cif"
    assert config.kmc_passes == 10000
    assert config.builtin_property_enabled == {}


@pytest.mark.unit
@pytest.mark.parametrize(
    "template_format,error",
    [("simulation", "Unknown run settings"), ("configuration", "Unknown configuration fields")],
)
def test_template_typos_are_rejected(tmp_path: Path, template_format, error):
    output_file = tmp_path / "input_template.yaml"
    write_template(output_file, template_format=template_format)
    text = output_file.read_text(encoding="utf-8").replace("temperature: 300.0", "temperaturee: 300.0")
    output_file.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        if template_format == "simulation":
            validate_simulation_input(load_raw_data(output_file))
        else:
            Configuration.from_file(str(output_file))


@pytest.mark.unit
def test_write_template_overwrite_requires_force(tmp_path: Path):
    output_file = tmp_path / "input_template.yaml"
    write_template(output_file)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_template(output_file)

    write_template(output_file, force=True)
    assert output_file.exists()


@pytest.mark.unit
def test_kmcpy_init_subcommand_writes_template(tmp_path: Path):
    output_file = tmp_path / "custom_template.yaml"
    exit_code = kmcpy_main(["init", "--output", str(output_file)])

    assert exit_code == 0
    assert output_file.exists()


@pytest.mark.unit
def test_kmcpy_init_help_includes_examples(capsys):
    with pytest.raises(SystemExit) as exc_info:
        kmcpy_main(["init", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "Generate a commented YAML template" in output
    assert "Examples:" in output
    assert "kmcpy run --input input.yaml" in output
    assert "--format configuration" in output
