from pathlib import Path

import pytest

from kmcpy.cli.init import write_template
from kmcpy.cli.main import main as kmcpy_main
from kmcpy.simulator.config import Configuration


@pytest.mark.unit
def test_write_template_creates_parseable_config(tmp_path: Path):
    output_file = tmp_path / "input_template.yaml"
    written = write_template(output_file)

    assert written == output_file
    assert output_file.exists()

    template_text = output_file.read_text(encoding="utf-8")
    assert "structure_file" in template_text
    assert "model_file" in template_text
    assert "kmc_passes" in template_text
    assert "builtin_property_enabled" in template_text
    assert "property_callbacks" in template_text
    assert "# ----- Runtime fields -----" in template_text

    from_file_config = Configuration.from_file(str(output_file))
    assert from_file_config.structure_file == "path/to/structure.cif"
    assert from_file_config.kmc_passes == 10000
    assert from_file_config.builtin_property_enabled == {}


@pytest.mark.unit
def test_configuration_from_file_rejects_unknown_template_keys(tmp_path: Path):
    output_file = tmp_path / "input_template.yaml"
    write_template(output_file)
    template_text = output_file.read_text(encoding="utf-8")
    template_text = template_text.replace(
        "temperature: 300.0",
        "temperaturee: 300.0",
    )
    output_file.write_text(template_text, encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown configuration fields"):
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
    assert "run_kmc --input input.yaml" in output
