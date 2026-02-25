from pathlib import Path

import pytest

from kmcpy.cli.init import write_template
from kmcpy.cli.main import main as kmcpy_main
from kmcpy.io.config_io import ConfigIO
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
    assert "# ----- Runtime parameters -----" in template_text

    raw_data = ConfigIO._load_yaml_section(str(output_file), "kmc", "default")
    config = Configuration.from_dict(raw_data)
    assert config.structure_file == "path/to/structure.cif"
    assert config.kmc_passes == 10000
    assert config.builtin_property_enabled == {}


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
