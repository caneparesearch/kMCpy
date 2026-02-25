from pathlib import Path

import pytest

from kmcpy.cli.main import main as kmcpy_main
from kmcpy.cli.pack_model import write_model_bundle
from kmcpy.io.config_io import ConfigIO


@pytest.mark.unit
def test_kmcpy_pack_model_subcommand(tmp_path: Path):
    root = Path(__file__).parent / "files" / "input"
    output = tmp_path / "model.json"

    exit_code = kmcpy_main(
        [
            "pack-model",
            "--kra-lce",
            str(root / "lce.json"),
            "--kra-fit",
            str(root / "fitting_results.json"),
            "--site-lce",
            str(root / "lce_site.json"),
            "--site-fit",
            str(root / "fitting_results_site.json"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert output.exists()
    loaded = ConfigIO.load_model_bundle(str(output))
    assert loaded["format"] == "kmcpy.model_bundle.v1"
    assert "kra" in loaded


@pytest.mark.unit
def test_pack_model_requires_site_args_in_pair(tmp_path: Path):
    root = Path(__file__).parent / "files" / "input"
    with pytest.raises(ValueError, match="both --site-lce and --site-fit"):
        write_model_bundle(
            output=tmp_path / "bundle.json",
            kra_lce=str(root / "lce.json"),
            kra_fit=str(root / "fitting_results.json"),
            site_lce=str(root / "lce_site.json"),
            site_fit=None,
        )
