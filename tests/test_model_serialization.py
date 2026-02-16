from pathlib import Path

from kmcpy.models.composite_lce_model import CompositeLCEModel
from kmcpy.models.local_cluster_expansion import LocalClusterExpansion


def test_local_cluster_expansion_from_legacy_json_sets_name():
    root = Path(__file__).parent / "files" / "input"
    model = LocalClusterExpansion.from_json(str(root / "lce.json"))

    payload = model.as_dict()

    assert model.name == "LocalClusterExpansion"
    assert payload["name"] == "LocalClusterExpansion"


def test_composite_lce_model_as_dict_with_legacy_json_inputs():
    root = Path(__file__).parent / "files" / "input"
    model = CompositeLCEModel.from_json(
        lce_fname=str(root / "lce.json"),
        fitting_results=str(root / "fitting_results.json"),
        lce_site_fname=str(root / "lce_site.json"),
        fitting_results_site=str(root / "fitting_results_site.json"),
    )

    payload = model.as_dict()

    assert payload["site_model"]["name"] == "LocalClusterExpansion"
    assert payload["kra_model"]["name"] == "LocalClusterExpansion"
