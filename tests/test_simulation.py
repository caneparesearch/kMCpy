"""Tests for Simulation, its slot plugins, and input files."""

import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest
from monty.serialization import dumpfn

import kmcpy
from kmcpy.cli.main import main as kmcpy_main
from kmcpy.models.base import MODEL_FILETYPE, BaseModel
from kmcpy.models.registry import MODEL_CLASS_REGISTRY
from kmcpy.simulation import validate_simulation_input
from kmcpy.simulator.config import Configuration
from kmcpy.simulator.kmc import KMC

FILES = Path(__file__).parent / "files"
CIF = FILES / "EntryWithCollCode15546_Na4Zr2Si3O12_573K.cif"
SITE_MAPPING = {"Na": ["Na", "X"], "Zr": "Zr", "Si": ["Si", "P"], "O": "O"}
NASICON_EVENTS = dict(cutoffs={("Na+", "Na+"): 4.0, ("Na+", "Si4+"): 4.0}, labels=("Na1", "Na2"))
RUN_SETTINGS = dict(
    temperature=298,
    attempt_frequency=5e12,
    equilibration_passes=1,
    kmc_passes=100,
    random_seed=12345,
    name="NASICON",
)
# The flat Configuration needs these explicitly; Simulation derives them.
DERIVED_SETTINGS = dict(mobile_ion_specie="Na", mobile_ion_charge=1.0, elementary_hop_distance=3.47782)
EXAMPLES = Path(__file__).parent.parent / "example"


@pytest.fixture(scope="module")
def supercell():
    return kmcpy.LatticeStructure.from_cif(CIF, SITE_MAPPING, primitive=True).make_supercell((2, 1, 1))


@pytest.mark.integration
def test_simulation_matches_configuration_workflow(supercell, tmp_path):
    reference = KMC.from_config(
        Configuration(
            structure_file=str(CIF),
            model_file=str(FILES / "input" / "model.json"),
            event_file=str(FILES / "input" / "events.json"),
            initial_state_file=str(FILES / "input" / "initial_state.json"),
            supercell_shape=(2, 1, 1),
            site_mapping=SITE_MAPPING,
            convert_to_primitive_cell=True,
            **RUN_SETTINGS,
            **DERIVED_SETTINGS,
        )
    ).run(output_dir=tmp_path / "reference")

    simulation = kmcpy.Simulation(
        supercell,
        events=kmcpy.HopEvents(**NASICON_EVENTS),
        model=FILES / "input" / "model.json",
        state=FILES / "input" / "initial_state.json",
        **RUN_SETTINGS,
    )
    tracker = simulation.run(output_dir=tmp_path / "simulation")
    assert simulation.config.mobile_ion_specie == "Na"
    assert simulation.config.mobile_ion_charge == 1.0
    # Derived from the events; 3.47782 is the same length rounded.
    assert simulation.config.elementary_hop_distance == pytest.approx(3.47782, abs=1e-5)

    fixture_events = kmcpy.EventLib.from_file(str(FILES / "input" / "events.json")).events
    assert [(e.mobile_ion_indices, e.local_env_indices) for e in simulation.event_lib().events] == [
        (e.mobile_ion_indices, e.local_env_indices) for e in fixture_events
    ]
    # Only the correlation factor (last value) uses the hop length.
    result, expected = tracker.return_current_info(), reference.return_current_info()
    assert list(result[:-1]) == list(expected[:-1])
    assert result[-1] == pytest.approx(expected[-1], rel=1e-5)
    assert (tmp_path / "simulation" / "results_NASICON.csv.gz").exists()


@pytest.mark.unit
def test_models_and_states_are_interchangeable(supercell, tmp_path):
    simulation = kmcpy.Simulation(
        supercell,
        events=kmcpy.HopEvents(**NASICON_EVENTS),
        model=kmcpy.LocalBarrierModel.constant_barrier(300.0),
        state=kmcpy.RandomOccupation({"Na": 0.5}, seed=3),
        temperature=600,
        kmc_passes=5,
        equilibration_passes=0,
        random_seed=1,
    )
    simulation.run(output_dir=tmp_path / "first")
    # Each build starts from a fresh state, so repeated runs reproduce.
    first = simulation.build().run(output_dir=tmp_path / "a").return_current_info()
    second = simulation.build().run(output_dir=tmp_path / "b").return_current_info()
    assert list(first) == list(second)


@pytest.mark.unit
def test_random_occupation_places_exact_fractions(supercell):
    state = kmcpy.RandomOccupation({"Na": 0.75, "P": 0.5}, seed=7).build(supercell)
    allowed = supercell.active_site_order.allowed_species_by_active_site
    labels = [states[occupation] for states, occupation in zip(allowed, state.occupations)]

    assert labels.count("Na") == 12 and labels.count("X") == 4
    assert labels.count("P") == 6 and labels.count("Si") == 6
    again = kmcpy.RandomOccupation({"Na": 0.75, "P": 0.5}, seed=7).build(supercell)
    assert again.occupations == state.occupations
    vacancies = kmcpy.RandomOccupation({"Va": 0.25}, seed=7).build(supercell)
    assert sum(states[o] == "X" for states, o in zip(allowed, vacancies.occupations)) == 4


@pytest.mark.unit
def test_simulation_rejects_settings_owned_by_parts(supercell):
    parts = dict(events=kmcpy.HopEvents(), model=kmcpy.LocalBarrierModel.constant_barrier(1.0), state=[0] * 28)
    with pytest.raises(ValueError, match=r"event_file \(set via events=\)"):
        kmcpy.Simulation(supercell, event_file="events.json", **parts)
    with pytest.raises(ValueError, match="Unknown configuration fields"):
        kmcpy.Simulation(supercell, temprature=300, **parts)
    assert kmcpy.Simulation(supercell, **parts).config.mobile_ion_specie == "Na"


@pytest.mark.unit
def test_registered_models_load_by_type(tmp_path):
    @kmcpy.register_model("test_constant_rate")
    class ConstantRate(BaseModel):
        def __init__(self, rate=1.0):
            super().__init__(name="ConstantRate")
            self.rate = rate

        def compute_probability(self, event, runtime_config, simulation_state):
            return self.rate

        def as_dict(self):
            return {"rate": self.rate}

        @classmethod
        def from_dict(cls, data):
            return cls(data["rate"])

    try:
        path = tmp_path / "model.json"
        dumpfn({"filetype": MODEL_FILETYPE, "model_type": "test_constant_rate", "rate": 2.5}, path)
        model = BaseModel.load(path)
        assert isinstance(model, ConstantRate) and model.rate == 2.5

        with pytest.raises(ValueError, match="already registered"):
            kmcpy.register_model("local_barrier")(ConstantRate)
    finally:
        MODEL_CLASS_REGISTRY.pop("test_constant_rate", None)


@pytest.mark.integration
def test_examples_run(tmp_path):
    minimal = runpy.run_path(str(EXAMPLES / "minimal_example.py"))
    tracker = minimal["main"](output_dir=tmp_path / "minimal", kmc_passes=100)
    assert (tmp_path / "minimal" / "results_MinimalExample.csv.gz").exists()
    assert np.isclose(tracker.return_current_info()[4], 1.1823906621661553)

    tutorial = runpy.run_path(str(EXAMPLES / "tutorial_nasicon.py"))
    for argv in (["--kmc-passes", "5"], ["--model", "constant", "--na-fraction", "0.5", "--kmc-passes", "5"]):
        tutorial["main"](argv + ["--output-dir", str(tmp_path / "tutorial")])
    assert (tmp_path / "tutorial" / "properties_NASICON_constant.json.gz").exists()


# --- Input files -------------------------------------------------------------

NASICON_INPUT = """\
lattice_structure:
  structure: nasicon.cif
  site_mapping: {Na: [Na, X], Si: [Si, P]}
  supercell_shape: [2, 1, 1]
  primitive: true
events: {type: hop, cutoffs: [[Na+, Na+, 4.0], [Na+, Si4+, 4.0]], labels: [Na1, Na2]}
model: model.json
state: initial_state.json
run:
  temperature: 298
  attempt_frequency: 5.0e+12
  equilibration_passes: 1
  kmc_passes: 100
  random_seed: 12345
  name: NASICON
  output_dir: results
"""


@pytest.fixture
def nasicon_input(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy(CIF, project / "nasicon.cif")
    shutil.copy(FILES / "input" / "model.json", project / "model.json")
    shutil.copy(FILES / "input" / "initial_state.json", project / "initial_state.json")
    (project / "input.yaml").write_text(NASICON_INPUT)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # relative paths must resolve from the file
    return project


@pytest.mark.integration
def test_input_file_matches_python_setup(nasicon_input, supercell):
    expected = kmcpy.Simulation(
        supercell,
        events=kmcpy.HopEvents(**NASICON_EVENTS),
        model=FILES / "input" / "model.json",
        state=FILES / "input" / "initial_state.json",
        **RUN_SETTINGS,
    ).run(output_dir=nasicon_input / "python")

    tracker = kmcpy.Simulation.from_file(nasicon_input / "input.yaml").run()
    assert list(tracker.return_current_info()) == list(expected.return_current_info())
    assert (nasicon_input / "results" / "results_NASICON.csv.gz").exists()

    assert kmcpy_main(["run", "--input", str(nasicon_input / "input.yaml"), "--output_dir", "cli"]) == 0
    assert (nasicon_input.parent / "elsewhere" / "cli" / "results_NASICON.csv.gz").exists()


@pytest.mark.unit
def test_inline_components_from_input(tmp_path):
    shutil.copy(CIF, tmp_path / "nasicon.cif")
    (tmp_path / "input.yaml").write_text(
        """\
lattice_structure: {structure: nasicon.cif, site_mapping: {Na: [Na, X], Si: [Si, P]},
                    supercell_shape: [2, 1, 1], primitive: true}
events: {type: hop, cutoff: 4.0}
model: {type: local_barrier, default_barrier: 300.0}
state: {type: random, fractions: {Na: 0.5}, seed: 2}
run: {temperature: 600, kmc_passes: 3, equilibration_passes: 0, random_seed: 1}
"""
    )
    simulation = kmcpy.Simulation.from_file(tmp_path / "input.yaml")

    assert isinstance(simulation.model, kmcpy.LocalBarrierModel)
    assert isinstance(simulation.state, kmcpy.RandomOccupation)
    simulation.run(output_dir=tmp_path / "out")


@pytest.mark.unit
@pytest.mark.parametrize(
    "change,error",
    [
        (lambda d: d.update(extra={}), "Unknown simulation input sections"),
        (lambda d: d.pop("model"), r"missing sections: \['model'\]"),
        (lambda d: d.update(events={"type": "teleport"}), "Unknown events type 'teleport'"),
        (lambda d: d.update(model={"type": "magic"}), "Unknown model type 'magic'"),
        (lambda d: d["run"].update(temprature=300), r"Unknown run settings: \['temprature'\]"),
        (lambda d: d["run"].update(event_file="x.json"), r"Unknown run settings: \['event_file'\]"),
        (lambda d: d["lattice_structure"].update(supercell=[1, 1, 1]), "Unknown lattice_structure keys"),
        (lambda d: d["lattice_structure"].pop("structure"), "lattice_structure needs 'structure'"),
    ],
)
def test_input_validation_errors(change, error):
    data = {
        "lattice_structure": {"structure": "x.cif", "site_mapping": {"Li": ["Li", "X"]}},
        "events": "events.json",
        "model": "model.json",
        "state": "state.json",
        "run": {"temperature": 300},
    }
    validate_simulation_input(data)
    change(data)
    with pytest.raises(ValueError, match=error):
        validate_simulation_input(data)


@pytest.mark.integration
def test_example_notebook_runs(tmp_path, monkeypatch):
    import json
    import re

    import matplotlib

    matplotlib.use("Agg")
    (tmp_path / "files").symlink_to(EXAMPLES / "files")
    monkeypatch.chdir(tmp_path)

    notebook = json.loads((EXAMPLES / "NASICON.ipynb").read_text())
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and not "".join(cell["source"]).lstrip().startswith("#")
    )
    namespace = {}
    exec(compile(code.replace("plt.show()", "plt.close('all')"), "NASICON.ipynb", "exec"), namespace)
    assert np.isclose(namespace["info"][4], 1.1823906621661553)

    # The YAML cell describes the constant-barrier run from the notebook.
    yaml = re.search(r"```yaml\n(.*?)```", "".join(notebook["cells"][-1]["source"]), re.S).group(1)
    (tmp_path / "input.yaml").write_text(yaml)
    info = kmcpy.Simulation.from_file(tmp_path / "input.yaml").run().return_current_info()
    assert list(info) == list(namespace["info_constant"])


@pytest.mark.unit
def test_charge_and_hop_length_are_derived():
    from pymatgen.core import Lattice, Structure

    rock_salt = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.2), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    # 2x2x2: each of the 12 Mg neighbors within the cutoff is a distinct site.
    supercell = kmcpy.LatticeStructure(rock_salt, {"Mg": ["Mg", "X"]}).make_supercell(2)
    parts = dict(model=kmcpy.LocalBarrierModel.constant_barrier(500.0), state=[0] * 31 + [1])

    mg = kmcpy.Simulation(supercell, events=kmcpy.HopEvents(cutoff=3.0), **parts, temperature=1000, kmc_passes=1)
    assert mg.config.mobile_ion_specie == "Mg" and mg.config.mobile_ion_charge == 2.0
    mg.build()
    assert mg.config.elementary_hop_distance == pytest.approx(4.2 / np.sqrt(2))

    # Hops of two different lengths: warn and use their root mean square.
    cell = Structure(Lattice.orthorhombic(3.0, 4.0, 10.0), ["Na"] * 3, [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]])
    na_lattice = kmcpy.LatticeStructure(cell, {"Na": ["Na", "X"]})
    events = kmcpy.EventLib()
    events.add_event(kmcpy.Event(mobile_ion_indices=(0, 1), local_env_indices=()))
    events.add_event(kmcpy.Event(mobile_ion_indices=(0, 2), local_env_indices=()))
    events.set_index_metadata(na_lattice.active_site_order)
    events.generate_event_dependencies()
    mixed = kmcpy.Simulation(
        na_lattice,
        events=events,
        model=parts["model"],
        state=[0, 1, 1],
        mobile_ion_charge=1.0,
        temperature=1000,
        kmc_passes=1,
    )
    with pytest.warns(UserWarning, match="hop lengths from 1.500 to 2.000"):
        mixed.build()
    assert mixed.config.elementary_hop_distance == pytest.approx(np.sqrt((1.5**2 + 2.0**2) / 2))
