"""Characterization tests for LocalBarrierModel rule parsing, matching, and I/O."""

import re
from itertools import product

import pytest

from kmcpy.event import Event
from kmcpy.models import BarrierRule, LocalBarrierModel
from kmcpy.simulator.state import State

STATE_ALIAS_MESSAGE = (
    "must be a nonnegative integer state index or one of "
    "['match', 'mismatch', 'occupied', 'other', 'template', 'vacancy', 'vacant']"
)

ALL_RULE_TYPES = [
    {"name": "gate", "barrier": 100, "mobile_ion_indices": [0, 1], "local_env_indices": [2, 3]},
    {
        "type": "exact",
        "mobile_ion_indices": [0, 1],
        "local_env_indices": [1, 2, 3],
        "occupations": [0, "vacant", 1, "occupied"],
        "properties": {"barrier": 250, "prefactor": 2},
    },
    {"pattern": ["*", 1, "occupied", "2"], "barrier": 260.5},
    {"occupation": "vacant", "sites": [2, 3], "min_count": 1, "max_count": 2, "barrier": 270},
    {"type": "state_count", "state": 2, "count": 0, "properties": {"barrier": 280}},
    {"species": "Si", "min_count": 2, "barrier": 290},
    {"name": "mixed", "species": ["Si", "Al"], "sites": "all", "count": 3, "barrier": 295},
    {"type": "constant", "barrier": 999},
]

EXPECTED_RULE_PAYLOADS = [
    {
        "name": "gate",
        "type": "constant",
        "properties": {"barrier": 100.0},
        "mobile_ion_indices": [0, 1],
        "local_env_indices": [2, 3],
    },
    {
        "name": "rule_1",
        "type": "exact",
        "properties": {"barrier": 250.0, "prefactor": 2.0},
        "mobile_ion_indices": [0, 1],
        "local_env_indices": [1, 2, 3],
        "occupations": [0, 1, 1, 0],
    },
    {
        "name": "rule_2",
        "type": "pattern",
        "properties": {"barrier": 260.5},
        "pattern": ["*", 1, 0, 2],
        "sites": "canonical",
    },
    {
        "name": "rule_3",
        "type": "state_count",
        "properties": {"barrier": 270.0},
        "sites": [2, 3],
        "state": 1,
        "min_count": 1,
        "max_count": 2,
    },
    {
        "name": "rule_4",
        "type": "state_count",
        "properties": {"barrier": 280.0},
        "sites": "local_env",
        "state": 2,
        "count": 0,
    },
    {
        "name": "rule_5",
        "type": "species_count",
        "properties": {"barrier": 290.0},
        "sites": "local_env",
        "species": "Si",
        "min_count": 2,
    },
    {
        "name": "mixed",
        "type": "species_count",
        "properties": {"barrier": 295.0},
        "sites": "all",
        "species": ["Si", "Al"],
        "count": 3,
    },
    {"name": "rule_7", "type": "constant", "properties": {"barrier": 999.0}},
]

SITE_SPECIES = {
    1: {0: "P", 1: "Si", 2: "Al"},
    2: {0: "Si", 1: "Al", 2: "P"},
    3: {0: "Si", 1: "P", 2: "Al"},
    0: {0: "Na", 1: "X", 2: "Si"},
}


@pytest.mark.unit
def test_rule_payloads_are_normalized_and_round_trip():
    model = LocalBarrierModel(
        rules=ALL_RULE_TYPES,
        default_barrier=300,
        site_species={1: {0: "P", "vacant": "Si"}, "2": {0: "Si", 1: "Al"}},
    )
    payload = model.as_dict()

    assert payload["rules"] == EXPECTED_RULE_PAYLOADS
    assert list(payload["rules"][3]) == [
        "name", "type", "properties", "sites", "state", "min_count", "max_count"
    ]
    assert payload["site_species"] == {"1": {"0": "P", "1": "Si"}, "2": {"0": "Si", "1": "Al"}}
    assert payload["default_properties"] == {"barrier": 300.0}
    assert LocalBarrierModel.from_dict(payload).as_dict() == payload


MATCH_RULES = {
    "gate": {"barrier": 100, "mobile_ion_indices": [0, 1], "local_env_indices": [2, 3]},
    "exact": {
        "type": "exact",
        "mobile_ion_indices": [0, 1],
        "local_env_indices": [1, 2, 3],
        "occupations": [0, "vacant", 1, "occupied"],
        "barrier": 250,
    },
    "pattern": {"pattern": ["*", 1, "occupied", "2"], "barrier": 260.5},
    "pattern_mobile": {"pattern": [0, "*"], "sites": "mobile_ion", "barrier": 1},
    "state_from": {"state": 0, "sites": "from", "count": 1, "barrier": 1},
    "state_to": {"state": 2, "sites": "to", "count": 1, "barrier": 1},
    "state_range": {"occupation": "vacant", "sites": [2, 3], "min_count": 1, "max_count": 2, "barrier": 270},
    "state_zero": {"type": "state_count", "state": 2, "count": 0, "barrier": 280},
    "species_si": {"species": "Si", "min_count": 2, "barrier": 290},
    "species_mixed": {"species": ["Si", "Al"], "sites": "all", "count": 3, "barrier": 295},
    "species_canon": {"species": ["P"], "sites": "canonical", "max_count": 1, "barrier": 295},
}
MATCH_EVENTS = {
    "e01": Event(mobile_ion_indices=(0, 1), local_env_indices=(1, 2, 3)),
    "e32": Event(mobile_ion_indices=(3, 2), local_env_indices=(0, 1)),
    "gate": Event(mobile_ion_indices=(0, 1), local_env_indices=(2, 3)),
}
# Number of the 81 three-state occupations of 4 sites that each rule matches.
EXPECTED_MATCH_COUNTS = {
    ("gate", "e01"): 0, ("gate", "e32"): 0, ("gate", "gate"): 81,
    ("exact", "e01"): 1, ("exact", "e32"): 0,
    ("pattern", "e01"): 3, ("pattern", "e32"): 3,
    ("pattern_mobile", "e01"): 27, ("pattern_mobile", "e32"): 27,
    ("state_from", "e01"): 27, ("state_from", "e32"): 27,
    ("state_to", "e01"): 27, ("state_to", "e32"): 27,
    ("state_range", "e01"): 45, ("state_range", "e32"): 45,
    ("state_zero", "e01"): 24, ("state_zero", "e32"): 36,
    ("species_si", "e01"): 21, ("species_si", "e32"): 9,
    ("species_mixed", "e01"): 28, ("species_mixed", "e32"): 28,
    ("species_canon", "e01"): 60, ("species_canon", "e32"): 60,
}


@pytest.mark.unit
@pytest.mark.parametrize("rule_name,event_name", sorted(EXPECTED_MATCH_COUNTS))
def test_rule_match_counts(rule_name, event_name):
    model = LocalBarrierModel(
        rules=[MATCH_RULES[rule_name]],
        default_barrier=-1,
        site_species=SITE_SPECIES,
    )
    event = MATCH_EVENTS[event_name]
    matches = sum(
        model.compute(simulation_state=State(occupations=list(occupation)), event=event) != -1
        for occupation in product(range(3), repeat=4)
    )
    assert matches == EXPECTED_MATCH_COUNTS[(rule_name, event_name)]


INVALID_RULES = [
    ("not a dict", TypeError, "Each local barrier rule must be a dictionary"),
    ({"type": "bogus", "barrier": 1}, ValueError,
     "Unsupported local barrier rule type 'bogus'. Supported types: "
     "['constant', 'exact', 'pattern', 'species_count', 'state_count']"),
    ({"type": "", "barrier": 1}, ValueError, "Rule 'type' must be a non-empty string"),
    ({"barrier": True}, TypeError, "'barrier' must be numeric"),
    ({"properties": {}}, ValueError, "'properties' must be a non-empty object"),
    ({"properties": {"barrier": "x"}}, TypeError, "Property 'barrier' must be a numeric value"),
    ({"properties": {"": 1.0}}, ValueError, "Property names must be non-empty strings"),
    ({"type": "exact", "mobile_ion_indices": [0, 1], "local_env_indices": [2], "occupations": [0, 1], "barrier": 1},
     ValueError, "Exact rule occupation length must match canonical site count (3), got 2"),
    ({"type": "exact", "mobile_ion_indices": "01", "local_env_indices": [], "occupations": [0, 1], "barrier": 1},
     TypeError, "'mobile_ion_indices' must be a list or tuple of integers"),
    ({"type": "exact", "mobile_ion_indices": [], "local_env_indices": [], "occupations": [0], "barrier": 1},
     ValueError, "'mobile_ion_indices' must be non-empty"),
    ({"type": "exact", "mobile_ion_indices": [0, 1.5], "local_env_indices": [], "occupations": [0, 1], "barrier": 1},
     TypeError, "'mobile_ion_indices' must contain integers only"),
    ({"type": "exact", "mobile_ion_indices": [0, 1], "local_env_indices": [], "occupations": [], "barrier": 1},
     ValueError, "'occupations' must be non-empty"),
    ({"pattern": [], "barrier": 1}, ValueError, "'pattern' must be non-empty"),
    ({"pattern": "01", "barrier": 1}, TypeError, "'pattern' must be a list or tuple"),
    ({"pattern": ["?"], "barrier": 1}, ValueError, "'pattern' " + STATE_ALIAS_MESSAGE),
    ({"pattern": [-1], "barrier": 1}, ValueError, "'pattern' " + STATE_ALIAS_MESSAGE),
    ({"pattern": [0], "sites": "nowhere", "barrier": 1}, ValueError,
     "Unsupported sites selector 'nowhere'. Supported selectors: "
     "['all', 'canonical', 'from', 'local_env', 'mobile_ion', 'to']"),
    ({"pattern": [0], "sites": [0.5], "barrier": 1}, TypeError, "'sites' must contain integers only"),
    ({"state": "occupied", "barrier": 1}, ValueError,
     "Count rules must provide at least one of 'count', 'min_count', or 'max_count'"),
    ({"state": "occupied", "count": 1, "min_count": 0, "barrier": 1}, ValueError,
     "'count' cannot be combined with min_count or max_count"),
    ({"state": "occupied", "count": -1, "barrier": 1}, ValueError, "'count' must be non-negative"),
    ({"state": "occupied", "count": 1.0, "barrier": 1}, TypeError, "'count' must be an integer"),
    ({"state": True, "count": 1, "barrier": 1}, TypeError, "'state' " + STATE_ALIAS_MESSAGE),
    ({"type": "state_count", "count": 1, "barrier": 1}, ValueError, "'state' " + STATE_ALIAS_MESSAGE),
    ({"species": [], "count": 1, "barrier": 1}, ValueError, "'species' must be a string or non-empty list"),
    ({"species": [""], "count": 1, "barrier": 1}, ValueError, "'species' entries must be non-empty strings"),
    ({"species": 3, "count": 1, "barrier": 1}, ValueError, "'species' must be a string or non-empty list"),
]


@pytest.mark.unit
@pytest.mark.parametrize("rule,error,message", INVALID_RULES)
def test_invalid_rules_are_rejected(rule, error, message):
    with pytest.raises(error, match=f"^{re.escape(message)}$"):
        LocalBarrierModel(rules=[rule])


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs,error,message",
    [
        ({"site_species": [1]}, TypeError, "'site_species' must be a mapping"),
        ({"site_species": {"a": {0: "Si"}}}, TypeError, "site_species keys must be site indices"),
        ({"site_species": {1: [0]}}, TypeError,
         "site_species values must map occupation states to species strings"),
        ({"site_species": {1: {0: ""}}}, ValueError,
         "site_species species values must be non-empty strings"),
        ({"default_barrier": "x"}, TypeError, "'default_barrier' must be numeric"),
        ({"default_properties": {}}, ValueError, "'properties' must be a non-empty object"),
    ],
)
def test_invalid_model_options_are_rejected(kwargs, error, message):
    with pytest.raises(error, match=f"^{re.escape(message)}$"):
        LocalBarrierModel(**kwargs)


@pytest.mark.unit
def test_duplicate_exact_rules_are_rejected():
    rule = {
        "type": "exact",
        "mobile_ion_indices": [0, 1],
        "local_env_indices": [2],
        "occupations": [0, 1, 0],
        "barrier": 1,
    }
    with pytest.raises(ValueError, match=re.escape(
        "Duplicate exact local-barrier rule detected: mobile_ion_indices=(0, 1), "
        "canonical_sites=(0, 1, 2), occupations=(0, 1, 0)"
    )):
        LocalBarrierModel(rules=[rule, dict(rule)])


@pytest.mark.unit
def test_rule_evaluation_errors():
    state = State(occupations=[0, 1, 0])
    event = Event(mobile_ion_indices=(0, 1), local_env_indices=(2,))

    pattern_model = LocalBarrierModel(rules=[{"pattern": [0, 1], "sites": "canonical", "barrier": 1}])
    with pytest.raises(ValueError, match=re.escape(
        "Pattern rule 'rule_0' has length 2 but selected 3 sites"
    )):
        pattern_model.compute(simulation_state=state, event=event)

    species_model = LocalBarrierModel(rules=[{"species": "Si", "count": 1, "barrier": 1}])
    with pytest.raises(ValueError, match=re.escape(
        "species_count rules require site_species for every counted site; missing site 2"
    )):
        species_model.compute(simulation_state=state, event=event)


@pytest.mark.unit
def test_add_rule_helpers_return_names_and_build_equivalent_rules():
    model = LocalBarrierModel(default_barrier=300.0)
    names = [
        model.add_exact_rule([0, 1], [], [0, 1], barrier=250.0),
        model.add_state_count_rule("vacant", barrier=270.0, sites=[2, 3], min_count=1, max_count=2),
        model.add_species_count_rule(["Si", "Al"], properties={"barrier": 295.0}, name="mixed", count=3),
        model.add_pattern_rule(["*", 1], barrier=260.0, sites="mobile_ion"),
    ]

    assert names == ["rule_0", "rule_1", "mixed", "rule_3"]
    assert model.as_dict()["rules"] == [
        {"name": "rule_0", "type": "exact", "properties": {"barrier": 250.0},
         "mobile_ion_indices": [0, 1], "local_env_indices": [], "occupations": [0, 1]},
        {"name": "rule_1", "type": "state_count", "properties": {"barrier": 270.0},
         "sites": [2, 3], "state": 1, "min_count": 1, "max_count": 2},
        {"name": "mixed", "type": "species_count", "properties": {"barrier": 295.0},
         "sites": "local_env", "species": ["Si", "Al"], "count": 3},
        {"name": "rule_3", "type": "pattern", "properties": {"barrier": 260.0},
         "pattern": ["*", 1], "sites": "mobile_ion"},
    ]


@pytest.mark.unit
def test_barrier_rule_objects_can_be_built_and_added_directly():
    rule = BarrierRule.from_dict(
        {"state": "vacant", "sites": [2, 3], "min_count": 1, "barrier": 270},
        default_name="crowded",
    )

    assert rule == BarrierRule(
        name="crowded",
        type="state_count",
        properties={"barrier": 270.0},
        sites=(2, 3),
        state=1,
        min_count=1,
    )
    assert BarrierRule.from_dict(rule.as_dict()) == rule

    model = LocalBarrierModel(default_barrier=300.0)
    model.add_rule(rule)
    assert model.rules == [rule]
    state = State(occupations=[0, 1, 1, 0])
    event = Event(mobile_ion_indices=(0, 1), local_env_indices=(2, 3))
    assert model.compute(simulation_state=state, event=event) == 270.0

