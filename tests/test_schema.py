import json, os, copy, pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from topoheat.thermal_bridge import ThermoelasticBridge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = json.load(open(os.path.join(ROOT, "topoheat", "schema", "schema_v3.json")))
SPEC = json.load(open(os.path.join(ROOT, "example_problem.json")))

def test_metaschema_valid():
    Draft202012Validator.check_schema(SCHEMA)

def test_good_spec_valid():
    Draft202012Validator(SCHEMA).validate(SPEC)

@pytest.mark.parametrize("mut", [
    lambda s: s["loads"][0].__setitem__("force_vector", [0, -3, 0]),
    lambda s: s.pop("material"),
    lambda s: s["material"].__setitem__("poisson_ratio", 0.7),
    lambda s: s.__setitem__("mystery", 1),
])
def test_bad_spec_rejected(mut):
    s = copy.deepcopy(SPEC); mut(s)
    with pytest.raises(ValidationError):
        Draft202012Validator(SCHEMA).validate(s)

def test_thermal_bridge_assembly():
    b = ThermoelasticBridge([4, 3], "node_major")
    heat, ftn, ftv, robin = b.assemble_thermal(
        [{"node_indices": [0, 1, 2, 3], "temperature": 373.15}],
        [{"node_indices": [19], "power": 5.0}],
        [{"node_indices": [16, 17, 18, 19], "film_coefficient": 10.0, "ambient_temperature": 293.15}])
    assert b.thermal_dof_count() == 20
    assert abs(heat[19] - (5 + 10 * 293.15)) < 1e-9
