"""Validation tests for schema_v4.json.

Mirrors the style of the existing tests: load the Draft202012 validator, then
assert that good specs validate and malformed specs are rejected. Demonstrates
backward compatibility (a v3 single-material spec still validates) and the new
multi_material rules.

Run directly (python3 test_schema_v4.py) or under pytest.
"""
import copy
import json
import os

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

HERE = os.path.dirname(os.path.abspath(__file__))


def _validator():
    schema = json.load(open(os.path.join(HERE, "..", "topoheat", "schema", "schema_v4.json")))
    Draft202012Validator.check_schema(schema)  # the schema itself is well formed
    return Draft202012Validator(schema)


def _load(name):
    return json.load(open(os.path.join(HERE, "..", name)))


def is_valid(v, spec):
    return v.is_valid(spec)


def _two_good_materials():
    return [
        {"name": "invar_36", "youngs_modulus": 1.41e11, "poisson_ratio": 0.29,
         "thermal_expansion": 1.2e-6, "conductivity": 10, "density": 8100},
        {"name": "aluminum_6061", "youngs_modulus": 6.9e10, "poisson_ratio": 0.33,
         "thermal_expansion": 23.6e-6, "conductivity": 167, "density": 2700},
    ]


def test_v3_single_material_backward_compatible():
    """(a) The existing v3 single-material spec still validates under v4."""
    v = _validator()
    spec = _load("example_problem.json")
    assert "material_selection" not in spec  # genuinely a v3-style spec
    assert is_valid(v, spec), list(v.iter_errors(spec))


def test_multimaterial_example_validates():
    """(b) example_multimaterial.json validates under v4."""
    v = _validator()
    spec = _load("example_multimaterial.json")
    assert spec["material_selection"]["mode"] == "multi_material"
    assert is_valid(v, spec), list(v.iter_errors(spec))


def test_rejections():
    """(c) At least four malformed multi-material specs are rejected."""
    v = _validator()
    base = _load("example_multimaterial.json")

    # 1) multi_material with only ONE material
    s1 = copy.deepcopy(base)
    s1["materials"] = [_two_good_materials()[0]]
    assert not is_valid(v, s1), "one material under multi_material should fail"

    # 2) a materials entry MISSING conductivity
    s2 = copy.deepcopy(base)
    s2["materials"] = _two_good_materials()
    del s2["materials"][1]["conductivity"]
    assert not is_valid(v, s2), "missing conductivity should fail"

    # 3) a materials entry with poisson_ratio 0.7 (out of [0, 0.5])
    s3 = copy.deepcopy(base)
    s3["materials"] = _two_good_materials()
    s3["materials"][0]["poisson_ratio"] = 0.7
    assert not is_valid(v, s3), "poisson_ratio 0.7 should fail"

    # 4) a STRAY key inside a materials entry (additionalProperties false)
    s4 = copy.deepcopy(base)
    s4["materials"] = _two_good_materials()
    s4["materials"][0]["color"] = "silver"
    assert not is_valid(v, s4), "stray key in materials entry should fail"

    # bonus rejections that further exercise the conditionals:
    # 5) sweep mode with only one material (minItems 2 required)
    s5 = copy.deepcopy(base)
    s5["material_selection"] = {"mode": "sweep"}
    s5["materials"] = [_two_good_materials()[0]]
    assert not is_valid(v, s5), "sweep with one material should fail"

    # 6) multi_material with THREE materials (maxItems 2)
    s6 = copy.deepcopy(base)
    s6["materials"] = _two_good_materials() + [_two_good_materials()[0]]
    assert not is_valid(v, s6), "three materials under multi_material should fail"


if __name__ == "__main__":
    test_v3_single_material_backward_compatible()
    print("PASS (a) v3 single-material spec backward compatible under v4")
    test_multimaterial_example_validates()
    print("PASS (b) example_multimaterial.json validates under v4")
    cases = test_rejections()
    print("PASS (c) rejection cases:", cases)
    print("ALL ASSERTIONS PASSED")
