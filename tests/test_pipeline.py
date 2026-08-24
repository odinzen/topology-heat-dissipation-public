import os, json, pytest
from jsonschema.exceptions import ValidationError
from topoheat.engine import validate, run_from_spec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(ROOT, "example_problem.json")

def test_contract_accepts_good():
    validate(json.load(open(SPEC)))

def test_contract_rejects_bad():
    s = json.load(open(SPEC)); s["loads"][0]["force_vector"] = [0, -3, 0]
    with pytest.raises(ValidationError):
        validate(s)

def test_pipeline_end_to_end(tmp_path):
    res = run_from_spec(SPEC, iters=15, out_prefix=str(tmp_path / "pytest_pipeline"))
    assert abs(res["volume_fraction"] - 0.45) < 0.03
    assert res["compliance_final"] > 0
