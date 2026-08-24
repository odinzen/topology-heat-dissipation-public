import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, copy
from jsonschema.exceptions import ValidationError
from topoheat.engine import validate, run_from_spec

good = json.load(open("example_problem.json"))

# 1) the contract rejects physically invalid specifications before the solver
bad_cases = {
    "3-component force in 2D": lambda s: s["loads"][0].__setitem__("force_vector",[0,-3,0]),
    "missing material": lambda s: s.pop("material"),
    "poisson over half": lambda s: s["material"].__setitem__("poisson_ratio",0.7),
    "stray key": lambda s: s.__setitem__("mystery",1),
}
for name, mut in bad_cases.items():
    s = copy.deepcopy(good); mut(s)
    try:
        validate(s); print("NOT REJECTED:", name)
    except ValidationError:
        print("rejected:", name)

# 2) a valid spec flows all the way to a design
print("valid spec ->", json.dumps({k:run_from_spec("example_problem.json", iters=60)[k]
      for k in ["iterations","reduction","volume_fraction","gray_fraction"]}))
