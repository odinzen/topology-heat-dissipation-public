"""Single entry point that runs every verification and benchmark.

Each block prints a labeled result. The script exits nonzero if any check
fails its tolerance. Figures are derived artifacts; regenerate them with the
run scripts in this folder after the optimizations they depend on.
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np

fails = []
def check(name, ok, detail=""):
    print(("PASS" if ok else "FAIL"), name, detail)
    if not ok: fails.append(name)

t0 = time.time()

# Forward closed form benchmarks
import topoheat.thermoelastic_fem as fem
check("conduction 1D", fem.verify_conduction() < 1e-10)
ps, pd = fem.verify_patch_free()
check("thermal patch (free)", ps < 1e-8 and pd < 1e-10)
sx, sy, an = fem.verify_clamped()
check("clamped thermal stress 2D", abs(sx-an)/abs(an) < 1e-6, "%.1f vs %.1f"%(sx,an))

# Adjoint gradients
import topoheat.topopt_thermoelastic as one, topoheat.topopt_twoway as two, topoheat.projected_to as proj, topoheat.topopt3d as d3
check("one way gradient", one.gradient_check()[0] < 1e-4)
check("two way gradient", two.gradient_check()[0] < 1e-4)
check("projected gradient", proj.gradient_check() < 1e-4)
check("3D gradient", d3.gradient_check() < 1e-4)
check("3D heat conduction gradient", d3.gradient_check_heat() < 1e-4)

# hex8 element exactness
import topoheat.hex8 as hex8  # prints its own checks under __main__ only; call directly
E,nu,al,dT=210e3,0.3,12e-6,50.0
D=hex8.Dmat(E,nu); eps0=al*dT*np.array([1,1,1,0,0,0.0]); sig=-D@eps0
check("clamped thermal stress 3D", abs(sig[0]-(-E*al*dT/(1-2*nu)))<1e-6)


# Thermal stress objective gradient
import topoheat.stress_to as stress_to
check("thermal stress gradient", stress_to.gradient_check() < 1e-4)

# Heat-conduction objective gradient (self-adjoint J = q^T T), via a small spec
from topoheat.engine import SpecEngine
_nx, _ny = 12, 8
_nd = lambda ix, iy: iy * (_nx + 1) + ix
_sink = [_nd(ix, 0) for ix in range(_nx // 2 - 1, _nx // 2 + 2)]
_heat = [_nd(ix, iy) for iy in range(_ny + 1) for ix in range(_nx + 1)]
_spec = {"domain_dimensions": [_nx, _ny], "volume_fraction": 0.5,
         "units": {"length": "mm", "force": "N", "temperature": "K"}, "dof_ordering": "node_major",
         "material": {"youngs_modulus": 1.0, "poisson_ratio": 0.3, "thermal_expansion": 0.02,
                      "conductivity": 1.0, "density_floor": 0.001,
                      "penalization": {"stiffness": 3, "conductivity": 3}, "filter_radius": 1.5},
         "reference_temperature": 0.0, "objective": {"mode": "heat_conduction"},
         "coupling": {"mode": "one_way"},
         "supports": [{"node_indices": _sink, "fixed_components": [0, 1]}],
         "loads": [{"node_indices": [_nd(_nx // 2, 0)], "force_vector": [0.0, -1.0]}],
         "thermal_supports": [{"node_indices": _sink, "temperature": 0.0}],
         "thermal_loads": [{"node_indices": _heat, "power": 1.0 / len(_heat)}]}
_eng = SpecEngine(_spec, beta_hs=4.0)
_rng = np.random.default_rng(3)
_rho = _rng.uniform(0.3, 0.9, _eng.nel)
_, _g = _eng.full_gradient(_rho)
_hh, _err = 1e-6, 0.0
for _i in _rng.choice(_eng.nel, size=8, replace=False):
    _rp = _rho.copy(); _rp[_i] += _hh
    _rm = _rho.copy(); _rm[_i] -= _hh
    _fd = (_eng.objective(_eng.filter_forward(_rp))[0] - _eng.objective(_eng.filter_forward(_rm))[0]) / (2 * _hh)
    _err = max(_err, abs(_fd - _g[_i]) / (abs(_fd) + 1e-30))
check("heat conduction gradient", _err < 1e-4)

# End to end contract: a valid spec validates, a malformed one is rejected
import json, copy
from jsonschema.exceptions import ValidationError
from topoheat.engine import validate
spec = json.load(open(os.path.join(ROOT, "example_problem.json")))
ok_good = True
try:
    validate(spec)
except Exception:
    ok_good = False
bad = copy.deepcopy(spec); bad["loads"][0]["force_vector"] = [0, -3, 0]
ok_bad = False
try:
    validate(bad)
except ValidationError:
    ok_bad = True
check("contract validates good spec", ok_good)
check("contract rejects bad spec", ok_bad)


# Multi-material per-element gradients
import topoheat.multimaterial as multimaterial
_er, _em, *_ = multimaterial.gradient_check()
check("multimaterial density gradient", _er < 1e-4)
check("multimaterial material gradient", _em < 1e-4)

print("\nelapsed %.1fs" % (time.time()-t0))
if fails:
    print("FAILURES:", fails); sys.exit(1)
print("ALL CHECKS PASSED")
