"""Heat-conduction objective: adjoint gradient vs finite differences, and that
optimization actually lowers thermal compliance. Self-adjoint minimization of
J = q^T T at fixed material volume (the classic heat-sink layout problem)."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from topoheat.engine import SpecEngine, validate


def _spec(nx=12, ny=8, volfrac=0.5):
    nx1 = nx + 1
    node = lambda ix, iy: iy * nx1 + ix
    sink = [node(ix, 0) for ix in range(nx // 2 - 1, nx // 2 + 2)]
    heat = [node(ix, iy) for iy in range(ny + 1) for ix in range(nx + 1)]
    return {
        "domain_dimensions": [nx, ny], "volume_fraction": volfrac,
        "units": {"length": "mm", "force": "N", "temperature": "K"},
        "dof_ordering": "node_major",
        "material": {"youngs_modulus": 1.0, "poisson_ratio": 0.3, "thermal_expansion": 0.02,
                     "conductivity": 1.0, "density_floor": 0.001,
                     "penalization": {"stiffness": 3, "conductivity": 3}, "filter_radius": 1.5},
        "reference_temperature": 0.0,
        "objective": {"mode": "heat_conduction"},
        "coupling": {"mode": "one_way"},
        "supports": [{"node_indices": sink, "fixed_components": [0, 1]}],
        "loads": [{"node_indices": [node(nx // 2, 0)], "force_vector": [0.0, -1.0]}],
        "thermal_supports": [{"node_indices": sink, "temperature": 0.0}],
        "thermal_loads": [{"node_indices": heat, "power": 1.0 / len(heat)}],
    }


def test_heat_conduction_gradient_matches_fd():
    spec = _spec()
    validate(spec)
    eng = SpecEngine(spec, beta_hs=4.0)
    rng = np.random.default_rng(3)
    rho = rng.uniform(0.3, 0.9, eng.nel)
    _, g = eng.full_gradient(rho)
    h = 1e-6
    for i in rng.choice(eng.nel, size=8, replace=False):
        rp = rho.copy(); rp[i] += h
        rm = rho.copy(); rm[i] -= h
        fd = (eng.objective(eng.filter_forward(rp))[0]
              - eng.objective(eng.filter_forward(rm))[0]) / (2 * h)
        assert abs(fd - g[i]) / (abs(fd) + 1e-30) < 1e-4


def test_heat_conduction_optimization_reduces_compliance():
    spec = _spec()
    eng = SpecEngine(spec)
    _, _, hist = eng.run(iters=40, beta_schedule=True)
    assert hist[-1] < 0.5 * hist[0]   # thermal compliance at least halves


def test_heat_conduction_3d_gradient_matches_fd():
    from topoheat.topopt3d import gradient_check_heat
    assert gradient_check_heat() < 1e-4
