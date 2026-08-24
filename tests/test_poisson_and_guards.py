"""Coverage for the Poisson-ratio threading and the robustness guards added
in the handoff hardening pass (see HANDOFF_TO_FABLE5.md, items P1-1/P1-2/P1-4)."""
import os, json, copy
import numpy as np
import scipy.sparse as sp
import pytest

from topoheat.engine import SpecEngine
from topoheat.topopt_thermoelastic import ThermoelasticTO, KE0 as KE0_MODULE, _factor_or_raise

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(ROOT, "example_problem.json")


def test_default_nu_matches_module_matrix():
    # nu defaults to 0.30, so the instance stiffness is byte-identical to the
    # historical module matrix -> all prior nu=0.30 results are unchanged.
    to = ThermoelasticTO(8, 4, rmin=1.5)
    assert np.allclose(to.KE0, KE0_MODULE)


def test_nu_actually_enters_the_stiffness():
    # A different Poisson ratio must produce a different element stiffness;
    # previously the spec value was silently ignored.
    lo = ThermoelasticTO(8, 4, rmin=1.5, nu=0.30)
    hi = ThermoelasticTO(8, 4, rmin=1.5, nu=0.45)
    assert not np.allclose(lo.KE0, hi.KE0)


def test_spec_poisson_is_threaded_into_engine():
    spec = json.load(open(SPEC))
    spec["material"]["poisson_ratio"] = 0.45
    eng = SpecEngine(spec)
    ref = ThermoelasticTO(*spec["domain_dimensions"], nu=0.45, rmin=spec["material"]["filter_radius"])
    assert np.allclose(eng.KE0, ref.KE0)
    assert eng.nu == 0.45


def test_factor_or_raise_on_singular():
    singular = sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))  # rank deficient
    with pytest.raises(ValueError, match="singular"):
        _factor_or_raise(singular, "mechanical")


def test_bridge_node_coords_match_mesh_ordering():
    # Regression: the kinematic-stability check builds rigid-body modes from
    # NumericalBridge._node_coordinates, which MUST number nodes the same way the
    # solver mesh does (x inner). A transposed ordering silently breaks the check
    # on any non-square grid. Assert exact agreement on a non-square grid.
    import topoheat.thermoelastic_fem as fem
    from topoheat.numerical_bridge import NumericalBridge
    nx, ny = 6, 2
    coords = NumericalBridge([nx, ny], "node_major")._node_coordinates()
    mesh_coords = fem.Mesh(nx, ny, h=1.0).coords()
    assert np.array_equal(coords, mesh_coords)


def test_fixed_components_out_of_range_rejected():
    from topoheat.numerical_bridge import NumericalBridge
    nb = NumericalBridge([6, 2], "node_major")  # 2D -> valid components 0,1
    with pytest.raises(ValueError, match="out of range"):
        nb.assemble([{"node_indices": [0], "fixed_components": [5]}], [])


def test_iters_must_be_positive():
    from topoheat.engine import run_from_spec
    with pytest.raises(ValueError, match="iters must be"):
        run_from_spec(SPEC, iters=0)


def test_under_constrained_spec_is_rejected():
    # Fix only the x component of a single node: leaves rigid translation in y
    # (and rotation) free, so the kinematic-stability check must reject it.
    spec = copy.deepcopy(json.load(open(SPEC)))
    spec["supports"] = [{"node_indices": [0], "fixed_components": [0]}]
    with pytest.raises(ValueError, match="rigid body"):
        SpecEngine(spec)


def test_convection_spec_now_builds_and_solves():
    # The Robin film term is now folded into the conduction matrix in the
    # single-material engine (+h on the diagonal, h*T_ref load correction in the
    # theta variable). A spec carrying convection must BUILD and SOLVE without the
    # old NotImplementedError, yielding a finite temperature field.
    spec = copy.deepcopy(json.load(open(SPEC)))
    spec["convection"] = [{"node_indices": [10], "film_coefficient": 5.0,
                           "ambient_temperature": 0.0}]
    eng = SpecEngine(spec)  # no longer raises
    assert eng.robin_diag[10] == 5.0
    st = eng.solve(np.full(eng.nel, spec["volume_fraction"]))
    assert np.all(np.isfinite(st["T"]))


def test_multimaterial_poisson_warns():
    # example_multimaterial.json carries per-material nu 0.29/0.33, which the
    # two-material path drops (matrices built at 0.30) — it must warn.
    from topoheat.spec_multimaterial import run_multimaterial_from_spec
    spec_path = os.path.join(ROOT, "example_multimaterial.json")
    with pytest.warns(RuntimeWarning, match="poisson_ratio"):
        run_multimaterial_from_spec(spec_path, iters=2)
