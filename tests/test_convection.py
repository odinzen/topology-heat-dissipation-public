"""Convection (Robin film boundary) in the single-material thermo leg.

The lumped nodal Robin term has no machine-precision continuous closed form, so
these tests verify it by exact physical limits and consistency:

  * h -> infinity drives a node's temperature to the ambient value (in theta);
  * h -> 0 is a no-op (field matches the no-convection solve);
  * a convection boundary regularizes the conduction matrix, so a spec with NO
    thermal Dirichlet support but WITH convection now solves (previously raised);
  * a normal convection spec no longer raises NotImplementedError.

The engine works in theta = T_abs - T_ref. With reference_temperature == 0 the
theta field equals the absolute field, which keeps these limits easy to read.
"""
import os
import json
import copy

import numpy as np

from topoheat.engine import SpecEngine, validate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(ROOT, "example_problem.json")


def _base_spec():
    return copy.deepcopy(json.load(open(SPEC)))


def _uniform_rho(eng):
    return np.full(eng.nel, eng.spec["volume_fraction"])


# A node strictly interior to the 61x31 node grid (x inner), not in the example's
# thermal supports (left column) or thermal loads (interior heated band). Row 10,
# column 35 -> node id = 10*61 + 35 = 645.
INTERIOR_NODE = 10 * 61 + 35


def test_high_h_drives_node_to_ambient():
    # A large film coefficient on an interior node should pin that node's solved
    # temperature to (Tamb - T_ref) in theta. T_ref == 0 here, so theta == Tamb.
    spec = _base_spec()
    Tamb = 7.0
    spec["convection"] = [{"node_indices": [INTERIOR_NODE],
                           "film_coefficient": 1e6,  # >> conductivity (1.0)
                           "ambient_temperature": Tamb}]
    validate(spec)
    eng = SpecEngine(spec)
    st = eng.solve(_uniform_rho(eng))
    theta_node = st["T"][INTERIOR_NODE]
    assert abs(theta_node - (Tamb - spec["reference_temperature"])) < 1e-3


def test_low_h_is_a_noop():
    # A vanishing film coefficient must leave the field essentially unchanged
    # relative to the same spec WITHOUT any convection block.
    base = _base_spec()
    eng_no = SpecEngine(base)
    st_no = eng_no.solve(_uniform_rho(eng_no))

    spec = _base_spec()
    spec["convection"] = [{"node_indices": [INTERIOR_NODE],
                           "film_coefficient": 1e-12,
                           "ambient_temperature": 100.0}]  # huge Tamb, tiny h -> no effect
    validate(spec)
    eng = SpecEngine(spec)
    st = eng.solve(_uniform_rho(eng))

    assert np.allclose(st["T"], st_no["T"], atol=1e-7)


def test_convection_only_is_well_posed():
    # No thermal Dirichlet support, but a convection boundary: the +h diagonal
    # regularizes the conduction matrix, so this must now solve (previously the
    # guard raised) and yield a finite field. Empty thermal_supports is schema-valid
    # (the array itself has no minItems; only each entry's node list does).
    spec = _base_spec()
    spec["thermal_supports"] = []
    spec["convection"] = [{"node_indices": [INTERIOR_NODE],
                           "film_coefficient": 3.0,
                           "ambient_temperature": 2.0}]
    validate(spec)  # confirms the empty-supports + convection spec is schema-valid
    eng = SpecEngine(spec)
    assert eng.fixed_temp.size == 0
    assert eng.robin_diag.any()
    st = eng.solve(_uniform_rho(eng))
    assert np.all(np.isfinite(st["T"]))


def test_convection_only_no_supports_no_convection_still_rejected():
    # The relaxed guard must still reject the genuinely singular case: no thermal
    # Dirichlet support AND no convection at all.
    import pytest
    spec = _base_spec()
    spec["thermal_supports"] = []
    eng = SpecEngine(spec)
    assert not eng.robin_diag.any()
    with pytest.raises(ValueError, match="singular"):
        eng.solve(_uniform_rho(eng))


def test_convection_spec_no_longer_raises_not_implemented():
    # Regression: building/solving a convection spec must NOT raise the old
    # NotImplementedError in the single-material engine.
    spec = _base_spec()
    spec["convection"] = [{"node_indices": [INTERIOR_NODE],
                           "film_coefficient": 5.0,
                           "ambient_temperature": 1.5}]
    eng = SpecEngine(spec)            # no raise on build
    eng.solve(_uniform_rho(eng))      # no raise on solve


def test_reference_temperature_shift_correction():
    # With a nonzero T_ref, the high-h limit must still pin theta to (Tamb - T_ref),
    # proving the h*T_ref load correction is applied. Use Tamb == T_ref so the
    # convection target in theta is exactly 0 regardless of the (large) film value.
    spec = _base_spec()
    spec["reference_temperature"] = 4.0
    spec["convection"] = [{"node_indices": [INTERIOR_NODE],
                           "film_coefficient": 1e6,
                           "ambient_temperature": 4.0}]  # Tamb == T_ref -> theta ~ 0
    validate(spec)
    eng = SpecEngine(spec)
    st = eng.solve(_uniform_rho(eng))
    assert abs(st["T"][INTERIOR_NODE]) < 1e-3


def test_reference_temperature_shift_correction_nonzero_target():
    # Isolating test for the h*T_ref load correction: choose Tamb != T_ref so the
    # physically correct shifted target (Tamb - T_ref) is a specific NONZERO value.
    # With Tamb == T_ref the film RHS h*(Tamb - T_ref) collapses to 0 and a wrong
    # (or missing) correction would still pass; here it cannot. A large film
    # coefficient pins the interior node's solved theta to exactly (Tamb - T_ref).
    spec = _base_spec()
    T_ref = 4.0
    Tamb = 10.0
    spec["reference_temperature"] = T_ref
    spec["convection"] = [{"node_indices": [INTERIOR_NODE],
                           "film_coefficient": 1e6,  # >> conductivity (1.0)
                           "ambient_temperature": Tamb}]
    validate(spec)
    eng = SpecEngine(spec)
    st = eng.solve(_uniform_rho(eng))
    target = Tamb - T_ref  # = 6.0, a specific nonzero theta value
    assert abs(target) > 1.0  # guard: the target really is nonzero
    assert abs(st["T"][INTERIOR_NODE] - target) < 1e-3
