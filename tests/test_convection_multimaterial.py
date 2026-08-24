"""Convection (Robin film boundary) in the MULTIMATERIAL thermo leg.

Mirrors tests/test_convection.py but drives the two-material path
(TwoMaterialTO + spec_multimaterial._build_bc_from_spec). The lumped nodal Robin
term has no machine-precision continuous closed form, so these tests verify it by
exact physical limits and consistency:

  * a convection spec no longer raises NotImplementedError and solves to a finite
    field;
  * h -> infinity drives a node's temperature toward the ambient value;
  * h -> 0 reproduces the no-convection field (allclose).

Unlike the single-material SpecEngine (which works in theta = T_abs - T_ref), the
multi-material path works in ABSOLUTE temperature (tref=0, no theta shift), so the
+h diagonal is exact against the bridge's +h*Tamb load with no T_ref correction.
example_multimaterial.json sets reference_temperature 0.0 and its two materials
carry differing poisson_ratio, so building the optimizer emits a RuntimeWarning
about the single representable nu — that is expected and filtered below.
"""
import os
import json
import copy
import warnings

import numpy as np
import pytest

from topoheat.spec_multimaterial import (
    validate_v4,
    _relative_materials,
    _build_bc_from_spec,
)
from topoheat.multimaterial import TwoMaterialTO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(ROOT, "example_multimaterial.json")

# An interior node (61x31 node grid), column 20 so it is NOT a thermal support
# (left column 0) and NOT in the heated band (columns 30..). Row 5, col 20.
INTERIOR_NODE = 5 * 61 + 20


def _base_spec():
    return copy.deepcopy(json.load(open(SPEC)))


def _build_optimizer(spec):
    """Build a TwoMaterialTO from the spec and wire its BCs, exactly as
    run_multimaterial_from_spec does (minus the optimization loop)."""
    matA, matB = _relative_materials(spec["materials"])
    nx, ny = spec["domain_dimensions"]
    mat = spec["material"]
    nu = float(mat.get("poisson_ratio", 0.30))
    # The example's two materials differ in nu; building emits a RuntimeWarning we
    # do not care about here. Filter it so the suite stays clean.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        to = TwoMaterialTO(
            nx, ny, matA=matA, matB=matB, nu=nu,
            volfrac=spec["volume_fraction"],
            rmin=mat["filter_radius"],
            penal=mat["penalization"]["stiffness"],
            penal_k=mat["penalization"]["conductivity"],
            penal_f=mat["penalization"]["stiffness"],
            floor=mat["density_floor"],
            heat=0.0, mech_load=0.0)
        _build_bc_from_spec(to, spec)
    return to


def _uniform_fields(to):
    rho = np.full(to.nel, 0.45)
    mu = np.full(to.nel, 0.5)
    return rho, mu


def test_convection_no_longer_raises_and_solves_finite():
    # Regression: a convection spec must NOT raise NotImplementedError in the
    # multi-material path, and must solve to a finite temperature field.
    spec = _base_spec()
    spec["convection"] = [{"node_indices": [INTERIOR_NODE],
                           "film_coefficient": 5.0,
                           "ambient_temperature": 1.5}]
    validate_v4(spec)
    to = _build_optimizer(spec)            # no NotImplementedError on build
    assert to.robin_diag.any()
    rho, mu = _uniform_fields(to)
    st = to.solve(rho, mu)                 # no raise on solve
    assert np.all(np.isfinite(st["T"]))


def test_high_h_drives_node_to_ambient():
    # A large film coefficient on an interior node should pin that node's solved
    # ABSOLUTE temperature to the ambient value (multimaterial works in absolute T).
    spec = _base_spec()
    Tamb = 7.0
    spec["convection"] = [{"node_indices": [INTERIOR_NODE],
                           "film_coefficient": 1e6,  # >> relative conductivity
                           "ambient_temperature": Tamb}]
    validate_v4(spec)
    to = _build_optimizer(spec)
    rho, mu = _uniform_fields(to)
    st = to.solve(rho, mu)
    assert abs(st["T"][INTERIOR_NODE] - Tamb) < 1e-3


def test_low_h_is_a_noop():
    # A vanishing film coefficient must leave the field essentially unchanged
    # relative to the same spec WITHOUT any convection block.
    base = _base_spec()
    to_no = _build_optimizer(base)
    rho, mu = _uniform_fields(to_no)
    st_no = to_no.solve(rho, mu)

    spec = _base_spec()
    spec["convection"] = [{"node_indices": [INTERIOR_NODE],
                           "film_coefficient": 1e-12,
                           "ambient_temperature": 100.0}]  # huge Tamb, tiny h
    validate_v4(spec)
    to = _build_optimizer(spec)
    st = to.solve(rho, mu)

    assert np.allclose(st["T"], st_no["T"], atol=1e-7)


def test_convection_only_is_well_posed():
    # No thermal Dirichlet support, but a convection boundary: the +h diagonal
    # regularizes the conduction matrix, so this must solve and yield a finite
    # field rather than raising the singular guard.
    spec = _base_spec()
    spec["thermal_supports"] = []
    spec["convection"] = [{"node_indices": [INTERIOR_NODE],
                           "film_coefficient": 3.0,
                           "ambient_temperature": 2.0}]
    validate_v4(spec)
    to = _build_optimizer(spec)
    assert to.fixed_temp.size == 0
    assert to.robin_diag.any()
    rho, mu = _uniform_fields(to)
    st = to.solve(rho, mu)
    assert np.all(np.isfinite(st["T"]))


def test_no_supports_no_convection_still_rejected():
    # The relaxed guard must still reject the genuinely singular case: no thermal
    # Dirichlet support AND no convection at all.
    spec = _base_spec()
    spec["thermal_supports"] = []
    spec.pop("convection", None)
    with pytest.raises(ValueError, match="singular"):
        _build_optimizer(spec)
