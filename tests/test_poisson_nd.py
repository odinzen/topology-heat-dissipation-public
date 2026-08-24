"""Poisson-ratio threading into the multimaterial (2D) and 3D thermoelastic
paths. Mirrors the single-material coverage in test_poisson_and_guards.py:

  - the spec/constructor nu must actually enter the per-instance element
    stiffness (a different nu -> a different self.KE0), and
  - the default nu=0.30 must reproduce the historical module matrix exactly, so
    every prior nu=0.30 result stays byte-identical.
"""
import os
import numpy as np
import pytest

from topoheat.multimaterial import TwoMaterialTO, KE0 as MM_KE0_MODULE
from topoheat.topopt3d import TO3D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- multimaterial (2D Q4) -------------------------------------------------
def test_multimaterial_nu_enters_stiffness():
    lo = TwoMaterialTO(8, 4, rmin=1.5, nu=0.30)
    hi = TwoMaterialTO(8, 4, rmin=1.5, nu=0.45)
    assert not np.allclose(lo.KE0, hi.KE0)


def test_multimaterial_default_nu_matches_module_matrix():
    # Default nu=0.30 -> self.KE0 is byte-identical to the historical module
    # KE0, so all prior nu=0.30 two-material results are unchanged.
    to = TwoMaterialTO(8, 4, rmin=1.5)
    assert np.array_equal(to.KE0, MM_KE0_MODULE)


# ---- 3D (hex8) -------------------------------------------------------------
def test_to3d_nu_enters_stiffness():
    lo = TO3D(4, 2, 2, rmin=1.5, nu=0.30)
    hi = TO3D(4, 2, 2, rmin=1.5, nu=0.45)
    assert not np.allclose(lo.KE0, hi.KE0)


def test_to3d_default_nu_matches_module_matrix():
    from topoheat.topopt3d import KE0 as TD_KE0_MODULE
    to = TO3D(4, 2, 2, rmin=1.5)
    assert np.array_equal(to.KE0, TD_KE0_MODULE)


# ---- spec path: differing per-material nu still warns -----------------------
def test_multimaterial_spec_differing_nu_warns():
    from topoheat.spec_multimaterial import run_multimaterial_from_spec
    spec_path = os.path.join(ROOT, "example_multimaterial.json")
    # example carries per-material nu 0.29/0.33 (differ) -> genuinely two-nu.
    with pytest.warns(RuntimeWarning, match="poisson_ratio"):
        run_multimaterial_from_spec(spec_path, iters=2)
