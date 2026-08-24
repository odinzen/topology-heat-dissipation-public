"""Smoke coverage for the two-way + projection optimizer combo
(topoheat/twoway_clean.py), which previously had zero tests.

ProjectedTwoWay is a multiple-inheritance combination of the projected
optimizer (ProjectedTO) and the two-way coupled optimizer (TwoWayTO). This
test runs a small, short optimization and checks the basic invariants:
the projected design respects the volume fraction target, and the compliance
history is finite, positive, and non-increasing end to end.
"""
import numpy as np

from topoheat.twoway_clean import ProjectedTwoWay


def test_projected_twoway_runs():
    volfrac = 0.45
    to = ProjectedTwoWay(16, 8, volfrac=volfrac, rmin=1.5, beta_hs=1.0,
                         beta_couple=2.0, relax=0.6, heat=2e-3, mech_load=0.05,
                         alpha=0.02, fp_max=40)
    rho, rp, hist = to.run(iters=12, beta_schedule=True)

    hist = np.asarray(hist, dtype=float)

    # Volume fraction of the projected design is sane and near the target.
    mean_vol = float(np.mean(rp))
    assert 0.0 < mean_vol < 1.0
    assert abs(mean_vol - volfrac) < 0.1

    # Compliance history is non-empty, finite, and positive.
    assert hist.size > 0
    assert np.all(np.isfinite(hist))
    assert np.all(hist > 0.0)

    # The optimizer made progress (or at least did not regress) overall.
    assert hist[-1] <= hist[0]
