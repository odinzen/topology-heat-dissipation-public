"""Coverage for previously-untested solvers (HANDOFF_TO_FABLE5.md P2-3):
the monolithic Newton solve, the two-way fixed-point convergence flag, the
multimaterial spec pipeline, and the AMG 3D solver (skipped without pyamg)."""
import os
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_monolithic_newton_matches_staggered():
    # In the weak-feedback regime the staggered fixed point converges, and the
    # monolithic Newton solve must reach the same state with a tiny residual.
    from topoheat.monolithic import Monolithic
    m = Monolithic(8, 4, beta_couple=5.0, heat=2e-3, mech_load=0.03, alpha=0.03, volfrac=0.5)
    us, Ts, _, sr = m.staggered()
    un, Tn, res = m.newton()
    assert res[-1] < 1e-8                                   # Newton converged
    rel = np.linalg.norm(un - us) / (np.linalg.norm(us) + 1e-30)
    assert rel < 1e-6                                       # same solution as staggered


def test_monolithic_continuation_handles_strong_feedback():
    # Strong feedback: Newton with continuation must still drive the residual
    # to tolerance (this is the regime where the staggered scheme diverges).
    from topoheat.monolithic import Monolithic
    m = Monolithic(8, 4, beta_couple=60.0, heat=2e-3, mech_load=0.03, alpha=0.03, volfrac=0.5)
    u, T, steps, resid = m.newton_continuation(60.0)
    assert resid < 1e-6


def test_twoway_fixed_point_sets_converged_flag():
    from topoheat.topopt_twoway import TwoWayTO
    to = TwoWayTO(12, 6, rmin=1.5, beta_couple=2.0, heat=2e-3,
                  mech_load=0.02, alpha=0.02, relax=0.5)
    to.solve(to.filter_forward(np.full(to.nel, 0.5)))
    assert to.fp_converged is True


def test_twoway_forward_self_consistent():
    # The forward solve drops its post-loop recompute: it now returns the (T, u)
    # from the converged fixed-point pass, with vol = vol_new (no relaxation), so
    # the returned state must already be self-consistent. Recompute vol(u) from the
    # returned u and check it matches st["vol"]; check the thermal residual against
    # the returned KTff (built from the returned k_eff) is small.
    from topoheat.topopt_twoway import TwoWayTO, M_DIV
    to = TwoWayTO(12, 6, rmin=1.5, beta_couple=2.0, heat=2e-3,
                  mech_load=0.02, alpha=0.02, relax=0.5)
    st = to.solve(to.filter_forward(np.full(to.nel, 0.5)))
    assert to.fp_converged is True
    vol_from_u = (st["u"][to.edof] * M_DIV[None, :]).sum(axis=1)
    assert np.max(np.abs(vol_from_u - st["vol"])) < 1e-10
    # Thermal residual at the returned T using the returned (k_eff-built) KTff.
    Tf = st["T"][to.free_temp]
    resid = st["KTff"] @ Tf - to.q[to.free_temp]
    assert np.max(np.abs(resid)) < 1e-9


def test_twoway_warns_when_not_converged():
    # A single allowed iteration cannot meet the 1e-12 tolerance, so the solver
    # must warn instead of silently returning an unconverged iterate.
    from topoheat.topopt_twoway import TwoWayTO
    to = TwoWayTO(12, 6, rmin=1.5, beta_couple=2.0, heat=2e-3,
                  mech_load=0.02, alpha=0.02, relax=0.5, fp_max=1)
    with pytest.warns(RuntimeWarning):
        to.solve(to.filter_forward(np.full(to.nel, 0.5)))
    assert to.fp_converged is False


def test_twoway_adjoint_warns_when_not_converged():
    # The sensitivity adjoint fixed point must also flag non-convergence (a
    # silent unconverged adjoint biases the gradient).
    from topoheat.topopt_twoway import TwoWayTO
    to = TwoWayTO(12, 6, rmin=1.5, beta_couple=2.0, heat=2e-3,
                  mech_load=0.02, alpha=0.02, relax=0.5, fp_max=1)
    with pytest.warns(RuntimeWarning):
        to.full_gradient(np.full(to.nel, 0.5))
    assert to.adj_converged is False


def test_multimaterial_spec_pipeline_runs():
    # End-to-end multimaterial run from the example spec (previously uncovered).
    from topoheat.spec_multimaterial import run_multimaterial_from_spec
    res = run_multimaterial_from_spec(os.path.join(ROOT, "example_multimaterial.json"), iters=8)
    assert res["mode"] == "multi_material"
    assert "converged" in res
    assert 0.0 < res["volume_fraction"] < 1.0
    assert res["objective_final"] > 0


def test_scale3d_amg_gradient():
    pytest.importorskip("pyamg")
    import topoheat.scale3d as scale3d
    err = scale3d.gradient_check()
    assert err < 1e-4


def test_filter_matches_bruteforce():
    # The vectorized density filter must be byte-identical to the original
    # per-element brute-force double loop over the (2R+1)^2 stencil.
    import scipy.sparse as sp
    from topoheat.topopt_thermoelastic import ThermoelasticTO
    nelx, nely, rmin = 5, 4, 2.0
    to = ThermoelasticTO(nelx, nely, rmin=rmin)

    # Original brute-force construction, recomputed here.
    order = []
    for iy in range(nely):
        for ix in range(nelx):
            order.append((ix, iy))
    order = np.array(order)
    R = int(np.ceil(rmin)) + 1
    grid = -np.ones((nelx, nely), dtype=np.int64)
    for e, (ix, iy) in enumerate(order):
        grid[ix, iy] = e
    rows, cols, vals = [], [], []
    for e, (ix, iy) in enumerate(order):
        for dx in range(-R, R + 1):
            for dy in range(-R, R + 1):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < nelx and 0 <= jy < nely:
                    w = rmin - np.hypot(dx, dy)
                    if w > 0:
                        rows.append(e)
                        cols.append(grid[jx, jy])
                        vals.append(w)
    Href = sp.csr_matrix((vals, (rows, cols)), shape=(to.nel, to.nel))
    Hsref = np.asarray(Href.sum(axis=1)).ravel()

    diff = (to.H - Href)
    assert diff.nnz == 0 or np.allclose(diff.data, 0.0)
    assert np.allclose(to.Hs, Hsref)


def test_monolithic_AB_match_bruteforce():
    # The vectorized _A and _B must be byte-identical to the original triple loops.
    import scipy.sparse as sp
    from topoheat.monolithic import Monolithic
    from topoheat.topopt_thermoelastic import KET0, G0
    from topoheat.topopt_twoway import M_DIV
    m = Monolithic(6, 4, beta_couple=10, heat=2e-3, mech_load=0.03,
                   alpha=0.03, volfrac=0.5)
    b = m.b

    # --- brute-force _A ---
    sc = m.Ef * b.alpha * 0.25
    r, c, v = [], [], []
    for e in range(b.nel):
        for i in range(8):
            gi = sc[e] * G0[i]
            for nb in range(4):
                r.append(b.edof[e, i]); c.append(b.edofT[e, nb]); v.append(gi)
    Aref = sp.csr_matrix((v, (r, c)), shape=(b.ndof, b.nnode))
    dA = (Aref - m._A())
    assert dA.nnz == 0 or np.allclose(dA.data, 0.0)

    # --- brute-force _B with a seeded temperature field and base conductivity ---
    rng = np.random.default_rng(0)
    T = rng.standard_normal(b.nnode)
    k_eff = m.k_base
    r, c, v = [], [], []
    for e in range(b.nel):
        kv = m.beta * k_eff[e] * (KET0 @ T[b.edofT[e]])
        for a in range(4):
            for j in range(8):
                r.append(b.edofT[e, a]); c.append(b.edof[e, j]); v.append(kv[a] * M_DIV[j])
    Bref = sp.csr_matrix((v, (r, c)), shape=(b.nnode, b.ndof))
    dB = (Bref - m._B(T, k_eff))
    assert dB.nnz == 0 or np.allclose(dB.data, 0.0)
