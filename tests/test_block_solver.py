"""Correctness coverage for the AMG field-split block solver
(topoheat/block_solver.py), which backs the manuscript's solver claims
but previously had zero tests.

The solver builds the monolithic coupled Jacobian J = [[Kuu, -Aut], [Btu, Ktt]],
solves it directly (spsolve), then solves it with GMRES preconditioned by an AMG
field split, and returns (ndof, iterations, rel_err, info). The core correctness
claim is that the field-split-preconditioned GMRES reproduces the direct solve
to GMRES tolerance, regardless of which preconditioner split is chosen (the split
preconditions; it does not change the operator).

These tests require pyamg; they skip gracefully where it is absent.
"""
import pytest

MODES = ["diagonal", "triangular", "schur"]


@pytest.mark.parametrize("mode", MODES)
def test_block_solver_matches_direct(mode):
    pytest.importorskip("pyamg")
    from topoheat.block_solver import run

    ndof, iters, err, info = run(16, 8, mode=mode)

    assert info == 0                      # GMRES reported convergence
    assert err < 1e-6                     # field-split GMRES reproduces direct solve
    assert 0 < iters < 400                # did real work, did not hit maxiter


def test_block_solver_modes_agree():
    pytest.importorskip("pyamg")
    from topoheat.block_solver import run

    # Every split converges to the SAME direct solution; the preconditioner
    # choice changes the iteration path, not the operator.
    errs = {}
    for mode in MODES:
        ndof, iters, err, info = run(16, 8, mode=mode)
        assert info == 0
        errs[mode] = err
    for mode, err in errs.items():
        assert err < 1e-6, f"mode {mode} disagreed with direct solve: err={err:.2e}"


def test_block_solver_holds_at_larger_grid():
    pytest.importorskip("pyamg")
    from topoheat.block_solver import run

    # The correctness claim must hold as the system grows. One modest larger
    # grid for a single mode keeps total runtime reasonable.
    ndof, iters, err, info = run(24, 12, mode="triangular")
    assert info == 0
    assert err < 1e-6
    assert 0 < iters < 400
