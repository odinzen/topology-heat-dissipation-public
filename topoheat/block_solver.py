"""Field split block preconditioning for the monolithic coupled Jacobian.

The coupled thermoelastic Jacobian is nonsymmetric because the two cross blocks
differ. A Krylov method for nonsymmetric systems, GMRES, is preconditioned by a
field split that approximates the inverse of each physical block with one
algebraic multigrid cycle. Two splits are compared: block diagonal (Jacobi) and
block lower triangular (Gauss Seidel), the latter reusing the mechanical to
thermal cross block.

  J = [ K_u   -A  ]      preconditioner approximates K_u^-1 and K_T^-1 by AMG
      [  B    K_T ]
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from scipy.sparse.linalg import gmres, LinearOperator
import pyamg
from .monolithic import Monolithic
from .topopt_thermoelastic import KET0
from .topopt_twoway import M_DIV


def _blocks(m, u, T):
    b = m.b; fm = b.free_mech; ft = b.free_temp
    k_eff = m._keff(m._vol(u)); KT = m._KT(k_eff)
    Kuu = m.KU[fm][:, fm].tocsr()
    Aut = m.A[fm][:, ft].tocsr()
    Btu = m._B(T, k_eff)[ft][:, fm].tocsr()
    Ktt = KT[ft][:, ft].tocsr()
    return Kuu, Aut, Btu, Ktt


def _rbm2d(m):
    b = m.b; co = b.mesh.coords()
    R = np.zeros((b.ndof, 3))
    R[0::2, 0] = 1.0; R[1::2, 1] = 1.0
    R[0::2, 2] = -co[:, 1]; R[1::2, 2] = co[:, 0]
    return R[b.free_mech, :]

def run(nelx, nely, beta=20.0, mode="triangular", tol=1e-8, inner=4, itol=1e-2):
    m = Monolithic(nelx, nely, beta_couple=beta, heat=2e-3, mech_load=0.03,
                   alpha=0.03, volfrac=0.5)
    u, T, res = m.newton()
    Kuu, Aut, Btu, Ktt = _blocks(m, u, T)
    nu, nt = Kuu.shape[0], Ktt.shape[0]
    J = sp.bmat([[Kuu, -Aut], [Btu, Ktt]], format="csr")
    rng = np.random.default_rng(0); rhs = rng.standard_normal(nu + nt)
    xref = spla.spsolve(J.tocsc(), rhs)

    mlu = pyamg.smoothed_aggregation_solver(Kuu, B=_rbm2d(m), max_coarse=300)
    mlt = pyamg.smoothed_aggregation_solver(Ktt, max_coarse=300)

    def solve_u(r):
        return mlu.solve(r, tol=itol, maxiter=inner, accel="cg")
    def solve_t(r):
        return mlt.solve(r, tol=itol, maxiter=inner, accel="cg")

    # Schur complement of K_u, approximated with the diagonal of K_u:
    #   S = K_T + B diag(K_u)^-1 A   (sparse), preconditioned by its own AMG.
    if mode == "schur":
        dinv = sp.diags(1.0 / Kuu.diagonal())
        Shat = (Ktt + Btu @ dinv @ Aut).tocsr()
        mls = pyamg.smoothed_aggregation_solver(Shat, max_coarse=300)
        def solve_s(r):
            return mls.solve(r, tol=itol, maxiter=inner, accel="cg")

    def prec(r):
        ru, rt = r[:nu], r[nu:]
        if mode == "diagonal":
            return np.concatenate([solve_u(ru), solve_t(rt)])
        if mode == "schur":
            zt = solve_s(rt)                 # S z_t = r_t
            zu = solve_u(ru + Aut @ zt)      # K_u z_u = r_u + A z_t
            return np.concatenate([zu, zt])
        yu = solve_u(ru)
        yt = solve_t(rt - Btu @ yu)          # block lower triangular
        return np.concatenate([yu, yt])

    M = LinearOperator((nu + nt, nu + nt), matvec=prec)
    it = [0]
    x, info = gmres(J, rhs, M=M, rtol=tol, maxiter=400, restart=50,
                    callback=lambda rk: it.__setitem__(0, it[0] + 1),
                    callback_type="pr_norm")
    err = np.linalg.norm(x - xref) / np.linalg.norm(xref)
    return nu + nt, it[0], err, info


if __name__ == "__main__":
    print("%-12s %8s %6s %6s %7s %12s" % ("grid","ndof","diag","tri","schur","rel err(schur)"))
    for nx, ny in [(16, 8), (32, 16), (48, 24), (64, 32)]:
        nd, itd, ed, _ = run(nx, ny, mode="diagonal")
        nd, itt, et, _ = run(nx, ny, mode="triangular")
        nd, its, es, _ = run(nx, ny, mode="schur")
        print("%-12s %8d %6d %6d %7d %12.2e" % ("%dx%d"%(nx,ny), nd, itd, itt, its, es))
