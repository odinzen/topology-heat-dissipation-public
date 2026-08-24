"""3D thermoelastic optimization with an AMG preconditioned CG solver.

Replaces the direct factorization with smoothed aggregation algebraic multigrid
used as a preconditioner for conjugate gradient. The elasticity near null space
(six rigid body modes) is supplied to the preconditioner, and each solve is warm
started from the previous design iteration. This lets the three dimensional study
scale past the grid the direct solver can handle in a fixed budget.
"""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import cg
import pyamg
from .topopt3d import TO3D


class TO3DIter(TO3D):
    def __init__(self, *a, cg_tol=1e-8, **k):
        super().__init__(*a, **k)
        self.cg_tol = cg_tol
        self._u = None; self._T = None; self._lamT = None
        self.stats = {}

    def _coords(self):
        c = np.zeros((self.nnode, 3))
        for iz in range(self.nz + 1):
            for iy in range(self.ny + 1):
                for ix in range(self.nx + 1):
                    c[self.nid(ix, iy, iz)] = (ix, iy, iz)
        return c

    def _rbm_free(self):
        c = self._coords(); R = np.zeros((self.ndof, 6))
        R[0::3, 0] = 1; R[1::3, 1] = 1; R[2::3, 2] = 1
        R[1::3, 3] = -c[:, 2]; R[2::3, 3] = c[:, 1]
        R[0::3, 4] = c[:, 2];  R[2::3, 4] = -c[:, 0]
        R[0::3, 5] = -c[:, 1]; R[1::3, 5] = c[:, 0]
        return R[self.free, :]

    def _pcg(self, A, b, x0, B=None, tag=""):
        A = A.tocsr()
        ml = pyamg.smoothed_aggregation_solver(A, B=B, max_coarse=400)
        M = ml.aspreconditioner(cycle="V")
        it = [0]
        x, info = cg(A, b, x0=x0, rtol=self.cg_tol, atol=0.0, maxiter=1000,
                     M=M, callback=lambda xk: it.__setitem__(0, it[0] + 1))
        self.stats[tag] = (it[0], info)
        if info != 0:
            import warnings
            warnings.warn(
                f"AMG-CG '{tag}' did not converge (info={info}, iters={it[0]}); "
                "the solve may be inaccurate", RuntimeWarning, stacklevel=2)
        return x

    def solve(self, rp):
        kv = self.k_(rp); sKt = (self.KET0.ravel()[None, :] * kv[:, None]).ravel()
        KT = sp.csr_matrix((sKt, (self.iKt, self.jKt)), shape=(self.nnode, self.nnode))
        KTff = KT[self.freeT][:, self.freeT]
        T = np.zeros(self.nnode)
        T[self.freeT] = self._pcg(KTff, self.q[self.freeT],
                                  self._T, tag="thermal")
        self._T = T[self.freeT].copy()
        Te = T[self.edofT].mean(1)
        coef = self.Ef_(rp) * self.alpha * (Te - self.tref)
        fth = np.zeros(self.ndof); np.add.at(fth, self.edof.ravel(),
                                             (coef[:, None] * self.G0[None, :]).ravel())
        Ev = self.E_(rp); sK = (self.KE0.ravel()[None, :] * Ev[:, None]).ravel()
        KU = sp.csr_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof))
        KUff = KU[self.free][:, self.free]
        if self._rbmf is None:
            self._rbmf = self._rbm_free()
        u = np.zeros(self.ndof)
        u[self.free] = self._pcg(KUff, (self.f + fth)[self.free],
                                 self._u, B=self._rbmf, tag="mech")
        self._u = u[self.free].copy()
        return dict(T=T, Te=Te, u=u, Ev=Ev, KT=KT, KU=KU)

    _rbmf = None

    def grad(self, rho):
        rp = self.filt(rho); J, st = self.obj(rp)
        u, T, Te = st["u"], st["T"], st["Te"]; ue = u[self.edof]
        lam_u = -2 * u
        w = self.Ef_(rp) * self.alpha * (1.0 / 8.0)
        s_e = (self.G0[None, :] * lam_u[self.edof]).sum(1)
        rhsT = np.zeros(self.nnode)
        np.add.at(rhsT, self.edofT.ravel(), np.repeat(w * s_e, 8).reshape(self.nel, 8).ravel())
        KTff = st["KT"][self.freeT][:, self.freeT]
        lamT = np.zeros(self.nnode)
        lamT[self.freeT] = self._pcg(KTff, rhsT[self.freeT], self._lamT, tag="adjT")
        self._lamT = lamT[self.freeT].copy()
        t1 = -self.dE_(rp) * np.einsum("ej,jk,ek->e", ue, self.KE0, ue)
        t2 = 2 * self.dEf_(rp) * self.alpha * (Te - self.tref) * (self.G0[None, :] * ue).sum(1)
        t3 = self.dk_(rp) * np.einsum("ej,jk,ek->e", lamT[self.edofT], self.KET0, T[self.edofT])
        return J, np.asarray(self.H @ ((t1 + t2 + t3) / self.Hs)).ravel()

    def obj(self, rp, st=None):
        st = st or self.solve(rp); ue = st["u"][self.edof]
        return float(np.sum(np.einsum("ej,jk,ek->e", ue, self.KE0, ue) * st["Ev"])), st


def correctness(seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    direct = TO3D(6, 3, 3, rmin=1.5)
    rp = rng.uniform(0.3, 0.9, direct.nel)
    sd = direct.solve(rp)
    it = TO3DIter(6, 3, 3, rmin=1.5, cg_tol=1e-10)
    si = it.solve(rp)
    du = np.linalg.norm(si["u"] - sd["u"]) / np.linalg.norm(sd["u"])
    dT = np.linalg.norm(si["T"] - sd["T"]) / (np.linalg.norm(sd["T"]) + 1e-30)
    return du, dT


def gradient_check(seed=0):
    rng = np.random.default_rng(seed)
    to = TO3DIter(4, 2, 2, rmin=1.5, cg_tol=1e-11)
    rho = rng.uniform(0.4, 0.9, to.nel)
    J, g = to.grad(rho); h = 1e-6; errs = []
    for i in rng.choice(to.nel, size=6, replace=False):
        rp = rho.copy(); rp[i] += h; rm = rho.copy(); rm[i] -= h
        to._u = to._T = None
        Jp, _ = to.obj(to.filt(rp)); Jm, _ = to.obj(to.filt(rm))
        errs.append(abs((Jp - Jm) / (2 * h) - g[i]) / (abs((Jp - Jm) / (2 * h)) + 1e-30))
    return float(max(errs))


if __name__ == "__main__":
    du, dT = correctness()
    print("PCG vs direct: rel u err %.2e, rel T err %.2e" % (du, dT))
    print("3D iterative gradient check max rel error:", gradient_check())
