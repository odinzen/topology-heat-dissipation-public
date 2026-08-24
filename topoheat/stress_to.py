"""Thermal stress objective: p-norm of the element von Mises stress.

Demonstrates the stress mode named in the schema and objective family. The
stress is a non self adjoint functional of the coupled state, so the full
adjoint is used. Correctness is established by a finite difference check.

  s_e = vonMises( D (B u_e - eps0_e) ),  eps0_e = alpha (T_e - T_ref) m
  J   = ( sum_e s_e^P )^(1/P)
"""
import numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
from . import thermoelastic_fem as fem
from .topopt_thermoelastic import ThermoelasticTO, KE0, KET0, G0

D0 = fem._Dmat(1.0, 0.3)             # unit modulus plane stress
B0 = fem._Bmat(0.0, 0.0, 1.0)        # strain displacement at element centre
MVM = np.array([[1.0, -0.5, 0.0], [-0.5, 1.0, 0.0], [0.0, 0.0, 3.0]])
MVEC = np.array([1.0, 1.0, 0.0])


class StressTO(ThermoelasticTO):
    def __init__(self, *a, pnorm=8.0, **kw):
        super().__init__(*a, **kw)
        self.P = pnorm

    def _stress(self, u, T):
        ue = u[self.edof]                        # nel x 8
        Te = T[self.edofT].mean(axis=1)
        eps = ue @ B0.T                          # nel x 3  (B0 u_e)
        eps0 = (self.alpha * (Te - self.tref))[:, None] * MVEC[None, :]
        w = (eps - eps0) @ D0.T                  # nel x 3 stress vectors
        s = np.sqrt(np.einsum("ej,jk,ek->e", w, MVM, w) + 1e-12)
        return w, s, Te

    def objective(self, rp, st=None):
        st = st or self.solve(rp)
        w, s, Te = self._stress(st["u"], st["T"])
        J = (np.sum(s ** self.P)) ** (1.0 / self.P)
        st["w"], st["s"] = w, s
        return float(J), st

    def sensitivity(self, rp):
        J, st = self.objective(rp)
        u, T = st["u"], st["T"]; w, s = st["w"], st["s"]
        Te = T[self.edofT].mean(axis=1)
        sumP = np.sum(s ** self.P)
        gfac = sumP ** (1.0 / self.P - 1.0)      # outer chain
        ge = gfac * s ** (self.P - 1)            # dJ/ds_e
        # dJ/dw_e = ge * (MVM w)/s
        dJdw = ge[:, None] * (w @ MVM.T) / s[:, None]
        # adjoint load on u: dJ/du = sum_e (D B0)^T dJdw_e  scattered
        dJdu = np.zeros(self.ndof)
        contrib_u = dJdw @ (D0 @ B0)             # nel x 8
        np.add.at(dJdu, self.edof.reshape(-1), contrib_u.reshape(-1))
        # adjoint load on T: dJ/dT via eps0 = alpha(Te-tref) m ; dw/dTe = -alpha D m
        dJdTe = -(self.alpha) * (dJdw @ (D0 @ MVEC))   # nel
        dJdT = np.zeros(self.nnode)
        np.add.at(dJdT, self.edofT.reshape(-1),
                  np.repeat(dJdTe * 0.25, 4).reshape(self.nel, 4).reshape(-1))
        # mechanical adjoint: K_u lam_u = -dJ/du
        lam_u = np.zeros(self.ndof)
        lam_u[self.free_mech] = st["Lu"].solve(-dJdu[self.free_mech])
        # thermal adjoint: K_T lam_T = A^T lam_u - dJ/dT
        w_node = self.Ef_of(rp) * self.alpha * 0.25
        s_e = (G0[None, :] * lam_u[self.edof]).sum(1)
        ATlam = np.zeros(self.nnode)
        np.add.at(ATlam, self.edofT.reshape(-1),
                  np.repeat(w_node * s_e, 4).reshape(self.nel, 4).reshape(-1))
        rhsT = ATlam - dJdT
        lam_T = np.zeros(self.nnode)
        lam_T[self.free_temp] = spla.spsolve(st["KTff"].tocsc(), rhsT[self.free_temp])
        # design derivative (J has no explicit rho dependence)
        ue = u[self.edof]; lam_ue = lam_u[self.edof]
        t_mech = self.dE_of(rp) * np.einsum("ej,jk,ek->e", lam_ue, KE0, ue)
        t_fth = -self.dEf_of(rp) * self.alpha * (Te - self.tref) * (G0[None, :] * lam_ue).sum(1)
        t_therm = self.dk_of(rp) * np.einsum("ej,jk,ek->e", lam_T[self.edofT], KET0, T[self.edofT])
        return J, t_mech + t_fth + t_therm, st

    def full_gradient(self, rho):
        rp = self.filter_forward(rho)
        J, dphys, st = self.sensitivity(rp)
        return J, self.filter_back(dphys)


def gradient_check(seed=3, nelx=12, nely=6):
    rng = np.random.default_rng(seed)
    to = StressTO(nelx, nely, rmin=1.5, heat=0.01, mech_load=1.0, alpha=0.02, pnorm=8.0)
    rho = rng.uniform(0.4, 0.9, to.nel)
    J0, g = to.full_gradient(rho)
    h = 1e-6; idx = rng.choice(to.nel, size=8, replace=False); errs = []
    for i in idx:
        rp = rho.copy(); rp[i] += h; rm = rho.copy(); rm[i] -= h
        Jp, _ = to.objective(to.filter_forward(rp)); Jm, _ = to.objective(to.filter_forward(rm))
        errs.append(abs((Jp - Jm) / (2 * h) - g[i]) / (abs((Jp - Jm) / (2 * h)) + 1e-30))
    return float(max(errs))


if __name__ == "__main__":
    print("thermal stress p-norm gradient check max rel error:", gradient_check())
