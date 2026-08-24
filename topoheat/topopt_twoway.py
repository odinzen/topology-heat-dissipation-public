"""Two way coupled thermoelastic topology optimization.

Conductivity responds to the local volumetric strain, so the thermal field and
the deformation form a closed loop. The forward problem is a staggered fixed
point with under relaxation; the sensitivity uses a coupled adjoint carrying the
cross term dK_T/du, solved by a staggered iteration that mirrors the forward
scheme. The mechanical factorization, constant within a design step, is reused.
Correctness is established by a finite difference check.

  K_T(rho, u) T = q       k_eff = k(rho) (1 + beta * vol(u))
  K_u(rho) u   = f_mech + f_th(rho, T)
  J = u^T K_u u
"""
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import thermoelastic_fem as fem
from .topopt_thermoelastic import ThermoelasticTO, KE0, KET0, G0

_B0 = fem._Bmat(0.0, 0.0, 1.0)
M_DIV = _B0[0, :] + _B0[1, :]


class TwoWayTO(ThermoelasticTO):
    def __init__(self, *args, beta_couple=2.0, fp_tol=1e-12, fp_max=200, relax=0.5, **kw):
        super().__init__(*args, **kw)
        self.beta_couple = beta_couple
        self.fp_tol, self.fp_max, self.relax = fp_tol, fp_max, relax

    # These delegate to the shared base-class assemblers. self.KE0/KET0/G0 are
    # built at the constructor nu; TwoWayTO is only ever constructed at the
    # default nu=0.30, so they are byte-identical to the module-level KE0/KET0/G0
    # this path historically used -- the routed matrix is unchanged.
    def _KT(self, k_eff):
        return self._assemble_KT(k_eff)

    def _KU(self, Ev):
        return self._assemble_KU(Ev)

    def _fth(self, rhophys, T):
        Te = T[self.edofT].mean(axis=1)
        coef = self.Ef_of(rhophys) * self.alpha * (Te - self.tref)
        return self._thermal_load(coef), Te

    def solve(self, rhophys):
        k_base = self.k_of(rhophys)
        Ev = self.E_of(rhophys)
        KU = self._KU(Ev)
        luU = spla.splu(KU[self.free_mech, :][:, self.free_mech].tocsc())

        def mech(f):
            u = np.zeros(self.ndof)
            u[self.free_mech] = luU.solve(f[self.free_mech])
            return u

        # Staggered fixed point. Each pass solves the thermal field at the current
        # vol's k_eff, then the mechanics, yielding vol_new. On convergence we set
        # vol = vol_new (no relaxation) and break: the just-computed (T, u) are then
        # consistent with the returned vol to within fp_tol, so the historical
        # post-loop recompute at the relaxed vol is unnecessary and is omitted. When
        # the loop does not contract we return the last iterate with vol = vol_new.
        vol = np.zeros(self.nel)
        T = np.zeros(self.nnode); KTff = None; Te = None; u = None
        k_eff = np.maximum(k_base, self.kmin)
        step = np.inf
        self.fp_converged = False
        for _ in range(self.fp_max):
            k_eff = np.maximum(k_base * (1.0 + self.beta_couple * vol), self.kmin)
            KT = self._KT(k_eff)
            KTff = KT[self.free_temp, :][:, self.free_temp].tocsc()
            T = np.zeros(self.nnode)
            T[self.free_temp] = spla.spsolve(KTff, self.q[self.free_temp])
            f_th, Te = self._fth(rhophys, T)
            u = mech(self.f_mech + f_th)
            vol_new = (u[self.edof] * M_DIV[None, :]).sum(axis=1)
            step = np.max(np.abs(vol_new - vol))
            if step < self.fp_tol:
                # Converged: adopt vol_new directly (no relaxation) so the (T, u)
                # just computed from this pass's k_eff stay consistent with it.
                vol = vol_new
                self.fp_converged = True
                break
            vol = (1.0 - self.relax) * vol + self.relax * vol_new
        if not self.fp_converged:
            # The staggered fixed point did not contract; the result is the last
            # iterate (vol = vol_new, the (T, u) from this pass). Strong
            # deformation-to-conductivity feedback needs the monolithic Newton
            # solve (monolithic.py) instead.
            vol = vol_new
            warnings.warn(
                f"two-way forward fixed point did not converge in {self.fp_max} "
                f"iterations (last step {step:.2e} > tol {self.fp_tol:.0e}); "
                "consider the monolithic Newton solver for strong feedback",
                RuntimeWarning, stacklevel=2)
        return dict(T=T, Te=Te, u=u, k_base=k_base, k_eff=k_eff, vol=vol,
                    Ev=Ev, KU=KU, luU=luU, KTff=KTff)

    def objective(self, rhophys, st=None):
        st = st or self.solve(rhophys)
        ue = st["u"][self.edof]
        return float(np.sum(np.einsum("ej,jk,ek->e", ue, KE0, ue) * st["Ev"])), st

    def sensitivity(self, rhophys):
        J, st = self.objective(rhophys)
        u, T, Te, vol = st["u"], st["T"], st["Te"], st["vol"]
        k_base, luU = st["k_base"], st["luU"]
        luT = spla.splu(st["KTff"])
        ue = u[self.edof]
        KuU = st["KU"] @ u

        w = self.Ef_of(rhophys) * self.alpha * 0.25
        lam_u = -2.0 * u
        lam_T = np.zeros(self.nnode)
        self.adj_converged = False
        for _ in range(self.fp_max):
            s_e = (G0[None, :] * lam_u[self.edof]).sum(axis=1)
            rhs_T = np.zeros(self.nnode)
            np.add.at(rhs_T, self.edofT.reshape(-1),
                      np.repeat(w * s_e, 4).reshape(self.nel, 4).reshape(-1))
            lam_T_new = np.zeros(self.nnode)
            lam_T_new[self.free_temp] = luT.solve(rhs_T[self.free_temp])
            lamTe = lam_T_new[self.edofT]; Tne = T[self.edofT]
            q_e = k_base * self.beta_couple * np.einsum("ej,jk,ek->e", lamTe, KET0, Tne)
            coupling = np.zeros(self.ndof)
            np.add.at(coupling, self.edof.reshape(-1),
                      (q_e[:, None] * M_DIV[None, :]).reshape(-1))
            lam_u_new = np.zeros(self.ndof)
            lam_u_new[self.free_mech] = luU.solve((-2.0 * KuU - coupling)[self.free_mech])
            d = np.max(np.abs(lam_u_new - lam_u)) / (np.max(np.abs(lam_u_new)) + 1e-30)
            lam_u, lam_T = lam_u_new, lam_T_new
            if d < 1e-12:
                self.adj_converged = True
                break
        if not self.adj_converged:
            # An unconverged adjoint silently biases the gradient; surface it (the
            # forward solve has the same guard). Strong feedback -> monolithic path.
            warnings.warn(
                f"two-way adjoint fixed point did not converge in {self.fp_max} "
                f"iterations (last rel. change {d:.2e}); gradients may be biased",
                RuntimeWarning, stacklevel=2)

        lam_ue = lam_u[self.edof]
        dE = self.dE_of(rhophys)
        t_mech = dE * (np.einsum("ej,jk,ek->e", ue, KE0, ue)
                       + np.einsum("ej,jk,ek->e", lam_ue, KE0, ue))
        df_scale = self.dEf_of(rhophys) * self.alpha * (Te - self.tref)
        t_fth = -df_scale * (G0[None, :] * lam_ue).sum(axis=1)
        lamTe = lam_T[self.edofT]; Tne = T[self.edofT]
        t_therm = self.dk_of(rhophys) * (1.0 + self.beta_couple * vol) \
                  * np.einsum("ej,jk,ek->e", lamTe, KET0, Tne)
        return J, t_mech + t_fth + t_therm, st

    def full_gradient(self, rho):
        rhophys = self.filter_forward(rho)
        J, dphys, st = self.sensitivity(rhophys)
        return J, self.filter_back(dphys)


def gradient_check(seed=1, nelx=12, nely=6, beta=2.0):
    rng = np.random.default_rng(seed)
    to = TwoWayTO(nelx, nely, rmin=1.5, beta_couple=beta,
                  heat=2e-3, mech_load=0.02, alpha=0.02, relax=0.5)
    rho = rng.uniform(0.4, 0.9, to.nel)
    st = to.solve(to.filter_forward(rho))
    print("forward vol range [%.2e, %.2e]  beta*|vol|max %.2e"
          % (st["vol"].min(), st["vol"].max(), beta * np.abs(st["vol"]).max()))
    J0, g = to.full_gradient(rho)
    h = 1e-6
    idx = rng.choice(to.nel, size=8, replace=False)
    errs = []
    for i in idx:
        rp = rho.copy(); rp[i] += h
        rm = rho.copy(); rm[i] -= h
        Jp, _ = to.objective(to.filter_forward(rp))
        Jm, _ = to.objective(to.filter_forward(rm))
        errs.append(abs((Jp - Jm) / (2 * h) - g[i]) / (abs((Jp - Jm) / (2 * h)) + 1e-30))
    return float(max(errs)), errs


if __name__ == "__main__":
    e, errs = gradient_check()
    print("two way gradient check max relative error:", e)
