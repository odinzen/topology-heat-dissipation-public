"""
Per-element two-material thermoelastic topology optimization.

Extends the verified ThermoelasticTO (one-way thermal -> mechanical) with a
SECOND per-element design field: a material fraction mu in [0, 1] that blends
two materials A and B given as relative properties (E, alpha, k).

Each element carries two independently density-filtered design fields:
    rho : SIMP-style density (presence of material, volume constrained)
    mu  : material fraction (which of A/B, no volume constraint)

Interpolation (p = penal, pk = penal_k):
    stiffness scale   S(rho,mu) = (floor + rho^p (1-floor)) * (EA + mu(EB-EA))
    conduction scale  K(rho,mu) = (kmin + rho^pk(1-kmin)) * (kA + mu(kB-kA))
    thermal-stress    F(rho,mu) = rho^p * (EA + mu(EB-EA))
    expansion         alpha(mu) = alphaA + mu(alphaB-alphaA)

Element matrices reuse the validated unit element matrices:
    Ke   = S(rho,mu) * KE0
    KTe  = K(rho,mu) * KET0
    f_th = F(rho,mu) * alpha(mu) * (Te - tref) * G0

State (one way, T independent of u):
    K_T(rho,mu) T = q
    K_u(rho,mu) u = f_mech + f_th(rho, mu, T)
Objective:
    J = u^T K_u u
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# KE0/KET0/G0 are kept imported (built at the module nu=0.30) for any external
# code/tests that compare against the historical 0.30 element matrices. Inside
# TwoMaterialTO we use the per-instance self.KE0/self.KET0/self.G0 built by the
# base class from the requested nu, so a spec-driven Poisson ratio enters here.
from .topopt_thermoelastic import ThermoelasticTO, NU, KE0, KET0, G0  # noqa: F401


class TwoMaterialTO(ThermoelasticTO):
    def __init__(self, nelx, nely,
                 matA=(1.0, 0.01, 1.0), matB=(0.4, 0.04, 3.0),
                 nu=NU,
                 **kwargs):
        # alpha is unused now (replaced by per-material expansion); keep the
        # base constructor happy by passing through whatever it accepts. nu is
        # forwarded so the per-instance element matrices honor the spec value
        # (default stays 0.30 -> byte-identical to the historical behavior).
        super().__init__(nelx, nely, nu=nu, **kwargs)
        self.EA, self.alphaA, self.kA = matA
        self.EB, self.alphaB, self.kB = matB

    # ---- interpolation in the material fraction mu -----------------------
    # E factor (relative modulus) and its mu derivative.
    def _Emix(self, mu):   return self.EA + mu * (self.EB - self.EA)
    def _dEmix(self):      return (self.EB - self.EA)
    # k factor (relative conductivity) and its mu derivative.
    def _kmix(self, mu):   return self.kA + mu * (self.kB - self.kA)
    def _dkmix(self):      return (self.kB - self.kA)
    # expansion alpha(mu) and its mu derivative.
    def _amix(self, mu):   return self.alphaA + mu * (self.alphaB - self.alphaA)
    def _damix(self):      return (self.alphaB - self.alphaA)

    # rho-only SIMP factors (shared structure with the base class).
    def _rhoE(self, rho):   return self.floor + rho ** self.penal * (1.0 - self.floor)
    def _drhoE(self, rho):  return self.penal * rho ** (self.penal - 1) * (1.0 - self.floor)
    def _rhoK(self, rho):   return self.kmin + rho ** self.penal_k * (1.0 - self.kmin)
    def _drhoK(self, rho):  return self.penal_k * rho ** (self.penal_k - 1) * (1.0 - self.kmin)
    def _rhoF(self, rho):   return rho ** self.penal           # thermal-stress modulus penalization
    def _drhoF(self, rho):  return self.penal * rho ** (self.penal - 1)

    # combined scales
    def S_of(self, rho, mu):   return self._rhoE(rho) * self._Emix(mu)
    def Kc_of(self, rho, mu):  return self._rhoK(rho) * self._kmix(mu)
    def F_of(self, rho, mu):   return self._rhoF(rho) * self._Emix(mu)

    # ---- forward solve ---------------------------------------------------
    def solve(self, rhophys, muphys):
        rho = np.asarray(rhophys, float)
        mu = np.asarray(muphys, float)

        # Thermal field on the design-dependent conduction scale.
        Kc = self.Kc_of(rho, mu)
        sKt = (self.KET0.reshape(-1)[None, :] * Kc[:, None]).reshape(-1)
        KT = sp.csr_matrix((sKt, (self.iKt, self.jKt)), shape=(self.nnode, self.nnode))
        # Robin film boundary: fold the per-node +h diagonal (set on the instance by
        # the spec BC builder) onto the conduction operator. The bridge already put
        # the matching +h*Tamb load into self.q. Guard with getattr so non-spec
        # construction (e.g. gradient_check, which builds TwoMaterialTO directly and
        # never sets robin_diag) is unaffected — there the attribute is absent/None.
        robin_diag = getattr(self, "robin_diag", None)
        if robin_diag is not None and robin_diag.any():
            KT = KT + sp.diags(robin_diag)
        T = np.zeros(self.nnode)
        KTff = KT[self.free_temp, :][:, self.free_temp]
        T[self.free_temp] = spla.spsolve(KTff.tocsc(), self.q[self.free_temp])

        # Element-average temperature and thermal load.
        Te = T[self.edofT].mean(axis=1)
        Fc = self.F_of(rho, mu)
        amix = self._amix(mu)
        coef = Fc * amix * (Te - self.tref)              # per-element scale on G0
        f_th = np.zeros(self.ndof)
        contrib = coef[:, None] * self.G0[None, :]
        np.add.at(f_th, self.edof.reshape(-1), contrib.reshape(-1))

        # Mechanical solve.
        Sv = self.S_of(rho, mu)
        sK = (self.KE0.reshape(-1)[None, :] * Sv[:, None]).reshape(-1)
        KU = sp.csr_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof))
        f_total = self.f_mech + f_th
        u = np.zeros(self.ndof)
        KUff = KU[self.free_mech, :][:, self.free_mech]
        Lu = spla.splu(KUff.tocsc())
        u[self.free_mech] = Lu.solve(f_total[self.free_mech])

        return dict(T=T, Te=Te, u=u, KT=KT, KU=KU, KTff=KTff, Lu=Lu,
                    Sv=Sv, Kc=Kc, Fc=Fc, amix=amix, coef=coef)

    def objective(self, rhophys, muphys, st=None):
        st = st or self.solve(rhophys, muphys)
        u = st["u"]
        ue = u[self.edof]
        Ku_u_e = np.einsum("ej,jk,ek->e", ue, self.KE0, ue) * st["Sv"]
        return float(np.sum(Ku_u_e)), st

    # ---- coupled adjoint: derivatives w.r.t. PHYSICAL rho and mu ----------
    def sensitivity(self, rhophys, muphys):
        rho = np.asarray(rhophys, float)
        mu = np.asarray(muphys, float)
        J, st = self.objective(rho, mu)
        u, T, Te = st["u"], st["T"], st["Te"]
        ue = u[self.edof]

        # Mechanical adjoint is explicit for J = u^T K_u u : lambda_u = -2 u.
        lam_u = -2.0 * u

        # Thermal adjoint: K_T lam_T = A^T lam_u.
        # f_th depends on T through coef = F*alpha*(Te - tref), with
        # Te = mean of element node temps, so d coef / d T_node = F*alpha*0.25.
        Fc = st["Fc"]
        amix = st["amix"]
        w = Fc * amix * 0.25                         # per-element node weight
        s_e = (self.G0[None, :] * lam_u[self.edof]).sum(axis=1)
        node_add = w * s_e
        rhs_T = np.zeros(self.nnode)
        np.add.at(rhs_T, self.edofT.reshape(-1),
                  np.repeat(node_add, 4).reshape(self.nel, 4).reshape(-1))
        lam_T = np.zeros(self.nnode)
        lam_T[self.free_temp] = spla.spsolve(st["KTff"].tocsc(), rhs_T[self.free_temp])

        lamTe = lam_T[self.edofT]
        Tne = T[self.edofT]
        G0u = (self.G0[None, :] * ue).sum(axis=1)         # u_e . G0
        keu = np.einsum("ej,jk,ek->e", ue, self.KE0, ue)  # u_e^T KE0 u_e
        ketLT = np.einsum("ej,jk,ek->e", lamTe, self.KET0, Tne)  # lam_T_e^T KET0 T_e

        dTe = (Te - self.tref)

        # ---- derivative w.r.t. PHYSICAL rho ----
        # term1 (mech stiffness): -dS/drho * u_e^T KE0 u_e
        dS_drho = self._drhoE(rho) * self._Emix(mu)
        t1 = -dS_drho * keu
        # term2 (thermal load): +2 * dF/drho * alpha * (Te-tref) * (u_e . G0)
        dF_drho = self._drhoF(rho) * self._Emix(mu)
        t2 = 2.0 * dF_drho * amix * dTe * G0u
        # term3 (conduction adjoint): dK/drho * lam_T_e^T KET0 T_e
        dK_drho = self._drhoK(rho) * self._kmix(mu)
        t3 = dK_drho * ketLT
        dJ_drho = t1 + t2 + t3

        # ---- derivative w.r.t. PHYSICAL mu ----
        # term1 (mech stiffness): -dS/dmu * u_e^T KE0 u_e
        dS_dmu = self._rhoE(rho) * self._dEmix()
        m1 = -dS_dmu * keu
        # term2 (thermal load): +2 * d(F*alpha)/dmu * (Te-tref) * (u_e . G0)
        #   F*alpha = rho^p * Emix(mu) * amix(mu)
        #   d/dmu = rho^p * (dEmix*amix + Emix*damix)
        dFa_dmu = self._rhoF(rho) * (self._dEmix() * amix + self._Emix(mu) * self._damix())
        m2 = 2.0 * dFa_dmu * dTe * G0u
        # term3 (conduction adjoint): dK/dmu * lam_T_e^T KET0 T_e
        dK_dmu = self._rhoK(rho) * self._dkmix()
        m3 = dK_dmu * ketLT
        dJ_dmu = m1 + m2 + m3

        return J, dJ_drho, dJ_dmu, st

    # ---- filter chain for both fields ------------------------------------
    def full_gradient(self, rho, mu):
        rhophys = self.filter_forward(rho)
        muphys = self.filter_forward(mu)
        J, dphys_rho, dphys_mu, st = self.sensitivity(rhophys, muphys)
        # Chain through the (identical) density filter for each field.
        return J, self.filter_back(dphys_rho), self.filter_back(dphys_mu)


# ============================================================================
# TASK 2 : finite-difference verification of BOTH gradients
# ============================================================================
def gradient_check(seed=0, nelx=12, nely=6, h=1e-6, nperturb=8):
    rng = np.random.default_rng(seed)
    to = TwoMaterialTO(nelx, nely, rmin=1.5,
                       matA=(1.0, 0.01, 1.0),      # stiff, low expansion
                       matB=(0.4, 0.04, 3.0),      # compliant, high expansion
                       heat=1.0, mech_load=1.0)
    rho = rng.uniform(0.3, 0.9, to.nel)
    mu = rng.uniform(0.3, 0.9, to.nel)

    J0, g_rho, g_mu = to.full_gradient(rho, mu)

    def obj(rraw, mraw):
        Jv, _ = to.objective(to.filter_forward(rraw), to.filter_forward(mraw))
        return Jv

    # --- rho gradient ---
    idx_r = rng.choice(to.nel, size=nperturb, replace=False)
    errs_r = []
    for i in idx_r:
        rp = rho.copy(); rp[i] += h
        rm = rho.copy(); rm[i] -= h
        fd = (obj(rp, mu) - obj(rm, mu)) / (2 * h)
        rel = abs(fd - g_rho[i]) / (abs(fd) + 1e-30)
        errs_r.append(rel)

    # --- mu gradient ---
    idx_m = rng.choice(to.nel, size=nperturb, replace=False)
    errs_m = []
    for i in idx_m:
        mp = mu.copy(); mp[i] += h
        mm = mu.copy(); mm[i] -= h
        fd = (obj(rho, mp) - obj(rho, mm)) / (2 * h)
        rel = abs(fd - g_mu[i]) / (abs(fd) + 1e-30)
        errs_m.append(rel)

    return float(max(errs_r)), float(max(errs_m)), errs_r, errs_m


if __name__ == "__main__":
    er, em, lr, lm = gradient_check()
    print("rho gradient max relative error:", er)
    print("mu  gradient max relative error:", em)
    print("rho per-element:", [f"{e:.2e}" for e in lr])
    print("mu  per-element:", [f"{e:.2e}" for e in lm])
    ok = (er < 1e-4) and (em < 1e-4)
    print("BOTH below 1e-4:", ok)
