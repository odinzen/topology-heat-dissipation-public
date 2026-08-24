"""
Density based thermoelastic topology optimization for the Odinzen engine.

Minimizes structural strain energy under a combined mechanical load and a
design dependent thermal load. The temperature field is solved on a
conductivity that interpolates with the design, so the sensitivity carries
a thermal adjoint path. The element matrices come from the validated Q4
routines in thermoelastic_fem, which passed the patch and clamped tests.

State, one way thermal to mechanical:
    K_T(rho) T = q                       design dependent conduction
    K_u(rho) u = f_mech + f_th(rho, T)   thermoelastic equilibrium
Objective:
    J = u^T K_u u                        twice the structural strain energy
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import thermoelastic_fem as fem

NU = 0.3
KE0 = fem.elem_stiffness(1.0, NU, 1.0)        # unit modulus stiffness (default nu)
KET0 = fem.elem_conduction(1.0, 1.0)          # unit conductivity conduction (nu independent)
G0 = fem.elem_thermal_force(1.0, NU, 1.0, 1.0, 1.0)  # unit thermal coupling vector (default nu)


def _factor_or_raise(Kff, what):
    """LU factorize a reduced stiffness, turning a singular matrix into an
    actionable error instead of a raw SciPy RuntimeError. A singular reduced
    stiffness means the design is under-constrained or has disconnected solid."""
    try:
        return spla.splu(Kff.tocsc())
    except RuntimeError as e:
        raise ValueError(
            f"{what} stiffness is singular: the problem is under-constrained or the "
            "solid material does not connect the loads to the supports") from e


class ThermoelasticTO:
    def __init__(self, nelx, nely, volfrac=0.45, penal=3.0, penal_k=3.0,
                 penal_f=3.0, rmin=2.0, floor=1e-3, alpha=0.5, kmin_ratio=1e-3,
                 heat=1.0, mech_load=1.0, tref=0.0, nu=NU):
        self.nelx, self.nely = nelx, nely
        self.volfrac, self.penal, self.penal_k, self.penal_f = volfrac, penal, penal_k, penal_f
        self.floor, self.alpha, self.tref = floor, alpha, tref
        self.kmin = kmin_ratio
        self.heat, self.mech_load = heat, mech_load
        # Element matrices built at the requested Poisson ratio. Defaults to the
        # module nu=0.30, so every nu=0.30 result is byte-identical to before;
        # a spec-driven nu now actually enters the stiffness instead of being
        # silently ignored. KET0 (conduction) is nu independent.
        self.nu = nu
        self.KE0 = fem.elem_stiffness(1.0, nu, 1.0)
        self.KET0 = fem.elem_conduction(1.0, 1.0)
        self.G0 = fem.elem_thermal_force(1.0, nu, 1.0, 1.0, 1.0)

        self.mesh = fem.Mesh(nelx, nely, h=1.0)
        self.nnode = self.mesh.nnode
        self.ndof = 2 * self.nnode
        self.nel = nelx * nely

        els = self.mesh.elements()
        self.edofT = np.array(els, dtype=np.int64)                 # nel x 4 (thermal)
        self.edof = np.empty((self.nel, 8), dtype=np.int64)        # nel x 8 (mech)
        self.edof[:, 0::2] = 2 * self.edofT
        self.edof[:, 1::2] = 2 * self.edofT + 1

        self._build_assembly_index()
        self._build_filter(rmin)
        self._build_bc()

    # ---- assembly bookkeeping --------------------------------------------
    def _build_assembly_index(self):
        e = np.arange(self.nel)
        self.iK = np.kron(self.edof, np.ones((8, 1), dtype=np.int64)).reshape(-1)
        self.jK = np.kron(self.edof, np.ones((1, 8), dtype=np.int64)).reshape(-1)
        self.iKt = np.kron(self.edofT, np.ones((4, 1), dtype=np.int64)).reshape(-1)
        self.jKt = np.kron(self.edofT, np.ones((1, 4), dtype=np.int64)).reshape(-1)

    def _build_filter(self, rmin):
        nelx, nely = self.nelx, self.nely
        nel = self.nel
        # element grid coordinates, consistent with mesh.elements() ordering
        # (iy outer, ix inner): element e maps to ix = e % nelx, iy = e // nelx.
        e = np.arange(nel)
        ix = e % nelx
        iy = e // nelx
        self.ecoord = np.column_stack([ix, iy]).astype(float) + 0.5  # nel x 2, (x, y)
        # Loop only over the small (dx, dy) offset stencil; for each offset with a
        # positive weight, vectorize the row/col/value contribution over all
        # elements. This is byte-identical to the per-element double loop because
        # the same (e, neighbor, w) triples are produced, just grouped by offset.
        R = int(np.ceil(rmin)) + 1
        rows, cols, vals = [], [], []
        for dx in range(-R, R + 1):
            for dy in range(-R, R + 1):
                w = rmin - np.hypot(dx, dy)
                if w <= 0:
                    continue
                jx, jy = ix + dx, iy + dy
                m = (jx >= 0) & (jx < nelx) & (jy >= 0) & (jy < nely)
                rows.append(e[m])
                cols.append((jy * nelx + jx)[m])
                vals.append(np.full(int(m.sum()), w))
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)
        self.H = sp.csr_matrix((vals, (rows, cols)), shape=(nel, nel))
        self.Hs = np.asarray(self.H.sum(axis=1)).ravel()

    def _build_bc(self):
        coords = self.mesh.coords()
        L = self.nelx
        # Mechanical: clamp the left edge.
        left = np.where(np.isclose(coords[:, 0], 0.0))[0]
        self.fixed_mech = np.concatenate([2 * left, 2 * left + 1])
        self.free_mech = np.setdiff1d(np.arange(self.ndof), self.fixed_mech)
        # Mechanical load: downward point force at the mid height of the right edge.
        ry = self.nely // 2
        tip = self.mesh.nid(self.nelx, ry)
        self.f_mech = np.zeros(self.ndof)
        self.f_mech[2 * tip + 1] = -self.mech_load
        # Thermal sink at the left edge, uniform heat load elsewhere.
        self.fixed_temp = left
        self.free_temp = np.setdiff1d(np.arange(self.nnode), self.fixed_temp)
        self.q = np.full(self.nnode, self.heat)
        self.q[self.fixed_temp] = 0.0

    # ---- interpolation ---------------------------------------------------
    def E_of(self, rho):
        return self.floor + rho ** self.penal * (1.0 - self.floor)

    def dE_of(self, rho):
        return self.penal * rho ** (self.penal - 1) * (1.0 - self.floor)

    def k_of(self, rho):
        return self.kmin + rho ** self.penal_k * (1.0 - self.kmin)

    def dk_of(self, rho):
        return self.penal_k * rho ** (self.penal_k - 1) * (1.0 - self.kmin)

    def Ef_of(self, rho):       # thermal stress coefficient interpolation
        return rho ** self.penal_f

    def dEf_of(self, rho):
        return self.penal_f * rho ** (self.penal_f - 1)

    # ---- shared forward-solve assembly -----------------------------------
    def _assemble_KT(self, kv):
        """Assemble the global conduction matrix (csr) from a per-element
        conductivity vector, scaling the unit conduction element matrix self.KET0."""
        sKt = (self.KET0.reshape(-1)[None, :] * kv[:, None]).reshape(-1)
        return sp.csr_matrix((sKt, (self.iKt, self.jKt)), shape=(self.nnode, self.nnode))

    def _assemble_KU(self, Ev):
        """Assemble the global mechanical stiffness (csr) from a per-element
        modulus vector, scaling the unit stiffness element matrix self.KE0."""
        sK = (self.KE0.reshape(-1)[None, :] * Ev[:, None]).reshape(-1)
        return sp.csr_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof))

    def _thermal_load(self, coef):
        """Assemble the global thermal load vector via np.add.at over self.G0,
        given a per-element coefficient (typically ef*alpha*(Te-tref))."""
        f_th = np.zeros(self.ndof)
        contrib = coef[:, None] * self.G0[None, :]
        np.add.at(f_th, self.edof.reshape(-1), contrib.reshape(-1))
        return f_th

    # ---- forward solve ---------------------------------------------------
    def solve(self, rhophys):
        # Thermal field on design dependent conductivity.
        if not self.free_temp.size or not np.size(self.fixed_temp):
            raise ValueError("thermal field has no Dirichlet temperature support; "
                             "the conduction matrix is singular")
        kv = self.k_of(rhophys)
        KT = self._assemble_KT(kv)
        T = np.zeros(self.nnode)
        KTff = KT[self.free_temp, :][:, self.free_temp]
        T[self.free_temp] = spla.spsolve(KTff.tocsc(), self.q[self.free_temp])

        # Element average temperature and thermal load.
        Te = T[self.edofT].mean(axis=1)
        ef = self.Ef_of(rhophys)
        coef = ef * self.alpha * (Te - self.tref)        # per element scale on G0
        f_th = self._thermal_load(coef)

        # Mechanical solve.
        Ev = self.E_of(rhophys)
        KU = self._assemble_KU(Ev)
        f_total = self.f_mech + f_th
        u = np.zeros(self.ndof)
        KUff = KU[self.free_mech, :][:, self.free_mech]
        Lu = _factor_or_raise(KUff, "mechanical")
        u[self.free_mech] = Lu.solve(f_total[self.free_mech])

        return dict(T=T, Te=Te, u=u, KT=KT, KU=KU, KTff=KTff, Lu=Lu, Ev=Ev, kv=kv, ef=ef)

    def objective(self, rhophys, st=None):
        st = st or self.solve(rhophys)
        u = st["u"]
        ue = u[self.edof]
        Ku_u_e = np.einsum("ej,jk,ek->e", ue, self.KE0, ue) * st["Ev"]
        return float(np.sum(Ku_u_e)), st

    # ---- sensitivity via the coupled adjoint -----------------------------
    def sensitivity(self, rhophys):
        J, st = self.objective(rhophys)
        u, T, Te = st["u"], st["T"], st["Te"]
        ue = u[self.edof]

        # Mechanical adjoint is explicit: lambda_u = -2 u.
        lam_u = -2.0 * u

        # Thermal adjoint: K_T lam_T = A^T lam_u, with A = d f_th / d T.
        # A maps node temperatures to mech load through coef = ef*alpha*(Te-tref)
        # and Te = mean of element node temps, so d coef / d T_node = ef*alpha*0.25.
        rhs_T = np.zeros(self.nnode)
        w = self.ef_cache = self.Ef_of(rhophys) * self.alpha * 0.25
        # element contribution of A^T lam_u: for each element, scalar
        # s_e = sum_i G0[i] * lam_u[edof_i]; distribute w_e * s_e to its 4 nodes.
        s_e = (self.G0[None, :] * lam_u[self.edof]).sum(axis=1)
        node_add = w * s_e
        np.add.at(rhs_T, self.edofT.reshape(-1),
                  np.repeat(node_add, 4).reshape(self.nel, 4).reshape(-1))
        lam_T = np.zeros(self.nnode)
        lam_T[self.free_temp] = spla.spsolve(st["KTff"].tocsc(), rhs_T[self.free_temp])

        # Assemble the design derivative on the physical density.
        # term1 = -u_e^T dKu u_e
        t1 = -self.dE_of(rhophys) * np.einsum("ej,jk,ek->e", ue, self.KE0, ue)
        # term2 = 2 u_e^T d f_th/d rho ; d f_th/d rho_e = dEf*alpha*(Te-tref)*G0
        df_scale = self.dEf_of(rhophys) * self.alpha * (Te - self.tref)
        t2 = 2.0 * df_scale * (self.G0[None, :] * ue).sum(axis=1)
        # term3 = lam_T_e^T dKt T_e
        lamTe = lam_T[self.edofT]
        Tne = T[self.edofT]
        t3 = self.dk_of(rhophys) * np.einsum("ej,jk,ek->e", lamTe, self.KET0, Tne)

        dJ_dphys = t1 + t2 + t3
        return J, dJ_dphys, st

    # ---- filter chain ----------------------------------------------------
    def filter_forward(self, rho):
        return np.asarray(self.H @ rho).ravel() / self.Hs

    def filter_back(self, dphys):
        return np.asarray(self.H @ (dphys / self.Hs)).ravel()

    def full_gradient(self, rho):
        rhophys = self.filter_forward(rho)
        J, dphys, st = self.sensitivity(rhophys)
        return J, self.filter_back(dphys)

    # ---- optimality criteria update --------------------------------------
    def oc(self, rho, dJ):
        dv = np.asarray(self.H @ (np.ones(self.nel) / self.Hs)).ravel()
        l1, l2, move = 0.0, 1e9, 0.2
        rho_new = rho.copy()
        while (l2 - l1) / (0.5 * (l1 + l2) + 1e-30) > 1e-3:
            lmid = 0.5 * (l1 + l2)
            be = np.maximum(0.0, -dJ / dv / lmid)
            rho_cand = np.maximum(0.0, np.maximum(rho - move,
                       np.minimum(1.0, np.minimum(rho + move, rho * np.sqrt(be)))))
            if np.mean(self.filter_forward(rho_cand)) > self.volfrac:
                l1 = lmid
            else:
                l2 = lmid
            rho_new = rho_cand
        return rho_new

    def run(self, iters=80, tol=1e-3, verbose=False):
        rho = np.full(self.nel, self.volfrac)
        history = []
        for it in range(iters):
            J, dJ = self.full_gradient(rho)
            history.append(J)
            rho_new = self.oc(rho, dJ)
            change = np.max(np.abs(rho_new - rho))
            rho = rho_new
            if verbose:
                print(f"it {it:3d}  J {J:.5e}  vol {np.mean(self.filter_forward(rho)):.3f}  ch {change:.3f}")
            if change < tol and it > 10:
                break
        return rho, self.filter_forward(rho), history


def gradient_check(seed=0, nelx=12, nely=6):
    rng = np.random.default_rng(seed)
    to = ThermoelasticTO(nelx, nely, rmin=1.5)
    rho = rng.uniform(0.3, 0.9, to.nel)
    J0, g = to.full_gradient(rho)
    h = 1e-6
    idx = rng.choice(to.nel, size=8, replace=False)
    errs = []
    for i in idx:
        rp = rho.copy(); rp[i] += h
        rm = rho.copy(); rm[i] -= h
        Jp, _ = to.objective(to.filter_forward(rp))
        Jm, _ = to.objective(to.filter_forward(rm))
        fd = (Jp - Jm) / (2 * h)
        rel = abs(fd - g[i]) / (abs(fd) + 1e-30)
        errs.append(rel)
    return float(max(errs)), errs


if __name__ == "__main__":
    maxerr, errs = gradient_check()
    print("gradient check max relative error:", maxerr)
    print("per element:", [f"{e:.2e}" for e in errs])
