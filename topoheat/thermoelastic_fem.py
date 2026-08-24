"""
Minimal Q4 thermoelastic finite element benchmark for the Odinzen engine.

Two physics on the same node grid:
  scalar steady state conduction   K_T T = q
  plane stress thermoelasticity    K_u u = f_mech + f_thermal(T)

The thermal load f_thermal is the consistent equivalent nodal force of the
free thermal strain eps0 = alpha * dT * [1, 1, 0]. The module validates each
piece against a closed form result, then demonstrates a two way fixed point
where conductivity depends on the local deformation.
"""

import numpy as np

GAUSS = np.array([-1.0, 1.0]) / np.sqrt(3.0)


def _shape_grads(xi, eta, h):
    # Bilinear Q4 on a square element of side h. Nodes counterclockwise:
    # (0,0), (h,0), (h,h), (0,h). Natural coords in [-1, 1].
    dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    # Jacobian of a square element is (h/2) * I, so d/dx = (2/h) d/dxi.
    dN_dx = (2.0 / h) * dN_dxi
    dN_dy = (2.0 / h) * dN_deta
    detJ = (h / 2.0) ** 2
    return dN_dx, dN_dy, detJ


def elem_conduction(k, h):
    Ke = np.zeros((4, 4))
    for xi in GAUSS:
        for eta in GAUSS:
            dN_dx, dN_dy, detJ = _shape_grads(xi, eta, h)
            B = np.vstack([dN_dx, dN_dy])  # 2 x 4
            Ke += k * (B.T @ B) * detJ
    return Ke


def _Dmat(E, nu):
    return E / (1 - nu ** 2) * np.array([[1, nu, 0],
                                         [nu, 1, 0],
                                         [0, 0, (1 - nu) / 2]])


def _Bmat(xi, eta, h):
    dN_dx, dN_dy, _ = _shape_grads(xi, eta, h)
    B = np.zeros((3, 8))
    B[0, 0::2] = dN_dx
    B[1, 1::2] = dN_dy
    B[2, 0::2] = dN_dy
    B[2, 1::2] = dN_dx
    return B


def elem_stiffness(E, nu, h):
    D = _Dmat(E, nu)
    Ke = np.zeros((8, 8))
    for xi in GAUSS:
        for eta in GAUSS:
            B = _Bmat(xi, eta, h)
            _, _, detJ = _shape_grads(xi, eta, h)
            Ke += (B.T @ D @ B) * detJ
    return Ke


def elem_thermal_force(E, nu, alpha, dT, h):
    # f = integral B^T D eps0 dOmega, eps0 = alpha dT [1, 1, 0].
    D = _Dmat(E, nu)
    eps0 = alpha * dT * np.array([1.0, 1.0, 0.0])
    f = np.zeros(8)
    for xi in GAUSS:
        for eta in GAUSS:
            B = _Bmat(xi, eta, h)
            _, _, detJ = _shape_grads(xi, eta, h)
            f += (B.T @ D @ eps0) * detJ
    return f


class Mesh:
    def __init__(self, nx, ny, h=1.0):
        self.nx, self.ny, self.h = nx, ny, h
        self.nnode = (nx + 1) * (ny + 1)

    def nid(self, ix, iy):
        return iy * (self.nx + 1) + ix

    def coords(self):
        xs = np.arange(self.nx + 1) * self.h
        ys = np.arange(self.ny + 1) * self.h
        XX, YY = np.meshgrid(xs, ys)
        return np.column_stack([XX.ravel(), YY.ravel()])

    def elements(self):
        els = []
        for iy in range(self.ny):
            for ix in range(self.nx):
                els.append([self.nid(ix, iy), self.nid(ix + 1, iy),
                            self.nid(ix + 1, iy + 1), self.nid(ix, iy + 1)])
        return els


def assemble_conduction(mesh, kfield):
    n = mesh.nnode
    K = np.zeros((n, n))
    for e, nodes in enumerate(mesh.elements()):
        Ke = elem_conduction(kfield[e], mesh.h)
        for a in range(4):
            for b in range(4):
                K[nodes[a], nodes[b]] += Ke[a, b]
    return K


def assemble_mech(mesh, E, nu, alpha, dTfield):
    n = mesh.nnode
    K = np.zeros((2 * n, 2 * n))
    F = np.zeros(2 * n)
    for e, nodes in enumerate(mesh.elements()):
        Ke = elem_stiffness(E, nu, mesh.h)
        fe = elem_thermal_force(E, nu, alpha, dTfield[e], mesh.h)
        edof = np.array([[2 * nd, 2 * nd + 1] for nd in nodes]).ravel()
        for a in range(8):
            F[edof[a]] += fe[a]
            for b in range(8):
                K[edof[a], edof[b]] += Ke[a, b]
    return K, F


def solve_dirichlet(K, F, fixed, values=None):
    n = K.shape[0]
    free = np.setdiff1d(np.arange(n), fixed)
    u = np.zeros(n)
    if values is not None:
        u[fixed] = values
    rhs = F[free] - K[np.ix_(free, fixed)] @ u[fixed]
    u[free] = np.linalg.solve(K[np.ix_(free, free)], rhs)
    return u


def elem_stress(mesh, u, E, nu, alpha, dTe, nodes):
    D = _Dmat(E, nu)
    edof = np.array([[2 * nd, 2 * nd + 1] for nd in nodes]).ravel()
    ue = u[edof]
    B = _Bmat(0.0, 0.0, mesh.h)  # element centre
    eps = B @ ue
    eps0 = alpha * dTe * np.array([1.0, 1.0, 0.0])
    return D @ (eps - eps0)


# ----------------------------------------------------------------------
# Verification cases
# ----------------------------------------------------------------------

def verify_conduction():
    mesh = Mesh(6, 4, h=1.0)
    kfield = np.ones(len(mesh.elements()))
    K = assemble_conduction(mesh, kfield)
    coords = mesh.coords()
    L = mesh.nx * mesh.h
    left = np.where(np.isclose(coords[:, 0], 0.0))[0]
    right = np.where(np.isclose(coords[:, 0], L))[0]
    fixed = np.concatenate([left, right])
    values = np.concatenate([np.full(left.size, 100.0), np.full(right.size, 0.0)])
    q = np.zeros(mesh.nnode)
    T = solve_dirichlet(K, q, fixed, values)
    analytic = 100.0 * (1 - coords[:, 0] / L)
    return float(np.max(np.abs(T - analytic)))


def verify_patch_free():
    # Uniform heating, minimal constraint to remove rigid body only.
    E, nu, alpha, dT = 210e3, 0.3, 12e-6, 50.0  # MPa, -, 1/K, K
    mesh = Mesh(5, 5, h=1.0)
    dTfield = np.full(len(mesh.elements()), dT)
    K, F = assemble_mech(mesh, E, nu, alpha, dTfield)
    # Pin node 0 in x and y, and bottom right node in y: statically minimal.
    n0 = mesh.nid(0, 0)
    nbr = mesh.nid(mesh.nx, 0)
    fixed = np.array([2 * n0, 2 * n0 + 1, 2 * nbr + 1])
    u = solve_dirichlet(K, F, fixed)
    stresses = [elem_stress(mesh, u, E, nu, alpha, dT, nodes)
                for nodes in mesh.elements()]
    max_stress = float(np.max(np.abs(np.array(stresses))))
    # Displacement should be uniform expansion u = alpha dT (x, y).
    coords = mesh.coords()
    ux = u[0::2]
    uy = u[1::2]
    exp_ux = alpha * dT * coords[:, 0]
    exp_uy = alpha * dT * coords[:, 1]
    disp_err = float(np.max(np.abs(ux - exp_ux)) + np.max(np.abs(uy - exp_uy)))
    return max_stress, disp_err


def verify_clamped():
    # Fully clamp all boundary displacements, uniform heating.
    E, nu, alpha, dT = 210e3, 0.3, 12e-6, 50.0
    mesh = Mesh(6, 6, h=1.0)
    dTfield = np.full(len(mesh.elements()), dT)
    K, F = assemble_mech(mesh, E, nu, alpha, dTfield)
    coords = mesh.coords()
    L = mesh.nx * mesh.h
    H = mesh.ny * mesh.h
    on_bnd = (np.isclose(coords[:, 0], 0) | np.isclose(coords[:, 0], L) |
              np.isclose(coords[:, 1], 0) | np.isclose(coords[:, 1], H))
    bnodes = np.where(on_bnd)[0]
    fixed = np.concatenate([[2 * b, 2 * b + 1] for b in bnodes])
    u = solve_dirichlet(K, F, fixed)
    # Stress of the central element.
    centre = mesh.elements()[len(mesh.elements()) // 2]
    sig = elem_stress(mesh, u, E, nu, alpha, dT, centre)
    analytic = -E * alpha * dT / (1 - nu)
    return float(sig[0]), float(sig[1]), analytic


def verify_two_way():
    # Conductivity rises with local volumetric strain: k = k0 (1 + beta*vol).
    # Staggered fixed point: thermal -> mechanical -> update k -> repeat.
    E, nu, alpha = 210e3, 0.3, 12e-6
    mesh = Mesh(6, 4, h=1.0)
    coords = mesh.coords()
    L = mesh.nx * mesh.h
    left = np.where(np.isclose(coords[:, 0], 0))[0]
    right = np.where(np.isclose(coords[:, 0], L))[0]
    tfixed = np.concatenate([left, right])
    tvals = np.concatenate([np.full(left.size, 100.0), np.full(right.size, 0.0)])
    n0 = mesh.nid(0, 0)
    nbr = mesh.nid(mesh.nx, 0)
    mfixed = np.array([2 * n0, 2 * n0 + 1, 2 * nbr + 1])

    nel = len(mesh.elements())
    k0, beta = 1.0, 5.0
    kfield = np.full(nel, k0)
    Tprev = None
    residuals = []
    for _ in range(12):
        KT = assemble_conduction(mesh, kfield)
        T = solve_dirichlet(KT, np.zeros(mesh.nnode), tfixed, tvals)
        if Tprev is not None:
            residuals.append(float(np.linalg.norm(T - Tprev)))
        Tprev = T.copy()
        # Element temperature and mechanical solve.
        dTe = np.array([np.mean(T[nodes]) for nodes in mesh.elements()])
        KU, F = assemble_mech(mesh, E, nu, alpha, dTe)
        u = solve_dirichlet(KU, F, mfixed)
        # Volumetric strain per element feeds conductivity.
        vol = []
        for nodes in mesh.elements():
            B = _Bmat(0.0, 0.0, mesh.h)
            edof = np.array([[2 * nd, 2 * nd + 1] for nd in nodes]).ravel()
            eps = B @ u[edof]
            vol.append(eps[0] + eps[1])
        kfield = k0 * (1 + beta * np.array(vol))
        kfield = np.clip(kfield, 0.1, None)
    return residuals


if __name__ == "__main__":
    c_err = verify_conduction()
    p_stress, p_disp = verify_patch_free()
    sx, sy, s_an = verify_clamped()
    res = verify_two_way()
    print("CONDUCTION 1D max abs error:", c_err)
    print("PATCH free expansion max abs stress (MPa):", p_stress)
    print("PATCH free expansion displacement error:", p_disp)
    print("CLAMPED central stress sx, sy (MPa):", sx, sy)
    print("CLAMPED analytic -E a dT /(1-nu) (MPa):", s_an)
    print("CLAMPED relative error:", abs(sx - s_an) / abs(s_an))
    print("TWO WAY fixed point residuals:", [round(r, 6) for r in res])
    print("TWO WAY monotone decreasing:", all(x > y for x, y in zip(res, res[1:])))
