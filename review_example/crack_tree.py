"""Sub-model 3 (2D): fatigue crack propagation through the pristine conductor tree.

Takes the pristine heat-conduction
topology-optimized tree (topoheat, imported as a dependency), drives a set of fatigue
cracks of different lengths across its primary limbs plus one conjugate-shear X, and
re-solves conduction to show the heat-absorption degradation.

Physics, one label per constant:
  - crack DRIVER: constrained thermal-cycling stress in the copper limb,
    dsigma = E_cu * alpha_cu * dT  (clamped alpha*dT bound, hand-checkable).
  - crack GROWTH: Paris law da/dN = C (dK)^m, dK = Y dsigma sqrt(pi a), integrated
    from a0 to the limb width -> cycles-to-sever. Different sites reach different
    lengths (different local strain / age): short, partial, and full-sever cracks.
  - crack DIRECTION: transverse cracks run perpendicular to the local limb axis (the
    fastest way to sever a conduction path). The X is a pair of conjugate cracks at
    +/-45 deg to the axis, the mixed-mode shear pattern thermal cycling can drive.
  - EFFECT: cracked cells lose conductivity; re-solving conduction gives the new
    temperature field. Severed limbs strand their collection area, which runs hot.

Paris constants for OFHC copper are lit-range / SCHEMATIC (no in-hand da/dN data), so
absolute cycles are illustrative; the pattern and coupling are the result.

Run:  python crack_tree.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

_TOPO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..")
if os.path.abspath(_TOPO) not in sys.path:
    sys.path.insert(0, os.path.abspath(_TOPO))

from topoheat.engine import SpecEngine, validate     # noqa: E402
import fatigue as fat                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
os.makedirs(CACHE, exist_ok=True)

# --- Paris-law crack growth for OFHC copper (lit-range / SCHEMATIC) -----------------
PARIS_C = 1.0e-11    # m/cycle / (MPa*m^0.5)^m  ; Cu ~1e-11..1e-12 (lit-range)
PARIS_M = 3.0        # Cu Paris exponent ~3      (lit-range)
Y_GEOM  = 1.12       # edge-crack geometry factor (gated, standard)
A0_M    = 2.0e-5     # initial crack ~20 um (SCHEMATIC)
CELL_MM = 1.0        # each element is 1 mm


def tree_spec(NX, NY, filt, volfrac=0.40):
    nnx = NX + 1
    def node(ix, iy):
        return iy * nnx + ix
    alln = [node(ix, iy) for iy in range(NY + 1) for ix in range(NX + 1)]
    half = NX // 2
    sink = [node(ix, 0) for ix in range(half - 2, half + 3)]
    return {
        "domain_dimensions": [NX, NY], "volume_fraction": volfrac,
        "units": {"length": "mm", "force": "N", "temperature": "K"},
        "dof_ordering": "node_major",
        "material": {"youngs_modulus": 1.0, "poisson_ratio": 0.3,
                     "thermal_expansion": 0.02, "conductivity": 1.0,
                     "density_floor": 0.001,
                     "penalization": {"stiffness": 3, "conductivity": 3},
                     "filter_radius": filt},
        "reference_temperature": 0.0,
        "objective": {"mode": "heat_conduction"}, "coupling": {"mode": "one_way"},
        "supports": [{"node_indices": sink, "fixed_components": [0, 1]}],
        "loads": [{"node_indices": [node(half, 0)], "force_vector": [0.0, -1.0]}],
        "thermal_supports": [{"node_indices": sink, "temperature": 0.0}],
        "thermal_loads": [{"node_indices": alln, "power": 1.0 / len(alln)}],
    }


def pristine(NX=160, NY=80, filt=4.0, iters=120):
    spec = tree_spec(NX, NY, filt)
    validate(spec)
    eng = SpecEngine(spec)
    cf = os.path.join(CACHE, f"tree_{NX}x{NY}_f{filt}.npy")
    if os.path.exists(cf):
        rp = np.load(cf)
    else:
        _, rp, _ = eng.run(iters=iters, beta_schedule=True)
        np.save(cf, rp)
    return eng, rp


def temp_field(eng, rp, NX, NY):
    T = eng.solve(rp)["T"]
    return T.reshape(NY + 1, NX + 1), float(T.max())


# --- crack geometry ----------------------------------------------------------------
def _grad(grid):
    from scipy.ndimage import gaussian_filter
    gs = gaussian_filter(grid, 1.0)
    gy, gx = np.gradient(gs)
    return gx, gy


def limb_axis(gx, gy, i, j, win=4, shape=None):
    j0, j1 = max(0, j - win), min(shape[0], j + win + 1)
    i0, i1 = max(0, i - win), min(shape[1], i + win + 1)
    Jxx = float(np.sum(gx[j0:j1, i0:i1] ** 2))
    Jyy = float(np.sum(gy[j0:j1, i0:i1] ** 2))
    Jxy = float(np.sum(gx[j0:j1, i0:i1] * gy[j0:j1, i0:i1]))
    w, v = np.linalg.eigh(np.array([[Jxx, Jxy], [Jxy, Jyy]]))
    return v[:, 0], v[:, 1]            # along (small eig), normal/across (large eig)


def snap_solid(grid, fx, fy, NX, NY, r=6):
    """Nearest solid cell (rho>0.5) to a fractional target."""
    i0, j0 = int(fx * NX), int(fy * NY)
    best = None
    for dj in range(-r, r + 1):
        for di in range(-r, r + 1):
            i, j = i0 + di, j0 + dj
            if 0 <= i < NX and 0 <= j < NY and grid[j, i] > 0.5:
                d = di * di + dj * dj
                if best is None or d < best[0]:
                    best = (d, i, j)
    return None if best is None else (best[1], best[2])


def _rot(v, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def limb_width(grid, seed, direction):
    """Solid extent through the seed along +/- direction (cells)."""
    i0, j0 = seed; dx, dy = direction
    def march(sgn):
        t = 0.0
        while True:
            i = int(round(i0 + sgn * t * dx)); j = int(round(j0 + sgn * t * dy))
            if not (0 <= i < grid.shape[1] and 0 <= j < grid.shape[0]) or grid[j, i] < 0.5:
                return t
            t += 0.5
    tp, tm = march(+1), march(-1)
    return tp + tm, tm


def crack_line(grid, seed, direction, a_cells, mode="edge"):
    """Cells cut + segment endpoints (i,j) for a crack of length a_cells."""
    i0, j0 = seed; dx, dy = direction
    _, tm = limb_width(grid, seed, direction)
    if mode == "center":
        sx, sy = i0 - 0.5 * a_cells * dx, j0 - 0.5 * a_cells * dy
    else:                                       # grow from the -direction edge inward
        sx, sy = i0 - tm * dx, j0 - tm * dy
    cells = set(); t = 0.0
    while t <= a_cells:
        for perp in (-0.5, 0.0, 0.5):
            pi, pj = -dy * perp, dx * perp
            i = int(round(sx + t * dx + pi)); j = int(round(sy + t * dy + pj))
            if 0 <= i < grid.shape[1] and 0 <= j < grid.shape[0]:
                cells.add((j, i))
        t += 0.5
    return cells, ((sx, sy), (sx + a_cells * dx, sy + a_cells * dy))


def cut_density(rp, NX, NY, cells, floor=0.001):
    g = rp.reshape(NY, NX).copy()
    for (j, i) in cells:
        g[j, i] = floor
    return g.ravel()


def cycles_vs_length(a_end_m, dT, n=200):
    E_MPa = fat._v("E"); alpha = fat._v("alpha")
    dsigma = E_MPa * alpha * dT
    a = np.linspace(A0_M, max(a_end_m, 2 * A0_M), n)
    dadN = PARIS_C * (Y_GEOM * dsigma * np.sqrt(np.pi * a)) ** PARIS_M
    N = np.concatenate([[0.0], np.cumsum(np.diff(a) / (0.5 * (dadN[1:] + dadN[:-1])))])
    return a, N, dsigma


# targets on primary/secondary limbs; (fx, fy, length-fraction-of-width, kind)
CRACK_TARGETS = [
    (0.30, 0.42, 1.00, "transverse"),   # full sever, primary left limb
    (0.71, 0.46, 0.55, "transverse"),   # partial, primary right limb
    (0.20, 0.60, 0.35, "transverse"),   # short, outer left
    (0.80, 0.34, 0.80, "transverse"),   # long, right
    (0.62, 0.38, 0.90, "x"),            # crisscross X on a right-of-centre limb
]


def build_cracks(grid, NX, NY):
    gx, gy = _grad(grid)
    all_cells, segs = set(), []
    for fx, fy, frac, kind in CRACK_TARGETS:
        seed = snap_solid(grid, fx, fy, NX, NY)
        if seed is None:
            continue
        _, normal = limb_axis(gx, gy, seed[0], seed[1], shape=(NY, NX))
        width, _ = limb_width(grid, seed, normal)
        a = max(2.0, frac * width)
        if kind == "x":
            for ang in (+45, -45):
                d = _rot(normal, ang)
                cells, seg = crack_line(grid, seed, d, a, mode="center")
                all_cells |= cells; segs.append((seg, kind, a))
        else:
            cells, seg = crack_line(grid, seed, normal, a, mode="edge")
            all_cells |= cells; segs.append((seg, kind, a))
    return all_cells, segs


def main():
    NX, NY, filt, dT = 240, 120, 5.0, 60.0
    eng, rp = pristine(NX, NY, filt)
    grid = rp.reshape(NY, NX)
    Tg0, peak0 = temp_field(eng, rp, NX, NY)

    all_cells, segs = build_cracks(grid, NX, NY)
    rp_cut = cut_density(rp, NX, NY, all_cells)
    Tg1, peak1 = temp_field(eng, rp_cut, NX, NY)
    crack_mask = (grid > 0.5) & (rp_cut.reshape(NY, NX) <= 0.5)   # material removed

    a_ref = np.array([s[2] for s in segs]) * CELL_MM * 1e-3
    _, _, dsig = cycles_vs_length(a_ref.max(), dT)
    print(f"pristine peak T = {peak0:.4g}")
    print(f"cracked  peak T = {peak1:.4g}  ({100*(peak1/peak0 - 1):.1f}% rise)")
    print(f"{len(segs)} cracks placed (incl. one X); driver dsigma = {dsig:.1f} MPa at dT={dT:.0f} K")

    np.savez(os.path.join(CACHE, "crack2d.npz"),
             grid=grid, Tg0=Tg0, Tg1=Tg1, peak0=peak0, peak1=peak1,
             crack_mask=crack_mask,
             segs=np.array([s[0] for s in segs], dtype=object),
             kinds=np.array([s[1] for s in segs]),
             lengths=np.array([s[2] for s in segs]), NX=NX, NY=NY)
    print("wrote", os.path.join(CACHE, "crack2d.npz"))


if __name__ == "__main__":
    main()
