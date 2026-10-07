"""Section 6.1 of the cooling review: the optimized tree against straight fins and a uniform plate.

Every comparison is at equal material (volume fraction 0.40), with heat generated over the whole
domain and a coolant sink at the centre of the base. Two designs:

  320 x 160, filter 5.0   the tree shown in Figure 5 (figure5_tree.py; sink 21 nodes wide)
  80 x 40,   filter 2.4   a coarser network (sink 5 nodes), the case where the peak-temperature
                          advantage over the uniform plate reverses

Baselines, each solved on the same problem as its tree:
  straight fins + base    a solid base strip with seven straight fins (3-row base and 4-wide fins
                          on the 80 x 40 grid, scaled with the grid), solid/void like the tree
  uniform plate           the same metal spread evenly. Under SIMP (k = rho^3) that is
                          rho = 0.40^(1/3), conductivity 0.40, NOT grey rho = 0.40, which the
                          model penalizes to conductivity 0.065 and would inflate the gain

Peak T is the hottest node; mean T equals the thermal compliance q^T T, the optimizer's own
objective and an average conduction resistance.

Run figure5_tree.py first (it caches the Figure 5 design), then:  python section61_comparisons.py
"""
import os
import sys

import numpy as np

import crack_tree as ct
from topoheat.engine import SpecEngine

HERE = os.path.dirname(os.path.abspath(__file__))
VOLFRAC = 0.40


def straight_fins(nx, ny):
    """Solid base strip plus seven straight fins, the layout used for the 80 x 40 comparison."""
    s = nx // 80
    f = np.full((ny, nx), 0.001)
    f[:3 * s, :] = 1.0
    for x in (np.linspace(4, 80 - 8, 7) * s).astype(int):
        f[3 * s:, x:x + 4 * s] = 1.0
    return f.ravel()


def figure5_design():
    sys.argv = ["figure5_tree.py", "320", "160", "5.0", "170"]  # figure5_tree reads its size from argv
    import figure5_tree as f5
    path = os.path.join(HERE, ".cache", "fig5_tree_320x160_f5.0.npy")
    if not os.path.exists(path):
        raise SystemExit("run figure5_tree.py 320 160 5.0 170 6 0.42 first")
    return SpecEngine(f5.build_spec()), np.load(path), 320, 160


def coarse_design():
    eng, tree = ct.pristine(80, 40, 2.4)  # same problem with a 5-node sink, cached under .cache/
    return eng, tree, 80, 40


def compare(eng, tree, nx, ny):
    designs = {"tree": tree, "fins": straight_fins(nx, ny),
               "uniform": np.full(nx * ny, VOLFRAC ** (1.0 / 3.0))}
    out = {}
    for name, rho in designs.items():
        mean_t, st = eng.objective(rho)
        temp = st["T"]
        out[name] = (float(temp.max()), float(mean_t), float(np.mean(temp > 0.5 * temp.max())))
    return out


def main():
    for label, build in [("Figure 5 design, 320 x 160, filter 5.0", figure5_design),
                         ("coarser network, 80 x 40, filter 2.4", coarse_design)]:
        r = compare(*build())
        print(f"== {label}")
        for name in ("tree", "fins", "uniform"):
            peak, mean_t, hot = r[name]
            print(f"   {name:8s} peak T {peak:7.3f}   mean T {mean_t:7.3f}   share above half its peak {hot:.2f}")
        t, f, u = r["tree"], r["fins"], r["uniform"]
        print(f"   tree vs straight fins: peak {100 * (1 - t[0] / f[0]):.0f}% lower")
        print(f"   tree vs uniform plate: mean {100 * (1 - t[1] / u[1]):.0f}% lower, "
              f"peak {100 * abs(t[0] / u[0] - 1):.0f}% {'higher' if t[0] > u[0] else 'lower'}")


if __name__ == "__main__":
    main()
