"""High-resolution, smooth render of the heat-conduction TO conductor tree.

Same physics as the 80 x 40 worked example, but on a fine mesh with the filter
scaled to preserve branch morphology, and a smooth (upsampled) greyscale render
so the branching tree reads cleanly. No baked title/caption on the image.

Figure 5 of the review:  python figure5_tree.py 320 160 5.0 170 6 0.42
(NX NY FILTER ITERS ZOOM SIGMAF; ZOOM and SIGMAF only smooth the rendered image)
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))  # the topoheat engine at the repo root
CACHE = os.path.join(HERE, ".cache"); OUT = os.path.join(HERE, "figures")
os.makedirs(CACHE, exist_ok=True); os.makedirs(OUT, exist_ok=True)
from topoheat.engine import SpecEngine, validate

NX     = int(sys.argv[1]) if len(sys.argv) > 1 else 240
NY     = int(sys.argv[2]) if len(sys.argv) > 2 else 120
FILTER = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
ITERS  = int(sys.argv[4]) if len(sys.argv) > 4 else 150
ZOOM   = int(sys.argv[5]) if len(sys.argv) > 5 else 6
SIGMAF = float(sys.argv[6]) if len(sys.argv) > 6 else 0.35   # anti-alias: higher = softer
VOLFRAC = 0.40
NNODES_X = NX + 1


def node(ix, iy):
    return iy * NNODES_X + ix


def build_spec():
    all_nodes = [node(ix, iy) for iy in range(NY + 1) for ix in range(NX + 1)]
    die = all_nodes
    half = NX // 2
    sinkw = max(2, NX // 32)                      # sink width scales with mesh
    base = [node(ix, 0) for ix in range(half - sinkw, half + sinkw + 1)]
    P_total = 1.0
    return {
        "domain_dimensions": [NX, NY],
        "volume_fraction": VOLFRAC,
        "units": {"length": "mm", "force": "N", "temperature": "K"},
        "dof_ordering": "node_major",
        "material": {
            "youngs_modulus": 1.0, "poisson_ratio": 0.3,
            "thermal_expansion": 0.02, "conductivity": 1.0,
            "density_floor": 0.001,
            "penalization": {"stiffness": 3, "conductivity": 3},
            "filter_radius": FILTER,
        },
        "reference_temperature": 0.0,
        "objective": {"mode": "heat_conduction"},
        "coupling": {"mode": "one_way"},
        "supports": [{"node_indices": base, "fixed_components": [0, 1]}],
        "loads": [{"node_indices": [node(half, 0)], "force_vector": [0.0, -1.0]}],
        "thermal_supports": [{"node_indices": base, "temperature": 0.0}],
        "thermal_loads": [{"node_indices": die, "power": P_total / len(die)}],
    }


def main():
    t0 = time.time()
    spec = build_spec()
    validate(spec)
    eng = SpecEngine(spec)
    print(f"mesh {NX}x{NY} = {NX*NY} elems, filter {FILTER}, {ITERS} iters ...", flush=True)
    rho, rp_opt, hist = eng.run(iters=ITERS, beta_schedule=True)
    print(f"solve done in {time.time()-t0:.1f}s  compliance {hist[0]:.4g} -> {hist[-1]:.4g}", flush=True)

    np.save(os.path.join(CACHE, f"fig5_tree_{NX}x{NY}_f{FILTER}.npy"), rp_opt)
    grid = rp_opt.reshape(NY, NX)

    from scipy.ndimage import zoom, gaussian_filter
    up = zoom(grid, ZOOM, order=3)                 # smooth cubic upsample
    up = gaussian_filter(up, sigma=ZOOM * SIGMAF)   # anti-alias (softer as SIGMAF rises)
    up = np.clip(up, 0, 1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # clean silhouette render, greyscale, no baked text
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.imshow(1 - up, cmap="gray", origin="lower",
              extent=[0, NX, 0, NY], vmin=0, vmax=1, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    p = os.path.join(OUT, f"tree_{NX}x{NY}_f{FILTER}.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
