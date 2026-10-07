"""Figures for the 3D cold-plate crack section. Greyscale house style, no baked caption.

Reads .cache/crack3d.npz (from crack_plate3d.py). Needs numpy, scipy and matplotlib.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter, zoom
from skimage.measure import marching_cubes

import crack_plate3d as cp3                       # voxelize + crack placement, reused

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300})

# render resolution for the plate surface (independent of the FEM solve grid)
RES3D = (208, 208, 18)


def _surface(occ, ax, zexag=2.6):
    vol = gaussian_filter(occ.astype(float), 1.1)          # smoother marching-cubes iso
    verts, faces, _, _ = marching_cubes(vol, level=0.5, step_size=1)
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2] * zexag
    ax.plot_trisurf(x, y, faces, z, color="0.72", edgecolor="none",
                    linewidth=0, antialiased=True, shade=True)
    ax.set_box_aspect((occ.shape[0], occ.shape[1], occ.shape[2] * zexag * 2.4))
    ax.view_init(elev=40, azim=-58)
    ax.set_axis_off()


def _hires_pair():
    nx, ny, nz = RES3D
    occ = cp3.voxelize(nx=nx, ny=ny, nz=nz)
    cracks = cp3.build_plate_cracks(occ.any(2))
    occ_cut = cp3.apply_cracks(occ, cracks, nz)     # surface cracks to their own depth
    return occ, occ_cut


def fig_plate_3d(d):
    occ, occ_cut = _hires_pair()
    fig = plt.figure(figsize=(11.4, 4.9))
    for k, (o, title) in enumerate([(occ, "(a) Pristine Cold Plate"),
                                    (occ_cut, "(b) Fatigue-Cracked")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        _surface(o, ax)
        ax.set_title(title, fontsize=10.5, y=0.97)
    fig.suptitle("Surface Fatigue Cracks Growing Into the Flat Cold Plate", fontsize=11)
    fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0, wspace=0.0)
    p = os.path.join(OUT, "crack3d_plate.png")
    fig.savefig(p); plt.close(fig)
    return p


def fig_plate_heat(d):
    NX, NY = int(d["NX"]), int(d["NY"])
    t0, t1 = d["top0"], d["top1"]
    T0, T1 = t0, t1
    dT = T1 - T0
    vmax = max(T0.max(), T1.max())
    segs = d["segs"]        # crack segments; footprint is [ix, iy], map is [iy, ix]
    casing = [pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()]

    def draw(ax):
        for (p0, p1) in segs:                          # swap to the map's [iy, ix] frame
            ax.plot([p0[1], p1[1]], [p0[0], p1[0]], color="#e8000b", lw=1.8,
                    solid_capstyle="round", path_effects=casing, zorder=5)

    fig, axs = plt.subplots(1, 3, figsize=(11.4, 3.7))
    for ax, F, title, vm, marks in [
        (axs[0], T0, "(a) Pristine", (0, vmax), False),
        (axs[1], T1, "(b) Cracked (cracks in red)", (0, vmax), True),
        (axs[2], dT, "(c) Added Heating (cracks in red)", (0, dT.max()), True)]:
        im = ax.imshow(F, cmap="gray_r", origin="lower", extent=[0, NX, 0, NY],
                       vmin=vm[0], vmax=vm[1], interpolation="bilinear")
        if marks:
            draw(ax)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_title(title, fontsize=9.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7)
    axs[0].set_ylabel("Die-face temperature (arb.)", fontsize=9)
    fig.suptitle("Cold-Plate Heat-Absorption Pattern (Die Face)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "crack3d_heat.png")
    fig.savefig(p); plt.close(fig)
    return p


def main():
    d = np.load(os.path.join(CACHE, "crack3d.npz"), allow_pickle=True)
    print("wrote", fig_plate_3d(d))
    print("wrote", fig_plate_heat(d))


if __name__ == "__main__":
    main()
