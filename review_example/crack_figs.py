"""Figures for the crack-in-the-tree section (2D). Greyscale house style, no baked
caption text: one Title-Case title, axis/colourbar labels only.

Reads .cache/crack2d.npz (written by crack_tree.py). Needs numpy, scipy and matplotlib.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from scipy.ndimage import zoom, gaussian_filter, binary_dilation

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.linewidth": 0.8, "savefig.dpi": 300})


def _smooth_tree(grid, z=6):
    up = gaussian_filter(zoom(grid, z, order=3), z * 0.4)
    return np.clip(up, 0, 1)


def fig_crack_map(d):
    grid, NX, NY = d["grid"], int(d["NX"]), int(d["NY"])
    segs, kinds = d["segs"], d["kinds"]
    up = _smooth_tree(grid)

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.imshow(1 - up, cmap="gray", origin="lower", extent=[0, NX, 0, NY],
              vmin=0, vmax=1, interpolation="bilinear")
    outline = [pe.Stroke(linewidth=3.4, foreground="0.15"), pe.Normal()]
    for seg, kind in zip(segs, kinds):
        (sx, sy), (ex, ey) = seg
        ax.plot([sx, ex], [sy, ey], color="white", lw=1.7, solid_capstyle="round",
                path_effects=outline, zorder=5)
    ax.set_xlim(0, NX); ax.set_ylim(0, NY)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Fatigue Cracks in the Conductor Tree", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02)
    p = os.path.join(OUT, "crack2d_map.png")
    fig.savefig(p); plt.close(fig)
    return p


def _overlay_cracks(ax, segs):
    """Draw the cracks as clean line segments (red with a thin dark casing)."""
    casing = [pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()]
    for seg in segs:
        (sx, sy), (ex, ey) = seg
        ax.plot([sx, ex], [sy, ey], color="#e8000b", lw=1.8, solid_capstyle="round",
                path_effects=casing, zorder=5)


def fig_heat_pattern(d):
    NX, NY = int(d["NX"]), int(d["NY"])
    up = lambda a: zoom(a, 3, order=1)
    T0, T1 = up(d["Tg0"]), up(d["Tg1"])
    dT = T1 - T0
    vmax = max(T0.max(), T1.max())
    segs = d["segs"]

    fig, axs = plt.subplots(1, 3, figsize=(11.4, 3.4))
    for ax, F, title, vm, marks in [
        (axs[0], T0, "(a) Pristine", (0, vmax), False),
        (axs[1], T1, "(b) Cracked (cracks in red)", (0, vmax), True),
        (axs[2], dT, "(c) Added Heating (cracks in red)", (0, dT.max()), True)]:
        im = ax.imshow(F, cmap="gray_r", origin="lower", extent=[0, NX, 0, NY],
                       vmin=vm[0], vmax=vm[1], interpolation="bilinear")
        if marks:
            _overlay_cracks(ax, segs)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_title(title, fontsize=9.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7)
    axs[0].set_ylabel("Temperature (arb. units)", fontsize=9)
    fig.suptitle("Heat-Absorption Pattern: Cracks Strand the Served Region",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT, "crack2d_heat.png")
    fig.savefig(p); plt.close(fig)
    return p


def main():
    d = np.load(os.path.join(CACHE, "crack2d.npz"), allow_pickle=True)
    print("wrote", fig_crack_map(d))
    print("wrote", fig_heat_pattern(d))


if __name__ == "__main__":
    main()
