#!/usr/bin/env python3
"""
fig_connectivity.py
===================
Greyscale figure of the voxel connectivity filter. Panel A renders the optimized
design as a single connected body. Panel B reports the three connected component
sizes on a log scale, the dominant body against the two isolated specks the filter
removes. Reads the committed voxel lists, so the figure regenerates without the
optimizer density field.

Output: figures/fig_connectivity.png
"""

import csv
import os
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "geometry", "truck_step", "voxels_raw.csv")
CLEAN = os.path.join(REPO, "geometry", "truck_step", "voxels.csv")
OUT = os.path.join(REPO, "figures", "fig_connectivity.png")

BODY = 0.78
EDGE = (0.25, 0.25, 0.25, 0.55)
_NB = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def load(path):
    pts = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            pts.append(tuple(int(round(float(v))) for v in row[:3]))
    return pts


def components(points):
    vox = set(points)
    seen = set()
    comps = []
    for s in vox:
        if s in seen:
            continue
        q = deque([s]); seen.add(s); cur = 0
        while q:
            p = q.popleft(); cur += 1
            for d in _NB:
                n = (p[0] + d[0], p[1] + d[1], p[2] + d[2])
                if n in vox and n not in seen:
                    seen.add(n); q.append(n)
        comps.append(cur)
    return sorted(comps, reverse=True)


def render_design(ax, points):
    pts = np.array(points)
    dims = tuple(pts.max(axis=0) + 1)
    g = np.zeros(dims, dtype=bool)
    g[pts[:, 0], pts[:, 1], pts[:, 2]] = True
    col = np.zeros(g.shape + (4,))
    for i in range(3):
        col[..., i] = BODY
    col[..., 3] = 1.0
    ax.voxels(g, facecolors=col, edgecolor=EDGE, linewidth=0.15)
    ax.set_box_aspect(dims)
    ax.view_init(elev=24, azim=-58)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.fill = False
        a.pane.set_edgecolor("black")
        a.pane.set_linewidth(0.9)
        a.line.set_color("black")
    ax.grid(False)
    ax.set_title("A | OPTIMIZED DESIGN, ONE CONNECTED BODY",
                 fontsize=10.5, fontweight="bold", pad=0)


def bar_sizes(ax, sizes):
    labels = ["MAIN BODY", "SPECK 1", "SPECK 2"]
    kept = [True, False, False]
    y = np.arange(len(sizes))[::-1]
    for yi, s, k in zip(y, sizes, kept):
        if k:
            ax.barh(yi, s, color="0.55", edgecolor="black", linewidth=1.0)
            ax.text(s ** 0.5, yi, f"{s} voxels", va="center", ha="center",
                    fontsize=10.5, color="white", fontweight="bold")
        else:
            ax.barh(yi, s, color="white", edgecolor="black", linewidth=1.0,
                    hatch="////")
            ax.text(s * 1.7, yi, f"{s} voxels", va="center", fontsize=9.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlim(1, 4000)
    ax.set_xlabel("Component size, voxel count", fontsize=10)
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_linewidth(1.1)
        ax.spines[sp].set_color("black")
    ax.grid(False)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_title("B | CONNECTED COMPONENT SIZES, LOG SCALE",
                 fontsize=10.5, fontweight="bold", pad=6)
    ax.text(0.99, 0.06, "hatched bars: removed by the filter",
            transform=ax.transAxes, ha="right", fontsize=8.0, style="italic",
            color="0.2")


def main():
    raw = load(RAW)
    sizes = components(raw)
    clean = load(CLEAN)

    fig = plt.figure(figsize=(11.6, 5.2))
    fig.patch.set_facecolor("white")
    axA = fig.add_subplot(1, 2, 1, projection="3d")
    render_design(axA, clean)
    axB = fig.add_subplot(1, 2, 2)
    bar_sizes(axB, sizes)

    fig.suptitle(
        "VOXEL CONNECTIVITY FILTER: ISOLATED SPECKS REMOVED BEFORE CAD EXPORT",
        fontsize=12.5, fontweight="bold", y=0.98)
    fig.text(0.5, 0.02,
             "Greyscale render | source geometry/truck_step voxels.csv | "
             "6 connected component analysis, scripts/voxels_export.py",
             ha="center", fontsize=8.0, style="italic", color="0.25")

    fig.subplots_adjust(left=0.02, right=0.97, top=0.88, bottom=0.13, wspace=0.16)
    fig.savefig(OUT, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}")
    print(f"  components {sizes}; raw {len(raw)}, clean {len(clean)}")


if __name__ == "__main__":
    main()
