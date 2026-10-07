"""Composite flagship figure for the crack section: the fatigue-damage cascade from the
2D conductor tree to the 3D cold plate, with crack depth.

Blue-to-red (turbo) heat maps for readability; cracks marked in red everywhere; a depth
cross-section shows a crack penetrating the plate thickness with the conducting ligament
below. Reads .cache/crack2d.npz and .cache/crack3d.npz. Needs numpy, scipy and matplotlib.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter, zoom
from skimage.measure import marching_cubes

import crack_plate3d as cp3
import crack_cycles as cc

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
OUT = os.path.join(HERE, "figures")
CMAP = "turbo"
RED = "#d40000"
CASE = [pe.Stroke(linewidth=4.0, foreground="white"), pe.Normal()]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "savefig.dpi": 300})


def _segs(ax, segs, swap):
    for (p0, p1) in segs:
        x = ([p0[1], p1[1]] if swap else [p0[0], p1[0]])
        y = ([p0[0], p1[0]] if swap else [p0[1], p1[1]])
        ax.plot(x, y, color=RED, lw=2.0, solid_capstyle="round",
                path_effects=CASE, zorder=6)


def _heat(ax, F, NX, NY, vmax, title, segs=None, swap=False):
    im = ax.imshow(F, cmap=CMAP, origin="lower", extent=[0, NX, 0, NY],
                   vmin=0, vmax=vmax, interpolation="bilinear")
    if segs is not None:
        _segs(ax, segs, swap)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_title(title, fontsize=9.5)
    return im


def _plate3d(ax, RES=(176, 176, 16), zexag=2.6):
    nx, ny, nz = RES
    occ = cp3.voxelize(nx=nx, ny=ny, nz=nz)
    cracks = cp3.build_plate_cracks(occ.any(2))
    occ_cut = cp3.apply_cracks(occ, cracks, nz)
    vol = gaussian_filter(occ_cut.astype(float), 1.1)
    verts, faces, _, _ = marching_cubes(vol, level=0.5)
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2] * zexag,
                    color="0.72", edgecolor="none", antialiased=True, shade=True)
    ztop = nz * zexag
    for cr in cracks:
        drop = cr["depth"] * nz * zexag
        for (p0, p1) in cr["segs"]:
            x0, y0, x1, y1 = p0[1], p0[0], p1[1], p1[0]     # foot [ix,iy] -> plate axes
            ax.plot([x0, x1], [y0, y1], [ztop, ztop], color=RED, lw=2.2,
                    path_effects=CASE, zorder=10)
            xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            ax.plot([xm, xm], [ym, ym], [ztop, ztop - drop], color=RED, lw=1.5, zorder=10)
    ax.set_box_aspect((nx, ny, nz * zexag * 2.4))
    ax.view_init(elev=40, azim=-58); ax.set_axis_off()
    ax.set_title("(c) Cold Plate: Cracks + Depth", fontsize=9.5)


def _failure(ax, ds):
    """Branch effectiveness vs crack depth: the partial-crack failure threshold."""
    x = ds["depths"] * 100.0; eff = ds["eff"]; dc = float(ds["d_crit"]) * 100.0
    ax.axvspan(0, dc, color="#2ca02c", alpha=0.10)
    ax.axvspan(dc, 100, color=RED, alpha=0.10)
    ax.plot(x, eff, "-o", color="0.1", lw=1.8, ms=3, zorder=5)
    ax.axvline(dc, color=RED, ls="--", lw=1.3)
    ax.text(dc + 3, 88, "fails at\n~%.0f%% depth" % dc, color=RED, fontsize=8, va="top")
    ax.text(4, eff.min() + 6, "shallow:\nligament\nconducts", color="#217a21",
            fontsize=7.5, va="bottom")
    ax.set_xlim(0, 100); ax.set_ylim(min(50, eff.min() - 5), 103)
    ax.set_xlabel("crack depth (% thickness)", fontsize=8)
    ax.set_ylabel("branch effectiveness (%)", fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(True, color="0.9", lw=0.5)
    ax.set_title("(f) Partial Crack → Failure Threshold", fontsize=9.5)


def _cycles(ax, dsm):
    """Paris growth maps each branch's critical depth to a cycles-to-failure."""
    labels = list(dsm["labels"]); dcrit = dsm["dcrit"]; order = np.argsort(dcrit)
    a_curve, N_curve = cc.depth_vs_cycles(cc.THICKNESS_MM * 1e-3)
    depth_pct = a_curve / (cc.THICKNESS_MM * 1e-3) * 100.0
    ax.plot(N_curve, depth_pct, color="0.15", lw=2.2, zorder=3)
    Nmax = N_curve.max()
    for row, k in enumerate(np.argsort([cc.cycles_to(dcrit[j] * cc.THICKNESS_MM * 1e-3)
                                        for j in range(len(dcrit))])):
        col = cc.COLORS[k % len(cc.COLORS)]
        yc = dcrit[k] * 100.0
        nf = cc.cycles_to(dcrit[k] * cc.THICKNESS_MM * 1e-3)
        ax.axhline(yc, color=col, ls=":", lw=0.9, alpha=0.55)
        ax.plot([nf], [yc], "o", color=col, ms=7, zorder=5)
        ax.text(Nmax * 0.02, 96 - row * 7.5,
                "%-16s crit %2.0f%%  ->  %.2e cyc" % (labels[k], dcrit[k] * 100.0, nf),
                color=col, fontsize=9, va="top", family="DejaVu Sans Mono")
    ax.set_xlim(0, Nmax * 1.02); ax.set_ylim(0, 100)
    ax.set_xlabel("Cycles"); ax.set_ylabel("Crack depth (% thickness)")
    ax.set_title("(g) Paris Growth -> Cycles-to-Failure per Branch  "
                 "(critical depth spans 6-47% of thickness, life spans only ~17%)",
                 fontsize=10)
    ax.grid(True, color="0.92", lw=0.6)


def main():
    d2 = np.load(os.path.join(CACHE, "crack2d.npz"), allow_pickle=True)
    d3 = np.load(os.path.join(CACHE, "crack3d.npz"), allow_pickle=True)
    ds = np.load(os.path.join(CACHE, "depth_sweep.npz"))
    dsm = np.load(os.path.join(CACHE, "depth_sweep_multi.npz"), allow_pickle=True)

    NX2, NY2 = int(d2["NX"]), int(d2["NY"])
    up2 = lambda a: zoom(a, 3, order=1)
    T2p, T2c = up2(d2["Tg0"]), up2(d2["Tg1"])
    v2 = max(T2p.max(), T2c.max())

    NX3, NY3 = int(d3["NX"]), int(d3["NY"])
    T3p, T3c = d3["top0"], d3["top1"]
    v3 = max(T3p.max(), T3c.max())

    fig = plt.figure(figsize=(12.2, 10.4))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.8],
                           hspace=0.24, wspace=0.12)

    a = fig.add_subplot(gs[0, 0]); im2 = _heat(a, T2p, NX2, NY2, v2, "(a) Tree: Pristine")
    b = fig.add_subplot(gs[0, 1]); _heat(b, T2c, NX2, NY2, v2, "(b) Tree: Cracked",
                                         segs=d2["segs"], swap=False)
    fig.colorbar(im2, ax=[a, b], fraction=0.025, pad=0.01).set_label("T (arb.)", fontsize=8)

    c = fig.add_subplot(gs[0, 2], projection="3d"); _plate3d(c)

    e = fig.add_subplot(gs[1, 0]); im3 = _heat(e, T3p, NX3, NY3, v3, "(d) Plate Die Face: Pristine")
    f = fig.add_subplot(gs[1, 1]); _heat(f, T3c, NX3, NY3, v3, "(e) Plate Die Face: Cracked",
                                         segs=d3["segs"], swap=True)
    fig.colorbar(im3, ax=[e], fraction=0.045, pad=0.02).set_label("T (arb.)", fontsize=8)

    g = fig.add_subplot(gs[1, 2]); _failure(g, ds)

    h = fig.add_subplot(gs[2, :]); _cycles(h, dsm)

    fig.suptitle("Fatigue-Damage Cascade: Conductor Tree to Cold Plate, Crack Depth, and Life",
                 fontsize=12.5, y=0.99)
    p = os.path.join(OUT, "crack_composite.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
