"""Standalone versions of the composite panels, for use as separate manuscript figures.
Reuses the exact panel code from crack_composite so the split figures match the composite.

  crack_fig_tree.png    : conductor-tree die heat, pristine vs cracked
  crack_fig_plate.png   : 3D cold plate (cracks + depth) and die-face heat pristine/cracked
  crack_fig_failure.png : the partial-crack failure threshold (single branch)
  crack_fig_cycles.png  : Paris growth -> cycles-to-failure across branches

Needs numpy, scipy and matplotlib.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

import crack_composite as cx

CACHE, OUT = cx.CACHE, cx.OUT


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

    # 1) tree heat, pristine vs cracked
    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.6))
    a = cx._heat(axs[0], T2p, NX2, NY2, v2, "(a) Pristine")
    cx._heat(axs[1], T2c, NX2, NY2, v2, "(b) Cracked (cracks in red)",
             segs=d2["segs"], swap=False)
    fig.colorbar(a, ax=axs, fraction=0.03, pad=0.02).set_label("T (arb.)", fontsize=8)
    p = os.path.join(OUT, "crack_fig_tree.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("wrote", p)

    # 2) cold plate (3D) + die-face heat pristine/cracked
    fig = plt.figure(figsize=(12.6, 4.2))
    axc = fig.add_subplot(1, 3, 1, projection="3d")
    cx._plate3d(axc); axc.set_title("(a) Cold Plate: Cracks + Depth", fontsize=10)
    axd = fig.add_subplot(1, 3, 2)
    im = cx._heat(axd, T3p, NX3, NY3, v3, "(b) Die Face: Pristine")
    axe = fig.add_subplot(1, 3, 3)
    cx._heat(axe, T3c, NX3, NY3, v3, "(c) Die Face: Cracked (cracks in red)",
             segs=d3["segs"], swap=True)
    fig.colorbar(im, ax=[axd, axe], fraction=0.03, pad=0.02).set_label("T (arb.)", fontsize=8)
    p = os.path.join(OUT, "crack_fig_plate.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("wrote", p)

    # 3) failure threshold, standalone
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    cx._failure(ax, ds)
    ax.set_title("")   # no baked title; the manuscript caption carries it
    p = os.path.join(OUT, "crack_fig_failure.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("wrote", p)

    # 4) cycles-to-failure, standalone
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    cx._cycles(ax, dsm)
    ax.set_title("")   # no baked title; the manuscript caption carries it
    p = os.path.join(OUT, "crack_fig_cycles.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    main()
