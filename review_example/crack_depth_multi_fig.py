"""Failure threshold across several branches: each branch's served-region effectiveness
vs crack depth, with its critical depth. Shows the spread across the plate hierarchy.
Reads .cache/depth_sweep_multi.npz. Needs numpy, scipy and matplotlib.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
OUT = os.path.join(HERE, "figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300})
COLORS = ["#1f4e79", "#c00000", "#2e7d32", "#e08b00", "#6a3d9a", "#00838f"]


def main():
    d = np.load(os.path.join(CACHE, "depth_sweep_multi.npz"), allow_pickle=True)
    depths = d["depths"] * 100.0
    labels = list(d["labels"]); curves = d["curves"]
    dcrit = d["dcrit"]; sev = d["severity"]

    fig, (axc, axb) = plt.subplots(1, 2, figsize=(11.8, 4.4),
                                   gridspec_kw=dict(width_ratios=[1.5, 1]))

    ymin = max(0, np.nanmin(curves) - 6)
    for k, lab in enumerate(labels):
        col = COLORS[k % len(COLORS)]
        axc.plot(depths, curves[k], "-o", color=col, lw=1.8, ms=3, label=lab)
        if np.isfinite(dcrit[k]):
            xc = dcrit[k] * 100.0
            yc = np.interp(xc, depths, curves[k])
            axc.plot([xc], [yc], "v", color=col, ms=8, zorder=6)
    axc.set_xlim(0, 100); axc.set_ylim(ymin, 103)
    axc.set_xlabel("Crack depth (% of plate thickness)")
    axc.set_ylabel("Served-region effectiveness (%)")
    axc.set_title("(a) Each Branch Has Its Own Failure Threshold", fontsize=10.5)
    axc.grid(True, color="0.9", lw=0.6)
    axc.legend(fontsize=8, loc="lower left", frameon=False)

    # (b) critical depth per branch (bars); redundant branches flagged
    order = np.argsort([dc if np.isfinite(dc) else 9 for dc in dcrit])
    y = np.arange(len(order))
    for row, k in enumerate(order):
        col = COLORS[k % len(COLORS)]
        if np.isfinite(dcrit[k]):
            axb.barh(row, dcrit[k] * 100.0, color=col, edgecolor="0.2", height=0.6)
            axb.text(dcrit[k] * 100.0 + 1.5, row, "%.0f%%" % (dcrit[k] * 100.0),
                     va="center", fontsize=8.5, color="0.1")
        else:
            axb.text(2, row, "redundant (no threshold)", va="center", fontsize=8.5,
                     color="0.4", style="italic")
        axb.text(-2, row, labels[k], va="center", ha="right", fontsize=8.5, color=col)
    axb.set_yticks([]); axb.set_ylim(-0.6, len(order) - 0.4)
    axb.set_xlim(0, 100); axb.set_xlabel("Critical depth (% of thickness)")
    axb.set_title("(b) The Spread in Critical Depth", fontsize=10.5)
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)

    finite = [dc * 100 for dc in dcrit if np.isfinite(dc)]
    span = ("critical depth spans %.0f–%.0f%% of thickness" % (min(finite), max(finite))
            if finite else "")
    fig.suptitle("Depth to Failure Across Branches: " + span, fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "crack_depth_branches.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
