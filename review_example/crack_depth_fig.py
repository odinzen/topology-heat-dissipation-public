"""Depth-to-failure figure: branch effectiveness vs crack depth, with the critical depth
at which a partial crack becomes a thermal failure (well before full severance), and a
cross-section schematic of the thinning ligament. Reads .cache/depth_sweep.npz.

Blue-to-red heat coloring in the cross-section; house-style otherwise. Needs numpy, scipy and matplotlib.
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
CMAP = "turbo"
GREEN, RED = "#2ca02c", "#d40000"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 300})


def _curve(ax, d):
    x = d["depths"] * 100.0
    eff = d["eff"]
    dc = float(d["d_crit"]) * 100.0
    ax.axvspan(0, dc, color=GREEN, alpha=0.10)
    ax.axvspan(dc, 100, color=RED, alpha=0.10)
    ax.plot(x, eff, "-o", color="0.1", lw=2.0, ms=4, zorder=5)
    ax.axvline(dc, color=RED, ls="--", lw=1.4)
    ax.text(dc + 1.5, 92, "critical depth ≈ %.0f%%\nbranch reroutes / fails" % dc,
            color=RED, fontsize=9, va="top")
    ax.text(4, 70, "shallow cracks:\nligament still conducts\n(branch effective)",
            color=GREEN, fontsize=9, va="center")
    ax.annotate("full severance (100%) is rare;\na partial crack past ~1/3 depth\nalready"
                " fails, and is far more likely",
                xy=(100, eff[-1]), xytext=(55, 82), fontsize=9, color="0.15",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="0.3", lw=1.1))
    ax.set_xlim(0, 100); ax.set_ylim(min(50, eff.min() - 5), 103)
    ax.set_xlabel("Crack depth (% of plate thickness)")
    ax.set_ylabel("Branch effectiveness (%)")
    ax.set_title("(a) A Partial Crack Is Enough: The Failure Threshold", fontsize=10.5)
    ax.grid(True, color="0.9", lw=0.6)


def _section(ax, dc):
    """Three plate cross-sections: healthy, at threshold, failed. Top = die (hot)."""
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 10); ax.axis("off")
    x0, W, th = 2.4, 6.6, 1.4
    ny, nx = 40, 120
    v = np.linspace(0, 1, ny)[:, None] * np.ones((1, nx))
    xx = np.linspace(-1, 1, nx)[None, :] * np.ones((ny, 1))
    rows = [(0.20, GREEN, "20% deep\nbranch OK"),
            (dc / 100.0, "#e08b00", "~%.0f%% deep\nthreshold" % dc),
            (0.60, RED, "60% deep\nfailed")]
    for k, (c, col, lab) in enumerate(rows):
        yc = 7.7 - k * 3.0
        hot = c if c > dc / 100.0 - 1e-6 else 0.15    # failed rows run hot above the crack
        tip = 1.0 - c
        pocket = np.exp(-(xx ** 2) / 0.02) * np.exp(-((v - tip) ** 2) / 0.010)
        field = np.clip(0.18 + 0.82 * hot * v ** 1.2 + 0.6 * hot * pocket, 0, 1)
        ax.imshow(field, cmap=CMAP, extent=[x0, x0 + W, yc - th / 2, yc + th / 2],
                  origin="lower", vmin=0, vmax=1, aspect="auto", zorder=2)
        ax.add_patch(plt.Rectangle((x0, yc - th / 2), W, th, fill=False,
                                   ec=col, lw=2.2, zorder=5))
        cw, dn = 0.5, c * th
        ax.add_patch(plt.Rectangle((x0 + W / 2 - cw / 2, yc + th / 2 - dn), cw, dn,
                                   facecolor="white", ec=RED, lw=1.6, zorder=6))
        ax.text(x0 - 0.35, yc, lab, ha="right", va="center", fontsize=8.5, color=col)
    ax.text(x0 + W / 2, 7.7 + th / 2 + 0.35, "die side (hot)", ha="center",
            fontsize=8, color="0.2")
    ax.text(x0 + W / 2, 1.7 - th / 2 - 0.45, "coolant side (sink)", ha="center",
            fontsize=8, color="0.2")
    ax.set_title("(b) The Thinning Ligament", fontsize=10.5)


def main():
    d = np.load(os.path.join(CACHE, "depth_sweep.npz"))
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
    _curve(axs[0], d)
    _section(axs[1], float(d["d_crit"]) * 100.0)
    fig.suptitle("When Does a Cracked Branch Stop Working? Depth to Thermal Failure",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "crack_depth_failure.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
