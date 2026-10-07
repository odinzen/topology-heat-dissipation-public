"""Figure 11 reworked: relative mechanical-fatigue ranking of the limbs (no absolute life).

Each limb's critical depth is converted to a mechanical-fatigue life via Paris growth, then
normalized to the earliest-failing limb. The point is the ranking and the compression: critical
depths spanning 6-47% of the wall collapse to a ~17% spread in relative life, led by the outer
twig. No absolute service life is claimed (constants schematic, corrosion excluded).
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache"); OUT = os.path.join(HERE, "figures")
PARIS_C, PARIS_M, Y, A0 = 1.0e-11, 3.0, 1.12, 2.0e-5
E, ALPHA, DT, TH = 117000.0, 16.6e-6, 60.0, 8.0
DSIGMA = E * ALPHA * DT
COLORS = ["#6a3d9a", "#1f4e79", "#c00000", "#2e7d32", "#e08b00"]

d = np.load(os.path.join(CACHE, "depth_sweep_multi.npz"), allow_pickle=True)
labels = list(d["labels"]); dcrit = np.array(d["dcrit"])

def cyc(ac, n=4000):
    a = np.linspace(A0, ac, n)
    dadN = PARIS_C * (Y * DSIGMA * np.sqrt(np.pi * a)) ** PARIS_M
    return float(np.sum(0.5 * (1 / dadN[1:] + 1 / dadN[:-1]) * np.diff(a)))

Nf = np.array([cyc(dc * TH * 1e-3) for dc in dcrit])
order = np.argsort(Nf)
rel = Nf / Nf.min()

fig, ax = plt.subplots(figsize=(7.0, 3.6))
for row, k in enumerate(order):
    col = COLORS[k % len(COLORS)]
    ax.hlines(row, 1.0, rel[k], color=col, lw=2, alpha=0.6)
    ax.plot(rel[k], row, "o", color=col, ms=10, zorder=5)
    ax.text(rel[k] + 0.006, row, f"{labels[k]}  (crit {dcrit[k]*100:.0f}%)",
            va="center", fontsize=9, color=col)
ax.axvline(1.0, color="0.75", lw=1.0, ls="--")
ax.set_yticks([]); ax.set_ylim(-0.6, len(order) - 0.4)
ax.set_xlim(0.98, 1.30)
ax.set_xlabel("Relative mechanical-fatigue life  (× earliest-failing limb)")
ax.set_title("Fatigue ranking: wide depth spread, narrow life spread", fontsize=10.5)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
for ext, kw in ((".png", {}), (".tif", {"pil_kwargs": {"compression": "tiff_lzw"}})):
    fig.savefig(os.path.join(OUT, "crack_rank" + ext), dpi=300 if ext == ".tif" else 200,
                bbox_inches="tight", **kw)
plt.close(fig)
print("rel lives:", [(labels[k], round(float(rel[k]), 3), int(dcrit[k]*100)) for k in order])
print("wrote", os.path.join(OUT, "crack_rank.png/.tif"))
