import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""TASK 4: house-style greyscale two-panel figure (topology + material field)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shutil, os

rp = np.load("mm_rho.npy")
mp = np.load("mm_mu.npy")
nelx, nely = (np.load("mm_shape.npy")).astype(int)

rho2 = rp.reshape(nely, nelx)
mu2 = mp.reshape(nely, nelx)
solid = rho2 > 0.5

# Material field shown only where material is present (masked elsewhere).
mu_masked = np.ma.masked_where(~solid, mu2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.4))

# ---- Panel 1 : topology (solid black, void white -> show 1-rho) ----
ax1.imshow(1.0 - rho2, cmap="gray", vmin=0.0, vmax=1.0,
           origin="lower", interpolation="nearest", aspect="equal")
ax1.set_title("DENSITY / TOPOLOGY", fontsize=10, fontweight="bold", pad=6)

# ---- Panel 2 : material field mu (A dark, B light), masked by rho>0.5 ----
cmap2 = plt.cm.gray.copy()
cmap2.set_bad(color="white")                       # void shown white
ax2.imshow(mu_masked, cmap=cmap2, vmin=0.0, vmax=1.0,
           origin="lower", interpolation="nearest", aspect="equal")
ax2.set_title("MATERIAL FIELD  (A DARK  -  B LIGHT)",
              fontsize=10, fontweight="bold", pad=6)

# House style: full rectangular box, four thin black spines, no ticks.
for ax in (ax1, ax2):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(0.8)

fig.suptitle("TWO-MATERIAL THERMOELASTIC TOPOLOGY OPTIMIZATION",
             fontsize=13, fontweight="bold", y=0.99)
fig.text(0.5, 0.015,
         "Stiff low-expansion material A occupies the load paths; "
         "compliant high-conductivity material B appears in low-stress / thermal regions.",
         ha="center", va="bottom", fontsize=8, style="italic")

fig.patch.set_facecolor("white")
plt.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.14, wspace=0.06)

tmp = "/tmp/fig16.png"
fig.savefig(tmp, dpi=150, facecolor="white")
plt.close(fig)

dst = "/sessions/sweet-focused-ritchie/mnt/outputs/fig16_multimaterial.png"
shutil.copy(tmp, dst)
sz = os.path.getsize(dst)
assert sz > 0, "figure is empty!"
print("saved", dst, "bytes:", sz)
