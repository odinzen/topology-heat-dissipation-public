"""Figure: surface chunk vs full-width slot on a primary limb (reworked).

(a) geometry schematic: a shallow surface chunk (a real fatigue flaw) versus the
    full-width through-slot used in Figs 10-11 (the worst case). (b) critical crack depth
    versus chunk width: below about half the limb width a chunk never fails the limb at any
    depth, so the full-width slot is the conservative envelope. (c,d) die-face heating at
    the same 45% depth for the quarter-width chunk (benign) and the full-width slot (a
    stranded hot region), zoomed on the limb.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, FancyArrow

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache"); OUT = os.path.join(HERE, "figures")

d = np.load(os.path.join(CACHE, "crack_chunk.npz"), allow_pickle=True)
widths = list(d["widths"]); depths = d["depths"]; rise = d["rise"]
full_sev = float(rise[-1, -1]); half = full_sev / 2
mp = np.load(os.path.join(CACHE, "crack_chunk_maps.npz"), allow_pickle=True)
Tp = mp["Tp"]; seed = mp["seed"]; chunk_q = mp["chunk_q"]; chunk_f = mp["chunk_f"]
dT025 = d["map025"] - Tp; dT100 = d["map100"] - Tp
map_depth = float(d["map_depth"])

# combine with extra widths for the critical-depth curve
allw = list(widths); allrise = [rise[i] for i in range(len(widths))]
ep = os.path.join(CACHE, "crack_chunk_extra.npz")
if os.path.exists(ep):
    e = np.load(ep, allow_pickle=True)
    for i, w in enumerate(list(e["widths"])):
        allw.append(float(w)); allrise.append(e["rise"][i])
order = np.argsort(allw); allw = np.array(allw)[order]; allrise = [allrise[k] for k in order]

def crit_depth(r):
    r = np.asarray(r)
    return float(np.interp(half, r, depths)) * 100 if np.nanmax(r) >= half else None
crit = [crit_depth(r) for r in allrise]

fig = plt.figure(figsize=(11.8, 5.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1.02, 1, 1], height_ratios=[1, 1],
                      hspace=0.42, wspace=0.30)

# ---- (a) geometry schematic: two limb cross-sections ----
axa = fig.add_subplot(gs[:, 0]); axa.set_xlim(0, 10); axa.set_ylim(0, 10)
axa.axis("off"); axa.set_title("(a) Defect geometry", fontsize=10.5)
GREY = "#c7ccd1"

def limb(y0, kind, label):
    H, x0, x1 = 2.2, 1.6, 8.4
    rect = Rectangle((x0, y0), x1 - x0, H, facecolor=GREY, edgecolor="0.25", lw=1.4)
    axa.add_patch(rect)
    cx = (x0 + x1) / 2
    if kind == "chunk":
        cw, cdep = 1.8, 1.1      # narrow, shallow scoop bitten into the top surface
        sc = Ellipse((cx, y0 + H), cw, cdep, facecolor="white", edgecolor="#c00000", lw=1.6)
        axa.add_patch(sc); sc.set_clip_path(rect)
    else:
        sd = 1.35                # full-width cut, leaves a thin ligament
        axa.add_patch(Rectangle((x0 + 0.12, y0 + H - sd), x1 - x0 - 0.24, sd,
                      facecolor="white", edgecolor="#c00000", lw=1.6))
        axa.text(cx, y0 + 0.36, "thin ligament", ha="center", fontsize=7.5, color="0.3")
    axa.text(cx, y0 + H + 0.42, label, ha="center", fontsize=8.6, color="#c00000")
    return H, x0, x1

axa.text(5.0, 9.55, "grey = conductor,  white = cracked away", ha="center",
         fontsize=7.8, color="0.45")
limb(5.9, "chunk", "shallow surface chunk")
_, x0, x1 = limb(1.5, "slot", "full-width slot")
# dimension labels on the lower cross-section
axa.annotate("", xy=(1.15, 1.5), xytext=(1.15, 3.7), arrowprops=dict(arrowstyle="<->", color="0.45"))
axa.text(0.55, 2.6, "wall\nthickness", rotation=90, va="center", ha="center", fontsize=7.8, color="0.35")
axa.annotate("", xy=(1.6, 0.95), xytext=(8.4, 0.95), arrowprops=dict(arrowstyle="<->", color="0.45"))
axa.text(5.0, 0.5, "limb width", ha="center", fontsize=7.8, color="0.35")

# ---- (b) critical depth vs chunk width ----
axb = fig.add_subplot(gs[0, 1:])
wpct = allw * 100
solid = [(w, c) for w, c in zip(wpct, crit) if c is not None]
never = [w for w, c in zip(wpct, crit) if c is None]
wthr = (max(never) + min(w for w, c in solid)) / 2 if never and solid else 40
axb.axvspan(0, wthr, color="#e5f2e5", zorder=0)
axb.text(wthr / 2 + 3, 62, "chunk never fails\n(any depth)", ha="center", fontsize=8.3, color="#2e7d32")
if solid:
    xs, ys = zip(*solid)
    axb.plot(xs, ys, "-o", color="#c00000", ms=6, lw=2)
axb.axhline(0, color="0.7", lw=0.6)
axb.annotate("full-width slot\n(Fig 10)", xy=(100, solid[-1][1]), xytext=(80, 46),
             fontsize=8, color="0.3", arrowprops=dict(arrowstyle="->", color="0.5"))
axb.set_xlim(15, 105); axb.set_ylim(0, 75)
axb.set_xlabel("Chunk width (% of limb width)")
axb.set_ylabel("Critical crack depth\n(% of wall)")
axb.grid(True, color="0.92", lw=0.6)
axb.set_title("(b) A chunk must span the limb before depth matters", fontsize=10)

# ---- (c,d) zoomed heating maps ----
iy0, iy1, ix0, ix1 = 28, 56, 12, 52
def zoom(M): return M[iy0:iy1, ix0:ix1]
vmax = max(zoom(dT100).max(), 1e-6)
def draw(ax, dTM, cells, title):
    im = ax.imshow(zoom(dTM), origin="lower", cmap="inferno", vmin=0, vmax=vmax,
                   extent=[ix0, ix1, iy0, iy1], aspect="equal")
    ax.plot([cells[:, 1].min(), cells[:, 1].max()], [cells[0, 0], cells[0, 0]],
            "-", color="#39ff14", lw=2.4, solid_capstyle="round")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9.5)
    return im
axc = fig.add_subplot(gs[1, 1]); draw(axc, dT025, chunk_q, f"(c) Quarter-width chunk (depth {map_depth*100:.0f}%)")
axd = fig.add_subplot(gs[1, 2]); im = draw(axd, dT100, chunk_f, f"(d) Full-width slot (depth {map_depth*100:.0f}%)")
cb = fig.colorbar(im, ax=axd, fraction=0.046, pad=0.04); cb.set_label("Temperature rise (arb.)", fontsize=8)

p = os.path.join(OUT, "crack_chunk.png")
fig.savefig(p, dpi=200, bbox_inches="tight")
pt = os.path.join(OUT, "crack_chunk.tif")
fig.savefig(pt, dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("wrote", p, "and", pt)
print("crit(width):", [(round(w), None if c is None else round(c)) for w, c in zip(wpct, crit)])
