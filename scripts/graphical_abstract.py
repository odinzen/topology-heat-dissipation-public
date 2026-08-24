"""Graphical abstract for the Odinzen engine manuscript.

Greyscale house style: full rectangular boxes with black spines, white canvas,
no gridlines, bold title at the top, source note at the bottom. Series are
separated by line texture, never by colour. The figure is self contained: it
crops the committed three dimensional render for the design thumbnail and draws
every schematic and summary chart natively, so it regenerates from the
repository without any transient result file.

Layout: a four stage pipeline across the top (contract, two way engine, verified
designs, manufacturable geometry) above two result panels (the material ranking
reversal and the preconditioner scaling), closed by a one line verification band.
Run from the scripts directory; it writes ../figures/graphical_abstract.png.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, Polygon

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "black"})

W, H = 12.0, 7.4
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, fc="0.96", lw=1.3):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="black", lw=lw, zorder=1))


def header(x, y, t, fs=10.0):
    ax.text(x, y, t, ha="center", va="center", fontsize=fs, fontweight="bold", zorder=3)


def body(x, y, t, fs=8.2):
    ax.text(x, y, t, ha="center", va="center", fontsize=fs, zorder=3)


def arrow(x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                 mutation_scale=17, lw=1.6, color="black", zorder=4))


def to_fig(x, y, w, h):
    return [x / W, y / H, w / W, h / H]


fig.text(0.5, 0.965,
         "A VALIDATED, DECOUPLED ENGINE FOR TWO WAY COUPLED THERMOELASTIC TOPOLOGY OPTIMIZATION",
         ha="center", va="top", fontsize=12.5, fontweight="bold")

top_y, top_h = 4.55, 1.95
mid = top_y + top_h / 2

x1, w1 = 0.35, 2.45
box(x1, top_y, w1, top_h)
header(x1 + w1 / 2, top_y + top_h - 0.32, "HARDENED JSON\nCONTRACT", 9.8)
body(x1 + w1 / 2, top_y + 0.72,
     "schema validated\nbridge assembled\ninvalid specs rejected\nkinetic side constraints", 7.8)

x2, w2 = x1 + w1 + 0.32, 2.55
box(x2, top_y, w2, top_h)
header(x2 + w2 / 2, top_y + top_h - 0.30, "TWO WAY ENGINE", 9.8)
ax.text(x2 + w2 / 2, top_y + 1.02,
        r"$T \rightarrow \varepsilon_{\mathrm{th}} \rightarrow u \rightarrow k(u) \rightarrow T$",
        ha="center", va="center", fontsize=10.5, zorder=3)
body(x2 + w2 / 2, top_y + 0.46,
     "coupled adjoint, certified\nHeaviside projection", 7.8)

x3, w3 = x2 + w2 + 0.32, 2.55
box(x3, top_y, w3, top_h)
header(x3 + w3 / 2, top_y + top_h - 0.28, "VERIFIED DESIGNS", 9.8)
try:
    img = plt.imread("../geometry/preview3d.png")
    h0, w0 = img.shape[0], img.shape[1]
    crop = img[int(0.21 * h0):int(0.84 * h0), int(0.21 * w0):int(0.80 * w0)]
    ins = fig.add_axes(to_fig(x3 + 0.30, top_y + 0.46, w3 - 0.60, 1.02))
    ins.imshow(crop, interpolation="bilinear", aspect="auto")
    ins.set_xticks([]); ins.set_yticks([])
    for s in ins.spines.values():
        s.set_edgecolor("black"); s.set_linewidth(0.9)
except FileNotFoundError:
    pass
body(x3 + w3 / 2, top_y + 0.26, "two way | 3D hex8 | MBB", 7.8)

x4, w4 = x3 + w3 + 0.32, 2.55
box(x4, top_y, w4, top_h)
header(x4 + w4 / 2, top_y + top_h - 0.28, "MANUFACTURABLE\nGEOMETRY", 9.8)
cx, cy = x4 + w4 / 2, top_y + 0.92
fw, fh, dp = 1.30, 0.62, 0.42
ax.add_patch(Rectangle((cx - fw / 2, cy - fh / 2), fw, fh, facecolor="0.88", edgecolor="black", lw=1.1, zorder=3))
ax.add_patch(Polygon([(cx - fw / 2, cy + fh / 2), (cx - fw / 2 + dp, cy + fh / 2 + dp * 0.7),
                      (cx + fw / 2 + dp, cy + fh / 2 + dp * 0.7), (cx + fw / 2, cy + fh / 2)],
                     closed=True, facecolor="0.74", edgecolor="black", lw=1.1, zorder=3))
ax.add_patch(Polygon([(cx + fw / 2, cy + fh / 2), (cx + fw / 2 + dp, cy + fh / 2 + dp * 0.7),
                      (cx + fw / 2 + dp, cy - fh / 2 + dp * 0.7), (cx + fw / 2, cy - fh / 2)],
                     closed=True, facecolor="0.62", edgecolor="black", lw=1.1, zorder=3))
ax.add_patch(Circle((cx - 0.28, cy), 0.085, facecolor="white", edgecolor="black", lw=1.0, zorder=4))
ax.add_patch(Circle((cx + 0.28, cy), 0.085, facecolor="white", edgecolor="black", lw=1.0, zorder=4))
body(x4 + w4 / 2, top_y + 0.26, "STL | STEP | glTF, watertight", 7.8)

arrow(x1 + w1 + 0.02, x2 - 0.02, mid)
arrow(x2 + w2 + 0.02, x3 - 0.02, mid)
arrow(x3 + w3 + 0.02, x4 - 0.02, mid)

pan_y, pan_h = 1.30, 2.55
pan_top = pan_y + pan_h

header(3.05, pan_top + 0.20, "MATERIAL RANKING REVERSAL", 9.6)
header(8.95, pan_top + 0.20, "PRECONDITIONER SCALING", 9.6)

axA = fig.add_axes(to_fig(0.95, pan_y, 4.35, pan_h))
xx = [0, 1]
series = {
    "AlSi10Mg":    ([2.70, 0.72], "-",  1.7),
    "CuCrZr":      ([1.50, 0.38], "-",  1.9),
    "Invar 36":    ([1.35, 0.36], "-",  1.5),
    "Inconel 718": ([0.97, 1.25], (0, (5, 4)), 1.4),
    "Ti6Al4V":     ([1.65, 1.07], (0, (1, 1.4)), 1.4),
    "316L":        ([1.00, 1.00], (0, (3, 3)), 1.2),
}
label_y = {"AlSi10Mg": 0.72, "CuCrZr": 0.47, "Invar 36": 0.23,
           "Inconel 718": 1.30, "Ti6Al4V": 1.07, "316L": 0.90}
for name, (vals, st, lw) in series.items():
    col = "black" if st == "-" else "0.5"
    axA.plot(xx, vals, linestyle=st, color=col, lw=lw, marker="o", markersize=5,
             markerfacecolor=("black" if st == "-" else "white"),
             markeredgecolor="black", markeredgewidth=1.0)
    axA.text(1.05, label_y[name], name, fontsize=7.0, va="center", ha="left",
             fontweight=("bold" if st == "-" else "normal"))
axA.axhline(1.0, color="0.8", lw=0.8, zorder=0)
axA.set_xlim(-0.12, 1.78); axA.set_ylim(0, 3.0)
axA.set_xticks([0, 1]); axA.set_xticklabels(["LOW HEAT", "HIGH HEAT"], fontsize=8)
axA.set_ylabel("Relative compliance, 316L = 1", fontsize=8.5)
axA.tick_params(labelsize=7.5)
axA.text(0.5, 2.74, "solid improves under heat, governed by E alpha / k",
         ha="center", fontsize=7.2, style="italic")
for s in axA.spines.values():
    s.set_edgecolor("black"); s.set_linewidth(1.1)
axA.grid(False)

axB = fig.add_axes(to_fig(6.80, pan_y, 4.35, pan_h))
dof = np.array([6.8e3, 1.3e4, 2.8e4, 7.4e4])
cg = np.array([11, 11, 11, 11])
gmres = np.array([9, 12, 14, 16])
axB.plot(dof, cg, linestyle="-", color="black", lw=1.9, marker="o", markersize=7,
         markerfacecolor="black", label="AMG preconditioned CG")
axB.plot(dof, gmres, linestyle=(0, (5, 4)), color="0.45", lw=1.5, marker="s", markersize=7,
         markerfacecolor="white", markeredgecolor="black", label="field split GMRES")
axB.set_xscale("log")
axB.set_xlim(5e3, 1.1e5); axB.set_ylim(0, 42)
axB.set_xlabel("Degrees of freedom, count", fontsize=8.5)
axB.set_ylabel("Solver iterations, count", fontsize=8.5)
axB.tick_params(labelsize=7.5)
axB.annotate("count near eleven", xy=(2.8e4, 11), xytext=(1.05e4, 25),
             fontsize=7.2, arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black"))
axB.legend(frameon=True, edgecolor="black", fontsize=7.4, loc="upper right")
for s in axB.spines.values():
    s.set_edgecolor("black"); s.set_linewidth(1.1)
axB.grid(False)

box(0.35, 0.42, 11.30, 0.62, fc="1.0")
ax.text(6.0, 0.73,
        "adjoint gradients 5.6e-8 to 4.9e-5   |   closed form FE exact, minus 180 and minus 315 MPa   |   "
        "MBB reproduced   |   monolithic solve at strong feedback   |   twenty checks pass",
        ha="center", va="center", fontsize=8.2)

fig.text(0.5, 0.018, "Greyscale, fully reproducible from a single command | scripts/reproduce_all.py",
         ha="center", va="bottom", fontsize=8, style="italic")

fig.savefig("../figures/graphical_abstract.png", dpi=200, bbox_inches="tight")
print("wrote ../figures/graphical_abstract.png")
