"""
House style greyscale figures for the Odinzen engine manuscript.
Pure monochrome, full rectangular box, white canvas, no gridlines,
bold title at top, small note at bottom, hatching instead of color.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from . import thermoelastic_fem as fem

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.1,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})


def box_all_spines(ax):
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(1.1)
        ax.spines[s].set_color("black")
    ax.grid(False)


def coupling_schematic(path):
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_xticks([])
    ax.set_yticks([])
    box_all_spines(ax)

    fig.text(0.5, 0.955, "TWO WAY THERMOMECHANICAL COUPLING LOOP",
             ha="center", va="top", fontsize=13, fontweight="bold")

    boxes = {
        "thermal": (1.0, 4.3, 3.0, 1.4, "STEADY STATE\nCONDUCTION\nK$_T$ T = q", "////"),
        "strain": (6.0, 4.3, 3.0, 1.4, "THERMAL STRAIN\n$\\varepsilon_0$ = $\\alpha\\,$T", "\\\\\\\\"),
        "mech": (6.0, 1.0, 3.0, 1.4, "MECHANICAL\nEQUILIBRIUM\nK$_u$ u = f + f$_{th}$", "xxxx"),
        "update": (1.0, 1.0, 3.0, 1.4, "CONDUCTIVITY\nUPDATE\nk = k(u)", "...."),
    }
    centres = {}
    for key, (x, y, w, h, label, hatch) in boxes.items():
        ax.add_patch(Rectangle((x, y), w, h, facecolor="0.92",
                               edgecolor="black", hatch=hatch, linewidth=1.1))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=9.5, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="black",
                          boxstyle="round,pad=0.35", linewidth=0.8))
        centres[key] = (x + w / 2, y + h / 2)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(centres[a], centres[b],
                     arrowstyle="-|>", mutation_scale=16, lw=1.4,
                     color="black", shrinkA=46, shrinkB=46))

    arrow("thermal", "strain")
    arrow("strain", "mech")
    arrow("mech", "update")
    arrow("update", "thermal")

    fig.text(0.5, 0.045,
             "Odinzen engine | steady state field with deformation dependent "
             "conductivity | fixed point iteration to convergence",
             ha="center", va="bottom", fontsize=8, style="italic")
    fig.subplots_adjust(left=0.04, right=0.96, top=0.9, bottom=0.12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def verification_panels(path):
    mesh = fem.Mesh(8, 4, h=1.0)
    kfield = np.ones(len(mesh.elements()))
    K = fem.assemble_conduction(mesh, kfield)
    coords = mesh.coords()
    L = mesh.nx * mesh.h
    left = np.where(np.isclose(coords[:, 0], 0.0))[0]
    right = np.where(np.isclose(coords[:, 0], L))[0]
    fixed = np.concatenate([left, right])
    vals = np.concatenate([np.full(left.size, 100.0), np.full(right.size, 0.0)])
    T = fem.solve_dirichlet(K, np.zeros(mesh.nnode), fixed, vals)
    mid = mesh.ny // 2
    row = [mesh.nid(ix, mid) for ix in range(mesh.nx + 1)]
    xpos = coords[row, 0] / L
    Tfe = T[row]
    xa = np.linspace(0, 1, 100)
    Ta = 100.0 * (1 - xa)

    p_stress, _ = fem.verify_patch_free()
    sx, _, s_an = fem.verify_clamped()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 4.4))
    fig.text(0.5, 0.955,
             "ODINZEN ENGINE VERIFICATION AGAINST CLOSED FORM SOLUTIONS",
             ha="center", va="top", fontsize=12, fontweight="bold")

    axA.set_title("A | STEADY STATE CONDUCTION", fontsize=10, fontweight="bold")
    axA.plot(xa, Ta, color="black", lw=1.6, label="Analytic")
    axA.plot(xpos, Tfe, linestyle="none", marker="o", markersize=7,
             markerfacecolor="white", markeredgecolor="black",
             markeredgewidth=1.3, label="Finite element")
    axA.set_xlabel("Position, dimensionless", fontsize=9.5)
    axA.set_ylabel("Temperature, K", fontsize=9.5)
    axA.set_xlim(-0.03, 1.03)
    axA.set_ylim(-5, 105)
    axA.legend(frameon=True, edgecolor="black", fontsize=8.5, loc="upper right")
    box_all_spines(axA)

    axB.set_title("B | THERMOELASTIC RESPONSE", fontsize=10, fontweight="bold")
    labels = ["FREE\nEXPANSION", "FULLY\nCLAMPED"]
    mags = [abs(p_stress), abs(sx)]
    axB.bar([0, 1], mags, width=0.55, edgecolor="black",
            facecolor="0.85", linewidth=1.2, hatch="////")
    axB.axhline(abs(s_an), color="black", lw=1.3, linestyle=(0, (5, 4)))
    axB.text(1.0, abs(s_an) + 6, "analytic |$E\\,\\alpha\\,$T / (1 $-$ $\\nu$)|",
             ha="center", va="bottom", fontsize=8.2)
    axB.set_xticks([0, 1])
    axB.set_xticklabels(labels, fontsize=9)
    axB.set_ylabel("Normal stress magnitude, MPa", fontsize=9.5)
    axB.set_ylim(0, 210)
    box_all_spines(axB)

    fig.text(0.5, 0.04,
             "Q4 plane stress, E = 210 GPa, nu = 0.30, alpha = 12 microstrain "
             "per K, T rise = 50 K | free expansion stress below 1e-9 MPa",
             ha="center", va="bottom", fontsize=7.6, style="italic")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.18, wspace=0.32)
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    coupling_schematic("fig1_coupling.png")
    verification_panels("fig2_verification.png")
    print("figures written")
