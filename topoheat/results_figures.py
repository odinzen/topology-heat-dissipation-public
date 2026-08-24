"""
House style greyscale results figures for the thermoelastic optimizer.
Monochrome, full box, white canvas, bold top title, small bottom note.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .topopt_thermoelastic import ThermoelasticTO

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": "black",
    "axes.linewidth": 1.1, "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
})


def box(ax):
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(1.1); ax.spines[s].set_color("black")
    ax.grid(False)


def solid(rp, nelx, nely):
    return (1.0 - rp).reshape(nely, nelx)


def fig_topology_and_field(path):
    rp = np.load("main_rho.npy")
    to = ThermoelasticTO(60, 30, volfrac=0.45, rmin=2.4, heat=0.005, mech_load=3.0, alpha=0.02)
    st = to.solve(rp)
    T = st["T"].reshape(31, 61)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 3.6))
    fig.text(0.5, 0.96, "OPTIMIZED DESIGN AND ITS COUPLED TEMPERATURE FIELD",
             ha="center", va="top", fontsize=12, fontweight="bold")

    axA.set_title("A | OPTIMIZED TOPOLOGY", fontsize=10, fontweight="bold")
    axA.imshow(solid(rp, 60, 30), cmap="gray", vmin=0, vmax=1, origin="lower",
               interpolation="nearest", aspect="equal")
    axA.set_xticks([]); axA.set_yticks([]); box(axA)

    axB.set_title("B | TEMPERATURE ON THE DESIGN", fontsize=10, fontweight="bold")
    im = axB.imshow(T, cmap="gray_r", origin="lower", interpolation="bilinear",
                    aspect="equal")
    axB.set_xticks([]); axB.set_yticks([]); box(axB)
    cb = fig.colorbar(im, ax=axB, fraction=0.025, pad=0.02)
    cb.set_label("Temperature, dimensionless", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.text(0.5, 0.04,
             "Sink and clamp at the left edge, point load at the right | the "
             "temperature field is solved on the design dependent conductivity",
             ha="center", va="bottom", fontsize=8, style="italic")
    fig.subplots_adjust(left=0.02, right=0.97, top=0.84, bottom=0.15, wspace=0.06)
    fig.savefig(path, dpi=200); plt.close(fig)


def fig_history(path):
    hist = np.load("main_hist.npy")
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fig.text(0.5, 0.95, "COMPLIANCE CONVERGENCE HISTORY",
             ha="center", va="top", fontsize=13, fontweight="bold")
    ax.semilogy(np.arange(1, len(hist) + 1), hist, color="black", lw=1.7)
    ax.set_xlabel("Iteration, count", fontsize=10)
    ax.set_ylabel("Structural strain energy, dimensionless", fontsize=10)
    ax.set_xlim(0, len(hist)); box(ax)
    fig.text(0.5, 0.04,
             "Optimality criteria update with density filter | monotone descent "
             "to a fivefold reduction at fixed volume",
             ha="center", va="bottom", fontsize=8, style="italic")
    fig.subplots_adjust(left=0.13, right=0.96, top=0.88, bottom=0.16)
    fig.savefig(path, dpi=200); plt.close(fig)


def fig_mesh(path):
    panels = [("30 BY 15", np.load("mesh_30x15.npy"), 30, 15),
              ("45 BY 22", np.load("mesh_45x22.npy"), 45, 22),
              ("60 BY 30", np.load("main_rho.npy"), 60, 30)]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.0))
    fig.text(0.5, 0.97, "MESH INDEPENDENCE UNDER A FIXED PHYSICAL FILTER RADIUS",
             ha="center", va="top", fontsize=12, fontweight="bold")
    for ax, (title, rp, nx, ny) in zip(axes, panels):
        ax.imshow(solid(rp, nx, ny), cmap="gray", vmin=0, vmax=1, origin="lower",
                  interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax