"""Turn the branch-specific critical depths into cycles-to-failure via the Paris law.

A surface crack grows through the plate thickness by da/dN = C (dK)^m,
dK = Y * dsigma * sqrt(pi * a), driven by the thermal-cycling stress dsigma = E*alpha*dT.
A branch fails when its crack depth reaches its critical depth (from the multi-branch
sweep). Integrating the Paris law from the initial flaw to each critical depth gives
cycles-to-failure per branch.

Key point: because fatigue growth accelerates (da/dN ~ a^(m/2)), most of the life is spent
growing the shallow initial flaw, so the cycles-to-failure are compressed relative to the
wide spread in critical depth. Constants are SCHEMATIC / lit-range (Paris C, m, initial
flaw, plate thickness); the ordering and the compression are the result.

Needs numpy, scipy and matplotlib.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import crack_tree as ct
import fatigue as fat

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
OUT = os.path.join(HERE, "figures")
COLORS = ["#1f4e79", "#c00000", "#2e7d32", "#e08b00", "#6a3d9a", "#00838f"]

THICKNESS_MM = 8.0                 # cold-plate base thickness (SCHEMATIC)
DT = 60.0                          # junction thermal swing (K)
DSIGMA = fat._v("E") * fat._v("alpha") * DT    # MPa, clamped alpha*dT bound
CYCLES_PER_DAY = 6.0               # illustrative daily load cycles


def _dadN(a_m):
    return ct.PARIS_C * (ct.Y_GEOM * DSIGMA * np.sqrt(np.pi * a_m)) ** ct.PARIS_M


def cycles_to(a_crit_m, n=4000):
    a = np.linspace(ct.A0_M, a_crit_m, n)
    y = 1.0 / _dadN(a)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(a)))


def depth_vs_cycles(a_max_m, n=4000):
    a = np.linspace(ct.A0_M, a_max_m, n)
    N = np.concatenate([[0.0], np.cumsum(np.diff(a) / (0.5 * (_dadN(a[1:]) + _dadN(a[:-1]))))])
    return a, N


def main():
    d = np.load(os.path.join(CACHE, "depth_sweep_multi.npz"), allow_pickle=True)
    labels = list(d["labels"]); dcrit = d["dcrit"]
    order = np.argsort(dcrit)

    Nf = {}
    print(f"driver dsigma = {DSIGMA:.0f} MPa (E*alpha*dT, dT={DT:.0f} K); thickness "
          f"{THICKNESS_MM:.0f} mm; Paris C={ct.PARIS_C:g}, m={ct.PARIS_M:g} (SCHEMATIC)\n")
    for k in order:
        a_c = dcrit[k] * THICKNESS_MM * 1e-3
        nf = cycles_to(a_c)
        Nf[k] = nf
        yr = nf / (CYCLES_PER_DAY * 365.25)
        print(f"{labels[k]:>18}: crit depth {dcrit[k]*100:4.0f}%  a_crit {dcrit[k]*THICKNESS_MM:4.2f} mm"
              f"  Nf {nf:8.3e} cyc  (~{yr:6.0f} yr @ {CYCLES_PER_DAY:.0f}/day)")
    nfa = np.array([Nf[k] for k in order])
    print(f"\ncritical depth spans {dcrit[order].min()*100:.0f}-{dcrit[order].max()*100:.0f}% "
          f"but cycles-to-failure spans only {nfa.min():.2e}-{nfa.max():.2e} "
          f"({100*(nfa.max()/nfa.min()-1):.0f}% spread): acceleration compresses the life.")

    # figure: universal depth-vs-cycles curve + each branch's threshold -> its Nf
    a_curve, N_curve = depth_vs_cycles(THICKNESS_MM * 1e-3)
    depth_pct = (a_curve / (THICKNESS_MM * 1e-3)) * 100.0

    fig, (axc, axb) = plt.subplots(1, 2, figsize=(11.8, 4.4),
                                   gridspec_kw=dict(width_ratios=[1.5, 1]))
    axc.plot(N_curve, depth_pct, color="0.15", lw=2.2, zorder=3)
    for k in order:
        col = COLORS[k % len(COLORS)]
        yc = dcrit[k] * 100.0
        axc.axhline(yc, color=col, ls=":", lw=1.0, alpha=0.7)
        axc.plot([Nf[k]], [yc], "o", color=col, ms=7, zorder=5)
        axc.annotate(labels[k], xy=(Nf[k], yc), xytext=(Nf[k] * 1.03, yc + 1.5),
                     fontsize=7.5, color=col)
    axc.set_xlabel("Cycles"); axc.set_ylabel("Crack depth (% of thickness)")
    axc.set_xlim(0, N_curve.max() * 1.02); axc.set_ylim(0, 100)
    axc.set_title("(a) Paris Growth Maps Each Threshold to a Life", fontsize=10.5)
    axc.grid(True, color="0.92", lw=0.6)

    rows = list(order)
    for row, k in enumerate(rows):
        col = COLORS[k % len(COLORS)]
        axb.barh(row, Nf[k], color=col, edgecolor="0.2", height=0.6)
        axb.text(Nf[k] * 1.01, row, "%.2e" % Nf[k], va="center", fontsize=8, color="0.1")
        axb.text(-0.02 * nfa.max(), row, labels[k], va="center", ha="right",
                 fontsize=8.5, color=col)
    axb.set_yticks([]); axb.set_ylim(-0.6, len(rows) - 0.4)
    axb.set_xlim(0, nfa.max() * 1.25); axb.set_xlabel("Cycles to failure")
    axb.set_title("(b) Similar Lives Despite the Depth Spread", fontsize=10.5)
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)

    fig.suptitle("From Critical Depth to Cycles-to-Failure (Paris law; constants SCHEMATIC)",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "crack_cycles.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
