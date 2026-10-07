"""Surface-chunk (partial-width) crack vs the full-width through-slot on one limb.

The multi-branch sweep cuts a full-width crack across a limb and grows it in depth: the
worst case, because at any depth it blocks the whole limb. A real fatigue flaw is a
surface chunk that is limited in width as well as depth and grows outward. Here we take a
single primary limb and remove a centered surface patch of a given width fraction and
depth, re-solving conduction, so the full-width slot (width fraction 1.0) is recovered as
the conservative envelope and narrower chunks show how much margin the idealization hides.

Constants SCHEMATIC. Needs numpy, scipy and matplotlib.
"""
from __future__ import annotations

import os
import numpy as np

import crack_plate3d as cp3
import crack_tree as ct
import crack_depth_sweep_multi as ms   # reuse plate grid, sector mask, served-peak

CACHE = cp3.CACHE
NX, NY, NZ = ms.NX, ms.NY, ms.NZ

LIMB = (0.50, 0.75, "top primary")                 # representative primary limb
WIDTHS = [0.25, 0.50, 0.75, 1.00]                  # chunk width as fraction of limb width
DEPTHS = np.array([0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.0])
MAP_DEPTH = 0.45                                   # depth for the chunk-vs-slot maps


def ordered_chunk_cells(foot, gx, gy, fx, fy):
    """Crack cells across the limb, ordered along the crack line, plus the seed."""
    seed = ct.snap_solid(foot.astype(float), fx, fy, NX, NY, r=10)
    _, normal = ct.limb_axis(gx, gy, seed[0], seed[1], shape=(NY, NX))
    width, _ = ct.limb_width(foot.astype(float), seed, normal)
    cells, _ = ct.crack_line(foot.astype(float), seed, normal, max(3.0, width), mode="center")
    n = np.array(normal, float); n /= (np.hypot(*n) + 1e-9)
    # order cells by signed projection onto the crack direction (across the width)
    cl = sorted(cells, key=lambda ji: (ji[0] - seed[0]) * n[0] + (ji[1] - seed[1]) * n[1])
    return seed, cl, width


def central(cells, wfrac):
    """Contiguous central wfrac of an ordered cell list (the chunk)."""
    if wfrac >= 1.0:
        return list(cells)
    k = max(1, int(round(wfrac * len(cells))))
    lo = (len(cells) - k) // 2
    return list(cells[lo:lo + k])


def main():
    occ = cp3.voxelize(nx=NX, ny=NY, nz=NZ)
    plate = cp3.Plate(NX, NY, NZ, volfrac=0.3, rmin=1.5)
    plate.plate_bc()
    foot = occ.any(2)
    footT = foot.T
    gx, gy = ct._grad(foot.astype(float))

    fx, fy, lab = LIMB
    seed, cells, width = ordered_chunk_cells(foot, gx, gy, fx, fy)
    mask = ms._sector_mask(seed, footT)
    m0 = ms._served_peak(plate, cp3.rp_from_occ(plate, occ), mask)
    print(f"limb '{lab}': width {width:.1f} cells, served pristine peak {m0:.4f}", flush=True)

    rise = np.full((len(WIDTHS), len(DEPTHS)), np.nan)   # % served-peak rise vs pristine
    maps = {}
    total = len(WIDTHS) * len(DEPTHS); done = 0
    for wi, wf in enumerate(WIDTHS):
        chunk = central(cells, wf)
        for di, c in enumerate(DEPTHS):
            d = int(round(c * NZ))
            occ_c = occ.copy()
            if d > 0:
                for (j, i) in chunk:
                    occ_c[i, j, NZ - d:] = False
            rp = cp3.rp_from_occ(plate, occ_c)
            served = ms._served_peak(plate, rp, mask)
            rise[wi, di] = 100.0 * (served / m0 - 1.0)
            # stash top-face field for the chunk-vs-slot maps at MAP_DEPTH
            if abs(c - MAP_DEPTH) < 1e-9 and wf in (0.25, 1.00):
                T = plate.solve(rp)["T"].reshape(NZ + 1, NY + 1, NX + 1)[NZ]
                maps[wf] = T[:NY, :NX].copy()
            done += 1
            print(f"  w={wf:.2f} d={c:.2f} -> served rise {rise[wi,di]:6.1f}%  ({done}/{total})", flush=True)
        # incremental save
        np.savez(os.path.join(CACHE, "crack_chunk.npz"),
                 widths=np.array(WIDTHS), depths=DEPTHS, rise=rise,
                 m0=m0, width_cells=width, seed=np.array(seed),
                 mask=mask, footT=footT,
                 map025=maps.get(0.25, np.zeros((NY, NX))),
                 map100=maps.get(1.00, np.zeros((NY, NX))),
                 map_depth=MAP_DEPTH)

    # critical depth per width: depth at which rise reaches half the full-slot severance rise
    full_sev = rise[-1, -1]                     # width 1.0, depth 1.0
    print(f"\nfull-slot severance rise {full_sev:.0f}%; half = {full_sev/2:.0f}%", flush=True)
    for wi, wf in enumerate(WIDTHS):
        r = rise[wi]
        if np.nanmax(r) >= full_sev / 2:
            dc = float(np.interp(full_sev / 2, r, DEPTHS))
            print(f"  width {wf:.2f}: critical depth {dc*100:.0f}% of wall", flush=True)
        else:
            print(f"  width {wf:.2f}: never reaches half-severance (benign to full depth)", flush=True)
    print("wrote", os.path.join(CACHE, "crack_chunk.npz"), flush=True)


if __name__ == "__main__":
    main()
