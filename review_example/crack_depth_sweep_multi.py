"""Failure threshold across several branches. For each of a set of limbs (primary,
diagonal, side, outer twig) grow a full-width surface crack progressively deeper and
re-solve conduction, tracking the plate peak temperature. Gives each branch its own
effectiveness-vs-depth curve and critical depth, and so the spread across the hierarchy.

Constants SCHEMATIC. Needs numpy, scipy and matplotlib.
"""
from __future__ import annotations

import gc
import os
import numpy as np

import crack_plate3d as cp3
import crack_tree as ct

CACHE = cp3.CACHE
NX, NY, NZ = 56, 56, 16                  # finer-thickness plate; small in-plane for speed/memory

# branches spanning the hierarchy: (fx, fy, label)
BRANCHES = [
    (0.50, 0.75, "top primary"),
    (0.75, 0.50, "side primary"),
    (0.69, 0.69, "diagonal limb"),
    (0.38, 0.71, "secondary branch"),
    (0.55, 0.87, "outer twig"),
]
DEPTHS = np.array([0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.65, 0.82, 1.0])
CX, CY = NX / 2.0, NY / 2.0


def _branch(foot, gx, gy, fx, fy):
    seed = ct.snap_solid(foot.astype(float), fx, fy, NX, NY, r=10)
    if seed is None:
        return None
    _, normal = ct.limb_axis(gx, gy, seed[0], seed[1], shape=(NY, NX))
    width, _ = ct.limb_width(foot.astype(float), seed, normal)
    cells, _ = ct.crack_line(foot.astype(float), seed, normal, max(3.0, width), mode="center")
    return seed, cells


def _sector_mask(seed, footT):
    """Top-face cells this branch serves: same angular sector, further out than the seed."""
    ix, iy = seed
    th = np.arctan2(iy - CY, ix - CX); r0 = np.hypot(ix - CX, iy - CY)
    Y, X = np.mgrid[0:NY, 0:NX]                        # [iy, ix]
    dang = np.abs(np.angle(np.exp(1j * (np.arctan2(Y - CY, X - CX) - th))))
    rad = np.hypot(X - CX, Y - CY)
    return (dang < np.radians(26)) & (rad > max(3.0, 0.85 * r0)) & footT


def _served_peak(plate, rp, mask):
    T = plate.solve(rp)["T"].reshape(NZ + 1, NY + 1, NX + 1)[NZ]   # top face [iy, ix]
    vals = T[:NY, :NX][mask]
    return float(vals.max()) if vals.size else 0.0


def main():
    occ = cp3.voxelize(nx=NX, ny=NY, nz=NZ)
    plate = cp3.Plate(NX, NY, NZ, volfrac=0.3, rmin=1.5)
    plate.plate_bc()
    foot = occ.any(2)
    footT = foot.T                                    # [iy, ix] material mask
    gx, gy = ct._grad(foot.astype(float))

    labels, curves, dcrit, severity = [], [], [], []
    for fx, fy, lab in BRANCHES:
        try:
            br = _branch(foot, gx, gy, fx, fy)
            if br is None:
                print("skip (no seed):", lab, flush=True); continue
            seed, cells = br
            mask = _sector_mask(seed, footT)
            if mask.sum() < 5:
                print("skip (empty sector):", lab, flush=True); continue
            m0 = _served_peak(plate, cp3.rp_from_occ(plate, occ), mask)
            served = []
            for c in DEPTHS:
                d = int(round(c * NZ))
                occ_c = occ.copy()
                if d > 0:
                    for (j, i) in cells:
                        occ_c[i, j, NZ - d:] = False
                served.append(_served_peak(plate, cp3.rp_from_occ(plate, occ_c), mask))
            served = np.array(served)
            rise_frac = (served - m0) / max(served[-1] - m0, 1e-9)
            sev = 100.0 * (served[-1] / m0 - 1.0)
            dc = float(np.interp(0.5, rise_frac, DEPTHS)) if sev > 8.0 else np.nan
            labels.append(lab); curves.append(m0 / served * 100.0)
            dcrit.append(dc); severity.append(sev)
            dcs = f"{dc*100:.0f}%" if np.isfinite(dc) else "none (redundant)"
            print(f"{lab:>18}: served rise {sev:5.0f}%  critical depth {dcs}", flush=True)
            # incremental save so a later crash never loses completed branches
            np.savez(os.path.join(CACHE, "depth_sweep_multi.npz"),
                     depths=DEPTHS, labels=np.array(labels), curves=np.array(curves),
                     dcrit=np.array(dcrit), severity=np.array(severity))
            gc.collect()
        except Exception as e:
            print(f"ERROR on {lab}: {type(e).__name__}: {e}", flush=True)

    print("done; wrote", os.path.join(CACHE, "depth_sweep_multi.npz"), flush=True)
    print("wrote", os.path.join(CACHE, "depth_sweep_multi.npz"))


if __name__ == "__main__":
    main()
