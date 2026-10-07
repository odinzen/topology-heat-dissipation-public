"""Depth-to-failure sweep: grow a single crack on one primary limb of the cold plate
progressively deeper, re-solve conduction at each depth, and track when the branch stops
carrying heat effectively. Finds the critical depth (the knee) at which a partial crack
becomes a thermal failure, well before full severance.

Constants SCHEMATIC where noted. Needs numpy, scipy and matplotlib.
"""
from __future__ import annotations

import os
import numpy as np

import crack_plate3d as cp3

CACHE = cp3.CACHE
# Finer through-thickness resolution than the 96x96x8 display grid so the depth sweep
# is smooth (the plate thickness is the same; only its z-discretization is finer).
NX, NY, NZ = 80, 80, 20


def main():
    occ = cp3.voxelize(nx=NX, ny=NY, nz=NZ)
    plate = cp3.Plate(NX, NY, NZ, volfrac=0.3, rmin=1.5)
    plate.plate_bc()

    rp0 = cp3.rp_from_occ(plate, occ)
    peak0 = float(plate.solve(rp0)["T"][plate.freeT].max())

    cracks = cp3.build_plate_cracks(occ.any(2))
    cr = cracks[0]                                     # top primary limb (severs at full depth)

    depths = np.linspace(0.0, 1.0, 21)
    peaks = []
    for c in depths:
        d = int(round(c * NZ))
        occ_c = occ.copy()
        if d > 0:
            for (j, i) in cr["cells"]:
                occ_c[i, j, NZ - d:] = False
        rp = cp3.rp_from_occ(plate, occ_c)
        peaks.append(float(plate.solve(rp)["T"][plate.freeT].max()))
    peaks = np.array(peaks)

    eff = peak0 / peaks * 100.0                        # branch effectiveness (%, 100 = pristine)
    rise = (peaks - peak0) / (peaks[-1] - peak0)       # 0 pristine -> 1 fully severed
    # critical depth: where the served region has lost half the way to the severed state
    d_crit = float(np.interp(0.5, rise, depths))

    print(f"pristine peak T {peak0:.4g}, severed peak T {peaks[-1]:.4g} "
          f"({100*(peaks[-1]/peak0-1):.0f}% rise)")
    print(f"critical depth (50% of the way to severed): {d_crit*100:.0f}% of thickness")
    for c, p, e in zip(depths, peaks, eff):
        print(f"  depth {c*100:5.0f}%  peakT {p:6.3f}  effectiveness {e:5.1f}%")

    np.savez(os.path.join(CACHE, "depth_sweep.npz"),
             depths=depths, peaks=peaks, eff=eff, rise=rise,
             peak0=peak0, d_crit=d_crit)
    print("wrote", os.path.join(CACHE, "depth_sweep.npz"))


if __name__ == "__main__":
    main()
