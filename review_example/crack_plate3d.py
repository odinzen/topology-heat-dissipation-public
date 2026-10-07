"""Sub-model 3 (3D): fatigue cracks in the manufacturable flat cold plate.

Takes the flat cold-plate geometry we
already have (the branching heat-collection plate, recovered by voxelizing its surface
mesh), drives through-thickness fatigue cracks of different lengths across its limbs
plus one conjugate-shear X, and re-solves 3D conduction (topoheat HeatTO3D) to show
the heat-absorption degradation on the plate.

Same physics labels as the 2D crack_tree.py: Paris-law growth, transverse crack
direction, conduction re-solve. Constants lit-range / SCHEMATIC where noted.

Run:  python crack_plate3d.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_fill_holes

_TOPO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..")
if os.path.abspath(_TOPO) not in sys.path:
    sys.path.insert(0, os.path.abspath(_TOPO))

from topoheat.topopt3d import HeatTO3D           # noqa: E402
import crack_tree as ct                            # noqa: E402  (2D crack helpers reused)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
os.makedirs(CACHE, exist_ok=True)
MESH = os.path.join(CACHE, "flatplate_mesh.json")  # written by figure6_flatplate.py

NX, NY, NZ = 96, 96, 8


def voxelize(mesh_path=MESH, nx=NX, ny=NY, nz=NZ):
    cf = os.path.join(CACHE, f"plate_occ_{nx}x{ny}x{nz}.npy")
    if os.path.exists(cf):
        return np.load(cf)
    d = json.load(open(mesh_path))
    V = np.array(d["verts"]); F = np.array(d["faces"])
    lo, hi = V.min(0), V.max(0)
    def to_ijk(P):
        return (P - lo) / (hi - lo) * (np.array([nx, ny, nz]) - 1)
    shell = np.zeros((nx, ny, nz), bool)
    u = np.linspace(0, 1, 16); bu, bv = np.meshgrid(u, u)
    m = (bu + bv) <= 1.0; bu, bv = bu[m], bv[m]
    for tri in F:
        a, b, c = V[tri]
        pts = a[None] + bu[:, None] * (b - a)[None] + bv[:, None] * (c - a)[None]
        idx = np.clip(np.round(to_ijk(pts)).astype(int), 0, [nx-1, ny-1, nz-1])
        shell[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    occ = binary_fill_holes(shell)
    np.save(cf, occ)
    return occ


class Plate(HeatTO3D):
    """Cold-plate conduction: distributed die heat over the plate, coolant sink at the
    central collection port (base centre). Reuses HeatTO3D's thermal solve."""
    def plate_bc(self, sink_r=3):
        nx, ny, nz = self.nx, self.ny, self.nz
        cx, cy = nx // 2, ny // 2
        sink = [self.nid(ix, iy, iz)
                for iz in range(min(3, nz + 1))
                for iy in range(ny + 1) for ix in range(nx + 1)
                if abs(ix - cx) <= sink_r and abs(iy - cy) <= sink_r]
        self.ftemp = np.unique(sink)
        self.freeT = np.setdiff1d(np.arange(self.nnode), self.ftemp)
        self.q = np.full(self.nnode, 1.0 / self.nnode)
        self.q[self.ftemp] = 0.0


def rp_from_occ(plate, occ, floor=1e-3):
    r = occ[plate.cen[:, 0], plate.cen[:, 1], plate.cen[:, 2]].astype(float)
    r[r < 0.5] = floor
    return r


def top_face_T(plate, T):
    Tg = T.reshape(plate.nz + 1, plate.ny + 1, plate.nx + 1)
    return Tg[plate.nz]                       # die-side (top) face, indexed [iy, ix]


# plate-specific crack targets on the radial limbs.
# (fx, fy, in-plane-length-frac, kind, depth-frac-of-thickness)
# A surface fatigue crack initiates at the die (top) face and grows down; depth-frac is
# how far it has penetrated the plate thickness. Only a full-depth crack severs the limb;
# a shallow crack leaves a conducting ligament below it.
PLATE_TARGETS = [
    (0.50, 0.74, 1.00, "transverse", 1.00),   # full sever, top limb (through-thickness)
    (0.74, 0.50, 0.55, "transverse", 0.60),   # mid-depth, right limb
    (0.28, 0.50, 0.35, "transverse", 0.35),   # shallow, left limb
    (0.68, 0.70, 0.80, "transverse", 0.80),   # deep, upper-right diagonal
    (0.35, 0.33, 0.90, "x", 0.90),            # deep crisscross X, lower-left diagonal
]


def build_plate_cracks(foot):
    """Per-crack in-plane cells + segments + penetration depth on the plate footprint."""
    gx, gy = ct._grad(foot.astype(float))
    ny, nx = foot.shape
    cracks = []
    for fx, fy, frac, kind, depth in PLATE_TARGETS:
        seed = ct.snap_solid(foot.astype(float), fx, fy, nx, ny, r=8)
        if seed is None:
            continue
        _, normal = ct.limb_axis(gx, gy, seed[0], seed[1], shape=(ny, nx))
        width, _ = ct.limb_width(foot.astype(float), seed, normal)
        a = max(2.0, frac * width)
        dirs = [ct._rot(normal, +45), ct._rot(normal, -45)] if kind == "x" else [normal]
        mode = "center" if kind == "x" else "edge"
        cells, segs = set(), []
        for dvec in dirs:
            cc, seg = ct.crack_line(foot.astype(float), seed, dvec, a, mode=mode)
            cells |= cc; segs.append(seg)
        cracks.append(dict(cells=cells, segs=segs, kind=kind, depth=depth, seed=seed))
    return cracks


def apply_cracks(occ, cracks, nz):
    """Cut each crack from the top face down to its depth (leave the ligament below)."""
    occ_cut = occ.copy()
    for cr in cracks:
        d = max(1, int(round(cr["depth"] * nz)))
        for (j, i) in cr["cells"]:
            occ_cut[i, j, nz - d:] = False
    return occ_cut


def main():
    occ = voxelize()
    plate = Plate(NX, NY, NZ, volfrac=0.3, rmin=1.5)
    plate.plate_bc()
    print(f"plate {NX}x{NY}x{NZ}, solid voxels {occ.sum()}", flush=True)

    rp0 = rp_from_occ(plate, occ)
    T0 = plate.solve(rp0)["T"]; peak0 = float(T0[plate.freeT].max())
    top0 = top_face_T(plate, T0)
    Tvol0 = T0.reshape(NZ + 1, NY + 1, NX + 1)
    print(f"pristine peak T = {peak0:.4g}", flush=True)

    foot = occ.any(2)                          # in-plane footprint
    cracks = build_plate_cracks(foot)
    occ_cut = apply_cracks(occ, cracks, NZ)    # surface cracks, each to its own depth
    rp1 = rp_from_occ(plate, occ_cut)
    T1 = plate.solve(rp1)["T"]; peak1 = float(T1[plate.freeT].max())
    top1 = top_face_T(plate, T1)
    Tvol1 = T1.reshape(NZ + 1, NY + 1, NX + 1)
    segs = [s for cr in cracks for s in cr["segs"]]
    kinds = [cr["kind"] for cr in cracks for _ in cr["segs"]]
    depths = [cr["depth"] for cr in cracks for _ in cr["segs"]]
    # per-crack metadata for the depth cross-section (seed row, depth, kind)
    seed_iy = np.array([cr["seed"][1] for cr in cracks])
    seed_ix = np.array([cr["seed"][0] for cr in cracks])
    crack_depth = np.array([cr["depth"] for cr in cracks])
    crack_kind = np.array([cr["kind"] for cr in cracks])
    print(f"cracked  peak T = {peak1:.4g}  ({100*(peak1/peak0 - 1):.1f}% rise), "
          f"{len(cracks)} cracks (incl. X); depths (frac) "
          f"{[cr['depth'] for cr in cracks]}", flush=True)

    np.savez(os.path.join(CACHE, "crack3d.npz"),
             occ=occ, occ_cut=occ_cut, top0=top0, top1=top1,
             Tvol0=Tvol0, Tvol1=Tvol1,
             seed_ix=seed_ix, seed_iy=seed_iy,
             crack_depth=crack_depth, crack_kind=crack_kind,
             peak0=peak0, peak1=peak1,
             segs=np.array(segs, dtype=object),
             kinds=np.array(kinds), depths=np.array(depths), NX=NX, NY=NY, NZ=NZ)
    print("wrote", os.path.join(CACHE, "crack3d.npz"), flush=True)


if __name__ == "__main__":
    main()
