#!/usr/bin/env python3
"""
voxels_export.py
================
Turn a 3D topology-optimized density field into the occupied-voxel list that the
Truck STEP exporter (geometry/truck_step) consumes as voxels.csv.

This is the canonical voxel-export step. It thresholds the density field and then
keeps only structurally connected material, dropping isolated specks the filter
and threshold leave behind. Those specks are face-disconnected from the main body,
carry no load, and turn one clean part into several floating solids downstream.
Removing them here means every export is connected by construction.

Field ordering matches geometry_export.py and scalefigs.py:
    element e = iz*(nx*ny) + iy*nx + ix
    field = transpose(rho.reshape(nz, ny, nx), (2, 1, 0))  ->  [ix, iy, iz]

Two input modes:

  From the optimizer density field (the normal path):
      python voxels_export.py field scaleopt_rp.npy scaleopt_shape.npy voxels.csv

  Re-filter an existing voxels.csv in place (no density field needed):
      python voxels_export.py refilter voxels.csv voxels.csv

Connectivity is 6-connected (face adjacency), which is exactly the adjacency that
fuses into one solid in the STEP and CAD stages. By default only the single
largest component survives. Raise the keep threshold with --min-voxels to retain
every component at or above a given size instead.
"""

import argparse
import csv
import sys

import numpy as np
from scipy import ndimage


# 6-connected structuring element: faces touch, edges and corners do not.
_STRUCT = ndimage.generate_binary_structure(3, 1)


def field_to_occupancy(rho_path, shape_path, level):
    """Load a flat density vector + [nx,ny,nz] shape, threshold to a bool grid."""
    rho = np.load(rho_path).astype(np.float64).ravel()
    shp = np.load(shape_path).astype(int).ravel()
    nx, ny, nz = int(shp[0]), int(shp[1]), int(shp[2])
    if rho.size != nx * ny * nz:
        raise ValueError(
            f"density length {rho.size} != nx*ny*nz = {nx * ny * nz}")
    field = np.transpose(rho.reshape(nz, ny, nx), (2, 1, 0))  # [ix, iy, iz]
    return field > level, (nx, ny, nz)


def occupancy_from_csv(csv_path):
    """Read an x,y,z voxel list into a bool grid sized to its bounding box."""
    pts = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            pts.append(tuple(int(round(float(v))) for v in row[:3]))
    if not pts:
        raise ValueError(f"no voxels found in {csv_path}")
    arr = np.array(pts, dtype=int)
    mn = arr.min(axis=0)
    arr -= mn  # shift to origin so the grid starts at 0
    dims = arr.max(axis=0) + 1
    occ = np.zeros(tuple(dims), dtype=bool)
    occ[arr[:, 0], arr[:, 1], arr[:, 2]] = True
    return occ, tuple(int(v) for v in mn)  # offset so coordinates round-trip


def keep_connected(occ, min_voxels=None):
    """Label 6-connected components; keep the largest, or all >= min_voxels.

    Returns the filtered bool grid plus a small report dict.
    """
    labels, n = ndimage.label(occ, structure=_STRUCT)
    if n == 0:
        return occ, {"components": 0, "kept": 0, "removed": 0, "sizes": []}
    sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n + 1))
    sizes = sizes.astype(int)
    order = np.argsort(sizes)[::-1]  # largest first, 0-based label index

    if min_voxels is None:
        keep_labels = {int(order[0]) + 1}
    else:
        keep_labels = {int(i) + 1 for i in range(n) if sizes[i] >= min_voxels}

    mask = np.isin(labels, list(keep_labels))
    kept = int(occ.sum()) - int((occ & ~mask).sum())
    report = {
        "components": int(n),
        "kept": int(mask.sum()),
        "removed": int(occ.sum()) - int(mask.sum()),
        "sizes": sorted(sizes.tolist(), reverse=True),
        "kept_components": len(keep_labels),
    }
    return mask, report


def write_csv(occ, out_path, offset=(0, 0, 0)):
    """Write occupied voxel coordinates (one x,y,z per line), offset restored."""
    xs, ys, zs = np.nonzero(occ)
    ox, oy, oz = offset
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        for x, y, z in zip(xs, ys, zs):
            w.writerow([x + ox, y + oy, z + oz])
    return len(xs)


def _report(report, n_in, n_out):
    print(f"  components        : {report['components']}  "
          f"(sizes: {report['sizes'][:8]}"
          f"{' ...' if len(report['sizes']) > 8 else ''})")
    print(f"  kept components   : {report.get('kept_components', 1)}")
    print(f"  voxels in / out   : {n_in} -> {n_out}  "
          f"(removed {n_in - n_out})")


def main(argv=None):
    p = argparse.ArgumentParser(description="Export connected voxels to CSV.")
    sub = p.add_subparsers(dest="mode", required=True)

    pf = sub.add_parser("field", help="threshold a density field, then filter")
    pf.add_argument("rho_npy")
    pf.add_argument("shape_npy")
    pf.add_argument("out_csv")
    pf.add_argument("--level", type=float, default=0.5,
                    help="density threshold (default 0.5, matches the mesher)")
    pf.add_argument("--min-voxels", type=int, default=None,
                    help="keep every component this size or larger "
                         "(default: keep only the largest)")

    pr = sub.add_parser("refilter", help="re-filter an existing voxels.csv")
    pr.add_argument("in_csv")
    pr.add_argument("out_csv")
    pr.add_argument("--min-voxels", type=int, default=None)

    a = p.parse_args(argv)

    if a.mode == "field":
        occ, (nx, ny, nz) = field_to_occupancy(a.rho_npy, a.shape_npy, a.level)
        print(f"field grid          : {nx} x {ny} x {nz}  "
              f"(threshold {a.level}, occupied {int(occ.sum())})")
        offset = (0, 0, 0)
    else:
        occ, offset = occupancy_from_csv(a.in_csv)
        print(f"loaded voxels.csv   : {int(occ.sum())} occupied, "
              f"offset {offset}")

    n_in = int(occ.sum())
    filtered, report = keep_connected(occ, min_voxels=a.min_voxels)
    n_out = write_csv(filtered, a.out_csv, offset=offset)
    _report(report, n_in, n_out)
    print(f"wrote {a.out_csv}  ({n_out} voxels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
