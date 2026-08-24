#!/usr/bin/env python3
"""
geometry_export.py
==================
Convert a 3D topology-optimized density field into CAD-ready surface meshes.

Pipeline (permissive-only stack):
  numpy (BSD)  ->  skimage.measure.marching_cubes (BSD)  ->  trimesh (MIT)

For each input density field we:
  1. load the flat density vector and its [nx, ny, nz] grid shape,
  2. reshape using element ordering e = iz*(nx*ny) + iy*nx + ix  ->  [iz, iy, ix],
     then transpose to a clean [ix, iy, iz] (x, y, z) field,
  3. pad with one layer of zeros on all six faces so the isosurface closes
     into a watertight solid (no open boundaries clipped by the grid edge),
  4. extract the level-0.5 isosurface with marching cubes,
  5. build a trimesh.Trimesh, fix winding / normals, fill holes,
  6. export the SAME geometry to STL (binary), OBJ, PLY and GLB (glTF binary).

License: this file is part of a permissive (MIT/BSD/Apache) export stack.
See GEOMETRY_LICENSES.md.
"""

import os
import sys
import numpy as np
from skimage import measure
import trimesh

# Reuse the canonical connectivity filter so the mesh path and the voxel path
# (scripts/voxels_export.py) drop the same disconnected specks. Add this file's
# directory to the path so the import resolves however the script is launched.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voxels_export import keep_connected

# Input and output directory. The earlier hardcoded session path is dead; resolve
# from TOPOHEAT_GEOMETRY_OUTDIR, falling back to the current working directory.
OUTDIR = os.environ.get("TOPOHEAT_GEOMETRY_OUTDIR", os.getcwd())


def load_field(rho_path, shape_path):
    """Load a flat density vector + [nx,ny,nz] shape -> [ix,iy,iz] float field."""
    rho = np.load(rho_path).astype(np.float64).ravel()
    shp = np.load(shape_path).astype(int).ravel()
    nx, ny, nz = int(shp[0]), int(shp[1]), int(shp[2])
    assert rho.size == nx * ny * nz, (
        f"density length {rho.size} != nx*ny*nz = {nx*ny*nz}")
    # element e = iz*(nx*ny) + iy*nx + ix  ->  reshape(nz, ny, nx) = [iz,iy,ix]
    field_zyx = rho.reshape(nz, ny, nx)
    # transpose to (x, y, z) ordering for the mesher
    field_xyz = np.transpose(field_zyx, (2, 1, 0))  # [ix, iy, iz]
    return field_xyz, (nx, ny, nz)


def field_to_mesh(field, level=0.5):
    """Pad with zeros, marching-cubes at `level`, return a cleaned trimesh."""
    # pad one zero layer on every side so the surface closes
    padded = np.pad(field, pad_width=1, mode="constant", constant_values=0.0)

    # marching cubes; spacing=1 voxel.
    verts, faces, normals, _ = measure.marching_cubes(
        padded, level=level, allow_degenerate=False)

    # shift back so the padding offset is removed
    verts = verts - 1.0

    mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                           vertex_normals=normals, process=True)

    # cleanup pass
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.fix_normals()
    try:
        mesh.fill_holes()
    except Exception:
        pass
    return mesh


def export_all(mesh, stem):
    """Export the SAME mesh to STL(binary), OBJ, PLY, GLB."""
    out = {}
    targets = {
        "stl": dict(file_type="stl"),   # binary STL by default
        "obj": dict(file_type="obj"),
        "ply": dict(file_type="ply"),   # binary PLY by default
        "glb": dict(file_type="glb"),   # glTF binary
    }
    for ext, kw in targets.items():
        path = os.path.join(OUTDIR, f"{stem}.{ext}")
        mesh.export(path, **kw)
        out[ext] = (path, os.path.getsize(path))
    return out


def process(rho_name, shape_name, stem, level=0.5):
    print("=" * 64)
    print(f"DESIGN: {stem}   (from {rho_name})")
    field, (nx, ny, nz) = load_field(
        os.path.join(OUTDIR, rho_name), os.path.join(OUTDIR, shape_name))
    print(f"  grid (nx,ny,nz)   : {nx} x {ny} x {nz}  "
          f"(solid fraction {field.mean():.3f})")

    # Drop disconnected specks before meshing. Keep only material that is
    # face-connected to the main body at the isosurface threshold; otherwise
    # marching cubes emits a separate shell for every floating island, the same
    # fault that split the STEP export into three solids.
    occ = field > level
    keep, rep = keep_connected(occ)
    if rep["components"] > 1:
        field = np.where(keep, field, 0.0)
        print(f"  connectivity      : {rep['components']} components "
              f"{rep['sizes'][:8]}; kept largest, "
              f"removed {rep['removed']} voxel(s)")
    else:
        print(f"  connectivity      : single component "
              f"({rep['kept']} voxels)")

    mesh = field_to_mesh(field, level=level)

    bb = mesh.bounds
    print(f"  vertices          : {len(mesh.vertices)}")
    print(f"  faces             : {len(mesh.faces)}")
    print(f"  watertight        : {mesh.is_watertight}")
    print(f"  winding consistent: {mesh.is_winding_consistent}")
    print(f"  euler number      : {mesh.euler_number}")
    print(f"  bounding box min  : {bb[0]}")
    print(f"  bounding box max  : {bb[1]}")
    print(f"  bbox extents      : {mesh.extents}")
    if not mesh.is_watertight:
        comps = mesh.split(only_watertight=False)
        if len(comps):
            biggest = max(comps, key=lambda m: len(m.faces))
            print(f"  NOT watertight -> {len(comps)} component(s); "
                  f"largest = {len(biggest.faces)} faces, "
                  f"watertight={biggest.is_watertight}")

    exported = export_all(mesh, stem)
    for ext, (path, size) in exported.items():
        print(f"  export {ext.upper():3s}        : {os.path.basename(path)}  "
              f"({size:,} bytes)")
    return mesh, exported


def write_viewer_mesh_js(mesh, js_path):
    """Write inline geometry JSON for the file://-portable Three.js viewer."""
    import json
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    center = verts.mean(axis=0)
    verts_c = verts.copy()
    verts_c[:, 0] -= center[0]
    verts_c[:, 1] -= center[1]
    verts_c[:, 2] -= verts[:, 2].min()
    data = {
        "vertices": [round(float(v), 5) for v in verts_c.ravel()],
        "faces": [int(i) for i in faces.ravel()],
        "vertexCount": int(len(verts)),
        "faceCount": int(len(faces)),
    }
    with open(js_path, "w") as f:
        f.write("// Auto-generated by geometry_export.py - inline mesh for viewer.html\n")
        f.write("// Permissive stack (numpy/skimage BSD, trimesh MIT). file://-portable.\n")
        f.write("window.MESH_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print(f"  viewer mesh JS    : {os.path.basename(js_path)} "
          f"({os.path.getsize(js_path):,} bytes, "
          f"{data['vertexCount']} verts / {data['faceCount']} faces)")


def render_preview(mesh, png_path):
    """Static 3D sanity-check preview via matplotlib (Agg, no display)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    tris = verts[faces]
    coll = Poly3DCollection(tris, alpha=1.0)
    coll.set_facecolor((0.62, 0.62, 0.64))
    coll.set_edgecolor((0.30, 0.30, 0.30))
    coll.set_linewidth(0.05)
    ax.add_collection3d(coll)

    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[2], mx[2])
    try:
        ax.set_box_aspect(mx - mn)
    except Exception:
        pass
    ax.view_init(elev=22, azim=-60)
    ax.set_axis_off()
    ax.set_title("Topology-optimized 3D structure (isosurface @ 0.5)",
                 fontsize=11, color="0.2")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(png_path, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  preview PNG       : {os.path.basename(png_path)} "
          f"({os.path.getsize(png_path):,} bytes)")


def main():
    print(f"OUTDIR (in/out)     : {OUTDIR}")
    # PRIMARY design -> design3d.*  + viewer assets + preview
    primary_mesh, _ = process("scaleopt_rp.npy", "scaleopt_shape.npy", "design3d")
    write_viewer_mesh_js(primary_mesh, os.path.join(OUTDIR, "viewer_mesh.js"))
    render_preview(primary_mesh, os.path.join(OUTDIR, "preview3d.png"))

    # smaller alternative design -> design3d_d3.*
    process("d3_rho.npy", "d3_shape.npy", "design3d_d3")

    print("=" * 64)
    print("DONE.")


if __name__ == "__main__":
    main()
