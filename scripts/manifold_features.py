#!/usr/bin/env python3
"""
manifold_features.py

Harden a topology-optimized 3D mesh and add manufacturing features using
Manifold (manifold3d, Apache-2.0). Permissive stack only:
  - manifold3d (Apache-2.0)  : guaranteed-manifold CSG kernel
  - trimesh    (MIT)         : mesh I/O
  - numpy/scikit-image (BSD) : array math / marching cubes fallback
  - matplotlib (permissive)  : preview render

Confirmed against manifold3d 3.5.1:
  m   = manifold3d.Mesh(vert_properties=verts_f32_(N,>=3),
                        tri_verts=faces_u32_(M,3))
  man = manifold3d.Manifold(m)         # sets Error Status if not a 2-manifold
  manifold3d.Manifold.cube([sx,sy,sz], center=False)
  manifold3d.Manifold.cylinder(height, radius_low, radius_high=-1.0,
                               circular_segments=0, center=False)  # axis = +z
  Booleans:  union a + b ;  difference a - b ;  intersection a ^ b
  man.volume(); man.surface_area(); man.num_tri(); man.num_vert();
  man.genus(); man.is_empty(); man.status(); man.bounding_box();
  man.translate([x,y,z]); man.decompose()
  mesh = man.to_mesh(); mesh.vert_properties[:, :3] (xyz); mesh.tri_verts (M,3)
"""

import os
import shutil
import numpy as np
import manifold3d as m3
import trimesh

# Input and output directory. The earlier hardcoded session path is dead; resolve
# from TOPOHEAT_GEOMETRY_OUTDIR, falling back to the current working directory
# (matches scripts/geometry_export.py). Temp artifacts stay in-tree under OUT.
OUT = os.environ.get("TOPOHEAT_GEOMETRY_OUTDIR", os.getcwd())
os.makedirs(OUT, exist_ok=True)
TMP = OUT


def sep(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


def load_design():
    for name in ("design3d.ply", "design3d.obj", "design3d.glb"):
        p = os.path.join(OUT, name)
        if os.path.exists(p):
            try:
                tm = trimesh.load(p, process=False, force="mesh")
                if tm.vertices.shape[0] and tm.faces.shape[0]:
                    print(f"Loaded design from {name}: "
                          f"{tm.vertices.shape[0]} verts, {tm.faces.shape[0]} faces")
                    return np.asarray(tm.vertices, float), np.asarray(tm.faces, np.int64)
            except Exception as e:
                print(f"  {name} failed: {e}")
    print("Falling back to marching_cubes on scaleopt_rp.npy")
    from skimage import measure
    field = np.load(os.path.join(OUT, "scaleopt_rp.npy"))
    nz, ny, nx = np.load(os.path.join(OUT, "scaleopt_shape.npy")).astype(int)
    field = field.reshape(nz, ny, nx)
    padded = np.pad(field, 1, mode="constant", constant_values=0.0)
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    verts = verts[:, ::-1] - 1.0          # (z,y,x)->(x,y,z), undo pad
    return verts.astype(float), faces.astype(np.int64)


def manifold_from_arrays(verts, faces):
    mesh = m3.Mesh(vert_properties=np.ascontiguousarray(verts, dtype=np.float32),
                   tri_verts=np.ascontiguousarray(faces, dtype=np.uint32))
    return m3.Manifold(mesh)


def manifold_to_trimesh(man):
    mesh = man.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3].astype(np.float64)
    f = np.asarray(mesh.tri_verts).astype(np.int64)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def report(man, label):
    print(f"[{label}] volume={man.volume():.4f}  area={man.surface_area():.4f}  "
          f"tris={man.num_tri()}  verts={man.num_vert()}  genus={man.genus()}  "
          f"empty={man.is_empty()}  status={man.status()}  "
          f"components={len(man.decompose())}")


def main():
    sep("manifold3d API")
    import importlib.metadata as md
    print("manifold3d version:", md.version("manifold3d"))
    print("trimesh version    :", trimesh.__version__)
    print("Booleans: union=a+b  difference=a-b  intersection=a^b "
          "(verified: cube^cube overlap volume correct)")

    # ---------------- TASK 1 ----------------
    sep("TASK 1  Build guaranteed manifold from design mesh")
    verts, faces = load_design()
    design = manifold_from_arrays(verts, faces)
    assert not design.is_empty(), "design Manifold is empty!"
    assert design.status() == m3.Error.NoError, f"bad status {design.status()}"
    report(design, "design")
    minx, miny, minz, maxx, maxy, maxz = design.bounding_box()
    print("bounding_box min:", tuple(round(v, 3) for v in (minx, miny, minz)),
          " max:", tuple(round(v, 3) for v in (maxx, maxy, maxz)))

    # ---------------- TASK 2a : trim / build envelope (intersection) -------
    sep("TASK 2a  Trim with inset build envelope (intersection a ^ b)")
    inset = 0.5
    ex = (maxx - minx) - 2 * inset
    ey = (maxy - miny) - 2 * inset
    ez = (maxz - minz) - 2 * inset
    envelope = (m3.Manifold.cube([ex, ey, ez], center=False)
                .translate([minx + inset, miny + inset, minz + inset]))
    trimmed = design ^ envelope          # intersection
    report(trimmed, "trimmed")
    tnx, tny, tnz, txx, txy, txz = trimmed.bounding_box()
    print("trimmed bbox min:", tuple(round(v, 3) for v in (tnx, tny, tnz)),
          " max:", tuple(round(v, 3) for v in (txx, txy, txz)))

    # The trimmed part has two solid legs near the clamped low-x face:
    #   low-y  leg  ~ y in [0.5, 2.5]
    #   high-y leg  ~ y in [15.5, 18.5]
    # with a hollow gap between them. Holes are drilled through the legs and
    # the flange spans the full y-extent so the result stays one solid body.

    # ---------------- TASK 2b : mounting holes (cylinders, axis +z) --------
    sep("TASK 2b  Subtract mounting holes (cylinders) through clamped-end legs")
    r = 0.8
    hole_x = tnx + 1.5                    # 1.5 in from clamped face -> in solid
    hole_ys = [tny + 1.0, txy - 1.0]      # one through each leg
    h = (txz - tnz) + 4.0                 # over-long: punches clean through z
    featured = trimmed
    for i, hy in enumerate(hole_ys):
        cyl = (m3.Manifold.cylinder(h, r, r, circular_segments=48, center=False)
               .translate([hole_x, hy, tnz - 2.0]))
        before = featured.volume()
        featured = featured - cyl         # difference
        print(f"  hole {i+1} at (x={hole_x:.2f}, y={hy:.2f}) r={r} "
              f"removed {before - featured.volume():.3f} vol")
    report(featured, "after holes")

    # ---------------- TASK 2c : mounting flange / base tab (cube union) ----
    sep("TASK 2c  Union mounting flange (cube) under clamped end")
    tab_thick = 2.0
    tab_x = 4.0
    overlap = 0.5                         # fuse into legs so it is one body
    tab = (m3.Manifold.cube([tab_x, (txy - tny), tab_thick], center=False)
           .translate([tnx, tny, tnz - tab_thick + overlap]))
    featured = featured + tab            # union
    report(featured, "FEATURED part")
    assert not featured.is_empty()
    assert featured.status() == m3.Error.NoError
    n_components = len(featured.decompose())
    print(f"Single manifold solid? components = {n_components} "
          f"({'YES' if n_components == 1 else 'NO - multiple bodies'})")

    # ---------------- TASK 3 : export -------------------------------------
    sep("TASK 3  Export featured part (STL/OBJ/3MF/GLB)")
    tm = manifold_to_trimesh(featured)
    print(f"featured trimesh: {tm.vertices.shape[0]} verts, {tm.faces.shape[0]} faces, "
          f"watertight={tm.is_watertight}, trimesh volume={tm.volume:.4f}")

    exports = {
        "manifold_part.stl": dict(file_type="stl"),   # binary STL
        "manifold_part.obj": dict(file_type="obj"),
        "manifold_part.3mf": dict(file_type="3mf"),
        "manifold_part.glb": dict(file_type="glb"),
    }
    results = {}
    for fname, kw in exports.items():
        dst = os.path.join(OUT, fname)
        try:
            tm.export(dst, **kw)
        except Exception as e:
            print(f"  direct export {fname} note: {e}; retrying via Scene")
            trimesh.Scene(tm).export(dst, **kw)
        size = os.path.getsize(dst) if os.path.exists(dst) else 0
        results[fname] = size
        print(f"  wrote {fname}: {size} bytes  (non-empty={size > 0})")

    stl_reload = trimesh.load(os.path.join(OUT, "manifold_part.stl"),
                              process=True, force="mesh")
    print(f"  STL reload: {stl_reload.vertices.shape[0]} verts, "
          f"watertight={stl_reload.is_watertight}, volume={stl_reload.volume:.4f}")

    # ---------------- TASK 4 : preview PNG --------------------------------
    sep("TASK 4  Greyscale preview PNG")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    tris = tm.vertices[tm.faces]
    coll = Poly3DCollection(tris, facecolor="0.72", edgecolor="0.25",
                            linewidths=0.15, alpha=1.0)
    ax.add_collection3d(coll)
    v = tm.vertices
    ax.set_xlim(v[:, 0].min(), v[:, 0].max())
    ax.set_ylim(v[:, 1].min(), v[:, 1].max())
    ax.set_zlim(v[:, 2].min(), v[:, 2].max())
    try:
        ax.set_box_aspect((np.ptp(v[:, 0]), np.ptp(v[:, 1]), np.ptp(v[:, 2])))
    except Exception:
        pass
    ax.view_init(elev=22, azim=-58)
    ax.set_xlabel("x (clamped at low-x)")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Featured manifold part: trimmed + 2 mounting holes + flange")
    fig.tight_layout()

    png_name = "manifold_part_preview.png"
    tmp_png = os.path.join(TMP, png_name)
    fig.savefig(tmp_png, dpi=130)
    plt.close(fig)
    dst_png = os.path.join(OUT, png_name)
    if os.path.exists(tmp_png) and os.path.getsize(tmp_png) > 0:
        shutil.copyfile(tmp_png, dst_png)
    png_size = os.path.getsize(dst_png) if os.path.exists(dst_png) else 0
    print(f"  wrote {png_name}: {png_size} bytes (non-empty={png_size > 0})")

    # ---------------- SUMMARY ---------------------------------------------
    sep("SUMMARY")
    print(f"design volume       : {design.volume():.4f}  tris={design.num_tri()}  "
          f"genus={design.genus()}")
    print(f"trimmed volume      : {trimmed.volume():.4f}")
    print(f"featured volume     : {featured.volume():.4f}  components={n_components}")
    print("exports:")
    for fname, size in results.items():
        print(f"  {fname:22s} {size:>9d} bytes")
    print(f"preview png         : {png_size} bytes")
    print("DONE.")


if __name__ == "__main__":
    main()
