# Geometry Export & Viewer — License Audit

**Scope:** the 3D geometry pipeline that turns a topology-optimized density
field into CAD-ready surface meshes (`geometry_export.py`) and the interactive
web viewer (`viewer.html` + `viewer_mesh.js`).

**Bottom line:** every component of this geometry export and viewer stack is
released under a permissive license — **MIT, BSD-3-Clause, or Apache-2.0** — with
no GPL and no commercial dependencies. The stack is **product-safe**: it can be
redistributed and used inside a closed-source commercial product without
copyleft obligations, subject only to retaining the upstream copyright/license
notices.

---

## Components used

| Component | Role in this pipeline | Version | License | Permissive? |
|-----------|----------------------|---------|---------|-------------|
| **NumPy** | Load `.npy` density field, array math, zero-padding | 2.2.6 | BSD-3-Clause | Yes |
| **scikit-image** (`skimage.measure.marching_cubes`) | Extract the level-0.5 isosurface | 0.25.2 | BSD-3-Clause | Yes |
| **trimesh** | Build/clean mesh, fix normals, fill holes, export STL/OBJ/PLY/GLB | 4.12.2 | MIT | Yes |
| **SciPy** (transitive, via skimage/trimesh) | Numerical helpers | 1.15.3 | BSD-3-Clause | Yes |
| **NetworkX** (transitive, via trimesh) | Mesh graph ops | 3.4.2 | BSD-3-Clause | Yes |
| **Pillow** (transitive) | Image I/O helpers | 12.2.0 | MIT-CMU (MIT-style, permissive) | Yes |
| **Matplotlib** (verification only — `preview3d.png`) | Offscreen 3D sanity preview (Agg backend) | 3.10.9 | Matplotlib/PSF-based (BSD-compatible, permissive) | Yes |
| **Three.js** | WebGL rendering in `viewer.html` (loaded from cdnjs at runtime) | r128+ | MIT | Yes |
| **Three.js OrbitControls** | Orbit / zoom / pan camera (loaded from cdnjs) | r128+ | MIT | Yes |

> Matplotlib is used **only** to render the optional static `preview3d.png`
> sanity-check image. It is not part of the shippable export/viewer path; even
> so, its license (Matplotlib License, derived from the PSF license) is
> BSD-compatible and permissive — not copyleft.

---

## Export formats — all permissive

The mesh is written to four interchange formats, all produced by **trimesh (MIT)**
using open, royalty-free specifications:

- **STL** (binary) — `design3d.stl` — de-facto open format, no license encumbrance.
- **OBJ** — `design3d.obj` — open Wavefront text format.
- **PLY** — `design3d.ply` — open Stanford polygon format.
- **GLB** (glTF 2.0 binary) — `design3d.glb` — Khronos open standard, royalty-free.

All four files contain the **same** triangulated surface geometry and can be
opened directly in **FreeCAD** (LGPL — the CAD *application* the user opens files
in; it is not linked into or redistributed with this code, so it imposes no
obligation on our exported artifacts or our codebase).

---

## CDN delivery (viewer)

`viewer.html` loads Three.js and OrbitControls from
`https://cdnjs.cloudflare.com` (cdnjs serves the official MIT-licensed builds).
The default viewer embeds the geometry **inline** as `viewer_mesh.js`
(`window.MESH_DATA` = vertices + faces), so the page renders by simply opening
`viewer.html` from `file://` with no local web server. Only the Three.js library
files require internet access; the geometry itself is local and self-contained.

---

## STEP / true B-Rep — optional future path (NOT implemented here)

This pipeline exports **triangulated surface meshes** (STL/OBJ/PLY/GLB), which is
fully permissive. It does **not** export **STEP (ISO 10303) B-Rep solids**.

Producing genuine STEP B-Rep geometry in Python practically requires
**OpenCascade / OCCT** (via `pythonocc-core` or `cadquery`), which is licensed
under **LGPL-2.1**. LGPL is open source and *can* be used in a commercial product
via **dynamic linking** (keeping OCCT replaceable and shipping its license/notice),
but it is **not MIT-permissive** and carries copyleft-style obligations on the
OCCT component itself. Therefore STEP B-Rep export is documented here as an
**optional future path**, deliberately left unimplemented to keep the current
shipped stack 100% MIT/BSD/Apache-permissive.

---

## Confirmation

The geometry **export** stack (NumPy + scikit-image + trimesh) and the **viewer**
stack (Three.js + OrbitControls) are **entirely MIT/BSD permissive and
product-safe**, with no GPL and no commercial-license dependencies. The only
copyleft tool anywhere near this workflow is FreeCAD (LGPL), which is an external
application the user runs — it is neither imported, linked, nor redistributed by
this code.
