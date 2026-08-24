# Manifold mesh hardening + manufacturing features

## License posture (product-safe)
All libraries used here are permissive and safe to embed in a commercial product:

| Library        | License    | Role |
|----------------|------------|------|
| **manifold3d** | Apache-2.0 | guaranteed-manifold CSG kernel |
| trimesh        | MIT        | mesh load / export (STL/OBJ/3MF/GLB) |
| numpy          | BSD-3      | array math |
| scikit-image   | BSD-3      | marching-cubes fallback (only if mesh files missing) |
| matplotlib     | PSF/BSD    | greyscale preview render |

No GPL, no LGPL, no commercial/closed components. `manifold3d` (Apache-2.0) is the
core differentiator: every boolean it produces is provably an oriented 2-manifold
("watertight") or it returns an error status, which a raw triangle-soup mesh cannot
guarantee.

## What was demonstrated
Loaded the optimized watertight design (`design3d.ply`, 2368 verts / 4752 tris,
volume 1391.13, genus 5) and rebuilt it as a guaranteed `Manifold`. Then the
following Constructive Solid Geometry features were applied, each verified to stay
a single non-empty manifold:

1. **Trim / build envelope** — intersection (`a ^ b`) of the design with an
   axis-aligned box inset 0.5 mm inside the extents. Volume 1391.13 -> 1003.05.
2. **Mounting holes** — two `Manifold.cylinder` (r = 0.8, axis +z) subtracted
   (`a - b`) through the two solid legs at the clamped low-x face. Each hole
   removed 18.044 of material (confirmed they pass through solid, not the hollow
   middle gap). Volume 1003.05 -> 966.96; genus rose 5 -> 7 (two new through-holes).
3. **Mounting flange / base tab** — a `Manifold.cube` (4 x 19 x 2) unioned
   (`a + b`) under the clamped end, spanning the full y-extent so it fuses both
   legs into ONE body. Volume 966.96 -> 1108.66; `decompose()` reports 1 component.

Final featured part: 1436 verts / 2892 tris, volume **1108.66**, single manifold,
watertight on STL re-load.

## Confirmed manifold3d API (version 3.5.1, installed)
`manifold3d` exposes no `__version__`; use `importlib.metadata.version("manifold3d")`
-> `"3.5.1"`.

```python
import manifold3d as m3
import numpy as np

# --- mesh -> Manifold (the guaranteed-manifold ingest) ---
mesh = m3.Mesh(
    vert_properties = verts.astype(np.float32),   # shape (N, >=3); xyz in cols 0:3
    tri_verts       = faces.astype(np.uint32),    # shape (M, 3)
)
man = m3.Manifold(mesh)            # ctor overload Manifold(mesh); empty + Error status if not 2-manifold
assert man.status() == m3.Error.NoError

# --- primitives ---
m3.Manifold.cube([sx, sy, sz], center=False)                 # box, first octant by default
m3.Manifold.cylinder(height, radius_low, radius_high=-1.0,   # axis along +z
                     circular_segments=0, center=False)

# --- transforms ---
man.translate([x, y, z])           # also .rotate, .scale, .transform, .mirror

# --- booleans (operators) ---
union        = a + b
difference   = a - b               # tail removed from head
intersection = a ^ b               # VERIFIED: cube[0..2]^cube[1..3] -> volume 1.0 (the overlap)
# (also m3.Manifold.batch_boolean([list], op=m3.OpType.Add/Subtract/Intersect))

# --- properties ---
man.volume(); man.surface_area(); man.num_tri(); man.num_vert()
man.genus(); man.is_empty(); man.status(); man.bounding_box()  # (minx,miny,minz,maxx,maxy,maxz)
man.decompose()                    # list of connected components; len==1 => single solid

# --- Manifold -> mesh (round-trip out) ---
out = man.to_mesh()                # m3.Mesh
xyz   = np.asarray(out.vert_properties)[:, :3]   # (N,3) float32 (may be wider; slice 0:3)
faces = np.asarray(out.tri_verts)                # (M,3) int32
```

### Notes / deviations from the original hints
- `Mesh.__init__` keyword is `vert_properties` (float32, 2-D) and `tri_verts`
  (uint32, (M,3)) — exactly as hinted. There is **no** `Manifold.from_mesh`;
  the constructor overload `Manifold(mesh)` is the supported path.
- The boolean **`^` is intersection** in this build (verified numerically), matching
  the task hint. (It is NOT a symmetric-difference/XOR.)
- `to_mesh().tri_verts` comes back as **int32** (not uint32); cast as needed.
- `bounding_box()` returns a flat 6-tuple `(minx,miny,minz, maxx,maxy,maxz)`.
- STL exported via trimesh is **binary** by default and re-loads `watertight=True`.

## Reproduce
```
python3 manifold_features.py
```
Outputs into this folder: `manifold_part.stl/.obj/.3mf/.glb` and
`manifold_part_preview.png`.
