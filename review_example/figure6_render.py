"""Clean, edge-free shaded render of the flat cold-plate spreader for the manuscript.
Per-face lighting, no wireframe, no baked text (caption lives in the document)."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache"); OUT = os.path.join(HERE, "figures")
d = json.load(open(os.path.join(CACHE, "flatplate_mesh.json")))
verts = np.array(d["verts"]); faces = np.array(d["faces"]); NX, NY, NZ = d["bbox"]
tris = verts[faces]
n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
light = np.array([0.35, 0.45, 0.82]); light /= np.linalg.norm(light)
inten = np.clip(n @ light, 0, 1) * 0.72 + 0.28
base = np.array([0.46, 0.54, 0.66])
colors = np.clip(inten[:, None] * base[None, :], 0, 1)

fig = plt.figure(figsize=(13.5, 6))
for i, (el, az) in enumerate([(90, -90), (24, -58)]):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    p = Poly3DCollection(tris, facecolors=colors, edgecolor="none", antialiased=False)
    ax.add_collection3d(p)
    ax.set_xlim(0, NX); ax.set_ylim(0, NY); ax.set_zlim(0, NZ)
    ax.set_box_aspect((NX, NY, NZ * (1 if el == 90 else 5)))
    ax.view_init(elev=el, azim=az); ax.set_axis_off()
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0)
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "flatplate_clean.png"), dpi=210, bbox_inches="tight")
print("wrote flatplate_clean.png")
