"""Device-accurate flat cold plate: thin wide plate, heat in on the TOP face (the
chip), coolant drain at a central port on the bottom. The optimizer spreads
conductor IN-PLANE - a flat leaf-vein / river-delta network, not a ball."""
import os, sys, json, time, numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter
import pyamg
from skimage import measure
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))  # the topoheat engine at the repo root
CACHE = os.path.join(HERE, ".cache"); OUT = os.path.join(HERE, "figures")
os.makedirs(CACHE, exist_ok=True); os.makedirs(OUT, exist_ok=True)
from topoheat.topopt3d import HeatTO3D


class HeatTO3D_HR(HeatTO3D):
    def _filter(self, rmin):
        from math import ceil
        R = int(ceil(rmin)); nx, ny, nz = self.nx, self.ny, self.nz
        ix, iy, iz = self.cen[:, 0], self.cen[:, 1], self.cen[:, 2]
        eidx = np.arange(self.nel); rows, cols, vals = [], [], []
        for dx in range(-R, R + 1):
            for dy in range(-R, R + 1):
                for dz in range(-R, R + 1):
                    w = rmin - (dx * dx + dy * dy + dz * dz) ** 0.5
                    if w <= 0: continue
                    jx, jy, jz = ix + dx, iy + dy, iz + dz
                    ok = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny) & (jz >= 0) & (jz < nz)
                    rows.append(eidx[ok]); cols.append(jz[ok] * nx * ny + jy[ok] * nx + jx[ok])
                    vals.append(np.full(int(ok.sum()), w))
        self.H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(self.nel, self.nel))
        self.Hs = np.asarray(self.H.sum(1)).ravel()

    def solve(self, rp):
        kv = self.k_(rp)
        sKt = (self.KET0.ravel()[None, :] * kv[:, None]).ravel()
        KT = sp.csr_matrix((sKt, (self.iKt, self.jKt)), shape=(self.nnode, self.nnode))
        Kff = KT[self.freeT][:, self.freeT].tocsr()
        ml = pyamg.smoothed_aggregation_solver(Kff, max_coarse=800)
        x = ml.solve(self.q[self.freeT], tol=1e-8, accel="cg", maxiter=300)
        T = np.zeros(self.nnode); T[self.freeT] = x
        return dict(T=T, KT=KT)


NX = NY = 96; NZ = 8           # wide, thin plate
t0 = time.time()
to = HeatTO3D_HR(NX, NY, NZ, volfrac=0.12, rmin=2.3)   # scarce material -> forced to branch
print(f"built {to.nel} elements {time.time()-t0:.0f}s", flush=True)
# heat in on the TOP face (chip); coolant drain = central port on the bottom
top = [to.nid(ix, iy, NZ) for iy in range(NY + 1) for ix in range(NX + 1)]
c = NX // 2
drain = [to.nid(ix, iy, iz) for iz in range(0, 3)
         for iy in range(c - 2, c + 3) for ix in range(c - 2, c + 3)]
to.ftemp = np.array(sorted(set(drain)))
to.freeT = np.setdiff1d(np.arange(to.nnode), to.ftemp)
to.q = np.zeros(to.nnode); to.q[np.array(top)] = 2e-3; to.q[to.ftemp] = 0.0

rho = np.full(to.nel, to.volfrac); hist = []
for it in range(55):
    J, dJ = to.grad(rho); hist.append(J); rho = to.oc(rho, dJ)
    if it % 10 == 0: print(f"  it {it} J {J:.4g} t {time.time()-t0:.0f}s", flush=True)
rp = to.filt(rho)
print(f"done {time.time()-t0:.0f}s; compliance {hist[0]:.3g}->{hist[-1]:.3g}", flush=True)

field = rp.reshape(NZ, NY, NX).transpose(2, 1, 0)
field = gaussian_filter(field, 0.8); field = np.pad(field, 1)
verts, faces, normals, _ = measure.marching_cubes(field, level=0.5, step_size=1)
print(f"mesh {len(verts)} verts {len(faces)} faces", flush=True)
json.dump({"verts": np.round(verts, 3).tolist(), "faces": faces.astype(int).tolist(),
           "bbox": [NX, NY, NZ]}, open(os.path.join(CACHE, "flatplate_mesh.json"), "w"))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
fig = plt.figure(figsize=(15, 6.5))
for i, (el, az, ttl) in enumerate([(90, -90, "top-down (the leaf-vein spreader)"),
                                    (22, -60, "perspective (a thin flat plate)")]):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    m = Poly3DCollection(verts[faces], alpha=1.0); m.set_facecolor("#6b7a8f"); m.set_edgecolor((0, 0, 0, 0))
    ax.add_collection3d(m)
    ax.set_xlim(0, NX); ax.set_ylim(0, NY); ax.set_zlim(0, NZ)
    ax.set_box_aspect((NX, NY, NZ * 4)); ax.view_init(elev=el, azim=az); ax.set_axis_off()
    ax.set_title(ttl, fontsize=10)
fig.suptitle("Flat cold-plate heat spreader (chip on top, coolant port at center): in-plane branching network", fontsize=11)
fig.savefig(os.path.join(OUT, "flatplate.png"), dpi=175, bbox_inches="tight")
print("wrote flatplate_mesh.json and flatplate.png; total", f"{time.time()-t0:.0f}s", flush=True)
