"""Pristine die-face temperature and the chunk cell sets that Figure 12 (c, d) overlays.

Run after crack_chunk.py; writes .cache/crack_chunk_maps.npz for crack_chunk_fig.py.
"""
import numpy as np, os
import crack_plate3d as cp3, crack_tree as ct, crack_depth_sweep_multi as ms
import crack_chunk as cc
CACHE=cp3.CACHE; NX,NY,NZ=ms.NX,ms.NY,ms.NZ
occ=cp3.voxelize(nx=NX,ny=NY,nz=NZ)
plate=cp3.Plate(NX,NY,NZ,volfrac=0.3,rmin=1.5); plate.plate_bc()
foot=occ.any(2); gx,gy=ct._grad(foot.astype(float))
fx,fy,lab=cc.LIMB
seed,cells,width=cc.ordered_chunk_cells(foot,gx,gy,fx,fy)
# pristine top face
Tp=plate.solve(cp3.rp_from_occ(plate,occ))["T"].reshape(NZ+1,NY+1,NX+1)[NZ][:NY,:NX]
# chunk cells for width 0.25 and 1.0 (to mark)
q=cc.central(cells,0.25); f=cc.central(cells,1.0)
np.savez(os.path.join(CACHE,"crack_chunk_maps.npz"),
         Tp=Tp, seed=np.array(seed),
         chunk_q=np.array(list(q)), chunk_f=np.array(list(f)))
print("pristine served region and saved. limb width cells:",round(width,1))
print("quarter cells:",len(q),"full cells:",len(f))