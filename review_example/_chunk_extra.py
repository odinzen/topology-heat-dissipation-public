import numpy as np, os
import crack_plate3d as cp3, crack_tree as ct, crack_depth_sweep_multi as ms, crack_chunk as cc
CACHE=cp3.CACHE; NX,NY,NZ=ms.NX,ms.NY,ms.NZ
occ=cp3.voxelize(nx=NX,ny=NY,nz=NZ)
plate=cp3.Plate(NX,NY,NZ,volfrac=0.3,rmin=1.5); plate.plate_bc()
foot=occ.any(2); footT=foot.T; gx,gy=ct._grad(foot.astype(float))
fx,fy,lab=cc.LIMB
seed,cells,width=cc.ordered_chunk_cells(foot,gx,gy,fx,fy)
mask=ms._sector_mask(seed,footT)
m0=ms._served_peak(plate,cp3.rp_from_occ(plate,occ),mask)
WIDTHS=[0.35,0.60,0.85]; DEPTHS=cc.DEPTHS
rise=np.full((len(WIDTHS),len(DEPTHS)),np.nan)
for wi,wf in enumerate(WIDTHS):
    chunk=cc.central(cells,wf)
    for di,c in enumerate(DEPTHS):
        d=int(round(c*NZ)); occ_c=occ.copy()
        if d>0:
            for (j,i) in chunk: occ_c[i,j,NZ-d:]=False
        served=ms._served_peak(plate,cp3.rp_from_occ(plate,occ_c),mask)
        rise[wi,di]=100.0*(served/m0-1.0)
        print(f"w={wf} d={c:.2f} rise {rise[wi,di]:.1f}%",flush=True)
    np.savez(os.path.join(CACHE,"crack_chunk_extra.npz"),widths=np.array(WIDTHS),depths=DEPTHS,rise=rise,m0=m0)
print("wrote extra",flush=True)
