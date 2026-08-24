import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time, numpy as np
from topoheat.topopt3d import TO3D
from topoheat.scale3d import TO3DIter

grids = [(24,12,6),(40,20,10),(56,28,14)]
print("%-12s %8s %10s %12s %10s"%("grid","ndof","direct s","iter s","cg(mech)"))
for nx,ny,nz in grids:
    rho_seed = None
    nd = 3*(nx+1)*(ny+1)*(nz+1)
    # iterative: one forward solve timing
    it = TO3DIter(nx,ny,nz,volfrac=0.3,rmin=1.5)
    rp = np.full(it.nel, it.volfrac)
    t=time.time(); it.solve(it.filt(rp)); ti=time.time()-t
    cgm = it.stats.get("mech",(None,))[0]
    # direct: try, but cap by catching slowness via time
    td = None
    if nd <= 30000:
        d = TO3D(nx,ny,nz,volfrac=0.3,rmin=1.5)
        t=time.time(); d.solve(d.filt(np.full(d.nel,d.volfrac))); td=time.time()-t
    print("%-12s %8d %10s %12.2f %10s"%("%dx%dx%d"%(nx,ny,nz), nd,
          ("%.2f"%td) if td else "skipped(mem)", ti, cgm))
