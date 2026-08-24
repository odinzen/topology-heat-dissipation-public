import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time,numpy as np
from topoheat.topopt3d import TO3D
t0=time.time()
to=TO3D(24,12,6,volfrac=0.25,rmin=1.5,alpha=0.02,heat=2e-3,mech=1.0)
rho,rp,hist=to.run(iters=45)
print("elapsed %.1fs iters=%d Jred=%.1fx vol=%.3f gray=%.3f"%(
    time.time()-t0,len(hist),hist[0]/hist[-1],np.mean(rp),np.mean(4*rp*(1-rp))))
np.save("d3_rho.npy",rp); np.save("d3_shape.npy",np.array([24,12,6]))
