import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time, numpy as np
from topoheat.topopt_twoway import TwoWayTO
t0=time.time()
to=TwoWayTO(48,24,volfrac=0.45,rmin=2.0,beta_couple=2.0,
            heat=2e-3,mech_load=0.05,alpha=0.02,relax=0.6,fp_max=80)
rho,rp,hist=to.run(iters=60,tol=1e-3)
print("elapsed %.1fs iters=%d Jred=%.1fx vol=%.3f gray=%.3f"%(
    time.time()-t0,len(hist),hist[0]/hist[-1],np.mean(rp),np.mean(4*rp*(1-rp))))
np.save("twoway_rho.npy",rp); np.save("twoway_shape.npy",np.array([48,24]))
