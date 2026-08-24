import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time, numpy as np
from topoheat.projected_to import ProjectedTO
t0=time.time()
to=ProjectedTO(60,30,volfrac=0.45,rmin=2.4,beta_hs=1.0,
               heat=0.005,mech_load=3.0,alpha=0.02)
rho,rp,hist=to.run(iters=180,beta_schedule=True)
print("elapsed %.1fs Jred=%.1fx vol=%.3f gray=%.3f final_beta=%.1f"%(
    time.time()-t0,hist[0]/hist[-1],np.mean(rp),np.mean(4*rp*(1-rp)),to.beta_hs))
np.save("crisp_rho.npy",rp); np.save("crisp_hist.npy",np.array(hist))
