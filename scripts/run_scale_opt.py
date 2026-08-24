import sys, time, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from topoheat.scale3d import TO3DIter

NX,NY,NZ=40,20,10
BUDGET=26.0
to=TO3DIter(NX,NY,NZ,volfrac=0.2,rmin=1.6,alpha=0.02,heat=1.5e-3,mech=1.0,cg_tol=1e-7)
ckpt="scaleopt_rho.npy"; hk="scaleopt_hist.npy"
if os.path.exists(ckpt):
    rho=np.load(ckpt); hist=list(np.load(hk))
    # warm start fields not persisted; fine
else:
    rho=np.full(to.nel,to.volfrac); hist=[]
t0=time.time(); n0=len(hist)
while time.time()-t0 < BUDGET:
    J,dJ=to.grad(rho); hist.append(J); rho=to.oc(rho,dJ)
np.save(ckpt,rho); np.save(hk,np.array(hist))
rp=to.filt(rho)
print("grid %dx%dx%d ndof=%d | iters this call=%d total=%d | J=%.4e vol=%.3f gray=%.3f | cg mech=%s"%(
    NX,NY,NZ,to.ndof,len(hist)-n0,len(hist),hist[-1],np.mean(rp),np.mean(4*rp*(1-rp)),to.stats.get("mech",("?",))[0]))
np.save("scaleopt_rp.npy",rp); np.save("scaleopt_shape.npy",np.array([NX,NY,NZ]))
