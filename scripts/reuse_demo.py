import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time, numpy as np
from topoheat.scale3d import TO3DIter, correctness, gradient_check
print("correctness (PCG vs direct):", "%.1e, %.1e"%correctness())
print("gradient with iterative solver:", "%.1e"%gradient_check())
def timeit(rebuild_every, iters=4):
    to=TO3DIter(32,16,8,volfrac=0.25,rmin=1.6,alpha=0.02,heat=1.5e-3,cg_tol=1e-7)
    to.rebuild_every=rebuild_every
    rho=np.full(to.nel,to.volfrac); cg_counts=[]
    t=time.time()
    for _ in range(iters):
        J,dJ=to.grad(rho); rho=to.oc(rho,dJ); cg_counts.append(to.stats["mech"][0])
    return (time.time()-t)/iters, cg_counts
print("ndof = %d"%(3*33*17*9))
for re_ in [1,4]:
    tpit,cgc=timeit(re_)
    print("rebuild_every=%d : %.2f s/iter  cg(mech) per iter=%s"%(re_,tpit,cgc))
