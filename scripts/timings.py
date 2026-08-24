"""Measure wall times for the key operations. Single machine, single thread."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time, numpy as np
from topoheat.topopt_thermoelastic import ThermoelasticTO
from topoheat.topopt_twoway import TwoWayTO
from topoheat.topopt3d import TO3D
from topoheat.scale3d import TO3DIter

def t(f, n=1):
    f(); t0=time.time()
    for _ in range(n): f()
    return (time.time()-t0)/n

rows=[]

# 2D forward solve and one optimization iteration
to=ThermoelasticTO(60,30,rmin=2.4,heat=0.005,mech_load=3.0,alpha=0.02)
rho=np.full(to.nel,to.volfrac)
rows.append(("2D forward solve, 60 by 30 (3782 dof)", t(lambda: to.solve(to.filter_forward(rho)),3)))
rows.append(("2D optimization iteration (gradient + update)", t(lambda: to.oc(rho, to.full_gradient(rho)[1]),3)))

# 2D full optimization
t0=time.time(); to.run(iters=120); rows.append(("2D full optimization, 120 iterations", time.time()-t0))

# two way forward solve
tw=TwoWayTO(48,24,beta_couple=2.0,relax=0.6,heat=2e-3,mech_load=0.05,alpha=0.02,fp_max=80)
rho2=np.full(tw.nel,tw.volfrac)
rows.append(("2D two way fixed point solve, 48 by 24", t(lambda: tw.solve(tw.filter_forward(rho2)),2)))

# 3D direct vs AMG
d3=TO3D(24,12,6,rmin=1.5,alpha=0.02,heat=2e-3); r3=np.full(d3.nel,d3.volfrac)
rows.append(("3D forward solve, 24x12x6 (6825 dof), direct", t(lambda: d3.solve(d3.filt(r3)),2)))
i3=TO3DIter(24,12,6,rmin=1.5,alpha=0.02,heat=2e-3,cg_tol=1e-7); ri=np.full(i3.nel,i3.volfrac)
rows.append(("3D forward solve, 24x12x6, AMG preconditioned CG", t(lambda: i3.solve(i3.filt(ri)),2)))

# 3D large grid AMG (direct infeasible)
big=TO3DIter(56,28,14,rmin=1.5,alpha=0.02,heat=2e-3,cg_tol=1e-7); rb=np.full(big.nel,big.volfrac)
t0=time.time(); big.solve(big.filt(rb)); rows.append(("3D forward solve, 56x28x14 (74385 dof), AMG (direct infeasible)", time.time()-t0))

print("%-58s %10s"%("operation","seconds"))
for name,sec in rows:
    print("%-58s %10.3f"%(name,sec))
import json; json.dump({n:round(s,3) for n,s in rows}, open("timings.json","w"), indent=2)
