import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from topoheat.projected_to import ProjectedTO
from topoheat.stress_to import StressTO, gradient_check as stress_gc

# 1) stress objective: gradient verified, and a short descent reduces the p-norm
print("thermal stress gradient check:", "%.1e" % stress_gc())
to = StressTO(40, 20, rmin=2.0, heat=0.01, mech_load=1.0, alpha=0.02, pnorm=8.0, volfrac=0.5)
rho = np.full(to.nel, to.volfrac); J0 = None; hist = []
for it in range(25):
    J, dJ = to.full_gradient(rho); hist.append(J)
    # simple bounded steepest descent step (stress objective, volume not active)
    step = 0.05 / (np.max(np.abs(dJ)) + 1e-12)
    rho = np.clip(rho - step * dJ, 0.0, 1.0)
print("stress p-norm: start %.3e -> end %.3e (%.0f%% reduction)" %
      (hist[0], hist[-1], 100 * (1 - hist[-1] / hist[0])))

# 2) compliance mesh convergence: optimized cantilever, filter radius fixed in
#    physical units, point load fixed; final compliance should converge
print("\ncompliance mesh convergence (optimized, fixed physical filter):")
prev = None
for nx, ny, rm in [(30, 15, 1.2), (45, 22, 1.8), (60, 30, 2.4)]:
    o = ProjectedTO(nx, ny, volfrac=0.45, rmin=rm, heat=0.0, mech_load=1.0,
                    alpha=0.0, beta_hs=1.0)   # pure mechanical, point load
    _, rp, h = o.run(iters=120, beta_schedule=True)
    c = h[-1]
    rel = "" if prev is None else "  rel change %.3f" % (abs(c - prev) / prev)
    print("  %dx%d  compliance %.4f%s" % (nx, ny, c, rel)); prev = c

# 3) optimality: design change per iteration drives to zero
print("\noptimality (design change history, clean one way run):")
o = ProjectedTO(48, 24, volfrac=0.45, rmin=2.0, heat=0.005, mech_load=3.0,
                alpha=0.02, beta_hs=1.0)
rho = np.full(o.nel, o.volfrac); changes = []
for it in range(120):
    if it > 0 and it % 30 == 0 and o.beta_hs < 16: o.beta_hs = min(16, o.beta_hs * 2)
    J, dJ = o.full_gradient(rho); rn = o.oc(rho, dJ)
    changes.append(float(np.max(np.abs(rn - rho)))); rho = rn
print("  max design change: it10=%.3f it60=%.3f it120=%.3f" %
      (changes[9], changes[59], changes[-1]))
