"""Crisp two way coupled optimized design.

Two way solver plus Heaviside projection. The mechanical to thermal load ratio
matches the clean one way study; both loads are scaled down together, which
leaves the design unchanged but keeps deformations small so the two way fixed
point stays stable.
"""
import time, numpy as np
from .projected_to import ProjectedTO
from .topopt_twoway import TwoWayTO

class ProjectedTwoWay(ProjectedTO, TwoWayTO):
    pass

if __name__ == "__main__":
    t0 = time.time()
    to = ProjectedTwoWay(48, 24, volfrac=0.45, rmin=2.0, beta_hs=1.0,
                         beta_couple=2.0, relax=0.6, heat=2e-3, mech_load=0.05,
                         alpha=0.02, fp_max=80)
    rho, rp, hist = to.run(iters=200, beta_schedule=True)
    print("elapsed %.1fs iters=%d Jred=%.1fx vol=%.3f gray=%.3f beta=%.0f" %
          (time.time() - t0, len(hist), hist[0] / hist[-1], np.mean(rp),
           np.mean(4 * rp * (1 - rp)), to.beta_hs))
    np.save("tw_rp.npy", rp); np.save("tw_shape.npy", np.array([48, 24]))
