"""Reproduce the headline thermoelastic optimization and the gradient check."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from topoheat.topopt_thermoelastic import ThermoelasticTO, gradient_check

maxerr, _ = gradient_check()
print("adjoint vs finite difference, max relative error:", maxerr)

to = ThermoelasticTO(60, 30, volfrac=0.45, rmin=2.4, heat=0.005, mech_load=3.0, alpha=0.02)
rho, rp, hist = to.run(iters=130)
print("strain energy reduction: %.1f x" % (hist[0] / hist[-1]))
print("final volume fraction: %.3f" % np.mean(rp))
print("gray fraction: %.3f" % np.mean(4 * rp * (1 - rp)))
