"""Canonical MBB beam benchmark, pure mechanical compliance minimization.

Reuses the verified Q4 stiffness and the density filter. Boundary conditions
follow the standard half beam: symmetry on the left edge, a vertical roller at
the lower right corner, a downward unit load at the upper left corner. With the
thermal expansion set to zero the objective J = u^T K u is the compliance.
"""
import numpy as np
from .topopt_thermoelastic import ThermoelasticTO


class MBB(ThermoelasticTO):
    def _build_bc(self):
        coords = self.mesh.coords()
        L, Hh = self.nelx, self.nely
        left = np.where(np.isclose(coords[:, 0], 0.0))[0]
        botright = self.mesh.nid(self.nelx, 0)
        topleft = self.mesh.nid(0, self.nely)
        # symmetry: u_x = 0 on the left edge; roller: u_y = 0 at lower right
        self.fixed_mech = np.concatenate([2 * left, [2 * botright + 1]])
        self.free_mech = np.setdiff1d(np.arange(self.ndof), self.fixed_mech)
        self.f_mech = np.zeros(self.ndof)
        self.f_mech[2 * topleft + 1] = -1.0
        # thermal unused; keep a trivial sink so the solve is well posed
        self.fixed_temp = left
        self.free_temp = np.setdiff1d(np.arange(self.nnode), self.fixed_temp)
        self.q = np.zeros(self.nnode)


def run(nelx=60, nely=20, volfrac=0.5, penal=3.0, rmin=1.5, iters=120):
    to = MBB(nelx, nely, volfrac=volfrac, penal=penal, rmin=rmin,
             floor=1e-9, alpha=0.0, heat=0.0)
    rho, rp, hist = to.run(iters=iters, tol=1e-3)
    return to, rp, hist


if __name__ == "__main__":
    import time
    t0 = time.time()
    to, rp, hist = run()
    print("elapsed %.1fs iters=%d" % (time.time() - t0, len(hist)))
    print("MBB compliance c = %.4f" % hist[-1])
    print("volume fraction = %.3f" % np.mean(rp))
    print("gray fraction = %.3f" % np.mean(4 * rp * (1 - rp)))
    np.save("mbb_rho.npy", rp); np.save("mbb_hist.npy", np.array(hist))
