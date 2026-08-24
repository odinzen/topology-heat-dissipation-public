import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""TASK 3: short two-material optimization on a 60x30 clamped / heat-sink cantilever.

rho : OC update with the volume constraint (reuse base self.oc).
mu  : bounded gradient-descent (move-limited) clipped to [0,1], no volume constraint.
"""
import numpy as np
from topoheat.multimaterial import TwoMaterialTO

def run(nelx=60, nely=30, iters=120, volfrac=0.45, seed=0):
    to = TwoMaterialTO(nelx, nely, volfrac=volfrac, rmin=2.4,
                       matA=(1.0, 0.01, 1.0),     # stiff, low-expansion
                       matB=(0.4, 0.04, 3.0),     # compliant, high-expansion
                       heat=0.005, mech_load=3.0)
    rho = np.full(to.nel, volfrac)
    mu = np.full(to.nel, 0.5)
    mu_move = 0.1
    hist = []
    for it in range(iters):
        J, dJ_drho, dJ_dmu = to.full_gradient(rho, mu)
        hist.append(J)
        # rho : optimality-criteria with volume constraint (base routine).
        rho_new = to.oc(rho, dJ_drho)
        # mu : bounded steepest descent with adaptive, move-limited step.
        g = dJ_dmu
        gmax = np.max(np.abs(g)) + 1e-30
        step = mu_move / gmax
        mu_new = np.clip(mu - step * g, 0.0, 1.0)
        ch_rho = np.max(np.abs(rho_new - rho))
        ch_mu = np.max(np.abs(mu_new - mu))
        rho, mu = rho_new, mu_new
        rp = to.filter_forward(rho)
        if it % 10 == 0 or it == iters - 1:
            mp = to.filter_forward(mu)
            solid = rp > 0.5
            mbar = mp[solid].mean() if solid.any() else float("nan")
            print(f"it {it:3d}  J {J:.5e}  vol {rp.mean():.3f}  "
                  f"mu_bar(solid) {mbar:.3f}  ch_rho {ch_rho:.3f}  ch_mu {ch_mu:.3f}")
    rho_phys = to.filter_forward(rho)
    mu_phys = to.filter_forward(mu)
    Jf, _ = to.objective(rho_phys, mu_phys)
    return to, rho_phys, mu_phys, Jf, np.array(hist)

if __name__ == "__main__":
    to, rp, mp, Jf, hist = run()
    np.save("mm_rho.npy", rp)
    np.save("mm_mu.npy", mp)
    np.save("mm_hist.npy", hist)
    np.save("mm_shape.npy", np.array([to.nelx, to.nely]))
    solid = rp > 0.5
    print("---- converged ----")
    print(f"final objective J = {Jf:.6e}")
    print(f"volume fraction   = {rp.mean():.4f}")
    print(f"mu mean (all)     = {mp.mean():.4f}")
    print(f"mu mean (solid)   = {mp[solid].mean():.4f}")
    print(f"fraction of solid near A (mu<0.4) = {(mp[solid]<0.4).mean():.3f}")
    print(f"fraction of solid near B (mu>0.6) = {(mp[solid]>0.6).mean():.3f}")
