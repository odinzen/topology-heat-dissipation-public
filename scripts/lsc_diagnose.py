"""Diagnostic: exact Schur complement versus diagonal and LSC approximations.

S = K_T + B K_u^-1 A. We compare three approximations of the coupling term
M = B K_u^-1 A on a small mesh where the dense inverse is affordable:
  diagonal : M ~ B diag(K_u)^-1 A
  LSC      : M ~ (BA) (B K_u A)^-1 (BA)   (least squares commutator)
and report the relative error of each, plus the eigenvalue spread of the
preconditioned Schur, which is what drives GMRES.
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from topoheat.monolithic import Monolithic
from topoheat.block_solver import _blocks

def report(nelx, nely, beta):
    m = Monolithic(nelx, nely, beta_couple=beta, heat=2e-3, mech_load=0.03,
                   alpha=0.03, volfrac=0.5)
    u, T, res = m.newton()
    Kuu, Aut, Btu, Ktt = (b.toarray() for b in _blocks(m, u, T))
    Kui = np.linalg.inv(Kuu)
    M = Btu @ Kui @ Aut
    S = Ktt + M
    Mdiag = Btu @ np.diag(1.0 / np.diag(Kuu)) @ Aut
    BA = Btu @ Aut
    BKuA = Btu @ Kuu @ Aut
    Mlsc = BA @ np.linalg.inv(BKuA) @ BA
    nM = np.linalg.norm(M); nK = np.linalg.norm(Ktt)
    ed = np.linalg.norm(Mdiag - M) / nM
    el = np.linalg.norm(Mlsc - M) / nM

    def spread(Shat):
        # generalized eigenvalues of Shat^-1 S, real parts
        ev = np.linalg.eigvals(np.linalg.solve(Shat, S))
        ev = np.abs(ev)
        return ev.min(), ev.max(), ev.max() / ev.min()
    sd = spread(Ktt + Mdiag)
    sl = spread(Ktt + Mlsc)
    print("%dx%d beta=%d | ||M||/||K_T||=%.2f | rel err diag=%.3f lsc=%.3f"
          % (nelx, nely, beta, nM / nK, ed, el))
    print("   precond spectrum (min,max,ratio)  diag: %.2f %.2f %.1f   lsc: %.2f %.2f %.1f"
          % (sd[0], sd[1], sd[2], sl[0], sl[1], sl[2]))

if __name__ == "__main__":
    for beta in [20, 60]:
        report(16, 8, beta)
