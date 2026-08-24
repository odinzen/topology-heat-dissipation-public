"""Three field SIMP: density filter followed by a smoothed Heaviside projection.

Adds a projection between the filtered field and the physics, which drives the
design toward clean solid and void. A beta continuation sharpens the projection
across the run. The sensitivity carries the projection derivative. Verified by a
finite difference check.
"""
import numpy as np
from .topopt_thermoelastic import ThermoelasticTO


class ProjectedTO(ThermoelasticTO):
    def __init__(self, *args, beta_hs=1.0, eta=0.5, **kw):
        super().__init__(*args, **kw)
        self.beta_hs, self.eta = beta_hs, eta

    def _project(self, rt):
        b, e = self.beta_hs, self.eta
        num = np.tanh(b * e) + np.tanh(b * (rt - e))
        den = np.tanh(b * e) + np.tanh(b * (1 - e))
        return num / den

    def _dproject(self, rt):
        b, e = self.beta_hs, self.eta
        den = np.tanh(b * e) + np.tanh(b * (1 - e))
        return b * (1 - np.tanh(b * (rt - e)) ** 2) / den

    def _rtilde(self, rho):
        return np.asarray(self.H @ rho).ravel() / self.Hs

    def filter_forward(self, rho):
        return self._project(self._rtilde(rho))

    def full_gradient(self, rho):
        rt = self._rtilde(rho)
        rbar = self._project(rt)
        J, dphys, st = self.sensitivity(rbar)
        dJ_drt = dphys * self._dproject(rt)
        return J, np.asarray(self.H @ (dJ_drt / self.Hs)).ravel()

    def oc(self, rho, dJ):
        rt = self._rtilde(rho)
        dv = np.asarray(self.H @ (self._dproject(rt) / self.Hs)).ravel()
        l1, l2, move = 0.0, 1e9, 0.2
        rho_new = rho.copy()
        while (l2 - l1) / (0.5 * (l1 + l2) + 1e-30) > 1e-4:
            lmid = 0.5 * (l1 + l2)
            be = np.maximum(0.0, -dJ / dv / lmid)
            rho_cand = np.maximum(0.0, np.maximum(rho - move,
                       np.minimum(1.0, np.minimum(rho + move, rho * np.sqrt(be)))))
            if np.mean(self.filter_forward(rho_cand)) > self.volfrac:
                l1 = lmid
            else:
                l2 = lmid
            rho_new = rho_cand
        return rho_new

    def run(self, iters=160, tol=1e-3, beta_schedule=True, verbose=False):
        rho = np.full(self.nel, self.volfrac)
        history = []
        self.converged = False
        for it in range(iters):
            if beta_schedule and it > 0 and it % 30 == 0 and self.beta_hs < 16:
                self.beta_hs = min(16.0, self.beta_hs * 2.0)
            J, dJ = self.full_gradient(rho)
            history.append(J)
            rho_new = self.oc(rho, dJ)
            change = float(np.max(np.abs(rho_new - rho)))
            rho = rho_new
            if verbose:
                print("it %3d J %.4e beta %.1f gray %.3f ch %.3f" %
                      (it, J, self.beta_hs, np.mean(4 * self.filter_forward(rho) * (1 - self.filter_forward(rho))), change))
            # Early stop only once the projection is fully sharpened, so the
            # beta continuation is never cut short. Leaves fixed-iteration runs
            # (e.g. beta not yet at its cap) unchanged.
            if change < tol and (not beta_schedule or self.beta_hs >= 16) and it > 10:
                self.converged = True
                break
        return rho, self.filter_forward(rho), history


def gradient_check(seed=2, nelx=12, nely=6, beta_hs=4.0):
    rng = np.random.default_rng(seed)
    to = ProjectedTO(nelx, nely, rmin=1.5, beta_hs=beta_hs,
                     heat=0.005, mech_load=1.0, alpha=0.02)
    rho = rng.uniform(0.3, 0.9, to.nel)
    J0, g = to.full_gradient(rho)
    h = 1e-6
    idx = rng.choice(to.nel, size=8, replace=False)
    errs = []
    for i in idx:
        rp = rho.copy(); rp[i] += h
        rm = rho.copy(); rm[i] -= h
        Jp, _ = to.objective(to.filter_forward(rp))
        Jm, _ = to.objective(to.filter_forward(rm))
        errs.append(abs((Jp - Jm) / (2 * h) - g[i]) / (abs((Jp - Jm) / (2 * h)) + 1e-30))
    return float(max(errs))


if __name__ == "__main__":
    print("projected gradient check max relative error:", gradient_check())
