"""Monolithic Newton solve of the coupled thermoelastic problem.

The staggered fixed point contracts only when the deformation to conductivity
feedback is weak; for strong feedback it diverges. A monolithic Newton step on
the coupled residual, with the full Jacobian including the cross blocks,
converges regardless. A smooth exponential conductivity, k_eff = k_base
exp(beta vol), keeps the operator positive and differentiable so the comparison
is clean.

Unknowns u, T on the free sets. Residuals:
  R_u = K_u u - f_mech - f_th(T)
  R_T = K_T(u) T - q
Jacobian blocks: d R_u/d u = K_u, d R_u/d T = -A, d R_T/d u = B, d R_T/d T = K_T.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from .topopt_twoway import TwoWayTO, M_DIV
from .topopt_thermoelastic import KE0, KET0, G0


class Monolithic:
    def __init__(self, nelx, nely, beta_couple=20.0, **kw):
        b = TwoWayTO(nelx, nely, beta_couple=beta_couple, **kw)
        self.b = b; self.beta = beta_couple
        self.rp = np.full(b.nel, b.volfrac)
        self.k_base = b.k_of(self.rp); self.Ev = b.E_of(self.rp); self.Ef = b.Ef_of(self.rp)
        sK = (KE0.ravel()[None, :] * self.Ev[:, None]).ravel()
        self.KU = sp.csr_matrix((sK, (b.iK, b.jK)), shape=(b.ndof, b.ndof))
        self.A = self._A()

    def _keff(self, vol): return self.k_base * np.exp(np.clip(self.beta * vol, -30, 30))
    def _vol(self, u): return (u[self.b.edof] * M_DIV[None, :]).sum(1)

    def _KT(self, k_eff):
        sKt = (KET0.ravel()[None, :] * k_eff[:, None]).ravel()
        return sp.csr_matrix((sKt, (self.b.iKt, self.b.jKt)), shape=(self.b.nnode, self.b.nnode))

    def _fth(self, T):
        b = self.b; Te = T[b.edofT].mean(1); coef = self.Ef * b.alpha * (Te - b.tref)
        f = np.zeros(b.ndof); np.add.at(f, b.edof.ravel(), (coef[:, None] * G0[None, :]).ravel())
        return f

    def _A(self):
        b = self.b; sc = self.Ef * b.alpha * 0.25
        # local coupling block: outer(G0, ones4) scaled per element by sc.
        # Mirror the iK/jK kron pattern. Loop order is i (0..7) outer, nb (0..3)
        # inner, so rows repeat edof 4x along columns and cols tile edofT 8x.
        rows = np.repeat(b.edof, 4, axis=1).ravel()
        cols = np.tile(b.edofT, (1, 8)).ravel()
        vals = np.repeat(sc[:, None] * G0[None, :], 4, axis=1).ravel()
        return sp.csr_matrix((vals, (rows, cols)), shape=(b.ndof, b.nnode))

    def _B(self, T, k_eff):
        b = self.b
        # kv_e = beta * k_eff[e] * (KET0 @ T[edofT[e]]), length-4 per element.
        # Loop order a (0..3) outer, j (0..7) inner: rows repeat edofT 8x along
        # columns, cols tile edof 4x, vals are outer(kv, M_DIV) per element.
        Tel = T[b.edofT]                                   # nel x 4
        # einsum "ij,ej->ei" reproduces the per-element KET0 @ T_el contraction
        # bit-for-bit (a batched gemm would reorder the reduction and drift ~1 ULP).
        kv = self.beta * k_eff[:, None] * np.einsum("ij,ej->ei", KET0, Tel)  # nel x 4
        rows = np.repeat(b.edofT, 8, axis=1).ravel()
        cols = np.tile(b.edof, (1, 4)).ravel()
        vals = (kv[:, :, None] * M_DIV[None, None, :]).reshape(b.nel, -1).ravel()
        return sp.csr_matrix((vals, (rows, cols)), shape=(b.nnode, b.ndof))

    def newton_continuation(self, target, steps=6, tol=1e-10):
        betas=np.linspace(target/steps, target, steps)
        u=np.zeros(self.b.ndof); T=np.zeros(self.b.nnode); total=0
        for bt in betas:
            self.beta=bt
            u,T,res=self.newton(tol=tol,maxit=40,u=u,T=T); total+=len(res)-1
        return u,T,total,self.residual(u,T)

    def residual(self, u, T):
        b=self.b; k_eff=self._keff(self._vol(u)); KT=self._KT(k_eff)
        Ru=(self.KU@u-b.f_mech-self._fth(T))[b.free_mech]
        RT=(KT@T-b.q)[b.free_temp]
        return np.sqrt(Ru@Ru+RT@RT)

    def staggered(self, relax=0.5, maxit=200, tol=1e-10):
        b=self.b; KUff=self.KU[b.free_mech][:,b.free_mech].tocsc(); lu=spla.splu(KUff)
        vol=np.zeros(b.nel); u=np.zeros(b.ndof); T=np.zeros(b.nnode)
        for it in range(maxit):
            k_eff=self._keff(vol); KT=self._KT(k_eff)
            T=np.zeros(b.nnode); T[b.free_temp]=spla.spsolve(KT[b.free_temp][:,b.free_temp].tocsc(),b.q[b.free_temp])
            u=np.zeros(b.ndof); u[b.free_mech]=lu.solve((b.f_mech+self._fth(T))[b.free_mech])
            vn=self._vol(u); 
            if np.max(np.abs(vn-vol))<tol: vol=vn; break
            if np.any(~np.isfinite(vn)) or np.max(np.abs(vn))>1e3: return u,T,it,np.inf
            vol=(1-relax)*vol+relax*vn
        return u,T,it,self.residual(u,T)

    def newton(self, tol=1e-10, maxit=40, u=None, T=None):
        b=self.b; fm=b.free_mech; ft=b.free_temp
        u=np.zeros(b.ndof) if u is None else u.copy()
        T=np.zeros(b.nnode) if T is None else T.copy(); res=[]
        for _ in range(maxit):
            vol=self._vol(u); k_eff=self._keff(vol); KT=self._KT(k_eff)
            Ru=(self.KU@u-b.f_mech-self._fth(T))[fm]; RT=(KT@T-b.q)[ft]
            rn=np.sqrt(Ru@Ru+RT@RT); res.append(rn)
            if rn<tol: break
            B=self._B(T,k_eff)
            J=sp.bmat([[self.KU[fm][:,fm],-self.A[fm][:,ft]],[B[ft][:,fm],KT[ft][:,ft]]],format="csc")
            d=spla.spsolve(J,-np.concatenate([Ru,RT]))
            du=d[:fm.size]; dT=d[fm.size:]
            a=1.0
            while a>1e-4:
                ut=u.copy(); Tt=T.copy(); ut[fm]+=a*du; Tt[ft]+=a*dT
                if self.residual(ut,Tt) < (1-1e-4*a)*rn: break
                a*=0.5
            u[fm]+=a*du; T[ft]+=a*dT
        return u,T,res


if __name__=="__main__":
    print("%-6s %-26s %-30s"%("beta","staggered (iters,resid)","newton (iters,resid,match)"))
    for beta in [5,20,50]:
        m=Monolithic(16,8,beta_couple=beta,heat=2e-3,mech_load=0.03,alpha=0.03,volfrac=0.5)
        us,Ts,its,sr=m.staggered(); un,Tn,res=m.newton()
        mt=np.linalg.norm(un-us)/(np.linalg.norm(us)+1e-30)
        print("%-6d (%3d, %-9.2e) (%3d, %-9.2e, %.1e)"%(beta,its,sr,len(res)-1,res[-1],mt))
    print("--- strong feedback: staggered diverges, Newton with continuation converges ---")
    for beta in [100]:
        m=Monolithic(16,8,beta_couple=beta,heat=2e-3,mech_load=0.03,alpha=0.03,volfrac=0.5)
        us,Ts,its,sr=m.staggered()
        m2=Monolithic(16,8,beta_couple=beta,heat=2e-3,mech_load=0.03,alpha=0.03,volfrac=0.5)
        un,Tn,tot,nr=m2.newton_continuation(beta)
        print("beta %-4d staggered resid %-10.2e | newton-continuation (%d steps) resid %.2e"%(beta,sr,tot,nr))