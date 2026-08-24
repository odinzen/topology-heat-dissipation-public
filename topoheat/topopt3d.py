"""3D thermoelastic topology optimization (one way, design dependent thermal).

Hex8 elements, SIMP with density floor, 3D density filter, optimality criteria.
Sensitivity by the coupled one way adjoint, verified by finite differences.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from . import hex8

NU = 0.3
# Module-level matrices at the historical nu=0.30. Kept for any external code
# that compares against them; the TO3D instance builds its own from self.nu so a
# spec-driven Poisson ratio enters the stiffness. KET0 (conduction) is nu-free.
KE0 = hex8.KE_stiff(1.0, NU)
KET0 = hex8.KE_cond(1.0)
G0 = hex8.G_thermal(1.0, NU)

class TO3D:
    def __init__(self, nx, ny, nz, volfrac=0.3, penal=3.0, penal_k=3.0, penal_f=3.0,
                 rmin=1.5, floor=1e-3, kmin=1e-3, alpha=0.02, heat=2e-3, tref=0.0,
                 mech=1.0, nu=NU):
        self.nx,self.ny,self.nz=nx,ny,nz
        self.volfrac,self.penal,self.penal_k,self.penal_f=volfrac,penal,penal_k,penal_f
        self.floor,self.kmin,self.alpha,self.tref=floor,kmin,alpha,tref
        # Per-instance element matrices at the requested nu. Default 0.30 keeps
        # every existing result byte-identical to the module matrices above.
        self.nu=nu
        self.KE0=hex8.KE_stiff(1.0,nu)
        self.KET0=hex8.KE_cond(1.0)
        self.G0=hex8.G_thermal(1.0,nu)
        self.nel=nx*ny*nz
        self.nnode=(nx+1)*(ny+1)*(nz+1)
        self.ndof=3*self.nnode
        self._mesh(); self._filter(rmin); self._bc(heat,mech)
        self.iK=np.kron(self.edof,np.ones((24,1),int)).ravel()
        self.jK=np.kron(self.edof,np.ones((1,24),int)).ravel()
        self.iKt=np.kron(self.edofT,np.ones((8,1),int)).ravel()
        self.jKt=np.kron(self.edofT,np.ones((1,8),int)).ravel()
    def nid(self,ix,iy,iz): return iz*(self.nx+1)*(self.ny+1)+iy*(self.nx+1)+ix
    def _mesh(self):
        els=[]; cen=[]
        for iz in range(self.nz):
            for iy in range(self.ny):
                for ix in range(self.nx):
                    n=[self.nid(ix,iy,iz),self.nid(ix+1,iy,iz),self.nid(ix+1,iy+1,iz),self.nid(ix,iy+1,iz),
                       self.nid(ix,iy,iz+1),self.nid(ix+1,iy,iz+1),self.nid(ix+1,iy+1,iz+1),self.nid(ix,iy+1,iz+1)]
                    els.append(n); cen.append((ix,iy,iz))
        self.edofT=np.array(els); self.cen=np.array(cen)
        self.edof=np.zeros((self.nel,24),int)
        for a in range(8):
            self.edof[:,3*a]=3*self.edofT[:,a]
            self.edof[:,3*a+1]=3*self.edofT[:,a]+1
            self.edof[:,3*a+2]=3*self.edofT[:,a]+2
    def _filter(self,rmin):
        from math import ceil,hypot
        R=int(ceil(rmin)); idx={tuple(c):e for e,c in enumerate(self.cen)}
        rows,cols,val=[],[],[]
        for e,(ix,iy,iz) in enumerate(self.cen):
            for dx in range(-R,R+1):
                for dy in range(-R,R+1):
                    for dz in range(-R,R+1):
                        j=idx.get((ix+dx,iy+dy,iz+dz))
                        if j is not None:
                            w=rmin-hypot(hypot(dx,dy),dz)
                            if w>0: rows.append(e);cols.append(j);val.append(w)
        self.H=sp.csr_matrix((val,(rows,cols)),shape=(self.nel,self.nel))
        self.Hs=np.asarray(self.H.sum(1)).ravel()
    def _bc(self,heat,mech):
        # clamp x=0 face; downward load at far bottom edge centre; sink at x=0
        left=[self.nid(0,iy,iz) for iz in range(self.nz+1) for iy in range(self.ny+1)]
        left=np.array(left)
        self.fixed=np.concatenate([3*left,3*left+1,3*left+2])
        self.free=np.setdiff1d(np.arange(self.ndof),self.fixed)
        tip=self.nid(self.nx,self.ny//2,self.nz//2)
        self.f=np.zeros(self.ndof); self.f[3*tip+1]=-mech
        self.ftemp=left; self.freeT=np.setdiff1d(np.arange(self.nnode),left)
        self.q=np.full(self.nnode,heat); self.q[left]=0.0
    def E_(self,r):return self.floor+r**self.penal*(1-self.floor)
    def dE_(self,r):return self.penal*r**(self.penal-1)*(1-self.floor)
    def k_(self,r):return self.kmin+r**self.penal_k*(1-self.kmin)
    def dk_(self,r):return self.penal_k*r**(self.penal_k-1)*(1-self.kmin)
    def Ef_(self,r):return r**self.penal_f
    def dEf_(self,r):return self.penal_f*r**(self.penal_f-1)
    def filt(self,rho):return np.asarray(self.H@rho).ravel()/self.Hs
    def solve(self,rp):
        kv=self.k_(rp); sKt=(self.KET0.ravel()[None,:]*kv[:,None]).ravel()
        KT=sp.csr_matrix((sKt,(self.iKt,self.jKt)),shape=(self.nnode,self.nnode))
        T=np.zeros(self.nnode)
        T[self.freeT]=spla.spsolve(KT[self.freeT][:,self.freeT].tocsc(),self.q[self.freeT])
        Te=T[self.edofT].mean(1)
        coef=self.Ef_(rp)*self.alpha*(Te-self.tref)
        fth=np.zeros(self.ndof); np.add.at(fth,self.edof.ravel(),(coef[:,None]*self.G0[None,:]).ravel())
        Ev=self.E_(rp); sK=(self.KE0.ravel()[None,:]*Ev[:,None]).ravel()
        KU=sp.csr_matrix((sK,(self.iK,self.jK)),shape=(self.ndof,self.ndof))
        lu=spla.splu(KU[self.free][:,self.free].tocsc())
        u=np.zeros(self.ndof); u[self.free]=lu.solve((self.f+fth)[self.free])
        return dict(T=T,Te=Te,u=u,Ev=Ev,KT=KT,KU=KU,lu=lu)
    def obj(self,rp,st=None):
        st=st or self.solve(rp); ue=st["u"][self.edof]
        return float(np.sum(np.einsum("ej,jk,ek->e",ue,self.KE0,ue)*st["Ev"])),st
    def grad(self,rho):
        rp=self.filt(rho); J,st=self.obj(rp)
        u,T,Te=st["u"],st["T"],st["Te"]; ue=u[self.edof]
        lam_u=-2*u
        w=self.Ef_(rp)*self.alpha*(1.0/8.0)
        s_e=(self.G0[None,:]*lam_u[self.edof]).sum(1)
        rhsT=np.zeros(self.nnode)
        np.add.at(rhsT,self.edofT.ravel(),np.repeat(w*s_e,8).reshape(self.nel,8).ravel())
        lamT=np.zeros(self.nnode)
        lamT[self.freeT]=spla.spsolve(st["KT"][self.freeT][:,self.freeT].tocsc(),rhsT[self.freeT])
        t1=-self.dE_(rp)*np.einsum("ej,jk,ek->e",ue,self.KE0,ue)
        t2=2*self.dEf_(rp)*self.alpha*(Te-self.tref)*(self.G0[None,:]*ue).sum(1)
        t3=self.dk_(rp)*np.einsum("ej,jk,ek->e",lamT[self.edofT],self.KET0,T[self.edofT])
        dphys=t1+t2+t3
        return J,np.asarray(self.H@(dphys/self.Hs)).ravel()
    def oc(self,rho,dJ):
        dv=np.asarray(self.H@(np.ones(self.nel)/self.Hs)).ravel()
        l1,l2,mv=0,1e9,0.2; rn=rho
        while (l2-l1)/(0.5*(l1+l2)+1e-30)>1e-3:
            lm=0.5*(l1+l2); be=np.maximum(0,-dJ/dv/lm)
            rn=np.maximum(0,np.maximum(rho-mv,np.minimum(1,np.minimum(rho+mv,rho*np.sqrt(be)))))
            if np.mean(self.filt(rn))>self.volfrac: l1=lm
            else: l2=lm
        return rn
    def run(self,iters=40):
        rho=np.full(self.nel,self.volfrac); hist=[]
        for it in range(iters):
            J,dJ=self.grad(rho); hist.append(J); rho=self.oc(rho,dJ)
        return rho,self.filt(rho),hist

class HeatTO3D(TO3D):
    """3D heat-conduction topology optimization: self-adjoint minimization of the
    thermal compliance J = q^T T at fixed material volume (the 3D heat-sink layout).
    Thermal-only solve; the mechanical field is not needed for this objective."""
    def solve(self, rp):
        kv = self.k_(rp)
        sKt = (self.KET0.ravel()[None, :] * kv[:, None]).ravel()
        KT = sp.csr_matrix((sKt, (self.iKt, self.jKt)), shape=(self.nnode, self.nnode))
        T = np.zeros(self.nnode)
        T[self.freeT] = spla.spsolve(KT[self.freeT][:, self.freeT].tocsc(), self.q[self.freeT])
        return dict(T=T, KT=KT)

    def obj(self, rp, st=None):
        st = st or self.solve(rp)
        return float(self.q[self.freeT] @ st["T"][self.freeT]), st

    def grad(self, rho):
        rp = self.filt(rho); J, st = self.obj(rp)
        Tne = st["T"][self.edofT]
        dphys = -self.dk_(rp) * np.einsum("ej,jk,ek->e", Tne, self.KET0, Tne)
        return J, np.asarray(self.H @ (dphys / self.Hs)).ravel()


def gradient_check_heat(seed=1):
    rng = np.random.default_rng(seed)
    to = HeatTO3D(4, 3, 3, rmin=1.5)
    rho = rng.uniform(0.4, 0.9, to.nel)
    _, g = to.grad(rho); h = 1e-6; errs = []
    for i in rng.choice(to.nel, size=8, replace=False):
        rp = rho.copy(); rp[i] += h
        rm = rho.copy(); rm[i] -= h
        fd = (to.obj(to.filt(rp))[0] - to.obj(to.filt(rm))[0]) / (2 * h)
        errs.append(abs(fd - g[i]) / (abs(fd) + 1e-30))
    return float(max(errs))


def gradient_check(seed=0):
    rng=np.random.default_rng(seed)
    to=TO3D(4,2,2,rmin=1.5)
    rho=rng.uniform(0.4,0.9,to.nel)
    J,g=to.grad(rho); h=1e-6; errs=[]
    for i in rng.choice(to.nel,size=6,replace=False):
        rp=rho.copy();rp[i]+=h; rm=rho.copy();rm[i]-=h
        Jp,_=to.obj(to.filt(rp)); Jm,_=to.obj(to.filt(rm))
        errs.append(abs((Jp-Jm)/(2*h)-g[i])/(abs((Jp-Jm)/(2*h))+1e-30))
    return float(max(errs))

if __name__=="__main__":
    print("3D gradient check max relative error:",gradient_check())
