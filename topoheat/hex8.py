"""Hex8 trilinear element matrices for 3D thermoelasticity, 2x2x2 Gauss."""
import numpy as np

_g = np.array([-1, 1]) / np.sqrt(3.0)
# natural node coordinates, ordering: bottom face CCW then top face CCW
_NODES = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                   [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]], float)

def _shape(xi, eta, ze, h):
    s = _NODES
    N = 0.125*(1+s[:,0]*xi)*(1+s[:,1]*eta)*(1+s[:,2]*ze)
    dNdxi  = 0.125*s[:,0]*(1+s[:,1]*eta)*(1+s[:,2]*ze)
    dNdeta = 0.125*s[:,1]*(1+s[:,0]*xi)*(1+s[:,2]*ze)
    dNdze  = 0.125*s[:,2]*(1+s[:,0]*xi)*(1+s[:,1]*eta)
    grad = np.vstack([dNdxi, dNdeta, dNdze]) * (2.0/h)   # 3x8 d/dx
    detJ = (h/2.0)**3
    return N, grad, detJ

def Dmat(E, nu):
    lam = E*nu/((1+nu)*(1-2*nu)); mu = E/(2*(1+nu))
    D = np.zeros((6,6))
    D[:3,:3] = lam
    D[0,0]=D[1,1]=D[2,2]=lam+2*mu
    D[3,3]=D[4,4]=D[5,5]=mu
    return D

def Bmat(grad):
    B = np.zeros((6,24))
    B[0,0::3]=grad[0]; B[1,1::3]=grad[1]; B[2,2::3]=grad[2]
    B[3,0::3]=grad[1]; B[3,1::3]=grad[0]
    B[4,1::3]=grad[2]; B[4,2::3]=grad[1]
    B[5,0::3]=grad[2]; B[5,2::3]=grad[0]
    return B

def KE_stiff(E, nu, h=1.0):
    D = Dmat(E,nu); K = np.zeros((24,24))
    for xi in _g:
        for eta in _g:
            for ze in _g:
                _,grad,detJ=_shape(xi,eta,ze,h); B=Bmat(grad)
                K += B.T@D@B*detJ
    return K

def KE_cond(k, h=1.0):
    K=np.zeros((8,8))
    for xi in _g:
        for eta in _g:
            for ze in _g:
                _,grad,detJ=_shape(xi,eta,ze,h)
                K += k*(grad.T@grad)*detJ
    return K

def G_thermal(E, nu, h=1.0):
    D=Dmat(E,nu); m=np.array([1,1,1,0,0,0.0]); g=np.zeros(24)
    for xi in _g:
        for eta in _g:
            for ze in _g:
                _,grad,detJ=_shape(xi,eta,ze,h); B=Bmat(grad)
                g += B.T@D@m*detJ
    return g

def Bcentre(h=1.0):
    _,grad,_=_shape(0,0,0,h); return Bmat(grad)

if __name__=="__main__":
    # 3D thermoelastic patch and clamped checks on a single element.
    E,nu,al,dT=210e3,0.3,12e-6,50.0
    KE=KE_stiff(E,nu); g=G_thermal(E,nu); D=Dmat(E,nu)
    fth=al*dT*g
    # fully clamped single element: all dofs fixed -> u=0 -> stress=-D eps0
    eps0=al*dT*np.array([1,1,1,0,0,0.0]); sig=-D@eps0
    analytic=-E*al*dT/(1-2*nu)
    print("clamped sigma_xx %.3f  analytic %.3f  rel %.2e"%(sig[0],analytic,abs(sig[0]-analytic)/abs(analytic)))
    # free expansion: constrain only 6 rigid dofs, expect ~0 stress
    free=np.arange(24)
    fixed=[0,1,2, 1+3, 2+3*3, 2+3]  # a minimal set to remove rigid body
    # simpler: fix node0 (0,1,2), node1 y,z (4,5), node3 z (11)
    fixed=[0,1,2,4,5,11]
    fr=np.setdiff1d(free,fixed)
    u=np.zeros(24)
    u[fr]=np.linalg.solve(KE[np.ix_(fr,fr)], fth[fr])
    B=Bcentre(); eps=B@u; sig_free=D@(eps-eps0)
    print("free expansion max|stress| %.3e"%np.max(np.abs(sig_free)))
