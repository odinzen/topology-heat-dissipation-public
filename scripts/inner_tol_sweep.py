import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from topoheat.block_solver import run
print("%-10s %8s | outer GMRES iters by inner tolerance" % ("grid","ndof"))
for nx,ny in [(16,8),(32,16),(48,24),(64,32)]:
    row=[]
    for itol in [1e-2,1e-6,1e-10]:
        nd,it,err,info=run(nx,ny,mode="schur",inner=60,itol=itol,tol=1e-8)
        row.append((itol,it,err))
    ndof=2*(nx+1)*(ny+1)
    print("%-10s %8d | "%("%dx%d"%(nx,ny),ndof)+"  ".join("itol=%.0e:%d(%.0e)"%(t,it,e) for t,it,e in row))
