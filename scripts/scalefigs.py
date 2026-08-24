import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

# ---- 3D scaled design voxels ----
rp=np.load("scaleopt_rp.npy"); nx,ny,nz=np.load("scaleopt_shape.npy")
vol=np.transpose(rp.reshape(nz,ny,nx),(2,1,0)); filled=vol>0.4
shade=np.clip(0.85-0.7*vol,0,1); col=np.empty(filled.shape+(4,)); 
for i in range(3): col[...,i]=shade
col[...,3]=1.0
fig=plt.figure(figsize=(8.6,5.0)); ax=fig.add_subplot(111,projection="3d")
ax.voxels(filled,facecolors=col,edgecolor=(0,0,0,0.18),linewidth=0.15)
ax.set_box_aspect((nx,ny,nz)); ax.view_init(elev=22,azim=-60)
ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
fig.suptitle("SCALED 3D DESIGN | 40 BY 20 BY 10 | 28,413 DOF",fontsize=12,fontweight="bold",y=0.95)
fig.text(0.5,0.04,"Solved with AMG preconditioned conjugate gradient | conjugate gradient held at 11 iterations",
         ha="center",fontsize=8,style="italic")
fig.savefig("/tmp/fig9.png",dpi=200,bbox_inches="tight"); plt.close(fig)

# ---- scaling: CG iterations vs problem size ----
ndof=np.array([6825,28413,74385]); cg=np.array([10,11,11])
direct_ok=[True,True,False]
fig,ax=plt.subplots(figsize=(7.8,4.6))
for s in ("top","right","bottom","left"): ax.spines[s].set_linewidth(1.1); ax.spines[s].set_color("black")
ax.grid(False)
ax.plot(ndof,cg,marker="o",color="black",lw=1.7,markersize=8,markerfacecolor="white",markeredgewidth=1.5)
for x,y,ok in zip(ndof,cg,direct_ok):
    ax.annotate("direct: ok" if ok else "direct: infeasible",(x,y),textcoords="offset points",
                xytext=(0,12),ha="center",fontsize=8,style="italic")
ax.set_xlabel("Degrees of freedom, count",fontsize=10)
ax.set_ylabel("Conjugate gradient iterations, count",fontsize=10)
ax.set_ylim(0,20); ax.set_xlim(0,80000)
fig.suptitle("PRECONDITIONER SCALING: ITERATIONS INDEPENDENT OF MESH SIZE",fontsize=11,fontweight="bold")
fig.text(0.5,0.02,"Algebraic multigrid keeps the iteration count flat as the grid grows | the direct factorization fails at the largest grid",
         ha="center",fontsize=7.6,style="italic")
fig.subplots_adjust(left=0.11,right=0.96,top=0.9,bottom=0.16)
fig.savefig("/tmp/fig10.png",dpi=200); plt.close(fig)
print("ok")
