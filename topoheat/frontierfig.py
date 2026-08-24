import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ndof=np.array([432,1632,3600,6336])
diag=np.array([18,56,60,76]); tri=np.array([9,26,33,53]); schur=np.array([9,40,45,36])
fig,ax=plt.subplots(figsize=(7.8,4.6))
for s in ("top","right","bottom","left"): ax.spines[s].set_linewidth(1.1); ax.spines[s].set_color("black")
ax.grid(False)
ax.plot(ndof,diag,marker="s",color="black",lw=1.4,markersize=8,markerfacecolor="white",
        markeredgewidth=1.4,linestyle=(0,(5,4)),label="Block diagonal split")
ax.plot(ndof,tri,marker="^",color="black",lw=1.5,markersize=9,markerfacecolor="0.6",
        markeredgewidth=1.2,linestyle=(0,(1,1)),label="Block triangular split")
ax.plot(ndof,schur,marker="o",color="black",lw=1.8,markersize=8,markerfacecolor="black",
        label="Schur complement split")
ax.set_xlabel("Degrees of freedom, count",fontsize=10)
ax.set_ylabel("GMRES iterations, count",fontsize=10)
ax.set_xlim(0,7000); ax.set_ylim(0,90)
ax.legend(frameon=True,edgecolor="black",fontsize=9,loc="upper left")
fig.suptitle("FIELD SPLIT PRECONDITIONING OF THE COUPLED JACOBIAN",fontsize=11,fontweight="bold")
fig.text(0.5,0.02,"AMG inner blocks with elasticity rigid modes | the Schur split does not grow at the largest grid",
         ha="center",fontsize=7.4,style="italic")
fig.subplots_adjust(left=0.1,right=0.96,top=0.9,bottom=0.16)
import os
_OUT=os.environ.get("TOPOHEAT_FIG_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","figures"))
os.makedirs(_OUT, exist_ok=True)
fig.savefig(os.path.join(_OUT,"fig11.png"),dpi=200); print("ok ->", os.path.join(_OUT,"fig11.png"))
