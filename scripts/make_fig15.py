import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""fig15_materials.png -- greyscale grouped bar chart, house style.
Single Axes, save to /tmp then shutil.copy to the mounted outputs dir."""
import json, os, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "material_sweep.json")))
res = data["results"]
# keep a sensible plotting order: reference first, then by absolute compliance
order = sorted(res, key=lambda r: (r["material"] != data["reference"], r["compliance"]))
names = [r["material"].replace("_", "\n") for r in order]
comp = np.array([r["compliance_norm_steel"] for r in order])
spec = np.array([r["specific_compliance_norm_steel"] for r in order])

x = np.arange(len(order))
w = 0.38

fig, ax = plt.subplots(figsize=(9.0, 5.2))
b1 = ax.bar(x - w/2, comp, w, facecolor="0.35", edgecolor="black",
            linewidth=0.9, label="Compliance (steel = 1)")
b2 = ax.bar(x + w/2, spec, w, facecolor="white", edgecolor="black",
            linewidth=0.9, hatch="////", label="Specific compliance (steel = 1)")

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("Normalized value (steel = 1.0)", fontsize=11)
ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--")

# house style: full box, all four spines thin black, no grid, white bg
for s in ax.spines.values():
    s.set_visible(True); s.set_linewidth(0.8); s.set_color("black")
ax.set_facecolor("white"); fig.patch.set_facecolor("white")
ax.grid(False)
ax.tick_params(direction="out", length=3, color="black")

leg = ax.legend(loc="upper left", frameon=True, fontsize=9)
leg.get_frame().set_edgecolor("black"); leg.get_frame().set_linewidth(0.8)

fig.suptitle("Material selection sweep: compliance vs. specific compliance",
             fontsize=13, fontweight="bold", y=0.98)
fig.text(0.5, 0.012,
         "Thermoelastic topology optimization, 48x24 grid, identical boundary "
         "conditions; properties normalized to structural steel; nu held at 0.30.",
         ha="center", fontsize=8, style="italic")

fig.tight_layout(rect=[0, 0.045, 1, 0.95])

tmp = "/tmp/fig15.png"
fig.savefig(tmp, dpi=160, facecolor="white")
plt.close(fig)

dst = os.path.join(HERE, "fig15_materials.png")
shutil.copy(tmp, dst)
sz = os.path.getsize(dst)
assert sz > 0, "figure is empty"
print(f"fig15_materials.png written, {sz} bytes")
