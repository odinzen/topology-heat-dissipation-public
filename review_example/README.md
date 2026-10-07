# Worked example and reliability analysis for the cooling review

Scripts behind Section 6 of M. E. Bustamante, G. Bustamante, K. Lilova, "A review of materials
design for data center thermal management and waste heat recovery" (Applied Thermal Engineering,
under review). They use the `topoheat` engine in this repository as a dependency and add no new
solver.

Scope: heat conduction and mechanical crack damage only. Corrosion and composition change in
service are not modeled here (the review treats them separately, Section 4.2). The copper
fatigue and crack-growth constants in `fatigue.py` and `crack_tree.py` are schematic
(literature-typical, not fitted to a dataset), so Figure 11 and Table 4 report a ranking of
damage tolerance, not a service life.

## What regenerates what

| Review item | Script | Output |
|---|---|---|
| Figure 5 (optimized tree, 320 x 160) | `figure5_tree.py 320 160 5.0 170 6 0.42` | `figures/tree_320x160_f5.0.png` |
| Section 6.1 comparisons (tree vs straight fins and a uniform plate) | `section61_comparisons.py` | printed table |
| Figure 6 (thin 3D cold plate, vein network) | `figure6_flatplate.py`, then `figure6_render.py` | `figures/flatplate_clean.png` |
| Figure 8 (cracks in a 240 x 120 tree) | `crack_tree.py`, then `crack_split_figs.py` | `figures/crack_fig_tree.png` |
| Figure 9 (cracks in the flat cold plate) | `crack_plate3d.py`, then `crack_split_figs.py` | `figures/crack_fig_plate.png` |
| Figure 10 (partial depth failure threshold) | `crack_depth_sweep.py`, then `crack_split_figs.py` | `figures/crack_fig_failure.png` |
| Figure 11 and Table 4 (limb ranking, critical depths) | `crack_depth_sweep_multi.py`, `crack_cycles.py`, then `crack_rank_fig.py` | `figures/crack_rank.png` |
| Figure 12 (partial width surface damage) | `crack_chunk.py`, `_chunk_extra.py`, `crack_chunk_maps.py`, then `crack_chunk_fig.py` | `figures/crack_chunk.png` |

`fatigue.py` holds the copper constants and the strain-life model shared by the crack scripts.
`crack_figs.py`, `crack_plate3d_figs.py`, `crack_depth_fig.py`, `crack_depth_multi_fig.py` and
`crack_composite.py` draw companion views of the same results. `crack_plate3d.py` voxelizes the
Figure 6 plate, so `figure6_flatplate.py` must run first.

## Run order

Later scripts reuse designs cached in `.cache/` (not committed):

```
python figure5_tree.py 320 160 5.0 170 6 0.42
python figure6_flatplate.py
python figure6_render.py
python section61_comparisons.py
python crack_tree.py
python crack_plate3d.py
python crack_depth_sweep.py
python crack_depth_sweep_multi.py
python crack_chunk.py
python _chunk_extra.py
python crack_chunk_maps.py
python crack_cycles.py
python crack_composite.py
python crack_split_figs.py
python crack_rank_fig.py
python crack_chunk_fig.py
```

Needs numpy, scipy, matplotlib and scikit-image, plus pyamg for the multigrid solvers.

## Reproduced values

A clean run of the sequence above (fresh checkout, empty cache, 2026-10-07) reproduces every number
and figure in Section 6 of the review; the regenerated Figures 5, 6 and 8 to 12 match the
submitted ones.

| Item | Value |
|---|---|
| Figure 5 optimization | compliance 14.71 -> 1.347 (as in the original run) |
| Figure 5 tree vs straight fins + base (equal material) | peak temperature 36% lower |
| Figure 5 tree vs uniform plate of the same metal | mean temperature 38% lower; peak temperature 60% higher, at the corner farthest from the sink (14% vs 98% of the domain above half its peak) |
| Coarse 80 x 40 tree (filter 2.4) | 43% lower peak than straight fins; vs uniform plate 34% lower mean, 57% higher peak |
| Figure 10 | effectiveness holds near 100% for shallow cracks, falls to 61% past about 32% depth |
| Table 4 critical depths (outer twig, top, side, diagonal, secondary) | 6, 26, 26, 45, 47% of wall; 0.47, 2.12, 2.12, 3.59, 3.73 mm on the 8 mm base |
| Figure 11 | relative life spread 17%, outer twig first (absolute cycles are schematic) |
| Figure 12 | a chunk narrower than about 40% of the limb never disables it; full-width critical depth 22% |

The uniform-plate baseline spreads the same metal evenly (rho = 0.40^(1/3), conductivity 0.40).
An earlier draft compared against grey rho = 0.40, which the SIMP model penalizes to
conductivity 0.065 and which therefore overstates the gain.
