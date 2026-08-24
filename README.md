# topoheat — thermoelastic topology optimization for heat dissipation

`topoheat` designs cooling geometry. Given a domain, boundary conditions, a heat
load, and a material, it optimizes where to place material so the structure sheds
heat and carries mechanical load at once — two-way coupled thermoelastic topology
optimization, held to one standard: **every value is verified against a closed-form
solution or an exact physical limit.**

The method is standard density-based topology optimization (SIMP), applied and
verified rigorously. What it produces — an optimized geometry and its
thermal-resistance behavior — is the useful artifact: a design map from a heat-load
specification to a manufacturable heat-dissipating structure.

## What is inside

- A hardened generative-to-solver JSON contract that rejects physically invalid
  specifications before the solver runs.
- A forward thermoelastic FEM checked against closed-form solutions.
- One-way, two-way, projected, and 3D thermoelastic optimizers whose adjoint
  sensitivities are verified to finite-difference precision.
- A pure heat-conduction objective (`heat_conduction`): self-adjoint minimization of
  thermal compliance J = qᵀT at fixed material volume — the classic heat-sink layout
  problem — which grows the canonical dendritic conductor network. Available in 2D
  (`SpecEngine`) and 3D (`HeatTO3D`, hex8), both gradient-verified.
- An algebraic-multigrid preconditioned solver whose iteration count stays nearly
  flat as the mesh grows, and a monolithic Newton solve for the strong-feedback
  regime.
- Geometry export (STL/OBJ/PLY/glTF) and a bundled Three.js viewer.

## Layout

```
topoheat/          bridges, forward benchmark, one-way and two-way optimizers,
                   projection, MBB benchmark, hex8 element, 3D optimizer, figures
topoheat/schema/   hardened elastostatic (v2) and thermoelastic multiphysics (v3/v4) specs
scripts/           run scripts, reproduce_all.py, check_all_gradients.py
tests/             forward benchmarks, adjoint gradient checks, schema and bridge checks
figures/           coupling, forward validation, optimized design and field, history,
                   mesh independence, MBB, three-dimensional design, projection comparison
docs/              results and verification, geometry-stack license audits, CAD-kernel notes
geometry/          export scripts and the Three.js viewer
```

## Install and verify

```
conda env create -f environment.yml
conda activate topoheat
python scripts/reproduce_all.py    # runs every verification and benchmark
pytest                             # full suite
```

Or pip: `pip install -e ".[dev,amg,figures]"`.

### Verification status

`scripts/reproduce_all.py` passes all checks (~5 s):

```
PASS conduction 1D
PASS thermal patch (free)
PASS clamped thermal stress 2D        -180.0 vs -180.0 (exact)
PASS one way gradient
PASS two way gradient
PASS projected gradient
PASS 3D gradient
PASS clamped thermal stress 3D
PASS thermal stress gradient
PASS contract validates good spec / rejects bad spec
PASS multimaterial density gradient / material gradient
ALL CHECKS PASSED
```

Forward solutions match closed form to machine precision; clamped thermal stress is
exact at -180 MPa (2D) and -315 MPa (3D). Adjoint gradients agree with finite
differences below 1e-4 across all objectives. The optimizer reproduces the canonical
MBB beam. `pytest`: 80 passed.

## Geometry and CAD

The optimized 3D field exports to STL, OBJ, PLY, and glTF (`geometry/`), viewable in
any open-source CAD tool or the bundled Three.js viewer (`geometry/viewer.html`). The
export stack is MIT/BSD; see `docs/GEOMETRY_LICENSES.md`. Geometry export needs
`requirements-geometry.txt`.

## License

Code: MIT (`LICENSE`). Data and figures: CC BY 4.0 (`DATA_LICENSE.md`).

## Citation

See `CITATION.cff`. Archived on Zenodo (metadata in `.zenodo.json`).

## Acknowledgements

Developed with the assistance of Claude, an AI assistant by Anthropic, for
implementation, sensitivity verification, solver prototyping, and geometry tooling.
All results were reviewed and verified by the author.
