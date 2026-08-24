"""Multi-material extension of the JSON-driven engine.

A single v4 specification can request two-material thermoelastic topology
optimization. This module validates such a spec against schema_v4.json, reads
the two materials from the spec's `materials` library, normalizes their
properties RELATIVE to the first material, and drives the verified
TwoMaterialTO (per-element density rho + material fraction mu) optimizer.

Normalization (material 0 is the reference, "material A"):
    relative E_i     = E_i / E_0
    relative k_i     = k_i / k_0
    relative alpha_i = (alpha_i / alpha_0) * 0.02
The alpha map onto a base of 0.02 matches the thermal-expansion scale the
element thermal-coupling vector G0 was tuned to in the verified single-material
runs (example_problem.json uses thermal_expansion 0.02), so the coupled load
stays in the regime where the adjoint was finite-difference verified.

NOTE on Poisson's ratio: a single pair of unit element matrices (KE0, KET0, G0)
can encode exactly one nu. We thread the SIMP card's poisson_ratio into the
optimizer so the spec's nu actually enters the stiffness (default 0.30 keeps the
historical results byte-identical). The two-material optimizer blends modulus and
expansion only, so the two materials necessarily share this nu; if the per-material
poisson_ratio values differ from each other (a genuinely two-nu case that one
element matrix cannot represent) we warn and use the card nu for both.

The boundary conditions (mechanical supports/loads, thermal supports/loads,
convection) are built from the spec through the same NumericalBridge /
ThermoelasticBridge used by SpecEngine, so the multi-material run honors the
exact problem the spec describes rather than the optimizer's built-in defaults.
"""
import json
import os
import numpy as np

from jsonschema import Draft202012Validator

from . import engine  # reuse schema discovery + bridges from the single-material engine
from .numerical_bridge import NumericalBridge
from .thermal_bridge import ThermoelasticBridge
from .multimaterial import TwoMaterialTO

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_schema_v4():
    for c in [os.path.join(HERE, "schema", "schema_v4.json"),
              os.path.join(HERE, "..", "schema", "schema_v4.json"),
              os.path.join(HERE, "schema_v4.json"),
              os.path.join(os.getcwd(), "schema_v4.json")]:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("schema_v4.json not found")


SCHEMA_V4 = _find_schema_v4()


def validate_v4(spec, schema_path=SCHEMA_V4):
    """Validate a spec against schema v4 (raises on an invalid spec)."""
    Draft202012Validator(json.load(open(schema_path))).validate(spec)


# Base expansion scale onto which relative alpha is mapped (see module docstring).
ALPHA_BASE = 0.02


def _relative_materials(materials):
    """Return (matA, matB) as (relE, relAlpha, relK) tuples relative to material 0."""
    m0, m1 = materials[0], materials[1]
    E0 = float(m0["youngs_modulus"])
    k0 = float(m0["conductivity"])
    a0 = float(m0["thermal_expansion"])

    def rel(m):
        relE = float(m["youngs_modulus"]) / E0
        relK = float(m["conductivity"]) / k0
        # guard a0 == 0 (allowed by schema): fall back to a flat base expansion
        relA = (float(m["thermal_expansion"]) / a0 * ALPHA_BASE) if a0 != 0.0 else ALPHA_BASE
        return (relE, relA, relK)

    return rel(m0), rel(m1)


def _build_bc_from_spec(to, spec):
    """Override the optimizer's default BC arrays with the spec's BCs.

    Matches SpecEngine._build_bc: mechanical force/fixed set from NumericalBridge,
    thermal heat/Dirichlet set from ThermoelasticBridge. The two-material solve
    uses self.f_mech, self.q, self.free_mech, self.free_temp, self.fixed_temp.
    """
    dims = spec["domain_dimensions"]
    nb = NumericalBridge(dims, spec.get("dof_ordering", "node_major"))
    f_mech, fixed_mech = nb.assemble(spec["supports"], spec["loads"])
    # Parity with SpecEngine: reject kinematically under-constrained specs up front.
    if not nb.check_kinematic_stability(fixed_mech):
        raise ValueError("supports do not remove all rigid body modes: the "
                         "mechanical problem is kinematically under-constrained")
    to.f_mech = f_mech
    to.fixed_mech = fixed_mech
    to.free_mech = np.setdiff1d(np.arange(to.ndof), fixed_mech)

    tb = ThermoelasticBridge(dims, spec.get("dof_ordering", "node_major"))
    q, ftn, ftv, robin = tb.assemble_thermal(
        spec["thermal_supports"], spec["thermal_loads"], spec.get("convection"))
    # Robin film boundary: the bridge surfaces (nodes, h) pairs and has already
    # deposited the equivalent nodal heat h*Tamb into q. Fold the matching +h
    # into a per-node conduction-diagonal vector here; TwoMaterialTO.solve adds it
    # to its assembled KT. Unlike SpecEngine, the multi-material path works in
    # ABSOLUTE temperature (tref=0, no theta = T - T_ref shift — q and the
    # Dirichlet values are used as-is), so the +h diagonal is exact against the
    # bridge's +h*Tamb load and NO -h*T_ref correction is applied here.
    to.robin_diag = np.zeros(to.nnode)
    for nodes, h in robin:
        np.add.at(to.robin_diag, nodes, h)
    to.q = q
    to.fixed_temp = ftn
    to.free_temp = np.setdiff1d(np.arange(to.nnode), ftn)
    # Dirichlet thermal nodes carry a prescribed value, not a heat load. The
    # two-material solve fixes their temperature to zero (it solves the free DOFs
    # against T=0 on the fixed set), matching the example's temperature: 0.0.
    if ftn.size:
        to.q[ftn] = 0.0
    # A Dirichlet temperature support OR a convection film regularizes the
    # conduction operator (the +h diagonal makes KTff nonsingular). Reject only
    # when there is neither: pure conduction with no prescribed node is singular.
    if not to.fixed_temp.size and not to.robin_diag.any():
        raise ValueError("thermal field has no Dirichlet temperature support "
                         "and no convection boundary; the conduction matrix is "
                         "singular")


def run_multimaterial_from_spec(spec_path, iters=120):
    """Validate against schema v4, build a two-material optimizer from the spec,
    run it, and return a result dict (objective, volume, gray, material summary).
    """
    if iters < 1:
        raise ValueError(f"iters must be >= 1 (got {iters})")
    spec = json.load(open(spec_path))
    validate_v4(spec)  # raises on an invalid spec

    sel = spec.get("material_selection", {})
    if sel.get("mode") != "multi_material":
        raise ValueError("spec material_selection.mode must be 'multi_material'")
    materials = spec["materials"]
    if len(materials) != 2:
        raise ValueError("multi_material requires exactly two materials")

    # A single pair of element matrices (KE0/KET0/G0) can carry exactly one
    # Poisson ratio. If both materials share a nu, that value is threaded into
    # the optimizer below and nothing is lost. Only a GENUINELY two-nu spec --
    # the two materials disagree on nu -- is unrepresentable, so warn only then.
    per_mat_nus = [m["poisson_ratio"] for m in materials if "poisson_ratio" in m]
    if len(per_mat_nus) == 2 and abs(per_mat_nus[0] - per_mat_nus[1]) > 1e-9:
        import warnings
        warnings.warn(
            "multimaterial element matrices carry a single Poisson ratio; the "
            f"per-material poisson_ratio {per_mat_nus} differ, so the SIMP card "
            "nu is used for both and the difference is ignored",
            RuntimeWarning, stacklevel=2)

    matA, matB = _relative_materials(materials)

    nx, ny = spec["domain_dimensions"]
    mat = spec["material"]  # SIMP/filter knobs reused from the single-material card
    volfrac = spec["volume_fraction"]

    # Thread the SIMP card's Poisson ratio into the element matrices. When both
    # materials share a nu this represents the spec exactly; when they differ
    # (warned above) the card nu is the single representable value. Defaults to
    # 0.30 if the card omits it, preserving the historical byte-identical result.
    nu = float(mat.get("poisson_ratio", 0.30))
    to = TwoMaterialTO(
        nx, ny,
        matA=matA, matB=matB,
        nu=nu,
        volfrac=volfrac,
        rmin=mat["filter_radius"],
        penal=mat["penalization"]["stiffness"],
        penal_k=mat["penalization"]["conductivity"],
        penal_f=mat["penalization"]["stiffness"],
        floor=mat["density_floor"],
        heat=0.0, mech_load=0.0)  # BCs come from the spec, not these defaults

    _build_bc_from_spec(to, spec)

    # ---- optimization loop (rho via OC, mu via move-limited steepest descent) ----
    rho = np.full(to.nel, volfrac)
    mu = np.full(to.nel, 0.5)
    mu_move = 0.1
    hist = []
    converged = False
    for it in range(iters):
        J, dJ_drho, dJ_dmu = to.full_gradient(rho, mu)
        hist.append(J)
        rho_new = to.oc(rho, dJ_drho)
        gmax = np.max(np.abs(dJ_dmu)) + 1e-30
        mu_new = np.clip(mu - (mu_move / gmax) * dJ_dmu, 0.0, 1.0)
        change = max(float(np.max(np.abs(rho_new - rho))),
                     float(np.max(np.abs(mu_new - mu))))
        rho, mu = rho_new, mu_new
        if change < 1e-3 and it > 10:
            converged = True
            break

    rho_phys = to.filter_forward(rho)
    mu_phys = to.filter_forward(mu)
    Jf, _ = to.objective(rho_phys, mu_phys)

    solid = rho_phys > 0.5
    mu_solid = mu_phys[solid]
    result = {
        "mode": "multi_material",
        "grid": [int(nx), int(ny)],
        "iterations": len(hist),
        "converged": bool(converged),
        "objective_initial": float(hist[0]),
        "objective_final": float(Jf),
        "reduction": float(hist[0] / Jf) if Jf != 0 else float("inf"),
        "volume_fraction": float(np.mean(rho_phys)),
        "gray_fraction": float(np.mean(4 * rho_phys * (1 - rho_phys))),
        "materials": [materials[0]["name"], materials[1]["name"]],
        "relative_matA_E_alpha_k": list(matA),
        "relative_matB_E_alpha_k": list(matB),
        "material_field": {
            "mu_mean_all": float(np.mean(mu_phys)),
            "mu_mean_solid": float(mu_solid.mean()) if mu_solid.size else float("nan"),
            "solid_fraction_near_A_mu_lt_0.4": float((mu_solid < 0.4).mean()) if mu_solid.size else float("nan"),
            "solid_fraction_near_B_mu_gt_0.6": float((mu_solid > 0.6).mean()) if mu_solid.size else float("nan"),
        },
    }
    return result


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "example_multimaterial.json"
    res = run_multimaterial_from_spec(path)
    print(json.dumps(res, indent=2))
