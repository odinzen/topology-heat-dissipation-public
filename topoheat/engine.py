"""End to end engine: a validated JSON problem specification drives the optimizer.

Flow: load JSON -> validate against schema v3 -> build mechanical and thermal
boundary conditions through the bridges -> scale the material to the spec units
-> run the projected optimizer -> write the design and a result summary. This
ties the hardened contract to the numerics as a single pipeline.
"""
import json, os, numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
from jsonschema import Draft202012Validator

from .projected_to import ProjectedTO
from .topopt_thermoelastic import _factor_or_raise
from .numerical_bridge import NumericalBridge
from .thermal_bridge import ThermoelasticBridge

HERE = os.path.dirname(os.path.abspath(__file__))
def _find_schema():
    # schema/ now ships as package-data under topoheat/schema/, so the
    # first candidate (HERE/schema/...) resolves for wheels and editable installs
    # alike. The repo-relative and cwd fallbacks remain for legacy layouts.
    for c in [os.path.join(HERE, "schema", "schema_v3.json"),
              os.path.join(HERE, "..", "schema", "schema_v3.json"),
              os.path.join(HERE, "..", "..", "schema", "schema_v3.json"),
              os.path.join(HERE, "schema_v3.json"),
              os.path.join(os.getcwd(), "schema", "schema_v3.json"),
              os.path.join(os.getcwd(), "schema_v3.json")]:
        if os.path.exists(c): return c
    raise FileNotFoundError("schema_v3.json not found")
SCHEMA = _find_schema()


def validate(spec, schema_path=SCHEMA):
    Draft202012Validator(json.load(open(schema_path))).validate(spec)


class SpecEngine(ProjectedTO):
    """Projected one way thermoelastic optimizer configured entirely from a spec."""

    def __init__(self, spec, beta_hs=1.0):
        self.spec = spec
        mat = spec["material"]
        nx, ny = spec["domain_dimensions"]
        self.E0 = mat["youngs_modulus"]; self.k0 = mat["conductivity"]
        super().__init__(
            nx, ny, volfrac=spec["volume_fraction"],
            penal=mat["penalization"]["stiffness"],
            penal_k=mat["penalization"]["conductivity"],
            penal_f=mat["penalization"]["stiffness"],
            rmin=mat["filter_radius"], floor=mat["density_floor"],
            alpha=mat["thermal_expansion"], tref=0.0,
            heat=0.0, mech_load=0.0, nu=mat["poisson_ratio"], beta_hs=beta_hs)

    # material in absolute units (element matrices are for unit modulus)
    def E_of(self, r):  return self.E0 * super().E_of(r)
    def dE_of(self, r): return self.E0 * super().dE_of(r)
    def k_of(self, r):  return self.k0 * super().k_of(r)
    def dk_of(self, r): return self.k0 * super().dk_of(r)
    def Ef_of(self, r): return self.E0 * super().Ef_of(r)
    def dEf_of(self, r):return self.E0 * super().dEf_of(r)

    # ---- heat-conduction objective (spec mode "heat_conduction") ----------
    # Minimize thermal compliance J = q^T T at fixed material volume: the classic
    # heat-sink layout problem. It is self-adjoint (the adjoint temperature is T
    # itself), so dJ/drho_e = -dk/drho_e * (T_e^T KET0 T_e). This holds for a
    # design-independent heat load q with Dirichlet sinks (no convection term).
    # Any other mode defers to the thermoelastic objective in the base class.
    def objective(self, rhophys, st=None):
        if self.spec["objective"]["mode"] != "heat_conduction":
            return super().objective(rhophys, st)
        st = st or self.solve(rhophys)
        T = st["T"]
        J = float(self.q[self.free_temp] @ T[self.free_temp])
        return J, st

    def sensitivity(self, rhophys):
        if self.spec["objective"]["mode"] != "heat_conduction":
            return super().sensitivity(rhophys)
        J, st = self.objective(rhophys)
        Tne = st["T"][self.edofT]
        dJ = -self.dk_of(rhophys) * np.einsum("ej,jk,ek->e", Tne, self.KET0, Tne)
        return J, dJ, st

    def _build_bc(self):
        sp_ = self.spec
        nb = NumericalBridge(sp_["domain_dimensions"], "node_major")
        f_mech, fixed_mech = nb.assemble(sp_["supports"], sp_["loads"])
        # Fail fast on a spec whose supports leave a rigid body mode free, with
        # a clear message, rather than letting it surface as a singular solve.
        if not nb.check_kinematic_stability(fixed_mech):
            raise ValueError("supports do not remove all rigid body modes: the "
                             "mechanical problem is kinematically under-constrained")
        self.f_mech = f_mech
        self.fixed_mech = fixed_mech
        self.free_mech = np.setdiff1d(np.arange(self.ndof), fixed_mech)
        tb = ThermoelasticBridge(sp_["domain_dimensions"], "node_major")
        q, ftn, ftv, robin = tb.assemble_thermal(
            sp_["thermal_supports"], sp_["thermal_loads"], sp_.get("convection"))
        # Robin film boundary: the bridge surfaces (nodes, h) pairs and has already
        # deposited the equivalent nodal heat h*Tamb into q. Fold the matching +h
        # into a per-node conduction-diagonal vector here; solve() adds it to KT.
        # We work in theta = T - T_ref, and the +h diagonal breaks the constant-shift
        # invariance of pure conduction, so the consistent theta-load needs an
        # h*T_ref correction at the convection nodes:
        #   (K + hI) theta = q_other + h*Tamb - h*T_ref = q_other + h*(Tamb - T_ref).
        # q already carries +h*Tamb, so we subtract h*T_ref once below.
        self.robin_diag = np.zeros(self.nnode)
        for nodes, h in robin:
            np.add.at(self.robin_diag, nodes, h)
        q = q - self.robin_diag * sp_["reference_temperature"]
        self.q = q
        self.fixed_temp = ftn
        # work in theta = T - T_ref so the strain reference is zero
        self.temp_theta = ftv - sp_["reference_temperature"]
        self.free_temp = np.setdiff1d(np.arange(self.nnode), ftn)

    def solve(self, rp):
        # A Dirichlet temperature support OR a convection film regularizes the
        # conduction operator (the +h diagonal makes KTff nonsingular). Reject only
        # when there is neither: pure conduction with no prescribed node is singular.
        if not self.fixed_temp.size and not self.robin_diag.any():
            raise ValueError("thermal field has no Dirichlet temperature support "
                             "and no convection boundary; the conduction matrix is "
                             "singular")
        kv = self.k_of(rp)
        KT = self._assemble_KT(kv)
        if self.robin_diag.any():
            KT = KT + sp.diags(self.robin_diag)
        T = np.zeros(self.nnode)
        if self.fixed_temp.size:
            T[self.fixed_temp] = self.temp_theta
        KTf = KT[self.free_temp, :]
        KTff = KTf[:, self.free_temp]
        rhs = self.q[self.free_temp] - KTf[:, self.fixed_temp] @ T[self.fixed_temp]
        T[self.free_temp] = spla.spsolve(KTff.tocsc(), rhs)
        Te = T[self.edofT].mean(axis=1)
        ef = self.Ef_of(rp)
        coef = ef * self.alpha * (Te - self.tref)
        f_th = self._thermal_load(coef)
        Ev = self.E_of(rp)
        KU = self._assemble_KU(Ev)
        KUff = KU[self.free_mech, :][:, self.free_mech]
        Lu = _factor_or_raise(KUff, "mechanical")
        u = np.zeros(self.ndof)
        u[self.free_mech] = Lu.solve((self.f_mech + f_th)[self.free_mech])
        return dict(T=T, Te=Te, u=u, KT=KT, KU=KU, KTff=KTff, Lu=Lu, Ev=Ev, kv=kv, ef=ef)


def run_from_spec(spec_path, iters=120, out_prefix="engine_result"):
    if iters < 1:
        raise ValueError(f"iters must be >= 1 (got {iters})")
    spec = json.load(open(spec_path))
    validate(spec)                                  # raises on an invalid spec
    eng = SpecEngine(spec)
    rho, rp, hist = eng.run(iters=iters, beta_schedule=True)
    nx, ny = spec["domain_dimensions"]
    result = {
        "objective_mode": spec["objective"]["mode"],
        "coupling": spec["coupling"]["mode"],
        "iterations": len(hist),
        "compliance_initial": hist[0],
        "compliance_final": hist[-1],
        "reduction": (hist[0] / hist[-1]) if hist[-1] else float("inf"),
        "volume_fraction": float(np.mean(rp)),
        "gray_fraction": float(np.mean(4 * rp * (1 - rp))),
        "grid": [int(nx), int(ny)],
    }
    np.save(out_prefix + "_rho.npy", rp)
    json.dump(result, open(out_prefix + ".json", "w"), indent=2)
    return result


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        prog="topoheat",
        description="Run the thermoelastic topology optimizer from a validated JSON spec.")
    p.add_argument("spec", nargs="?", default="example_problem.json",
                   help="path to the problem spec JSON (schema v3)")
    p.add_argument("-n", "--iters", type=int, default=120,
                   help="optimizer iterations (default: 120)")
    p.add_argument("-o", "--out", default="engine_result",
                   help="output prefix for <out>_rho.npy and <out>.json (default: engine_result)")
    args = p.parse_args(argv)
    result = run_from_spec(args.spec, iters=args.iters, out_prefix=args.out)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
