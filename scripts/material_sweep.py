import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Material selection sweep for the thermoelastic topology optimizer.

Design note (unit/degeneracy pitfall): the engine's element matrices are written
for unit modulus and a balanced thermomechanical regime. Plugging absolute GPa
values would make the thermal load dominate and collapse the volume constraint.
So every material is normalized RELATIVE to structural steel:

    youngs_modulus    = E_m / E_steel            (steel -> 1.0)
    conductivity      = k_m / k_steel            (steel -> 1.0)
    thermal_expansion = 0.02 * alpha_m/alpha_st  (steel -> 0.02)

Poisson ratio is held at 0.30 for all materials. The engine's element matrices
use nu=0.30 fixed, so this is a deliberate approximation; the real materials span
nu 0.29..0.34, a small spread that we neglect here.

Boundary conditions are identical for every material (the applied problem):
left edge clamped + held at theta=0 (heat sink); a downward point load at the
right-edge midheight node; a distributed heat source over the right half.
"""
import json, os, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from topoheat.engine import validate, SpecEngine

NX, NY = 48, 24
VOLFRAC = 0.45

def nid(ix, iy):
    return iy * (NX + 1) + ix

def build_spec(E_rel, k_rel, alpha_eng):
    # left edge: all nodes ix=0
    left = [nid(0, iy) for iy in range(NY + 1)]
    # right-edge midheight node for the mechanical load
    mid_node = nid(NX, NY // 2)
    # right half: ix from NX//2 .. NX inclusive, all rows
    right_half = [nid(ix, iy) for iy in range(NY + 1)
                  for ix in range(NX // 2, NX + 1)]
    spec = {
        "domain_dimensions": [NX, NY],
        "volume_fraction": VOLFRAC,
        "units": {"length": "mm", "force": "N", "temperature": "K"},
        "dof_ordering": "node_major",
        "material": {
            "youngs_modulus": E_rel,
            "poisson_ratio": 0.30,
            "thermal_expansion": alpha_eng,
            "conductivity": k_rel,
            "density_floor": 1e-3,
            "penalization": {"stiffness": 3, "conductivity": 3},
            "filter_radius": 2.4,
        },
        "reference_temperature": 0.0,
        "objective": {"mode": "thermoelastic_compliance"},
        "coupling": {"mode": "one_way"},
        "supports": [{"node_indices": left, "fixed_components": [0, 1]}],
        "loads": [{"node_indices": [mid_node], "force_vector": [0.0, -3.0]}],
        "thermal_supports": [{"node_indices": left, "temperature": 0.0}],
        "thermal_loads": [{"node_indices": right_half, "power": 0.005}],
    }
    return spec

def main():
    lib = json.load(open("materials.json"))
    mats = {m["name"]: m for m in lib["materials"]}
    ref = lib["reference"]
    E_st = mats[ref]["E_GPa"]
    k_st = mats[ref]["k_W_mK"]
    a_st = mats[ref]["alpha_per_K"]
    rho_st = mats[ref]["density_kg_m3"]

    results = []
    for m in lib["materials"]:
        name = m["name"]
        E_rel = m["E_GPa"] / E_st
        k_rel = m["k_W_mK"] / k_st
        alpha_eng = 0.02 * (m["alpha_per_K"] / a_st)
        spec = build_spec(E_rel, k_rel, alpha_eng)
        validate(spec)
        eng = SpecEngine(spec)
        rho, rp, hist = eng.run(iters=100, beta_schedule=True)
        comp = float(hist[-1])
        vol = float(np.mean(rp))
        gray = float(np.mean(4 * rp * (1 - rp)))
        spec_comp = comp * (m["density_kg_m3"] / rho_st)
        results.append({
            "material": name,
            "E_rel": E_rel, "k_rel": k_rel, "alpha_eng": alpha_eng,
            "density_kg_m3": m["density_kg_m3"],
            "compliance": comp,
            "volume_fraction": vol,
            "gray_fraction": gray,
            "specific_compliance": spec_comp,
        })
        print(f"  ran {name:18s} comp={comp:.4e} vol={vol:.3f} gray={gray:.3f} spec_comp={spec_comp:.4e}")

    # normalize to steel
    steel = next(r for r in results if r["material"] == ref)
    c_st = steel["compliance"]
    sc_st = steel["specific_compliance"]
    for r in results:
        r["compliance_norm_steel"] = r["compliance"] / c_st
        r["specific_compliance_norm_steel"] = r["specific_compliance"] / sc_st

    rank_abs = sorted(results, key=lambda r: r["compliance"])
    rank_spec = sorted(results, key=lambda r: r["specific_compliance"])

    print("\n" + "=" * 78)
    print("MATERIAL SWEEP  (48x24 grid, nu=0.30 fixed approx, BCs identical)")
    print("=" * 78)
    hdr = f"{'material':18s}{'comp/steel':>12s}{'spec/steel':>12s}{'volume':>9s}{'gray':>8s}"
    print(hdr)
    print("-" * 78)
    for r in rank_abs:
        flag = "  <-- VOLUME COLLAPSE" if abs(r["volume_fraction"] - VOLFRAC) > 0.05 else ""
        print(f"{r['material']:18s}{r['compliance_norm_steel']:>12.4f}"
              f"{r['specific_compliance_norm_steel']:>12.4f}"
              f"{r['volume_fraction']:>9.3f}{r['gray_fraction']:>8.3f}{flag}")
    print("-" * 78)
    print(f"Ranked by ABSOLUTE compliance (stiffest first):")
    print("   " + " > ".join(r["material"] for r in rank_abs))
    print(f"Ranked by SPECIFIC compliance (best stiffness/mass first):")
    print("   " + " > ".join(r["material"] for r in rank_spec))
    print(f"\nWinner on absolute compliance : {rank_abs[0]['material']}")
    print(f"Winner on specific compliance : {rank_spec[0]['material']}")
    collapsed = [r["material"] for r in results
                 if abs(r["volume_fraction"] - VOLFRAC) > 0.05]
    print(f"Volume collapsed materials    : {collapsed if collapsed else 'none (all near 0.45)'}")

    out = {
        "grid": [NX, NY], "volume_fraction_target": VOLFRAC,
        "reference": ref, "poisson_note": "nu held at 0.30 (engine element matrices fixed at 0.30)",
        "normalization": {
            "youngs_modulus": "E_m / E_steel",
            "conductivity": "k_m / k_steel",
            "thermal_expansion": "0.02 * alpha_m/alpha_steel",
        },
        "results": results,
        "ranking_absolute": [r["material"] for r in rank_abs],
        "ranking_specific": [r["material"] for r in rank_spec],
        "winner_absolute": rank_abs[0]["material"],
        "winner_specific": rank_spec[0]["material"],
        "volume_collapsed": collapsed,
    }
    json.dump(out, open("material_sweep.json", "w"), indent=2)
    print("\nsaved material_sweep.json")

if __name__ == "__main__":
    main()
