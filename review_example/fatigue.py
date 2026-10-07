"""Sub-models 1 and 2: thermal cycling -> strain -> cycles-to-crack (copper).

Couples the public topoheat
thermoelastic solver to a Coffin-Manson strain-life law to get a first
cycles-to-failure (Nf) for a copper cold plate under a real temperature swing.

Physics, in two steps:

  1. cycling -> strain. A thermoelastic solve at a service dT gives the
     mechanically-constrained strain field. The fatigue driver is the
     constrained (stress-producing) strain, not the free thermal expansion:
     for a fully-constrained element it equals alpha*dT, which is the clean
     hand-checkable baseline used in the self-test.

  2. strain -> Nf. The total-strain-life (Coffin-Manson / Basquin) relation

         eps_a = (sigma_f'/E) (2 Nf)^b + eps_f' (2 Nf)^c

     inverted for Nf. The left term is elastic (Basquin), the right is plastic
     (Coffin-Manson); their sum is monotone decreasing in Nf, so a bracketed
     bisection on log10(Nf) is robust.

Every material constant is labelled measured / gated / SCHEMATIC. The copper
strain-life constants here are SCHEMATIC (literature-typical, ungated): they
must be gated by DOI + Crossref (rule 6) before any manuscript number rests on
them. The FEM elastic field is used directly as the strain-life input; a Neuber
notch correction to recover true local elastic-plastic strain at a stress
concentration is the documented refinement, not yet applied.

Run:  python fatigue.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

# Import the public topoheat thermoelastic solver as a dependency (not copied).
_TOPOHEAT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..")
if _TOPOHEAT not in sys.path:
    sys.path.insert(0, os.path.abspath(_TOPOHEAT))

from topoheat import thermoelastic_fem as te  # noqa: E402


@dataclass(frozen=True)
class Const:
    value: float
    unit: str
    status: str   # "measured" | "gated" | "SCHEMATIC" | "spec"
    source: str


# Constant sources, DOIs Crossref-verified 2026-08-23. Elastic/physical constants gate to
# SRM-grade primaries. No single authoritative room-temperature strain-life (4-parameter)
# or Paris set exists for OFHC copper, so those stay literature-consensus ("lit-range"),
# and the qualitative finding (copper is not the fatigue limiter) is robust to their scatter.
CITATIONS = {
    "ledbetter1974": "Ledbetter HM, Naimon ER. Elastic Properties of Metals and Alloys. II. "
                     "Copper. J Phys Chem Ref Data 1974;3:897-935. doi:10.1063/1.3253150",
    "hahn1970":      "Hahn TA. Thermal Expansion of Copper from 20 to 800 K - SRM 736. "
                     "J Appl Phys 1970;41:5096-5101. doi:10.1063/1.1658614",
    "hopcroft2010":  "Hopcroft MA, Nix WD, Kenny TW. What is the Young's Modulus of Silicon? "
                     "J Microelectromech Syst 2010;19:229-238. doi:10.1109/JMEMS.2009.2039697",
    "wortman1965":   "Wortman JJ, Evans RA. Young's Modulus, Shear Modulus, and Poisson's "
                     "Ratio in Silicon and Germanium. J Appl Phys 1965;36:153-156. "
                     "doi:10.1063/1.1713863",
    "okada1984":     "Okada Y, Tokumaru Y. Precise determination of lattice parameter and "
                     "thermal expansion coefficient of silicon between 300 and 1500 K. "
                     "J Appl Phys 1984;56:314-320. doi:10.1063/1.333965",
    "takahashi2012": "Takahashi S, Sano M, Watanabe A, Kitamura H. Prediction of fatigue life "
                     "of high-heat-load components made of oxygen-free copper. J Synchrotron "
                     "Radiat 2012;20:67-73. doi:10.1107/S0909049512041192",
    "seifi2018":     "Seifi R, Hosseini R. Experimental study of fatigue crack growth in raw "
                     "and annealed pure copper. Theor Appl Fract Mech 2018;94:1-9. "
                     "doi:10.1016/j.tafmec.2017.12.003",
}

# Annealed OFHC copper. Elastic/CTE gated to SRM primaries; strain-life constants are a
# literature-consensus range (no single authoritative RT set; exponents corroborated by
# takahashi2012 at elevated T, which reports them in a non-standard %-strain form).
COPPER = {
    "E":       Const(117_000.0, "MPa", "gated", "annealed OFHC engineering E ~115-117 GPa; single-xtal aggregate ~128 (ledbetter1974)"),
    "nu":      Const(0.34, "-", "gated", "Cu aggregate Poisson ratio (ledbetter1974)"),
    "alpha":   Const(16.6e-6, "1/K", "gated", "Cu CTE near RT, SRM 736 (hahn1970)"),
    "sigma_f": Const(500.0, "MPa", "lit-range", "fatigue strength coeff, annealed Cu ~400-600; no single authoritative source"),
    "b":       Const(-0.11, "-", "lit-range", "fatigue strength exponent, annealed Cu -0.10..-0.12; cf takahashi2012 beta -0.069 at 100-300C"),
    "eps_f":   Const(0.35, "-", "lit-range", "fatigue ductility coeff, annealed Cu ~0.3-1.0; no single authoritative source"),
    "c":       Const(-0.55, "-", "lit-range", "fatigue ductility exponent, annealed Cu -0.5..-0.6; cf takahashi2012 alpha -0.486"),
}


def _v(name: str) -> float:
    return COPPER[name].value


# Silicon die / substrate: the CTE-mismatch partner. The mismatch delta-alpha*dT at
# the bonded interface is the aggressive, realistic cold-plate fatigue driver, far
# stronger than copper's self-constrained expansion.
SUBSTRATE = {
    "E":     Const(150_000.0, "MPa", "gated", "Si isotropic-equiv E; anisotropic 130-188 GPa by orientation (hopcroft2010)"),
    "nu":    Const(0.22, "-", "gated", "Si Poisson ratio isotropic-equiv ~0.22 (wortman1965)"),
    "alpha": Const(2.57e-6, "1/K", "gated", "Si CTE at 300 K from the Okada formula (okada1984)"),
}


def _s(name: str) -> float:
    return SUBSTRATE[name].value


def strain_life_Nf(eps_amp: float, n_hi: float = 1e12) -> float:
    """Invert the total-strain-life relation for Nf (cycles to crack initiation).

    eps_amp is the strain amplitude (half the strain range) at the critical point.
    Returns Nf. The relation is monotone decreasing in Nf, so bisect on log10(Nf).
    """
    if eps_amp <= 0.0:
        return float("inf")
    E, sf, b, ef, c = _v("E"), _v("sigma_f"), _v("b"), _v("eps_f"), _v("c")

    def resid(nf: float) -> float:
        return (sf / E) * (2.0 * nf) ** b + ef * (2.0 * nf) ** c - eps_amp

    lo, hi = 0.5, n_hi                      # 0.5 cycle to a practical infinite-life cap
    r_lo, r_hi = resid(lo), resid(hi)
    if r_lo < 0.0:                          # strain so large even 1/2 cycle survives it: sub-cycle
        return lo
    if r_hi > 0.0:                          # below the (finite-life) endurance: effectively infinite
        return float("inf")
    for _ in range(200):
        mid = 10 ** (0.5 * (np.log10(lo) + np.log10(hi)))
        if resid(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return float(10 ** (0.5 * (np.log10(lo) + np.log10(hi))))


def principal_strain_amp(eps_mech: np.ndarray) -> float:
    """Max absolute principal strain of a plane mechanical strain (ex, ey, gamma_xy).

    gamma_xy is engineering shear (2*eps_xy). Max principal strain is the standard
    thermal-fatigue driver. For an equibiaxial constrained field (ex=ey=-alpha*dT,
    gamma=0) it returns alpha*dT, the hand-checkable baseline.
    """
    ex, ey, gxy = eps_mech
    ave = 0.5 * (ex + ey)
    rad = np.hypot(0.5 * (ex - ey), 0.5 * gxy)
    return float(max(abs(ave + rad), abs(ave - rad)))


def _plate_mesh(nx: int, ny: int, h: float) -> te.Mesh:
    return te.Mesh(nx, ny, h=h)


def _boundary_nodes(mesh: te.Mesh):
    coords = mesh.coords()
    L, H = mesh.nx * mesh.h, mesh.ny * mesh.h
    left = np.isclose(coords[:, 0], 0.0)
    right = np.isclose(coords[:, 0], L)
    bot = np.isclose(coords[:, 1], 0.0)
    top = np.isclose(coords[:, 1], H)
    return left, right, bot, top


def _elem_mech_strain(mesh: te.Mesh, u: np.ndarray, alpha: float, dTe: float, nodes) -> np.ndarray:
    """Constrained (stress-producing) mechanical strain of one element: eps - eps0."""
    edof = np.array([[2 * nd, 2 * nd + 1] for nd in nodes]).ravel()
    B = te._Bmat(0.0, 0.0, mesh.h)          # element centre
    eps = B @ u[edof]                        # total strain
    eps0 = alpha * dTe * np.array([1.0, 1.0, 0.0])
    return eps - eps0


def peak_strain_amp(dT: float, constraint: str = "clamped",
                    nx: int = 24, ny: int = 8, h: float = 1.0) -> float:
    """Peak strain amplitude in a copper plate under a service dT swing.

    Two physically distinct load cases:

    "clamped": whole plate heated by dT with every boundary dof fixed. This is the
    fully-constrained bound: every element sees the free thermal strain cancelled,
    so the peak equals alpha*dT (the conservative shortest-life case, hand-checkable).

    "patch": only a central die-footprint patch is hot (dT) in an otherwise cold
    plate whose edges are free (minimal constraint removes rigid-body modes only).
    The stress-producing strain now comes from the hot patch expanding against its
    cold surroundings, concentrated at the patch boundary. This is the real
    cold-plate loading and the answer is an FEM result, not a hand formula.

    Returns the strain *amplitude* (half the cold->hot range) at the critical element.
    """
    E, nu, alpha = _v("E"), _v("nu"), _v("alpha")
    mesh = _plate_mesh(nx, ny, h)
    els = mesh.elements()
    coords = mesh.coords()
    left, right, bot, top = _boundary_nodes(mesh)

    if constraint == "clamped":
        dTfield = np.full(len(els), dT)
        bnodes = np.where(left | right | bot | top)[0]
        fixed = np.concatenate([[2 * b, 2 * b + 1] for b in bnodes])
    elif constraint == "patch":
        # Central patch ~ one third of the plate span, full height.
        L, H = mesh.nx * mesh.h, mesh.ny * mesh.h
        dTfield = np.zeros(len(els))
        for e, nodes in enumerate(els):
            cx = np.mean(coords[nodes, 0])
            if L / 3.0 <= cx <= 2.0 * L / 3.0:
                dTfield[e] = dT
        # Free plate: pin one node in x,y and one more in y to kill rigid body only.
        n0 = mesh.nid(0, 0)
        nbr = mesh.nid(mesh.nx, 0)
        fixed = np.array([2 * n0, 2 * n0 + 1, 2 * nbr + 1])
    else:
        raise ValueError(f"unknown constraint {constraint!r}")

    K, F = te.assemble_mech(mesh, E, nu, alpha, dTfield)
    u = te.solve_dirichlet(K, F, fixed)
    amps = [principal_strain_amp(_elem_mech_strain(mesh, u, alpha, dTfield[e], nodes))
            for e, nodes in enumerate(els)]
    strain_range = float(np.max(amps))       # cold (0) -> hot (dT) full swing
    return 0.5 * strain_range


def _assemble_mech_multi(mesh, Ee, nue, alphae, dTfield):
    """Mechanical stiffness and thermal load with per-element material.

    Ee, nue, alphae are per-element arrays. The base solver assumes one material;
    a CTE-mismatch stack needs copper and substrate elements in the same mesh.
    """
    n = mesh.nnode
    K = np.zeros((2 * n, 2 * n))
    F = np.zeros(2 * n)
    for e, nodes in enumerate(mesh.elements()):
        Ke = te.elem_stiffness(Ee[e], nue[e], mesh.h)
        fe = te.elem_thermal_force(Ee[e], nue[e], alphae[e], dTfield[e], mesh.h)
        edof = np.array([[2 * nd, 2 * nd + 1] for nd in nodes]).ravel()
        for a in range(8):
            F[edof[a]] += fe[a]
            for b in range(8):
                K[edof[a], edof[b]] += Ke[a, b]
    return K, F


def peak_strain_amp_bimaterial(dT: float, cu_rows: int = 2, sub_rows: int = 10,
                               nx: int = 24, h: float = 1.0,
                               sub_E_scale: float = 1.0) -> float:
    """Peak copper strain amplitude in a copper-on-substrate stack under a dT swing.

    A thin copper layer (top cu_rows) bonded to a silicon substrate (sub_rows).
    Uniform heating; the plate is externally free (rigid-body pins only), so the
    strain comes purely from the internal CTE mismatch. The fatigue driver is the
    peak principal *mechanical* strain among the copper elements (referenced to
    copper's own free expansion). sub_E_scale stiffens the substrate for the
    rigid-substrate limit check (copper strain -> delta-alpha*dT).
    """
    ny = cu_rows + sub_rows
    mesh = _plate_mesh(nx, ny, h)
    els = mesh.elements()
    coords = mesh.coords()

    Ee = np.empty(len(els)); nue = np.empty(len(els)); alphae = np.empty(len(els))
    is_cu = np.zeros(len(els), dtype=bool)
    for e, nodes in enumerate(els):
        cy = np.mean(coords[nodes, 1])
        if cy >= sub_rows * h:               # top cu_rows are copper
            Ee[e], nue[e], alphae[e] = _v("E"), _v("nu"), _v("alpha")
            is_cu[e] = True
        else:
            Ee[e] = _s("E") * sub_E_scale
            nue[e], alphae[e] = _s("nu"), _s("alpha")
    dTfield = np.full(len(els), dT)

    K, F = _assemble_mech_multi(mesh, Ee, nue, alphae, dTfield)
    # Free plate: pin bottom-left in x,y and bottom-right in y (rigid body only).
    n0 = mesh.nid(0, 0)
    nbr = mesh.nid(mesh.nx, 0)
    fixed = np.array([2 * n0, 2 * n0 + 1, 2 * nbr + 1])
    u = te.solve_dirichlet(K, F, fixed)

    amps = [principal_strain_amp(_elem_mech_strain(mesh, u, alphae[e], dT, nodes))
            for e, nodes in enumerate(els) if is_cu[e]]
    return 0.5 * float(np.max(amps))


def nf_years(nf: float, cycles_per_day: float) -> float:
    return nf / (cycles_per_day * 365.25)


# ----------------------------------------------------------------------
# Report and self-test
# ----------------------------------------------------------------------

def _print_provenance() -> None:
    print("Copper constants (status | value | source):")
    for k, c in COPPER.items():
        print(f"  {k:>8} = {c.value:<12g} [{c.status:9}] {c.unit:5} {c.source}")


def report(dT: float = 60.0, cycles_per_day: float = 2.0) -> dict:
    """First Nf for copper at a service dT, clamped bound and bolted FEM case."""
    out = {}
    for name in ("clamped", "patch"):
        eps_a = peak_strain_amp(dT, constraint=name)
        nf = strain_life_Nf(eps_a)
        out[name] = dict(eps_amp=eps_a, Nf=nf, years=nf_years(nf, cycles_per_day))
    # CTE-mismatch (copper on silicon): the realistic, aggressive driver.
    eps_bi = peak_strain_amp_bimaterial(dT)
    nf_bi = strain_life_Nf(eps_bi)
    out["cte_mismatch"] = dict(eps_amp=eps_bi, Nf=nf_bi, years=nf_years(nf_bi, cycles_per_day))
    return out


def _selftest() -> None:
    # 1. Fully-clamped uniform dT: peak strain amplitude must equal alpha*dT/2.
    dT = 60.0
    eps_a = peak_strain_amp(dT, constraint="clamped")
    expect = 0.5 * _v("alpha") * dT
    assert abs(eps_a - expect) / expect < 1e-9, (eps_a, expect)

    # 2. Strain-life inversion round-trips in the finite-life regime. Amplitudes
    # below the 1e12-cycle elastic asymptote (~1.5e-4 here) are infinite by design,
    # so probe above it and assert the infinite-life boundary separately.
    E, sf, b, ef, c = _v("E"), _v("sigma_f"), _v("b"), _v("eps_f"), _v("c")
    for ea in (5e-3, 1e-3, 3e-4):
        nf = strain_life_Nf(ea)
        assert np.isfinite(nf), (ea, nf)
        recon = (sf / E) * (2 * nf) ** b + ef * (2 * nf) ** c
        assert abs(recon - ea) / ea < 1e-6, (ea, recon)
    assert strain_life_Nf(1e-5) == float("inf")   # well below the elastic asymptote

    # 3. Monotonicity: a larger swing must give fewer cycles.
    nfs = [strain_life_Nf(peak_strain_amp(t, "clamped")) for t in (20, 40, 60, 80)]
    assert all(a > b for a, b in zip(nfs, nfs[1:])), nfs

    # 4. The fully-clamped case bounds the strain: local die-patch heating cannot
    #    exceed it, so patch life >= clamped life.
    eps_clamp = peak_strain_amp(dT, "clamped")
    eps_patch = peak_strain_amp(dT, "patch")
    assert eps_patch <= eps_clamp * (1 + 1e-9), (eps_clamp, eps_patch)
    assert np.isfinite(strain_life_Nf(eps_patch))

    # 5. Rigid-substrate limit: a copper film on a very stiff, thick substrate is
    #    forced to the substrate's expansion, so its mechanical strain -> delta-alpha*dT.
    da = abs(_s("alpha") - _v("alpha"))
    eps_rigid = peak_strain_amp_bimaterial(dT, sub_E_scale=100.0)
    assert abs(eps_rigid - 0.5 * da * dT) / (0.5 * da * dT) < 0.02, (eps_rigid, 0.5 * da * dT)

    # 6. CTE mismatch on a real (Si-stiffness) substrate must out-strain copper's
    #    own self-constrained patch case: mismatch is the more aggressive driver.
    assert peak_strain_amp_bimaterial(dT) > eps_patch

    print("self-test OK: clamped baseline = alpha*dT, inversion round-trips, "
          "monotone in dT, clamped bounds patch, rigid-substrate limit = delta-alpha*dT, "
          "CTE mismatch out-strains self-constraint.")


def main() -> None:
    _print_provenance()
    print()
    _selftest()
    print()
    cpd = 2.0
    print(f"Copper cold-plate first Nf  (cycles_per_day = {cpd:g}, SCHEMATIC):")
    print(f"  {'dT (K)':>6} | {'constraint':>12} | {'eps_amp':>10} | "
          f"{'Nf (cycles)':>12} | {'life (yr)':>10}")
    for dT in (20.0, 40.0, 60.0, 80.0):
        r = report(dT, cycles_per_day=cpd)
        for name in ("clamped", "patch", "cte_mismatch"):
            d = r[name]
            nf_s = f"{d['Nf']:.3e}" if np.isfinite(d['Nf']) else "inf"
            yr_s = f"{d['years']:.3e}" if np.isfinite(d['years']) else "inf"
            print(f"  {dT:>6.0f} | {name:>12} | {d['eps_amp']:>10.3e} | "
                  f"{nf_s:>12} | {yr_s:>10}")
    print("\nAll Nf values are SCHEMATIC: copper strain-life constants are ungated "
          "literature-typical placeholders.")


if __name__ == "__main__":
    main()
