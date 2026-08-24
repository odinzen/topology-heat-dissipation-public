import numpy as np
import topoheat.thermoelastic_fem as fem
import topoheat.hex8 as hex8

def test_conduction_1d():
    assert fem.verify_conduction() < 1e-10

def test_thermal_patch_free():
    stress, disp = fem.verify_patch_free()
    assert stress < 1e-8 and disp < 1e-10

def test_clamped_thermal_stress_2d():
    sx, sy, an = fem.verify_clamped()
    assert abs(sx - an) / abs(an) < 1e-6

def test_clamped_thermal_stress_3d():
    E, nu, al, dT = 210e3, 0.3, 12e-6, 50.0
    D = hex8.Dmat(E, nu)
    eps0 = al * dT * np.array([1, 1, 1, 0, 0, 0.0])
    sig = -D @ eps0
    assert abs(sig[0] - (-E * al * dT / (1 - 2 * nu))) < 1e-6
