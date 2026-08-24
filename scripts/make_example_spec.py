"""Generate a complete, valid example problem specification (a heated cantilever)."""
import json, numpy as np
nx, ny = 60, 30
def nid(ix, iy): return iy * (nx + 1) + ix
left = [nid(0, iy) for iy in range(ny + 1)]               # clamped, heat sink
tip = nid(nx, ny // 2)                                    # mechanical load point
heat_nodes = [nid(ix, iy) for ix in range(nx // 2, nx + 1) for iy in range(ny + 1)]
spec = {
    "domain_dimensions": [nx, ny],
    "volume_fraction": 0.45,
    "units": {"length": "mm", "force": "N", "temperature": "K"},
    "dof_ordering": "node_major",
    "material": {
        "youngs_modulus": 1.0, "poisson_ratio": 0.3,
        "thermal_expansion": 0.02, "conductivity": 1.0,
        "density_floor": 1e-3,
        "penalization": {"stiffness": 3, "conductivity": 3},
        "filter_radius": 2.4,
    },
    "reference_temperature": 0.0,
    "objective": {"mode": "thermoelastic_compliance"},
    "coupling": {"mode": "one_way"},
    "supports": [{"node_indices": left, "fixed_components": [0, 1]}],
    "loads": [{"node_indices": [tip], "force_vector": [0.0, -3.0]}],
    "thermal_supports": [{"node_indices": left, "temperature": 0.0}],
    "thermal_loads": [{"node_indices": heat_nodes, "power": 0.004}],
}
json.dump(spec, open("example_problem.json", "w"), indent=2)
print("wrote example_problem.json |", len(left), "supports,", len(heat_nodes), "heat nodes")
