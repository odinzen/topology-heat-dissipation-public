"""Collected pytest coverage for the spec-to-solver bridges.

These checks were previously stranded in tests/verify.py and
tests/verify_thermal.py as standalone print/assert scripts that pytest never
collected. They are now proper test functions importing the bridges via their
package paths.
"""
import numpy as np
import pytest

from topoheat.numerical_bridge import NumericalBridge
from topoheat.thermal_bridge import ThermoelasticBridge


# --------------------------------------------------------------------------
# Mechanical bridge (NumericalBridge)
# --------------------------------------------------------------------------

def test_node_and_dof_counts():
    # 4x3 ELEMENTS -> 5x4 = 20 nodes, 40 dofs (2D).
    b = NumericalBridge([4, 3], "node_major")
    assert b.total_nodes == 20
    assert b.total_dofs == 40


def test_basic_mechanical_assemble_node_major():
    b = NumericalBridge([4, 3], "node_major")
    supports = [{"node_indices": [0, 1, 2, 3], "fixed_components": [0, 1]}]
    loads = [{"node_indices": [19], "force_vector": [0.0, -1.0]}]

    f, fixed = b.assemble(supports, loads)

    # Single -1.0 load on the y component of node 19 -> dof 19*2+1 = 39.
    nz = np.flatnonzero(f)
    assert nz.tolist() == [39]
    assert f[39] == pytest.approx(-1.0)

    # Both components of nodes 0..3 fixed -> dofs 0..7.
    assert fixed.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]


def test_dof_major_ordering_fixed_dofs():
    b = NumericalBridge([4, 3], "dof_major")
    supports = [{"node_indices": [0, 1, 2, 3], "fixed_components": [0, 1]}]
    loads = [{"node_indices": [19], "force_vector": [0.0, -1.0]}]

    _, fixed = b.assemble(supports, loads)
    # dof_major: index = component * total_nodes + node.
    # comp 0 nodes 0..3 -> 0..3 ; comp 1 nodes 0..3 -> 20..23.
    assert fixed.tolist() == [0, 1, 2, 3, 20, 21, 22, 23]


def test_kinematic_stability_cases():
    b = NumericalBridge([4, 3], "node_major")
    loads = [{"node_indices": [19], "force_vector": [0.0, -1.0]}]

    # Fully clamped left edge removes all rigid modes -> stable.
    _, fixed = b.assemble(
        [{"node_indices": [0, 1, 2, 3], "fixed_components": [0, 1]}], loads)
    assert bool(b.check_kinematic_stability(fixed)) is True

    # A single pinned node cannot remove the in-plane rotation -> unstable.
    _, fixed_pin = b.assemble(
        [{"node_indices": [0], "fixed_components": [0, 1]}], loads)
    assert bool(b.check_kinematic_stability(fixed_pin)) is False

    # Pin one node and add a roller on a second -> stable.
    _, fixed_pr = b.assemble(
        [{"node_indices": [0], "fixed_components": [0, 1]},
         {"node_indices": [16], "fixed_components": [1]}], loads)
    assert bool(b.check_kinematic_stability(fixed_pr)) is True


def test_node_index_out_of_bounds_raises():
    b = NumericalBridge([4, 3], "node_major")
    # total_nodes == 20, so node 20 is out of range.
    with pytest.raises(ValueError):
        b.assemble([], [{"node_indices": [20], "force_vector": [0.0, -1.0]}])
    with pytest.raises(ValueError):
        b.assemble(
            [{"node_indices": [-1], "fixed_components": [0]}], [])


def test_fixed_component_out_of_range_raises():
    b = NumericalBridge([4, 3], "node_major")  # spatial_dims == 2
    with pytest.raises(ValueError):
        b.assemble(
            [{"node_indices": [0], "fixed_components": [2]}], [])


def test_three_dimensional_assemble_and_stability():
    b = NumericalBridge([2, 2, 2], "node_major")
    assert b.total_nodes == 27  # 3x3x3 nodes
    face = list(range(b.total_nodes))
    _, fixed = b.assemble(
        [{"node_indices": face, "fixed_components": [0, 1, 2]}],
        [{"node_indices": [b.total_nodes - 1], "force_vector": [0, 0, -1]}])
    assert bool(b.check_kinematic_stability(fixed)) is True


# --------------------------------------------------------------------------
# Thermoelastic bridge (ThermoelasticBridge)
# --------------------------------------------------------------------------

def test_thermal_dof_counts():
    b = ThermoelasticBridge([4, 3], "node_major")
    assert b.thermal_dof_count() == 20            # one temp dof per node
    assert b.combined_dof_count() == 60           # 40 mech + 20 thermal


def test_thermal_assemble_heat_and_fixed_temps():
    b = ThermoelasticBridge([4, 3], "node_major")
    thermal_supports = [{"node_indices": [0, 1, 2, 3], "temperature": 373.15}]
    thermal_loads = [{"node_indices": [19], "power": 5.0}]
    convection = [{"node_indices": [16, 17, 18, 19],
                   "film_coefficient": 10.0, "ambient_temperature": 293.15}]

    heat, ftn, ftv, robin = b.assemble_thermal(
        thermal_supports, thermal_loads, convection)

    # Fixed temperature nodes and their prescribed values.
    assert ftn.tolist() == [0, 1, 2, 3]
    assert ftv.tolist() == pytest.approx([373.15, 373.15, 373.15, 373.15])

    # Node 19 carries both the point power and a convection term.
    assert heat[19] == pytest.approx(5.0 + 10.0 * 293.15)
    # Node 16 carries only the convection term.
    assert heat[16] == pytest.approx(10.0 * 293.15)
    # Convection touched nodes 16..19 -> exactly four nonzero heat entries.
    assert int(np.count_nonzero(heat)) == 4

    # Robin blocks surface the film coefficient for the matching nodes.
    assert len(robin) == 1
    nodes, h = robin[0]
    assert nodes.tolist() == [16, 17, 18, 19]
    assert h == pytest.approx(10.0)


def test_thermal_assemble_without_convection():
    b = ThermoelasticBridge([4, 3], "node_major")
    heat, ftn, ftv, robin = b.assemble_thermal(
        [{"node_indices": [0], "temperature": 350.0}],
        [{"node_indices": [10], "power": 2.5}],
        convection=None)
    assert robin == []
    assert heat[10] == pytest.approx(2.5)
    assert int(np.count_nonzero(heat)) == 1
    assert ftn.tolist() == [0]
    assert ftv.tolist() == pytest.approx([350.0])


def test_thermal_node_out_of_bounds_raises():
    b = ThermoelasticBridge([4, 3], "node_major")  # 20 nodes
    with pytest.raises(ValueError):
        b.assemble_thermal(
            [], [{"node_indices": [20], "power": 1.0}], None)
