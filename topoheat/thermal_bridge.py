import numpy as np
from .numerical_bridge import NumericalBridge


class ThermoelasticBridge(NumericalBridge):
    """
    Extends the mechanical bridge with the scalar thermal field. The thermal
    field carries one temperature degree of freedom per node, indexed directly
    by node number, independent of dof_ordering, which governs only the
    mechanical vector field. The class returns the global heat load vector,
    the prescribed temperature set, and any Robin film terms the solver needs
    to add to the conduction matrix diagonal.
    """

    def assemble_thermal(self, thermal_supports, thermal_loads, convection=None):
        heat = np.zeros(self.total_nodes, dtype=np.float64)

        for load in thermal_loads:
            nodes = np.asarray(load["node_indices"], dtype=np.int64)
            self._check_nodes(nodes)
            np.add.at(heat, nodes, float(load["power"]))

        robin = []
        if convection:
            for c in convection:
                nodes = np.asarray(c["node_indices"], dtype=np.int64)
                self._check_nodes(nodes)
                h = float(c["film_coefficient"])
                tamb = float(c["ambient_temperature"])
                # Equivalent nodal heat h*Tamb; the matching diagonal term h
                # is surfaced for the solver to fold into the conduction matrix.
                np.add.at(heat, nodes, h * tamb)
                robin.append((nodes, h))

        fixed_nodes_parts, fixed_values_parts = [], []
        for s in thermal_supports:
            nodes = np.asarray(s["node_indices"], dtype=np.int64)
            self._check_nodes(nodes)
            fixed_nodes_parts.append(nodes)
            fixed_values_parts.append(np.full(nodes.size, float(s["temperature"])))

        if fixed_nodes_parts:
            nodes_all = np.concatenate(fixed_nodes_parts)
            values_all = np.concatenate(fixed_values_parts)
            order = np.argsort(nodes_all, kind="stable")
            uniq, idx = np.unique(nodes_all[order], return_index=True)
            fixed_temp_nodes = uniq
            fixed_temp_values = values_all[order][idx]
        else:
            fixed_temp_nodes = np.empty(0, dtype=np.int64)
            fixed_temp_values = np.empty(0, dtype=np.float64)

        return heat, fixed_temp_nodes, fixed_temp_values, robin

    def thermal_dof_count(self):
        return self.total_nodes

    def combined_dof_count(self):
        return self.total_dofs + self.total_nodes
