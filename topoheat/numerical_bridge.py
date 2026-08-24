import numpy as np


class NumericalBridge:
    """
    Translates a validated problem specification into global load and
    constraint arrays for a downstream finite element solver.

    domain_dimensions counts ELEMENTS per axis. Node counts derive as
    element count plus one per axis, so a grid of NEX by NEY elements
    carries (NEX + 1) by (NEY + 1) nodes.

    dof_ordering selects the global layout:
      node_major: global index = node * spatial_dims + component
      dof_major:  global index = component * total_nodes + node
    The target solver dictates the choice; nothing here is fixed in place.
    """

    def __init__(self, domain_dimensions, dof_ordering="node_major"):
        self.element_dims = np.asarray(domain_dimensions, dtype=np.int64)
        self.spatial_dims = int(self.element_dims.size)
        self.node_dims = self.element_dims + 1
        self.total_nodes = int(np.prod(self.node_dims))
        self.total_dofs = self.total_nodes * self.spatial_dims
        if dof_ordering not in ("node_major", "dof_major"):
            raise ValueError("dof_ordering must be node_major or dof_major")
        self.dof_ordering = dof_ordering

    def _global_index(self, components, nodes):
        """
        Maps component and node arrays to global dof indices under the
        active ordering. Inputs broadcast against each other.
        """
        components = np.asarray(components, dtype=np.int64)
        nodes = np.asarray(nodes, dtype=np.int64)
        if self.dof_ordering == "node_major":
            return nodes * self.spatial_dims + components
        return components * self.total_nodes + nodes

    def _check_nodes(self, nodes):
        if nodes.size and (nodes.min() < 0 or nodes.max() >= self.total_nodes):
            raise ValueError("Node index falls outside the derived node grid")

    def assemble(self, supports, loads):
        """
        Builds the global force vector and the sorted unique array of fixed
        dofs. Each load scatters into the force vector through a single
        vectorized add over the outer pairing of components and nodes; each
        support gathers into the fixed set through one broadcast. No Python
        loop walks individual nodes or components.
        """
        force_vector = np.zeros(self.total_dofs, dtype=np.float64)
        components = np.arange(self.spatial_dims, dtype=np.int64)

        for load in loads:
            nodes = np.asarray(load["node_indices"], dtype=np.int64)
            self._check_nodes(nodes)
            forces = np.asarray(load["force_vector"], dtype=np.float64)
            grid = self._global_index(components[:, None], nodes[None, :])
            magnitudes = np.broadcast_to(forces[:, None], grid.shape)
            np.add.at(force_vector, grid.ravel(), magnitudes.ravel())

        fixed_parts = []
        for support in supports:
            nodes = np.asarray(support["node_indices"], dtype=np.int64)
            self._check_nodes(nodes)
            comps = np.asarray(support["fixed_components"], dtype=np.int64)
            if comps.size and (comps.min() < 0 or comps.max() >= self.spatial_dims):
                raise ValueError(
                    "fixed_components index out of range for the spatial dimension "
                    f"(got {comps.tolist()}, valid 0..{self.spatial_dims - 1})")
            grid = self._global_index(comps[:, None], nodes[None, :])
            fixed_parts.append(grid.ravel())

        if fixed_parts:
            fixed_dofs = np.unique(np.concatenate(fixed_parts))
        else:
            fixed_dofs = np.empty(0, dtype=np.int64)

        return force_vector, fixed_dofs

    def check_kinematic_stability(self, fixed_dofs):
        """
        Tests whether the supports remove every global rigid body mode.

        The routine builds the rigid body mode matrix R for the node grid,
        selects the rows the supports constrain, and checks that the
        selected block carries full column rank. Full column rank means no
        nonzero rigid mode survives the constraints.

        This is necessary, not sufficient. The check inspects only the global
        rigid modes of the full grid. It ignores element connectivity and the
        material density field, so it cannot detect internal mechanisms that
        appear when low density regions disconnect during optimization. True
        stability requires the assembled stiffness matrix, evaluated on the
        current density field, to be positive definite. Treat this as a fast
        precondition, not a guarantee.
        """
        dim = self.spatial_dims
        expected_modes = dim * (dim + 1) // 2
        if fixed_dofs.size < expected_modes:
            return False

        coords = self._node_coordinates()
        rigid_modes = self._rigid_body_modes(coords)

        # The projection C @ R reduces to the rows of R at the fixed dof
        # indices. No dense constraint matrix is allocated, and the rank
        # decomposition runs on an array of width at most six.
        projection = rigid_modes[fixed_dofs, :]
        rank = np.linalg.matrix_rank(projection, tol=1e-9)
        return rank == expected_modes

    def _node_coordinates(self):
        # Node i must map to the SAME physical coordinate the downstream mesh uses.
        # The solver numbers nodes with the first axis varying fastest
        # (Mesh.nid = iy*(nx+1)+ix, i.e. x inner), so ravel in Fortran order to
        # make axis 0 the fastest index. Using C order here transposes the grid
        # and breaks the rigid-body-mode check on any non-square (nx != ny) grid.
        axes = [np.arange(n) for n in self.node_dims]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.column_stack([m.ravel(order="F") for m in mesh])

    def _rigid_body_modes(self, coords):
        dim = self.spatial_dims
        n = self.total_nodes
        all_nodes = np.arange(n, dtype=np.int64)

        def placement(component, values):
            mode = np.zeros(self.total_dofs)
            idx = self._global_index(np.full(n, component, dtype=np.int64), all_nodes)
            mode[idx] = values
            return mode

        modes = []
        for d in range(dim):
            modes.append(placement(d, np.ones(n)))

        if dim == 2:
            r = np.zeros(self.total_dofs)
            r[self._global_index(np.zeros(n, np.int64), all_nodes)] = -coords[:, 1]
            r[self._global_index(np.ones(n, np.int64), all_nodes)] = coords[:, 0]
            modes.append(r)
        else:
            for a, b in [(1, 2), (2, 0), (0, 1)]:  # rotations about x, y, z
                r = np.zeros(self.total_dofs)
                r[self._global_index(np.full(n, a, np.int64), all_nodes)] = -coords[:, b]
                r[self._global_index(np.full(n, b, np.int64), all_nodes)] = coords[:, a]
                modes.append(r)

        return np.column_stack(modes)
