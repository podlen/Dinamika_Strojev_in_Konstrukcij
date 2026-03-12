import numpy as np


class MeshManager:
    """Dictionary-based mesh builder that converts to arrays for FEM solvers.

    Keynodes (structural points where boundary conditions are applied)
    are always stored at indices 0 .. N-1.  Interior nodes created by
    ``add_beam`` get IDs starting from N.

    Usage
    -----
    >>> mesh = MeshManager(Le_target=0.05)
    >>> mesh.set_keynodes([[0, 0], [1.2, 0], [1.2, 0.5]])
    >>> mesh.add_beam(0, 1)          # subdivide segment 0→1
    >>> mesh.add_beam(1, 2)          # subdivide segment 1→2
    >>> vozlisca, elementi = mesh.to_arrays()
    """

    def __init__(self, Le_target: float):
        self.Le = Le_target,
        self.nodes = {}
        self.elements = []
        self.keynode_count = 0
        self._next_node_id = 0

    # ── keynodes ────────────────────────────────────────────────────────
    def set_keynodes(self, keynodes_coords):
        """Define keynodes — always stored at indices 0 .. N-1.

        Parameters
        ----------
        keynodes_coords : list of [x, y]
            Coordinates of the structural key-points.
        """
        for i, coord in enumerate(keynodes_coords):
            self.nodes[i] = np.array(coord, dtype=float)
        self.keynode_count = len(keynodes_coords)
        self._next_node_id = self.keynode_count

    # ── beam subdivision ────────────────────────────────────────────────
    def add_beam(self, start_id: int, end_id: int):
        """Subdivide the segment between two existing nodes into elements.

        The number of elements is chosen so that each element length
        is close to (but not larger than) ``Le_target``.

        Parameters
        ----------
        start_id, end_id : int
            Node IDs (typically keynode indices) of the beam endpoints.
        """
        p1 = self.nodes[start_id]
        p2 = self.nodes[end_id]

        L = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        n_elem = max(1, int(np.ceil(L / self.Le)))

        x_coords = np.linspace(p1[0], p2[0], n_elem + 1)
        y_coords = np.linspace(p1[1], p2[1], n_elem + 1)

        last_node_id = start_id

        for i in range(1, n_elem):
            new_coords = np.array([x_coords[i], y_coords[i]])

            found, existing_id = self._find_node(new_coords)
            if found:
                current_node_id = existing_id
            else:
                current_node_id = self._next_node_id
                self.nodes[current_node_id] = new_coords
                self._next_node_id += 1

            self.elements.append([last_node_id, current_node_id])
            last_node_id = current_node_id

        # last sub-element connects to the end keynode
        self.elements.append([last_node_id, end_id])

    # ── conversion to numpy arrays ──────────────────────────────────────
    def to_arrays(self):
        """Convert the dictionary mesh into numpy arrays.

        Returns
        -------
        vozlisca : np.ndarray, shape (n_nodes, 2)
            Nodal coordinates ordered by node ID (keynodes first).
        elementi : np.ndarray, shape (n_elem, 2)
            Element connectivity (pairs of node indices).
        """
        n_nodes = len(self.nodes)
        vozlisca = np.empty((n_nodes, 2))
        for node_id, coords in self.nodes.items():
            vozlisca[node_id] = coords

        elementi = np.array(self.elements, dtype=int)
        return vozlisca, elementi

    @property
    def keynode_ids(self):
        """Return the list of keynode indices (always 0 .. N-1)."""
        return list(range(self.keynode_count))

    # ── internal helpers ────────────────────────────────────────────────
    def _find_node(self, coords, tol=1e-6):
        """Check whether a node at *coords* already exists.

        Returns (True, node_id) or (False, None).
        """
        for node_id, pos in self.nodes.items():
            if np.allclose(coords, pos, atol=tol):
                return True, node_id
        return False, None


