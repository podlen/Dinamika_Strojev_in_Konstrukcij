import numpy as np
import scipy.spatial as spatial

class LoadManager:
    """
    Za definicijo zunanjih obremenitev za harmonsko analizo.
    """
    def __init__(self, mesh, dof_manager):
        self.mesh = mesh
        self.dof_manager = dof_manager
        self.n_dof = dof_manager.n_dof
        
        self.F_glob = np.zeros(self.n_dof, dtype=complex) # complex za fazne zamike
        
        # Zgradimo KDTree za hitro iskanje vozlišč po koordinatah (če želimo obremenitev na točki)
        if len(self.mesh.nodes) > 0:
            coords = np.array([node.coords for node in mesh.nodes])
            self.tree = spatial.cKDTree(coords)
        else:
            self.tree = None

    
    def add_nodal_load(self, node_id: int, fx=0.0, fy=0.0, fz=0.0, mx=0.0, my=0.0, mz=0.0):
        """
        Doda točkovno obremenitev neposredno na določeno vozlišče (preko ID-ja).
        """
        idx = 6 * node_id
        self.F_glob[idx:idx+6] += np.array([fx, fy, fz, mx, my, mz], dtype=complex)

    def add_load_at_point(self, point: list, fx=0.0, fy=0.0, fz=0.0, mx=0.0, my=0.0, mz=0.0):
        """Poišče vozlišče, ki je najbližje podani točki v prostoru, in mu doda obremenitev."""
        dist, node_id = self.tree.query(point)
        self.add_nodal_load(node_id, fx, fy, fz, mx, my, mz)
        return node_id


    def get_global_force(self):
        """Vrne globalni kompleksni vektor obremenitev (za Harmonsko analizo)."""
        return self.F_glob.copy()

    def clear_loads(self):
        """
        Počisti vse dodane sile.
        """
        self.F_glob = np.zeros(self.n_dof, dtype=complex)