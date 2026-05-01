# modul za predpisovanje zunanjih obremenitev (Loads)

import numpy as np
import scipy.spatial as spatial

class LoadManager:
    """
    Upravljalnik obremenitev. Skrbi za dodajanje zunanjih sil 
    in sestavo globalnega vektorja obremenitev.
    """
    def __init__(self, mesh, dof_manager):
        """Inicializacija z mrežo in upravljalnikom prostostnih stopenj."""
        self.mesh = mesh
        self.dof_manager = dof_manager
        self.n_dof = dof_manager.n_dof
        
        # F_glob je zdaj COMPLEX, da podpira fazne zamike (F0 * e^(i*phi))
        self.F_glob = np.zeros(self.n_dof, dtype=complex)
        
        # Zgradimo KDTree za hitro iskanje vozlišč po koordinatah (če želimo obremenitev na točki)
        if len(self.mesh.nodes) > 0:
            coords = np.array([node.coords for node in mesh.nodes])
            self.tree = spatial.cKDTree(coords)
        else:
            self.tree = None

    # --- 1. HARMONSKA
    
    def add_nodal_load(self, node_id: int, fx=0.0, fy=0.0, fz=0.0, mx=0.0, my=0.0, mz=0.0):
        """
        Doda točkovno obremenitev neposredno na določeno vozlišče (preko ID-ja).
        Podpira tudi kompleksne vrednosti (za harmonsko analizo).
        """
        idx = 6 * node_id
        self.F_glob[idx:idx+6] += np.array([fx, fy, fz, mx, my, mz], dtype=complex)

    def add_load_at_point(self, point: list, fx=0.0, fy=0.0, fz=0.0, mx=0.0, my=0.0, mz=0.0):
        """Poišče vozlišče, ki je najbližje podani točki v prostoru, in mu doda obremenitev."""
        dist, node_id = self.tree.query(point)
        self.add_nodal_load(node_id, fx, fy, fz, mx, my, mz)
        return node_id



    # --- 3. DOSTOP DO VEKTORJEV ---

    def get_global_force(self):
        """Vrne globalni kompleksni vektor obremenitev (za Harmonsko analizo)."""
        return self.F_glob.copy()



    def clear_loads(self):
        """
        Počisti vse dodane harmonske / statične sile.
        Uporabno, ko v isti skripti prehajamo med različnimi analizami.
        """
        self.F_glob = np.zeros(self.n_dof, dtype=complex)