import numpy as np
import scipy.linalg as spla

class DOFManager:
    """
    Razred za upravljanje prostostnih stopenj in robnih pogojev.
    """
    def __init__(self, mesh):
        """Inicializacija z dano mrežo elementov."""
        self.mesh = mesh
        self.n_nodes = len(mesh.nodes)
        self.n_dof = 6 * self.n_nodes
        
        # Shranjujemo enačbe robnih pogojev kot dict
        # Format: {dof_index: coefficient} -> predstavlja enačbo coef * U[dof] = 0
        self.constraints = []
        
    def add_constraint(self, dof_coeffs: dict):
        """Doda poljubno enačbo robnega pogoja v obliki dict koeficientov."""
        self.constraints.append(dof_coeffs)

    def fix_node(self, node_id: int, dofs: list[int]):
        """
        Zaklene izbrane prostostne stopnje (od 0 do 5) za podano vozlišče.
        """
        for d in dofs:
            idx = 6 * node_id + d
            self.add_constraint({idx: 1.0})

    def fix_all(self, node_id: int):
        """Popolnoma vpne vozlišče (prepreči vse pomike in zasuke)."""
        self.fix_node(node_id, [0, 1, 2, 3, 4, 5])
        
    def pin_node(self, node_id: int):
        """Nepomična podpora vozlišča (translacije zaklenjene, rotacije proste)."""
        self.fix_node(node_id, [0, 1, 2])

    def auto_constrain_trusses(self):
        """
        Samodejno fiksira rotacije vozlišč, ki so povezana izključno s palicami,
        saj palice nimajo rotacijske togosti. Prepreči singularnost matrike.
        """
        has_frame = np.zeros(self.n_nodes, dtype=bool)
        for elem in self.mesh.elements:
            if elem.elem_type == 'Frame3D':
                has_frame[elem.n1.id] = True
                has_frame[elem.n2.id] = True
                
        for i in range(self.n_nodes):
            if not has_frame[i]:
                # Vozlišče nima nosilcev, fiksiramo rotacije (DOFs 3, 4, 5)
                self.fix_node(i, [3, 4, 5])

    def get_C_matrix(self):
        """Sestavi matriko pogojev C, ki opisuje sistem C * U = 0."""
        n_constraints = len(self.constraints)
        if n_constraints == 0:
            return None
            
        C = np.zeros((n_constraints, self.n_dof), dtype=float)
        for i, coeffs in enumerate(self.constraints):
            for dof_idx, val in coeffs.items():
                C[i, dof_idx] = val
        return C

    def get_L_matrix(self):
        """
        Izračuna in vrne  matriko L (ničelni prostor).
        Omogoča preslikavo: U_global = L * U_reduced.
        """
        C = self.get_C_matrix()
            
        if C is None or C.shape[0] == 0:
            return np.eye(self.n_dof)
            
        L = spla.null_space(C)
        return L