# modul za pripravo mreže

import numpy as np
from .geometry import Node, Truss3D, Frame3D, Part

class Mesh:
    """
    Razred za generiranje mreže (diskretizacijo).
    Makro-geometrijo (Part) razdeli na končne elemente in pripadajoča vozlišča.
    """
    def __init__(self):
        self.nodes = [] # List of Node objects
        self.elements = [] # List of Element objects
        self._node_id_map = {}
    
    def generate_mesh(self, part: Part, global_size: float = None, num_elements_per_line: int = None):
        """
        Zgenerira mrežo končnih elementov iz definiranih linij makro-geometrije.
        Uporabi določeno globalno velikost elementa (global_size) ali 
        določeno število elementov na linijo (num_elements_per_line).
        """
        # Kopiranje ključnih točk
        node_count = 0
        elem_count = 0
        
        # ustvarjanje objektov Node iz ključnih točk
        for kn_id, coords in part.keynodes.items():
            node = Node(node_count, coords.tolist())
            self.nodes.append(node)
            self._node_id_map[f"key_{kn_id}"] = node_count
            node_count += 1 
            
        for lid, (n1_id, n2_id) in part.lines.items():

            # lastnosti linije
            prop = part.properties.get(lid)
            if not prop:
                raise ValueError(f"Line {lid} has no properties assigned.")
                
            mat = prop['material']
            sec = prop['section']
            etype = prop['type']
            v_up = prop['v_up'] # Pridobimo orientacijski vektor
            
            # Pridobimo dejanska vozlišča (Node objects) za začetek in konec linije
            start_node_idx = self._node_id_map[f"key_{n1_id}"]
            end_node_idx = self._node_id_map[f"key_{n2_id}"]
            start_node = self.nodes[start_node_idx]
            end_node = self.nodes[end_node_idx]
            
            # CE JE TRUSS -> NE DISKRETIZIRAMO naredimo en element in gremo na naslednjo linijo.
            if etype == "Truss3D":
                elem = Truss3D(elem_count, start_node, end_node, mat, sec, v_up)
                self.elements.append(elem)
                elem_count += 1
                continue # Preskoči preostanek in pojdi na naslednjo linijo
            
            # --- Od tu naprej se koda izvede SAMO za Frame3D ---
            
            p1 = part.keynodes[n1_id] 
            p2 = part.keynodes[n2_id]
            L = np.linalg.norm(p2 - p1)
            
            if num_elements_per_line is not None:
                n_elem = num_elements_per_line
            elif global_size is not None:
                n_elem = max(1, int(np.ceil(L / global_size)))
            else:
                n_elem = 1
                
            # diskretizacija linije na elemente
            x_c = np.linspace(p1[0], p2[0], n_elem + 1)
            y_c = np.linspace(p1[1], p2[1], n_elem + 1)
            z_c = np.linspace(p1[2], p2[2], n_elem + 1)
            
            last_node_idx = start_node_idx
            
            # definiranje manjših elementov (vmesnih)
            for i in range(1, n_elem):
                coords = [x_c[i], y_c[i], z_c[i]]
                node = Node(node_count, coords)
                self.nodes.append(node)
                current_node_idx = node_count
                node_count += 1 
                
                # Dodajanje Frame3D elementa
                elem = Frame3D(elem_count, self.nodes[last_node_idx], node, mat, sec, v_up)
                self.elements.append(elem)
                elem_count += 1
                last_node_idx = current_node_idx
                
            # Zadnji element do končnega vozlišča (da nimamo luknje na koncu)
            elem = Frame3D(elem_count, self.nodes[last_node_idx], end_node, mat, sec, v_up)
            self.elements.append(elem)
            elem_count += 1