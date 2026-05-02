import numpy as np

class Material:
    """
    Podatki o materialu.
    Določa ime, modul elastičnosti (E), strižni modul (G), gostoto (rho) 
    in Rayleighova koeficienta dušenja (alpha, beta).
    """
    def __init__(self, name: str, E: float, G: float, rho: float, alpha: float = 0.0, beta: float = 0.0):
        self.name = name
        self.E = E
        self.G = G
        self.rho = rho
        # Rayleigh damping - za harmonsko analizo in časovno integracijo
        self.alpha = alpha 
        self.beta = beta

class Section:
    """
    Geometrijske lastnosti prereza.
    Vsebuje površino (A), vztrajnostna momenta (Iy, Iz), torzijski vztrajnostni moment (It) in polarni vztrajnostni moment (Ip).
    """
    def __init__(self, name: str, A: float, Iy: float, Iz: float, It: float,  Ip: float):
        self.name = name
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.It = It
        self.Ip = Ip

class Node:
    """
    Predstavlja vozlišče v prostoru s podanim ID-jem in koordinatami (x, y, z).
    """
    def __init__(self, node_id: int, coords: list[float]):
        self.id = node_id
        self.coords = np.array(coords, dtype=float)

class Element:
    """
    Osnovni končni element, ki povezuje dve vozlišči.
    Ima definirane materialne in geometrijske lastnosti ter orientacijski vektor (v_up).
    """
    def __init__(self, elem_id: int, n1: Node, n2: Node, material: Material, section: Section, v_up: list[float] = [0.0, 0.0, 1.0]):
        self.id = elem_id
        self.n1 = n1
        self.n2 = n2
        self.material = material
        self.section = section
        self.v_up = np.array(v_up, dtype=float)
    
    @property
    def length(self):
        """Izračuna in vrne dolžino elementa."""
        return np.linalg.norm(self.n2.coords - self.n1.coords)

# definiramo ali je element nosilec ali palica
class Truss3D(Element):
    elem_type = "Truss3D"

class Frame3D(Element):
    elem_type = "Frame3D"

class Part:
    """
    Razred za definicijo makro-geometrije konstrukcije.
    Upravlja s ključnimi točkami (keynodes) in linijami, ki jih povezujejo.
    """
    def __init__(self):
        self.keynodes = {}
        self.lines = {} # id -> (node1_id, node2_id)
        self.properties = {} # line_id -> {'material', 'section', 'type', 'v_up'}
        self._next_node_id = 0
        self._next_line_id = 0

    def add_keynode(self, coords: list[float]) -> int:
        """Doda ključno točko - točka, ki definira geometrijo modela."""
        nid = self._next_node_id 
        self.keynodes[nid] = np.array(coords, dtype=float) 
        self._next_node_id += 1
        return nid 

    def add_line(self, n1_id: int, n2_id: int) -> int:
        """Ustvari linijo med dvema podanima ključnima točkama."""
        if n1_id not in self.keynodes or n2_id not in self.keynodes:
            raise ValueError("Keynodes must be defined first.")
        lid = self._next_line_id
        self.lines[lid] = (n1_id, n2_id)
        self._next_line_id += 1
        return lid 

    def assign_property(self, line_ids: list[int], material: Material, section: Section, elem_type: str = 'Frame3D', v_up: list[float] = [0.0, 0.0, 1.0]):
        """
        Določi material, prerez, tip elementa (nosilec ali palica) in v_up izbranim linijam.
        """
        for lid in line_ids:
            self.properties[lid] = {
                'material': material, 
                'section': section, 
                'type': elem_type, 
                'v_up': np.array(v_up, dtype=float)
            }