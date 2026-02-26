# imports
import numpy as np
import scipy as sp



# update function so elements can have different A,E,rho
def MK_global(elementi, vozlisca, A, E, rho):
    """
    Assemble global mass and stiffness matrices for a 2D truss
    using a vectorised (loop-free) approach.

    Parameters
    ----------
    elementi : np.ndarray, shape (n_elem, 2)
        Element connectivity — each row holds the indices of the two
        nodes that define a bar element.
    vozlisca : np.ndarray, shape (n_nodes, 2)
        Nodal coordinates (x, y) for every node.
    A : float
        Cross-sectional area of each element [m²].
    E : float
        Young's modulus [Pa].
    rho : float
        Material density [kg/m³].

    Returns
    -------
    M_glob : np.ndarray, shape (n_dof, n_dof)
        Assembled global consistent mass matrix.
    K_glob : np.ndarray, shape (n_dof, n_dof)
        Assembled global stiffness matrix.
    """
    ndim = 2  # 2D truss (2 DOFs per node: ux, uy)
    n_ps = ndim * len(vozlisca)  # total number of DOFs

    # --- Element geometry (vectorised over all elements) --------------------
    # Coordinate differences between end and start node of each element
    dxy = vozlisca[elementi[:, 1]] - vozlisca[elementi[:, 0]]  # (n_elem, 2)
    Le_all = np.hypot(dxy[:, 0], dxy[:, 1])       # element lengths
    alpha_all = np.atan2(dxy[:, 1], dxy[:, 0])     # element orientations

    # --- Element mass matrices (consistent lumping) -------------------------
    # Template for 2-node bar element (4×4 consistent mass matrix pattern)
    template_M = np.array([
        [2, 0, 1, 0],
        [0, 2, 0, 1],
        [1, 0, 2, 0],
        [0, 1, 0, 2]
    ])

    # Scale each element's template by ρ·A·L/6 → (n_elem, 4, 4)
    scale_M = (rho * A * Le_all / 6)[:, None, None]
    M_all = scale_M * template_M[None, :, :]

    # --- Element stiffness matrices -----------------------------------------
    c = np.cos(alpha_all)  # direction cosines
    s = np.sin(alpha_all)

    # Direction vector d = [cos, sin, -cos, -sin] for the outer-product form
    d = np.stack([c, s, -c, -s], axis=1)           # (n_elem, 4)
    # K_e = (EA/L) · d ⊗ d  for each element → (n_elem, 4, 4)
    scale_K = (A * E / Le_all)[:, None, None]
    K_all = scale_K * (d[:, :, None] * d[:, None, :])

    # --- DOF mapping --------------------------------------------------------
    # Map each element's 4 local DOFs to global DOF indices
    # Element [i, j] → global DOFs [2i, 2i+1, 2j, 2j+1]
    dof_map = np.column_stack([
        2 * elementi[:, 0],
        2 * elementi[:, 0] + 1,
        2 * elementi[:, 1],
        2 * elementi[:, 1] + 1,
    ])  # (n_elem, 4)

    # --- Assembly (scatter-add) ---------------------------------------------
    # Expand DOF map to (row, col) pairs for every entry of every element matrix
    rows = dof_map[:, :, None].repeat(4, axis=2).reshape(-1)  # n_elem × 16
    cols = dof_map[:, None, :].repeat(4, axis=1).reshape(-1)  # n_elem × 16

    # Initialise global matrices and scatter-add element contributions
    M_glob = np.zeros((n_ps, n_ps))
    K_glob = np.zeros((n_ps, n_ps))

    np.add.at(M_glob, (rows, cols), M_all.reshape(-1))
    np.add.at(K_glob, (rows, cols), K_all.reshape(-1))

    return M_glob, K_glob


def solve_eigenproblem(M_glob, K_glob, C=None):
    """
    Solve the generalised eigenvalue problem  K φ = ω² M φ, optionally
    incorporating linear constraint equations via a constraint matrix C.

    When a constraint matrix *C* is provided the constraints C · U = 0
    are enforced by computing the null-space operator L (such that
    C · L = 0) and projecting the system into the reduced coordinate
    space Q where U = L · Q:

        L^T M L Q'' + L^T K L Q = L^T F

    The eigenvectors of the reduced system are mapped back to the full
    DOF space via U = L · Q.

    Parameters
    ----------
    M_glob : np.ndarray, shape (n_dof, n_dof)
        Global mass matrix.
    K_glob : np.ndarray, shape (n_dof, n_dof)
        Global stiffness matrix.
    C : np.ndarray, shape (n_constraints, n_dof) or None, optional
        Constraint matrix encoding the linear constraint equations
        C · U = 0 (boundary conditions, kinematic couplings, etc.).
        If *None* (default), no constraints are applied and the full
        system is solved directly.

    Returns
    -------
    eig_vals : np.ndarray, shape (n_free,)
        Eigenvalues (ω²) sorted in ascending order.
    eig_vecs : np.ndarray, shape (n_dof, n_free)
        Eigenvectors mapped back to the full DOF numbering.
        When constraints are present, rows corresponding to
        constrained DOFs are consistently handled through the
        null-space projection.
    """


    # change funciton - upoštevanje povezovalnih enačb in matrike C.

    # Teorija - matriko C tvorimo iz povezovalnih enačb. Lahko zapišemo enačbno C.U = 0, kjer C - constraint matrix.
    # Constraini nam spremenijo število P.S zato uvedemo nov vektor "vpetih" P.S - Q


    # rečemo, da U = L . Q -> kjer je L linearni operator, ki ga iščemo - to je matrika, ki preslika Q -> U
    # Iz C.U = 0 sledi:
    #       C . L . Q = 0 -> Q je neničelna matrika
    # C . L = 0 -> Matrika L predstavlja null space matrike C. To je linearni operator, ki vse elemente C presilka v 0


    # Izračunan imamo L - poglejmo gibalno enačbo
    # M.U'' + K.U = F
    # L.T . M . L . Q'' + L.T . K . L . Q = L.T . F
    # M_red.Q'' + K_red.Q = L.T.F
    # Vidimo da smo naredili podobno kot z modalno dekompozicjo.

    # z novo masno in togostno matriko izračunamo lastne frekvence in vektorje
    
    # lastne vektorje preslikamo v realni prostor preko enačbe U = L.Q
    
    
    
    # v nadaljevanju lahko pogledamo gibalno enacbo. M.U'' + K.U = F
    
    # implementacija v kodi

    # 1. Korak - izračun L in tvorba reduced masne in togostne matrike

    if C is not None:
        L = sp.linalg.null_space(C)
        M_red = L.T @ M_glob @ L
        K_red = L.T @ K_glob @ L

        # 2. Korak izračun lastnih vrednosti
        eig_vals, eig_vecs = sp.linalg.eigh(K_red, M_red)

        # 3. Korka preslikava lastnih vekotrjev v realne koordinate
        eig_vecs = L@eig_vecs

    else:
        eig_vals, eig_vecs = sp.linalg.eigh(K_glob, M_glob)
    
    return eig_vals, eig_vecs
    


