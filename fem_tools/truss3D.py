import numpy as np
import scipy.linalg as spla

def MK_global(elementi, vozlisca, A, E, rho):
    """
    Assemble global mass and stiffness matrices for a 3D Truss structure.

    Each node carries 3 DOFs: ux, uy, uz (translations in x, y, z).
    A 3D truss element only carries axial loads. The local 2x2 stiffness 
    matrix is rotated into the global 6x6 system using the transformation 
    matrix T. The consistent mass matrix is isotropic and therefore 
    formulated directly in the global system.

    Parameters
    ----------
    elementi : np.ndarray, shape (n_elem, 2)
        Element connectivity — each row [i, j] holds the start and end nodes.
    vozlisca : np.ndarray, shape (n_nodes, 3)
        Nodal coordinates (X, Y, Z) for every node.
    A : float
        Cross-sectional area [m²].
    E : float
        Young's modulus [Pa].
    rho : float
        Material density [kg/m³].

    Returns
    -------
    M_glob : np.ndarray, shape (n_dof, n_dof)
        Global consistent mass matrix.
    K_glob : np.ndarray, shape (n_dof, n_dof)
        Global stiffness matrix.
    """
    
    # 1. Local Axial Stiffness (2x2)
    def K_truss_local(A, E, L):
        return (A * E / L) * np.array([[ 1, -1],
                                       [-1,  1]])

    # 2. Consistent Mass Matrix (6x6)
    # Since mass acts identically in X, Y, and Z, this matrix is invariant 
    # to rotation and can be applied directly in the global coordinate system.
    def M_truss_e(A, L, rho):
        return (rho * A * L / 6) * np.array([[2, 0, 0, 1, 0, 0],
                                            [0, 2, 0, 0, 1, 0],
                                            [0, 0, 2, 0, 0, 1],
                                            [1, 0, 0, 2, 0, 0],
                                            [0, 1, 0, 0, 2, 0],
                                            [0, 0, 1, 0, 0, 2]])
    
    # 3. Transformation Matrix (2x6)
    # Maps global displacements (6 DOFs) to local axial displacements (2 DOFs)
    def T_mat(k, l, m):
        return np.array([[k, l, m, 0, 0, 0],
                         [0, 0, 0, k, l, m]])

    n_nodes = len(vozlisca)
    n_dof = n_nodes * 3
    
    M_glob = np.zeros((n_dof, n_dof))
    K_glob = np.zeros((n_dof, n_dof))
    
    for element in elementi:
        # Calculate 3D lengths
        dx, dy, dz = vozlisca[element[1]] - vozlisca[element[0]]
        Le = np.linalg.norm([dx, dy, dz]) 
        
        # Determine direction cosines
        k_cos = dx / Le
        l_cos = dy / Le
        m_cos = dz / Le

        # DOF indices for this element's nodes:[3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
        ind = (3 * element[:, None] + np.arange(3)).flatten()

        # Transform local 2x2 stiffness to global 6x6
        # T^T (6x2) @ K_local (2x2) @ T (2x6) = K_elem_global (6x6)
        T = T_mat(k_cos, l_cos, m_cos)
        K_elem_glob = T.T @ K_truss_local(A, E, Le) @ T
        
        # Add to global matrices
        K_glob[np.ix_(ind, ind)] += K_elem_glob
        M_glob[np.ix_(ind, ind)] += M_truss_e(A, Le, rho)

    return M_glob, K_glob


def solve_eigenproblem(M_glob, K_glob, C=None):
    """
    Solve the generalized eigenvalue problem K φ = ω² M φ.
    Incorporates multipoint constraint equations C * U = 0 via Null-Space Projection.
    """
    if C is not None:
        # Find the null space of the constraint matrix C
        L = spla.null_space(C)
        
        # Project K and M into the reduced, unconstrained subspace
        M_red = L.T @ M_glob @ L
        K_red = L.T @ K_glob @ L

        # Solve the eigenvalue problem in the reduced space
        eig_vals, eig_vecs_red = spla.eigh(K_red, M_red)

        # Map the eigenvectors back to the full DOF space (U = L * Q)
        eig_vecs = L @ eig_vecs_red
    else:
        # Solve unconstrained system directly
        eig_vals, eig_vecs = spla.eigh(K_glob, M_glob)
    
    return eig_vals, eig_vecs