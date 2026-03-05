# imports
import numpy as np
import scipy as sp



# update function so elements can have different A,E,rho
def MK_global(elementi, vozlisca, A, E, I, rho):


    # prliminary definitions
    def M_beam_e(A, Le, ρ):
        return ρ*A*Le/420*np.array([[156,     22*Le,    54,    -13*Le],
                            [22*Le,   4*Le**2,  13*Le, -3*Le**2],
                            [54,      13*Le,    156,   -22*Le],
                            [-13*Le, -3*Le**2, -22*Le,  4*Le**2]])
    def M_truss_e(A, L, ρ):
        return  ρ*A*L/6*np.array([
                                [2, 1],
                                [1, 2]
                                ])
    
    def K_beam_e(E, I, Le):
        return E*I/Le**3*np.array([[12,   6*Le,   -12,   6*Le],
                               [6*Le, 4*Le**2,-6*Le, 2*Le**2],
                               [-12, -6*Le,    12,  -6*Le],
                               [6*Le, 2*Le**2,-6*Le, 4*Le**2]])
    
    def K_truss_e(A, E, L):
        return A*E/L* np.array([[1,-1],
                                [-1, 1]])
    
    def T(fi):
        c = np.cos(fi)
        s = np.sin(fi)
        return np.array([[c, -s, 0, 0, 0, 0],
                        [s, c, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, c, -s, 0],
                        [0, 0, 0, s, c, 0],
                        [0, 0, 0, 0, 0, 1]])
    
    def M_ele_unrotated(L, A, rho):
        M = np.zeros(shape=(6,6))
        M[np.ix_([0,3], [0,3])] += M_truss_e(A, L, rho)
        M[np.ix_([1,2,4,5], [1,2,4,5])] += M_beam_e(A, L, rho)
        return M
    
    def K_ele_unrotated(L, A, E, I):
        K = np.zeros(shape=(6,6))
        K[np.ix_([0,3], [0,3])] += K_truss_e(A, E, L)
        K[np.ix_([1,2,4,5], [1,2,4,5])] += K_beam_e(E, I, L)
        return K
    
    # main code -> update to use without for loop

    M_glob, K_glob = np.zeros(shape=(len(vozlisca)*3,len(vozlisca)*3)), np.zeros(shape=(len(vozlisca)*3,len(vozlisca)*3))
    for element in elementi:
        dx, dy = vozlisca[element[1]] - vozlisca[element[0]]
        Le = np.hypot(dx, dy)
        ind = (3*element[:, None] + np.arange(3)).flatten()

        fi = np.atan2(dy, dx)

        M_glob[np.ix_(ind, ind)] += T(fi).T @ M_ele_unrotated(Le, A, rho) @ T(fi)
        K_glob[np.ix_(ind, ind)] += T(fi).T @ K_ele_unrotated(Le, A, E, I) @ T(fi)

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
    


