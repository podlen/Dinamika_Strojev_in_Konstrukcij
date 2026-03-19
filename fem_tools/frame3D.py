# code for 3d frame structures
import numpy as np
import scipy as sp


# Local DOF order used throughout this module:
# [u1, v1, w1, rx1, ry1, rz1, u2, v2, w2, rx2, ry2, rz2]
LOCAL_DOF_ORDER = (
	"u1", "v1", "w1", "rx1", "ry1", "rz1",
	"u2", "v2", "w2", "rx2", "ry2", "rz2",
)


def local_element_matrices(L, E, G, A, Iy, Iz, J, rho, Ip=None):
	"""
	Build local 12x12 stiffness and consistent mass matrices
	for a 2-node 3D Euler-Bernoulli frame element.

	Parameters
	----------
	L : float
		Element length.
	E, G : float
		Young's and shear modulus.
	A : float
		Cross-sectional area.
	Iy, Iz : float
		Second moments of area around local y and z axes.
	J : float
		Saint-Venant torsion constant (for stiffness).
	rho : float
		Material density.
	Ip : float or None, optional
		Polar mass moment quantity used in torsional inertia term.
		If None, J is used.

	Returns
	-------
	M : np.ndarray, shape (12, 12)
		Local consistent mass matrix.
	K : np.ndarray, shape (12, 12)
		Local stiffness matrix.
	"""
	if L <= 0.0:
		raise ValueError("Element length L must be positive.")

	if Ip is None:
		Ip = J

	# Axial 2x2 block
	k_ax = (E * A / L) * np.array([[1.0, -1.0], [-1.0, 1.0]])
	m_ax = (rho * A * L / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])

	# Torsion 2x2 block
	k_tor = (G * J / L) * np.array([[1.0, -1.0], [-1.0, 1.0]])
	m_tor = (rho * Ip * L / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])

	# Euler-Bernoulli bending 4x4 templates
	# tukaj se lahko doda tudi timoshenko nosilec - ko enkrat to vredu dela lahko dam karkoli
	kb = np.array(
		[
			[12.0, 6.0 * L, -12.0, 6.0 * L],
			[6.0 * L, 4.0 * L * L, -6.0 * L, 2.0 * L * L],
			[-12.0, -6.0 * L, 12.0, -6.0 * L],
			[6.0 * L, 2.0 * L * L, -6.0 * L, 4.0 * L * L],
		]
	)
	mb = np.array(
		[
			[156.0, 22.0 * L, 54.0, -13.0 * L],
			[22.0 * L, 4.0 * L * L, 13.0 * L, -3.0 * L * L],
			[54.0, 13.0 * L, 156.0, -22.0 * L],
			[-13.0 * L, -3.0 * L * L, -22.0 * L, 4.0 * L * L],
		]
	)

	k_bz = (E * Iz / (L ** 3)) * kb
	k_by = (E * Iy / (L ** 3)) * kb
	m_b = (rho * A * L / 420.0) * mb

	K = np.zeros((12, 12), dtype=float)
	M = np.zeros((12, 12), dtype=float)

	def _add_block(mat, idx, block):
		mat[np.ix_(idx, idx)] += block

	# Axial: [u1, u2]
	_add_block(K, [0, 6], k_ax)
	_add_block(M, [0, 6], m_ax)

	# Torsion: [rx1, rx2]
	_add_block(K, [3, 9], k_tor)
	_add_block(M, [3, 9], m_tor)

	# Bending about z -> disp.linalgcement v and rotation rz: [v1, rz1, v2, rz2]
	_add_block(K, [1, 5, 7, 11], k_bz)
	_add_block(M, [1, 5, 7, 11], m_b)

	# Bending about y -> disp.linalgcement w and rotation ry: [w1, ry1, w2, ry2]
	_add_block(K, [2, 4, 8, 10], k_by)
	_add_block(M, [2, 4, 8, 10], m_b)

	return M, K


def transformation_matrix(vozlisce1, vozlisce2):
	"""Return 12x12 local-to-global transformation matrix for a 3D frame element."""
	n1 = np.asarray(vozlisce1, dtype=float)
	n2 = np.asarray(vozlisce2, dtype=float)

	d = n2 - n1
	Le = np.linalg.norm(d)
	if Le <= 0.0:
		raise ValueError("Element length must be positive.")

	# Local x-axis along the element.
	x = d / Le

	# Pick a reference vector that is not parallel to x.
	v_ref = np.array([0.0, 0.0, 1.0])
	if abs(np.dot(x, v_ref)) > 0.99:
		v_ref = np.array([0.0, 1.0, 0.0])

	# Build an orthonormal local basis {x, y, z}.
	y = np.cross(v_ref, x)
	y /= np.linalg.norm(y)
	z = np.cross(x, y)

	lbd = np.vstack((x, y, z))
	T = sp.linalg.block_diag(lbd, lbd, lbd, lbd)
	return T


# assemble global mass and stifness matrix
def M_K_global(vozlisca, elementi, material_params):
	"""
	material_params = [E, G, A, Iy, Iz, J, rho, Ip]
	"""
	vozlisca = np.asarray(vozlisca, dtype=float)
	elementi = np.asarray(elementi, dtype=int)
	n_dof = 6 * len(vozlisca)
	K_glob = np.zeros((n_dof, n_dof), dtype=float)
	M_glob = np.zeros((n_dof, n_dof), dtype=float)

	for element in elementi:
		n1, n2 = int(element[0]), int(element[1])
		Le = np.linalg.norm(vozlisca[n2] - vozlisca[n1])
		M_ele, K_ele = local_element_matrices(Le, *material_params)
		T = transformation_matrix(vozlisca[n1], vozlisca[n2])

		idx = (6 * element[:, None] + np.arange(6)).flatten()
		M_glob[np.ix_(idx, idx)] += T.T @ M_ele @ T
		K_glob[np.ix_(idx, idx)] += T.T @ K_ele @ T

	return M_glob, K_glob



def solve_eigenproblem(M_glob, K_glob, C=None):
    """
    Solve the generalized eigenvalue problem K φ = ω² M φ.
    Incorporates multipoint constraint equations C * U = 0 via Null-Space Projection.
    """
    if C is not None:
        # Find the null space of the constraint matrix C
        L = sp.linalg.null_space(C)
        
        # Project K and M into the reduced, unconstrained subspace
        M_red = L.T @ M_glob @ L
        K_red = L.T @ K_glob @ L

        # Solve the eigenvalue problem in the reduced space
        eig_vals, eig_vecs_red = sp.linalg.eigh(K_red, M_red)

        # Map the eigenvectors back to the full DOF space (U = L * Q)
        eig_vecs = L @ eig_vecs_red
    else:
        # Solve unconstrained system directly
        eig_vals, eig_vecs = sp.linalg.eigh(K_glob, M_glob)
    
    return eig_vals, eig_vecs