import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.linalg as spla
import tqdm as tq

class GlobalAssembly:
    """
    Sestavljanje globalnih matrik sistema (masa, togost, dušenje).
    """
    def __init__(self, mesh, dof_manager):
        self.mesh = mesh
        self.dof_manager = dof_manager
        self.n_dof = dof_manager.n_dof

    def assemble_matrices(self):
        """Sestavi in vrne globalne matrike K, M in C za celoten sistem (brez robnih pogojev)."""
        Ne = len(self.mesh.elements)
        if Ne == 0:
            return sps.coo_matrix((self.n_dof, self.n_dof)), sps.coo_matrix((self.n_dof, self.n_dof)), sps.coo_matrix((self.n_dof, self.n_dof))
        
        # Lastnosti
        E = np.zeros(Ne)
        G = np.zeros(Ne)
        A = np.zeros(Ne)
        Iy = np.zeros(Ne)
        Iz = np.zeros(Ne)
        It = np.zeros(Ne)
        Ip = np.zeros(Ne)
        rho = np.zeros(Ne)
        alpha = np.zeros(Ne)
        beta = np.zeros(Ne)

        v_ref = np.zeros(shape=(Ne, 3)) 
        
        n1_idx = np.zeros(Ne, dtype=int)
        n2_idx = np.zeros(Ne, dtype=int)
        
        for i, e in enumerate(self.mesh.elements):
            n1_idx[i] = e.n1.id
            n2_idx[i] = e.n2.id
            E[i] = e.material.E
            rho[i] = e.material.rho
            A[i] = e.section.A
            alpha[i] = e.material.alpha
            beta[i] = e.material.beta
            
            # Palice imajo strižne in upogibne lastnosti enake nič
            if e.elem_type == 'Frame3D':
                G[i] = e.material.G
                Iy[i] = e.section.Iy
                Iz[i] = e.section.Iz
                Ip[i] = e.section.Ip
                It[i] = e.section.It
                v_ref[i] = e.v_up

        # Geometrijske lastnosti
        coords = np.array([node.coords for node in self.mesh.nodes])
        c1 = coords[n1_idx]
        c2 = coords[n2_idx]
        d = c2 - c1
        L = np.linalg.norm(d, axis=1)
        
        # Lokalne matrike
        K_local = np.zeros((Ne, 12, 12))
        M_local = np.zeros((Ne, 12, 12))
        
        # Aksialno
        k_ax = E * A / L
        m_ax = rho * A * L / 6.0
        
        K_local[:, 0, 0] = k_ax
        K_local[:, 6, 6] = k_ax
        K_local[:, 0, 6] = -k_ax
        K_local[:, 6, 0] = -k_ax
        
        M_local[:, 0, 0] = 2 * m_ax
        M_local[:, 6, 6] = 2 * m_ax
        M_local[:, 0, 6] = m_ax
        M_local[:, 6, 0] = m_ax
        
        # Torzija
        k_tor = G * It / L
        m_tor = rho * Ip * L / 6.0
        
        K_local[:, 3, 3] = k_tor
        K_local[:, 9, 9] = k_tor
        K_local[:, 3, 9] = -k_tor
        K_local[:, 9, 3] = -k_tor
        
        M_local[:, 3, 3] = 2 * m_tor
        M_local[:, 9, 9] = 2 * m_tor
        M_local[:, 3, 9] = m_tor
        M_local[:, 9, 3] = m_tor
        
        # Upogib okoli Z
        k_bz_12 = 12 * E * Iz / L**3
        k_bz_6  = 6 * E * Iz / L**2
        k_bz_4  = 4 * E * Iz / L
        k_bz_2  = 2 * E * Iz / L
        
        K_local[:, 1, 1] = k_bz_12
        K_local[:, 1, 5] = k_bz_6
        K_local[:, 1, 7] = -k_bz_12
        K_local[:, 1, 11] = k_bz_6
        
        K_local[:, 5, 1] = k_bz_6
        K_local[:, 5, 5] = k_bz_4
        K_local[:, 5, 7] = -k_bz_6
        K_local[:, 5, 11] = k_bz_2
        
        K_local[:, 7, 1] = -k_bz_12
        K_local[:, 7, 5] = -k_bz_6
        K_local[:, 7, 7] = k_bz_12
        K_local[:, 7, 11] = -k_bz_6
        
        K_local[:, 11, 1] = k_bz_6
        K_local[:, 11, 5] = k_bz_2
        K_local[:, 11, 7] = -k_bz_6
        K_local[:, 11, 11] = k_bz_4
        
        m_b = rho * A * L / 420.0
        M_local[:, 1, 1] = 156 * m_b
        M_local[:, 1, 5] = 22 * L * m_b
        M_local[:, 1, 7] = 54 * m_b
        M_local[:, 1, 11] = -13 * L * m_b
        
        M_local[:, 5, 1] = 22 * L * m_b
        M_local[:, 5, 5] = 4 * L**2 * m_b
        M_local[:, 5, 7] = 13 * L * m_b
        M_local[:, 5, 11] = -3 * L**2 * m_b
        
        M_local[:, 7, 1] = 54 * m_b
        M_local[:, 7, 5] = 13 * L * m_b
        M_local[:, 7, 7] = 156 * m_b
        M_local[:, 7, 11] = -22 * L * m_b
        
        M_local[:, 11, 1] = -13 * L * m_b
        M_local[:, 11, 5] = -3 * L**2 * m_b
        M_local[:, 11, 7] = -22 * L * m_b
        M_local[:, 11, 11] = 4 * L**2 * m_b

        # Upogib okol Y
        k_by_12 = 12 * E * Iy / L**3
        k_by_6  = 6 * E * Iy / L**2
        k_by_4  = 4 * E * Iy / L
        k_by_2  = 2 * E * Iy / L

        K_local[:, 2, 2] = k_by_12
        K_local[:, 2, 4] = k_by_6
        K_local[:, 2, 8] = -k_by_12
        K_local[:, 2, 10] = k_by_6
        
        K_local[:, 4, 2] = k_by_6
        K_local[:, 4, 4] = k_by_4
        K_local[:, 4, 8] = -k_by_6
        K_local[:, 4, 10] = k_by_2
        
        K_local[:, 8, 2] = -k_by_12
        K_local[:, 8, 4] = -k_by_6
        K_local[:, 8, 8] = k_by_12
        K_local[:, 8, 10] = -k_by_6
        
        K_local[:, 10, 2] = k_by_6
        K_local[:, 10, 4] = k_by_2
        K_local[:, 10, 8] = -k_by_6
        K_local[:, 10, 10] = k_by_4
        
        M_local[:, 2, 2] = 156 * m_b
        M_local[:, 2, 4] = 22 * L * m_b
        M_local[:, 2, 8] = 54 * m_b
        M_local[:, 2, 10] = -13 * L * m_b
        
        M_local[:, 4, 2] = 22 * L * m_b
        M_local[:, 4, 4] = 4 * L**2 * m_b
        M_local[:, 4, 8] = 13 * L * m_b
        M_local[:, 4, 10] = -3 * L**2 * m_b
        
        M_local[:, 8, 2] = 54 * m_b
        M_local[:, 8, 4] = 13 * L * m_b
        M_local[:, 8, 8] = 156 * m_b
        M_local[:, 8, 10] = -22 * L * m_b
        
        M_local[:, 10, 2] = -13 * L * m_b
        M_local[:, 10, 4] = -3 * L**2 * m_b
        M_local[:, 10, 8] = -22 * L * m_b
        M_local[:, 10, 10] = 4 * L**2 * m_b

        # Popravek za palice
        is_truss = np.array([e.elem_type == 'Truss3D' for e in self.mesh.elements])
        if np.any(is_truss):
            # Reset masnih matrik za palice
            M_local[is_truss, :, :] = 0.0
            
            # Masa za palico
            m_ax_truss = (rho * A * L / 6.0)[is_truss]
            
            # X smer (DOFs 0, 6)
            M_local[is_truss, 0, 0] = 2 * m_ax_truss
            M_local[is_truss, 6, 6] = 2 * m_ax_truss
            M_local[is_truss, 0, 6] = m_ax_truss
            M_local[is_truss, 6, 0] = m_ax_truss
            
            # Y smer (DOFs 1, 7)
            M_local[is_truss, 1, 1] = 2 * m_ax_truss
            M_local[is_truss, 7, 7] = 2 * m_ax_truss
            M_local[is_truss, 1, 7] = m_ax_truss
            M_local[is_truss, 7, 1] = m_ax_truss
            
            # Z smer (DOFs 2, 8)
            M_local[is_truss, 2, 2] = 2 * m_ax_truss
            M_local[is_truss, 8, 8] = 2 * m_ax_truss
            M_local[is_truss, 2, 8] = m_ax_truss
            M_local[is_truss, 8, 2] = m_ax_truss

        # Dušenje
        C_local = alpha[:, None, None] * M_local + beta[:, None, None] * K_local

        # Rotacija
        x = d / L[:, None]
        z = np.cross(x,v_ref)

        z_norm = np.linalg.norm(z, axis=1)       
        z = z/z_norm[:, None] # normiranje vektorja

        y = np.cross(z,x)

        lbd = np.stack((x,y,z), axis=1) # shape (Ne, 3, 3)
        
        T = np.zeros((Ne, 12, 12))
        T[:, 0:3, 0:3] = lbd
        T[:, 3:6, 3:6] = lbd
        T[:, 6:9, 6:9] = lbd
        T[:, 9:12, 9:12] = lbd
        
        # Množenje vseh elementov z einsum
        K_global_ele = np.einsum('nji, njk, nkl -> nil', T, K_local, T)
        M_global_ele = np.einsum('nji, njk, nkl -> nil', T, M_local, T)
        C_global_ele = np.einsum('nji, njk, nkl -> nil', T, C_local, T)

        # Indeski za globalno matriko.
        dofs = np.zeros((Ne, 12), dtype=int)
        dofs[:, 0:6] = 6 * n1_idx[:, None] + np.arange(6)
        dofs[:, 6:12] = 6 * n2_idx[:, None] + np.arange(6)

        row_idx = np.repeat(dofs, 12, axis=1).flatten()
        col_idx = np.tile(dofs, (1, 12)).flatten()

        K_sys = sps.coo_matrix((K_global_ele.flatten(), (row_idx, col_idx)), shape=(self.n_dof, self.n_dof))
        M_sys = sps.coo_matrix((M_global_ele.flatten(), (row_idx, col_idx)), shape=(self.n_dof, self.n_dof))
        C_sys = sps.coo_matrix((C_global_ele.flatten(), (row_idx, col_idx)), shape=(self.n_dof, self.n_dof))
        
        return K_sys, M_sys, C_sys


class Solvers:
    def __init__(self, K, M, C_mat, dof_manager):
        self.K = K.tocsr()
        self.M = M.tocsr()
        self.C_mat = C_mat.tocsr() if C_mat is not None else None
        self.dof_manager = dof_manager

    def project_matrix(self, matrix):
        """Pomožna funkcija za projekcijo globalne matrike v reduciran prostor prek matrike L."""
        L = self.dof_manager.get_L_matrix()
        reduced_matrix = L.T @ matrix @ L
        return reduced_matrix.toarray() if sps.issparse(reduced_matrix) else reduced_matrix


    # Modalna analiza
    def solve_eigen(self):
        """
        Modalna analiza (lastne vrednosti).
        Izračuna naravne frekvence in lastne oblike (mode) za reduciran sistem.
        """
        K_red_dense = self.project_matrix(self.K)
        M_red_dense = self.project_matrix(self.M)
        
        eig_vals, eig_vecs_red = spla.eigh(K_red_dense, M_red_dense)
        L = self.dof_manager.get_L_matrix()
        eig_vecs_full = L @ eig_vecs_red
        return eig_vals, eig_vecs_full
        
    
    # Harmonska analiza 
    def solve_harmonic(self, F_glob, omega_sweep):
        """
        Harmonska analiza z metodo modalne superpozicije.
        Izračuna odziv sistema na harmonsko vzbujanje preko spektra frekvenc (omega_sweep).
        """
        # 1. Projekcija matrik v nereduciran prostor
        L = self.dof_manager.get_L_matrix()
        K_red_dense = self.project_matrix(self.K)
        M_red_dense = self.project_matrix(self.M)
        
        if self.C_mat is not None:
            C_red_dense = self.project_matrix(self.C_mat)
        else:
            C_red_dense = np.zeros_like(K_red_dense)

        eig_vals, mod_mat = spla.eigh(K_red_dense, M_red_dense)
        
        F_modal = mod_mat.T @ L.T @ F_glob 
        
        C_modal_matrix = mod_mat.T @ C_red_dense @ mod_mat
        c_m = np.diag(C_modal_matrix) 

        w = omega_sweep[None, :]
        wn2 = eig_vals[:, None]
        c = c_m[:, None]
        F_m = F_modal[:, None]
        H = F_m / (-w**2 + 1j * w * c + wn2)

        U_red = mod_mat @ H           
        U_full = L @ U_red             

        return U_full


    
    def solve_integration_ivp(self, force_func, x0, dx0, t_span, t_eval, method='Radau'):
        """
        Časovna integracija z solve_ivp.
        
        Args:
            force_func: Funkcija f(t), ki vrne globalni vektor obremenitve ob času t.
            x0: Začetni pomiki.
            dx0: Začetne hitrosti.
            t_span: Interval integracije (t_start, t_end).
            t_eval: Časovne točke, v katerih želimo rešitev.
            method: Metoda integracije (privzeto 'Radau', dobro za toge enačbe).
        """
        import scipy.integrate as integrate

        L = self.dof_manager.get_L_matrix()
        K_red = self.project_matrix(self.K)
        M_red = self.project_matrix(self.M)
        
        if self.C_mat is not None:
            C_red = self.project_matrix(self.C_mat)
        else:
            C_red = np.zeros_like(K_red)

        N = M_red.shape[0]
        M_inv = np.linalg.inv(M_red)

        D = np.block([
            [-M_inv @ C_red, -M_inv @ K_red],
            [np.eye(N), np.zeros((N, N))]
        ])
        def int_fun(t, y):
            F_glob = force_func(t)
            F_red = L.T @ F_glob
            
            h = np.block([
                M_inv @ F_red,
                np.zeros(N)
            ])
            
            return D @ y + h

        y0 = np.concatenate([L.T@dx0, L.T@x0])
        res = integrate.solve_ivp(fun=int_fun, t_span=t_span, y0=y0, method=method, t_eval=t_eval)

        q_dot_red, q_red = np.split(res.y,2)
        
        U_full = L @ q_red

        return U_full
    
    def solve_integration_newmark(self, force_array: np.ndarray, dt: float, x_0: np.ndarray, v_0: np.ndarray, gamma: float, beta: float):
        
        L = self.dof_manager.get_L_matrix()
        K_red = self.project_matrix(self.K)
        M_red = self.project_matrix(self.M)
        C_red = self.project_matrix(self.C_mat) if self.C_mat is not None else np.zeros_like(K_red)

        F_red = L.T @ force_array
        u0_red = L.T @ x_0
        v0_red = L.T @ v_0

        n_dofs = F_red.shape[0]
        n_steps = F_red.shape[1]

        # Začetni pospešek
        # solve: M * a0 = F0 - C*v0 - K*u0
        RHS_0 = F_red[:, 0] - C_red @ v0_red - K_red @ u0_red
        a0_red = spla.solve(M_red, RHS_0)

        # Koeficienti
        a1 = (1 / (beta * dt**2)) * M_red + (gamma / (beta * dt)) * C_red
        a2 = (1 / (beta * dt)) * M_red + (gamma / beta - 1) * C_red
        a3 = (1 / (2 * beta) - 1) * M_red + dt * (gamma / (2 * beta) - 1) * C_red

        # Efektivna togostna matrika
        K_eff = K_red + a1
        
        # Faktorizacija
        lu, piv = spla.lu_factor(K_eff)

        u = np.zeros((n_dofs, n_steps))
        v = np.zeros((n_dofs, n_steps))
        a = np.zeros((n_dofs, n_steps))

        u[:, 0] = u0_red
        v[:, 0] = v0_red
        a[:, 0] = a0_red

        # Algoritem
        for i in tq.tqdm(range(n_steps - 1)):
            # Efektivna sila
            p = F_red[:, i+1] + a1 @ u[:, i] + a2 @ v[:, i] + a3 @ a[:, i]
            # Reševanje pomikov
            u[:, i+1] = spla.lu_solve((lu, piv), p)
            # Reševanje hitrosti 
            v[:, i+1] = (gamma / (beta * dt)) * (u[:, i+1] - u[:, i]) + (1 - gamma / beta) * v[:, i] + dt * (1 - gamma / (2 * beta)) * a[:, i]
            # Reševanje pospeškov
            a[:, i+1] = (1 / (beta * dt**2)) * (u[:, i+1] - u[:, i]) - (1 / (beta * dt)) * v[:, i] - (1 / (2 * beta) - 1) * a[:, i]
        
        U_full = L @ u
        V_full = L @ v
        A_full = L @ a

        return U_full, V_full, A_full

    def solve_integration_fd(self, force_array: np.ndarray, dt: float, x_0: np.ndarray, v_0: np.ndarray):
        L = self.dof_manager.get_L_matrix()
        K_glob_rp = self.project_matrix(self.K)
        M_glob_rp = self.project_matrix(self.M)
        C_glob_rp = self.project_matrix(self.C_mat) if self.C_mat is not None else np.zeros_like(K_glob_rp)

        F_glob_rp = L.T @ force_array
        Q0 = L.T @ x_0
        dQ0 = L.T @ v_0

        n_step = F_glob_rp.shape[1]

        q_mkr = []

        # ničti korak (k = 0, t = 0)
        q0 = np.hstack([dQ0, Q0])
        q_mkr_0 = Q0
        q_mkr.append(q_mkr_0)
        
        if n_step < 2:
            return L @ np.array(q_mkr).T

        # Prvi korak
        b1 = (2/dt**2 * M_glob_rp - K_glob_rp) @ Q0 + (2/dt * M_glob_rp - C_glob_rp) @ dQ0 + F_glob_rp[:, 0]
        q_mkr_1 = dt**2/2*np.linalg.inv(M_glob_rp)@b1
        q_mkr.append(q_mkr_1)

        # Preostali koraki
        D = 1/dt**2*M_glob_rp + 1/2/dt*C_glob_rp
        invD = np.linalg.inv(D) 

        for k in tq.tqdm(range(1,n_step-1)): 
            q_mkr_kplus1 = invD @ ((2/dt**2 * M_glob_rp - K_glob_rp) @ q_mkr[k] + (1/2/dt * C_glob_rp - 1/dt**2 * M_glob_rp) @ q_mkr[k-1] + F_glob_rp[:, k])
            q_mkr.append(q_mkr_kplus1)

        # prehod v koordinate u
        u_mkr = L @ np.array(q_mkr).T
        
        return u_mkr