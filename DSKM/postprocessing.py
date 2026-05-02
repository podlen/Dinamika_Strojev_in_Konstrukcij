import numpy as np
import matplotlib.pyplot as plt

try:
    import pyvista as pv
except ImportError:
    pv = None

class Visualizer:
    """
    Modul za vizualizacijo in procesiranje rezultatov.
    """
    def __init__(self, mesh, dof_manager=None, load_manager=None):
        if pv is None:
            raise ImportError("PyVista is required. Install via 'pip install pyvista'.")
        
        self.mesh = mesh
        self.dof_manager = dof_manager
        self.load_manager = load_manager
        
        self.pts = np.array([node.coords for node in mesh.nodes])
        
        self.keynode_indices =[]
        if hasattr(mesh, '_node_id_map'):
            self.keynode_indices = [idx for name, idx in mesh._node_id_map.items() if "key_" in name]

    def plot_part(self, part):
        """
        Prikaže makro-geometrijo. 
        """
        if pv is None: return
        plotter = pv.Plotter(title="Makro-geometrija")
        plotter.add_axes()

        pts = np.array(list(part.keynodes.values()))
        keys = list(part.keynodes.keys())
        
        lines =[]
        for n1, n2 in part.lines.values():
            idx1 = keys.index(n1)
            idx2 = keys.index(n2)
            lines.extend([2, idx1, idx2])
            
        # Dodamo vozlišča
        if len(pts) > 0:
            plotter.add_points(pts, color='red', point_size=20, render_points_as_spheres=True)
            
        # Mreža
        if lines:
            poly = pv.PolyData(pts, lines=lines)
            plotter.add_mesh(poly, color='orange', line_width=5, render_lines_as_tubes=True)
            
        plotter.show()

    def plot_mesh(self):
        """
        Prikaže diskretizirano mrežo.
        """
        plotter = pv.Plotter(title="Mreža")
        plotter.add_axes(label_size=(0.05, 0.05))

        # 1. Dodamo vozlišča
        if len(self.pts) > 0:
            plotter.add_points(self.pts, color='blue', point_size=12, render_points_as_spheres=True)

        # 2. Mreža
        lines =[]
        for e in self.mesh.elements:
            lines.extend([2, e.n1.id, e.n2.id])
            
        if lines:
            poly = pv.PolyData(self.pts, lines=lines)
            plotter.add_mesh(poly, color='lightblue', line_width=3)
            
        # Robni pogoji (Rdeče krogle)
        if self.dof_manager:
            support_pts = [self.pts[list(const.keys())[0] // 6] for const in self.dof_manager.constraints if len(const) == 1]
            if support_pts:
                plotter.add_points(np.array(support_pts), color='red', point_size=22, render_points_as_spheres=True)

        # 5. Sile (Magenta puščice)
        if self.load_manager:
            F = self.load_manager.get_global_force()
            for i in range(len(self.mesh.nodes)):
                f_vec = np.real(F[6*i : 6*i+3])
                mag = np.linalg.norm(f_vec)
                if mag > 1e-6:
                    direction = f_vec / mag
                    bbox_size = np.max(np.ptp(self.pts, axis=0)) if len(self.pts) > 0 else 1.0
                    arrow_len = bbox_size * 0.15
                    plotter.add_arrows(self.pts[i] - direction * arrow_len, direction, mag=arrow_len, color='magenta')

        plotter.show()


    def print_frequency_table(self, eig_vals):
        freqs_hz = np.sqrt(np.abs(eig_vals)) / (2 * np.pi)
        print("\n" + "="*30)
        print(f"{'Mode':<10} | {'Frekvenca [Hz]':<15}")
        print("-" * 30)
        for i, f in enumerate(freqs_hz):
            print(f"{i+1:<10} | {f:<15.3f}")
        print("="*30 + "\n")

    def plot_frf(self, omega_sweep, U_full, node_id, dof='uz'):
        dof_map = {'ux':0, 'uy':1, 'uz':2, 'rx':3, 'ry':4, 'rz':5}
        idx = 6 * node_id + dof_map[dof.lower()]
        
        response = U_full[idx, :]
        freqs_hz = omega_sweep / (2 * np.pi)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax1.semilogy(freqs_hz, np.abs(response), color='blue', lw=1.5)
        ax1.set_ylabel(f"Amplituda [{dof}] [m]")
        ax1.grid(True, which="both", ls="-", alpha=0.5)
        ax1.set_title(f"Harmonski odziv (FRF) - Vozlišče {node_id}")

        ax2.plot(freqs_hz, np.angle(response), color='red', lw=1.5)
        ax2.set_ylabel("Faza [rad]")
        ax2.set_xlabel("Frekvenca [Hz]")
        ax2.grid(True, alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def plot_frf_unwrap(self, f_rad, U_harm, node_id, dof):
        dof_map = {'ux':0, 'uy':1, 'uz':2, 'rx':3, 'ry':4, 'rz':5}
        idx = 6 * node_id + dof_map[dof.lower()]
        
        # Odziv vozlišča
        u_complex = U_harm[idx, :]
        amplitude = np.abs(u_complex)
        raw_phase = np.angle(u_complex)
        
        # Odpravljanje skokov 
        continuous_phase = np.unwrap(raw_phase)
        # Preverjanje v stopinjah
        phase_deg = np.degrees(continuous_phase)
        f_hz = f_rad / (2 * np.pi)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(f_hz, amplitude, 'b-')
        ax1.set_ylabel('Amplitude [mm]')
        ax1.grid(True)
        
        ax2.plot(f_hz, phase_deg, 'b-')
        ax2.set_ylabel('Phase [deg]')
        ax2.set_xlabel('Frequency [Hz]')
        ax2.set_yticks(np.arange(180, -541, -90)) 
        ax2.grid(True)
        
        plt.show()
    
    def _setup_open3d_scene(self, title, highlight_node=None):
        """Helper to set up the Open3D environment. Now supports highlighting a specific node"""
        import open3d as o3d
        ranges = np.ptp(self.pts, axis=0) if np.any(self.pts) else np.array([1, 1, 1])
        model_size = np.max(ranges)   
        
        o3d_lines = [[e.n1.id, e.n2.id] for e in self.mesh.elements]
        
        min_b = np.min(self.pts, axis=0) - ranges * 0.5 - np.array([model_size*0.1]*3)
        max_b = np.max(self.pts, axis=0) + ranges * 0.5 + np.array([model_size*0.1]*3)
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_b, max_b)
        bbox_ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
        bbox_ls.paint_uniform_color([1.0, 1.0, 1.0]) 
        
        # Geometrija
        ls_orig = o3d.geometry.LineSet()
        ls_orig.points = o3d.utility.Vector3dVector(self.pts)
        ls_orig.lines = o3d.utility.Vector2iVector(o3d_lines)
        ls_orig.colors = o3d.utility.Vector3dVector([[0.7, 0.7, 0.7]] * len(o3d_lines))
        
        ls_def = o3d.geometry.LineSet()
        ls_def.points = o3d.utility.Vector3dVector(self.pts)
        ls_def.lines = o3d.utility.Vector2iVector(o3d_lines)
        ls_def.colors = o3d.utility.Vector3dVector([[0.0, 0.4, 1.0]] * len(o3d_lines))
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pts)
        pcd.colors = o3d.utility.Vector3dVector([[1.0, 0.2, 0.2]] * len(self.pts))

        # Highlite node
        hl_sphere = None
        if highlight_node is not None and 0 <= highlight_node < len(self.pts):
            hl_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=model_size*0.04)
            hl_sphere.paint_uniform_color([1.0, 0.84, 0.0]) # Gold
            hl_sphere.translate(self.pts[highlight_node])
        
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=title, width=1200, height=800)
        vis.add_geometry(bbox_ls)
        vis.add_geometry(ls_orig)
        vis.add_geometry(ls_def)
        vis.add_geometry(pcd)
        if hl_sphere:
            vis.add_geometry(hl_sphere)
        
        # Add Coordinate axes
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=model_size*0.2, origin=np.min(self.pts, axis=0))
        vis.add_geometry(axes)

        opt = vis.get_render_option()
        opt.line_width = 3.0 
        opt.point_size = 5.0 
        opt.background_color = np.array([1.0, 1.0, 1.0])
        
        vis.poll_events()
        vis.update_renderer()
        vis.reset_view_point(True)
        
        return vis, ls_def, pcd, hl_sphere, model_size, o3d

    def animate_mode_shape(self, eig_vals, eig_vecs, mode_idx=0, scale=1.0):
        """3D animacija naravnih lastnih oblik (modalna analiza)."""
        vis, ls_def, pcd, _, model_size, o3d = self._setup_open3d_scene("DSKM - Modal Analysis")
        freqs =[np.sqrt(np.real(v)) / (2 * np.pi) if np.real(v) > 0 else 0.0 for v in eig_vals]
        
        def get_disp(m_idx, s):
            mode = eig_vecs[:, m_idx]
            disp = np.column_stack((mode[0::6], mode[1::6], mode[2::6]))
            max_val = np.max(np.linalg.norm(disp, axis=1))
            return disp * (model_size * 0.15 / max_val) * s if max_val > 1e-12 else disp

        state = {'t': 0.0, 'play': True, 'idx': mode_idx, 'scale': scale, 'disp': get_disp(mode_idx, scale), 'speed': 1.0}
        
        def update_title():
            print(f"\rMode: {state['idx']+1} | Freq: {freqs[state['idx']]:.2f} Hz | Speed: {state['speed']:.2f}x   ", end="")
            
        update_title()
        
        vis.register_key_callback(ord(' '), lambda v: state.update({'play': not state['play']}) or False)
        vis.register_key_callback(262, lambda v: state.update({'idx': (state['idx'] + 1) % len(freqs), 'disp': get_disp((state['idx'] + 1) % len(freqs), state['scale'])}) or update_title()) # Right
        vis.register_key_callback(263, lambda v: state.update({'idx': (state['idx'] - 1) % len(freqs), 'disp': get_disp((state['idx'] - 1) % len(freqs), state['scale'])}) or update_title()) # Left
        vis.register_key_callback(265, lambda v: state.update({'scale': state['scale']*1.25, 'disp': get_disp(state['idx'], state['scale']*1.25)}) or False) # Up
        vis.register_key_callback(264, lambda v: state.update({'scale': state['scale']/1.25, 'disp': get_disp(state['idx'], state['scale']/1.25)}) or False) # Down
        vis.register_key_callback(43, lambda v: state.update({'speed': state['speed']*1.25}) or update_title()) # + 
        vis.register_key_callback(61, lambda v: state.update({'speed': state['speed']*1.25}) or update_title()) # = / +
        vis.register_key_callback(45, lambda v: state.update({'speed': state['speed']/1.25}) or update_title()) # -
        vis.register_key_callback(ord('Q'), lambda v: v.close())

        import time
        try:
            while vis.poll_events():
                if state['play']:
                    state['t'] += 0.15 * state['speed']
                    pts_new = self.pts + state['disp'] * np.sin(state['t'])
                    ls_def.points = o3d.utility.Vector3dVector(pts_new)
                    pcd.points = o3d.utility.Vector3dVector(pts_new)
                    vis.update_geometry(ls_def)
                    vis.update_geometry(pcd)
                vis.update_renderer()
                time.sleep(0.01)
        except KeyboardInterrupt: pass
        finally: vis.destroy_window()

    def animate_harmonic(self, omega_sweep, U_full, start_idx=0, scale=1.0, highlight_node=None):
        """3D animacija harmonskega odziva zvezno preko frekvenčnega spektra (Ansys-style)."""
        vis, ls_def, pcd, hl_sphere, model_size, o3d = self._setup_open3d_scene("DSKM - Harmonic Response", highlight_node)
        freqs = omega_sweep / (2 * np.pi)
        
        def get_complex_disp(f_idx, s):
            U = U_full[:, f_idx]
            disp = np.column_stack((U[0::6], U[1::6], U[2::6]))
            max_val = np.max(np.linalg.norm(np.abs(disp), axis=1))
            return disp * (model_size * 0.15 / max_val) * s if max_val > 1e-12 else disp

        state = {
            't': 0.0, 
            'play': True, 
            'idx': start_idx, 
            'sweep_prog': float(start_idx), 
            'scale': scale, 
            'disp': get_complex_disp(start_idx, scale), 
            'speed': 1.0, 
            'auto_sweep': True
        }
        
        def update_title():
            # This \r carriage return updates the Jupyter cell output perfectly in real-time like a HUD
            print(f"\r>>> [ANSYS SWEEP] Vzbujevalna Frekvenca: {freqs[state['idx']]:.2f} Hz | Hitrost: {state['speed']:.2f}x | Auto-Sweep: {'ON ' if state['auto_sweep'] else 'OFF'} <<<   ", end="", flush=True)
            
        update_title()
        
        vis.register_key_callback(ord(' '), lambda v: state.update({'play': not state['play']}) or False)
        vis.register_key_callback(ord('M'), lambda v: state.update({'auto_sweep': not state['auto_sweep']}) or update_title()) # Toggle sweep
        
        # Manual arrow keys automatically pause the sweep so you can inspect a specific frequency manually
        vis.register_key_callback(262, lambda v: state.update({'auto_sweep': False, 'idx': (state['idx'] + 1) % len(freqs), 'sweep_prog': float((state['idx'] + 1) % len(freqs)), 'disp': get_complex_disp((state['idx'] + 1) % len(freqs), state['scale'])}) or update_title()) # Right
        vis.register_key_callback(263, lambda v: state.update({'auto_sweep': False, 'idx': (state['idx'] - 1) % len(freqs), 'sweep_prog': float((state['idx'] - 1) % len(freqs)), 'disp': get_complex_disp((state['idx'] - 1) % len(freqs), state['scale'])}) or update_title()) # Left
        vis.register_key_callback(265, lambda v: state.update({'scale': state['scale']*1.25, 'disp': get_complex_disp(state['idx'], state['scale']*1.25)}) or False) # Up
        vis.register_key_callback(264, lambda v: state.update({'scale': state['scale']/1.25, 'disp': get_complex_disp(state['idx'], state['scale']/1.25)}) or False) # Down
        vis.register_key_callback(43, lambda v: state.update({'speed': state['speed']*1.25}) or update_title()) # + 
        vis.register_key_callback(61, lambda v: state.update({'speed': state['speed']*1.25}) or update_title()) # = / +
        vis.register_key_callback(45, lambda v: state.update({'speed': state['speed']/1.25}) or update_title()) # -
        vis.register_key_callback(ord('Q'), lambda v: v.close())

        import time
        try:
            while vis.poll_events():
                if state['play']:
                    # Hitrost vibracije strukture
                    state['t'] += 0.15 * state['speed']
                    
                    # Ansys-Style zvezno spreminjanje vzbujevalne frekvence
                    if state['auto_sweep']:
                        state['sweep_prog'] = (state['sweep_prog'] + 0.5 * state['speed']) % len(freqs)
                        new_idx = int(state['sweep_prog'])
                        if new_idx != state['idx']:
                            state['idx'] = new_idx
                            state['disp'] = get_complex_disp(state['idx'], state['scale'])
                            update_title()
                            
                    # Računanje kompleksnega faznega zamika glede na frekvenco
                    active_disp = np.real(state['disp'] * np.exp(1j * state['t']))
                    pts_new = self.pts + active_disp
                    
                    ls_def.points = o3d.utility.Vector3dVector(pts_new)
                    pcd.points = o3d.utility.Vector3dVector(pts_new)
                    vis.update_geometry(ls_def)
                    vis.update_geometry(pcd)
                    
                    # Update the highlighted sphere's position
                    if hl_sphere is not None:
                        hl_sphere.translate(pts_new[highlight_node], relative=False)
                        vis.update_geometry(hl_sphere)
                        
                vis.update_renderer()
                time.sleep(0.01)
        except KeyboardInterrupt: pass
        finally: vis.destroy_window()

    def animate_transient(self, t_eval, U_full, scale=1.0, highlight_node=None):
        """3D animacija tranzientnega odziva z izbirnim poudarkom na specifičnem vozlišču."""
        vis, ls_def, pcd, hl_sphere, model_size, o3d = self._setup_open3d_scene("DSKM - Transient Response", highlight_node)
        
        U_3d = np.vstack([U_full[0::6], U_full[1::6], U_full[2::6]])
        global_max = np.max(np.abs(U_3d))
        scale_factor = (model_size * 0.15 / global_max) * scale if global_max > 1e-12 else 1.0

        state = {'play': True, 'step': 0, 'speed': 1}
        
        def update_title():
            print(f"\rTime: {t_eval[state['step']]:.3f} s | Step: {state['step']}/{len(t_eval)} | Speed: {state['speed']}x  ", end="")
            
        update_title()
        
        vis.register_key_callback(ord(' '), lambda v: state.update({'play': not state['play']}) or False)
        vis.register_key_callback(262, lambda v: state.update({'step': (state['step'] + 10) % len(t_eval)}) or update_title()) 
        vis.register_key_callback(263, lambda v: state.update({'step': (state['step'] - 10) % len(t_eval)}) or update_title()) 
        vis.register_key_callback(43, lambda v: state.update({'speed': max(1, state['speed']+1)}) or update_title()) 
        vis.register_key_callback(61, lambda v: state.update({'speed': max(1, state['speed']+1)}) or update_title()) 
        vis.register_key_callback(45, lambda v: state.update({'speed': max(1, state['speed']-1)}) or update_title()) 
        vis.register_key_callback(ord('Q'), lambda v: v.close())

        import time
        try:
            while vis.poll_events():
                if state['play']:
                    state['step'] = (state['step'] + state['speed']) % len(t_eval)
                    update_title()
                    
                U_step = U_full[:, state['step']]
                disp = np.column_stack((U_step[0::6], U_step[1::6], U_step[2::6]))
                pts_new = self.pts + disp * scale_factor
                
                ls_def.points = o3d.utility.Vector3dVector(pts_new)
                pcd.points = o3d.utility.Vector3dVector(pts_new)
                vis.update_geometry(ls_def)
                vis.update_geometry(pcd)
                
                if hl_sphere is not None:
                    hl_sphere.translate(pts_new[highlight_node], relative=False)
                    vis.update_geometry(hl_sphere)
                    
                vis.update_renderer()
                time.sleep(0.01)
        except KeyboardInterrupt: pass
        finally: vis.destroy_window()