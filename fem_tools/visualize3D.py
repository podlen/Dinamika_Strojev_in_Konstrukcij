import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


def animate_truss(elementi, vozlisca, eig_vals, eig_vecs, mode_idx, SCALE):
    """
    Create an animation of a single mode shape for a 3D truss.

    Parameters
    ----------
    elementi : np.ndarray, shape (n_elem, 2)
        Element connectivity — each row holds the start- and end-node
        indices of one bar element.
    vozlisca : np.ndarray, shape (n_nodes, 3)
        Nodal coordinates (x, y, z).
    eig_vals : np.ndarray, shape (n_dof,)
        Eigenvalues (ω²) returned by the generalised eigenproblem.
    eig_vecs : np.ndarray, shape (n_dof, n_dof)
        Eigenvector matrix — column ``mode_idx`` is the mode shape
        to animate.
    mode_idx : int
        Zero-based index of the mode to display.
    SCALE : float
        Maximum displacement amplitude used for visualisation.
    interactive : bool, optional
        If True, open in a GUI window (qt / tk) where you can rotate
        the 3D view while the animation plays.  If False (default),
        return an inline HTML animation for the notebook.

    Returns
    -------
    IPython.display.HTML or None
        HTML animation when ``interactive=False``; None when
        ``interactive=True`` (the window is shown via ``plt.show()``).
    """

    N_FRAMES = 40

    if mode_idx >= len(eig_vals):
        raise ValueError(
            f"mode_idx: {mode_idx} is out of range for "
            f"{len(eig_vals)} DOFs"
        )

    # ── helper: deformed positions at time t ────────────────────────────────
    def get_deformed(mode_idx, t):
        phi = eig_vecs[:, mode_idx].reshape(len(vozlisca), 3)
        max_disp = np.max(np.abs(phi))
        if max_disp > 0:
            phi = phi / max_disp * SCALE
        return vozlisca + phi * np.sin(t)

    # ── figure setup ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Compute axis limits with padding
    mins = vozlisca.min(axis=0)
    maxs = vozlisca.max(axis=0)
    span = (maxs - mins).max()
    pad = span * 0.5 + SCALE
    center = (mins + maxs) / 2

    ax.set_xlim(center[0] - span / 2 - pad, center[0] + span / 2 + pad)
    ax.set_ylim(center[1] - span / 2 - pad, center[1] + span / 2 + pad)
    ax.set_zlim(center[2] - span / 2 - pad, center[2] + span / 2 + pad)

    freq_hz = eig_vals[mode_idx]
    ax.set_title(f"Mode {mode_idx + 1} — {freq_hz:.1f} Hz")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    # ── undeformed (reference) geometry ─────────────────────────────────────
    for e in elementi:
        ax.plot(vozlisca[e, 0], vozlisca[e, 1], vozlisca[e, 2],
                color="lightgrey", lw=1.2, ls="--", zorder=1)
    ax.plot(vozlisca[:, 0], vozlisca[:, 1], vozlisca[:, 2],
            "o", color="lightgrey", ms=5, zorder=1)

    # ── initial deformed geometry (t = 0) ───────────────────────────────────
    nodes0 = get_deformed(mode_idx, 0)
    a_lines = []
    for e in elementi:
        ln, = ax.plot(nodes0[e, 0], nodes0[e, 1], nodes0[e, 2],
                       color="steelblue", lw=2.5, zorder=3)
        a_lines.append(ln)
    pts, = ax.plot(nodes0[:, 0], nodes0[:, 1], nodes0[:, 2],
                   "o", color="orange", ms=7, zorder=4)

    fig.tight_layout()

    # ── frame update ────────────────────────────────────────────────────────
    def update(frame):
        t = 2 * np.pi * frame / N_FRAMES
        nodes = get_deformed(mode_idx, t)
        for edge_idx, e in enumerate(elementi):
            a_lines[edge_idx].set_data_3d(
                nodes[e, 0], nodes[e, 1], nodes[e, 2]
            )
        pts.set_data_3d(nodes[:, 0], nodes[:, 1], nodes[:, 2])  # type: ignore[attr-defined]
        return a_lines + [pts]

    # ── create animation ────────────────────────────────────────────────────
    ani = animation.FuncAnimation(
        fig, update, frames=N_FRAMES, interval=1000 / 30, blit=False
    )

    plt.close()
    return HTML(ani.to_jshtml())
