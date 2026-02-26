import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import ipywidgets as widgets


def animate_mode_shapes(elementi, vozlisca, eig_vals, eig_vecs, mode_idx, SCALE):
    """
    Create an inline HTML animation of a single mode shape for a 2D truss.

    The eigenvector is normalised so that its maximum component equals
    ``SCALE``, then sinusoidally oscillated to produce a looping animation.
    The undeformed geometry is drawn as a dashed grey reference underneath.

    Parameters
    ----------
    elementi : np.ndarray, shape (n_elem, 2)
        Element connectivity — each row holds the start- and end-node
        indices of one bar element.
    vozlisca : np.ndarray, shape (n_nodes, 2)
        Nodal coordinates (x, y).
    eig_vals : np.ndarray, shape (n_dof,)
        Eigenvalues (ω²) returned by the generalised eigenproblem
        K φ = ω² M φ.  Used here only for the plot title (converted
        to Hz via f = √|λ| / 2π).
    eig_vecs : np.ndarray, shape (n_dof, n_dof)
        Eigenvector matrix — column ``mode_idx`` is the mode shape
        to animate.
    mode_idx : int
        Zero-based index of the mode to display.
    SCALE : float
        Maximum displacement amplitude used for visualisation (in the
        same length units as ``vozlisca``).

    Returns
    -------
    IPython.display.HTML
        An HTML object containing the JavaScript animation, suitable
        for inline display in a Jupyter notebook.

    Raises
    ------
    ValueError
        If ``mode_idx`` is out of range for the given eigenvectors.
    """

    # ── animation settings ──────────────────────────────────────────────────
    N_FRAMES = 120   # number of frames per oscillation cycle

    # ── input validation ────────────────────────────────────────────────────
    if mode_idx >= len(eig_vals):
        raise ValueError(
            f"mode_idx: {mode_idx} is out of range for "
            f"{len(eig_vals)} DOFs"
        )

    # ── helper: compute deformed nodal positions at time t ──────────────────
    def get_deformed(mode_idx, t):
        """Return deformed node coordinates for a sinusoidal oscillation at time *t*."""
        # Reshape the eigenvector from (n_dof,) to (n_nodes, 2)
        phi = eig_vecs[:, mode_idx].reshape(len(vozlisca), 2)
        # Normalise so the largest component equals SCALE
        max_displacement = np.max(np.abs(phi))
        if max_displacement > 0:
            phi = phi / max_displacement * SCALE
        return vozlisca + phi * np.sin(t)

    # ── set up the figure and axes ──────────────────────────────────────────
    fig = plt.figure(figsize=(10, 10))

    # Compute axis padding from the geometry bounding box
    x_span = vozlisca[:, 0].max() - vozlisca[:, 0].min()
    y_span = vozlisca[:, 1].max() - vozlisca[:, 1].min()
    span = max(x_span, y_span)
    pad = span * 0.5

    # Convert eigenvalue to frequency in Hz for the title
    freq_hz = eig_vals[mode_idx]
    plt.title(f"Mode {mode_idx + 1} — {freq_hz:.1f} Hz")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.xlim(vozlisca[:, 0].min() - pad, vozlisca[:, 0].max() + pad)
    plt.ylim(vozlisca[:, 1].min() - pad, vozlisca[:, 1].max() + pad)

    # ── draw the undeformed (reference) geometry ────────────────────────────
    for e in elementi:
        plt.plot(vozlisca[e, 0], vozlisca[e, 1],
                 color="lightgrey", lw=1.2, ls="--", zorder=1)
    plt.plot(vozlisca[:, 0], vozlisca[:, 1],
             "o", color="lightgrey", ms=6, zorder=1)

    # ── draw initial deformed geometry (t = 0) ──────────────────────────────
    nodes0 = get_deformed(mode_idx, 0)
    a_lines = []
    for e in elementi:
        ln, = plt.plot(nodes0[e, 0], nodes0[e, 1],
                       color="steelblue", lw=2.5, zorder=3)
        a_lines.append(ln)
    pts, = plt.plot(nodes0[:, 0], nodes0[:, 1],
                    "o", color="orange", ms=8, zorder=4)

    plt.tight_layout()

    # ── frame update callback for FuncAnimation ─────────────────────────────
    def update(frame):
        """Move every line and point marker to the deformed position at *frame*."""
        t = 2 * np.pi * frame / N_FRAMES
        nodes = get_deformed(mode_idx, t)
        # Update each element line
        for edge_idx, e in enumerate(elementi):
            a_lines[edge_idx].set_data(nodes[e, 0], nodes[e, 1])
        # Update node markers
        pts.set_data(nodes[:, 0], nodes[:, 1])
        return a_lines + [pts]

    # ── create and return the animation ─────────────────────────────────────
    ani = animation.FuncAnimation(
        fig, update, frames=N_FRAMES, interval=1000 / 30, blit=True
    )

    plt.close()  # prevent a static duplicate from appearing in the notebook
    return HTML(ani.to_jshtml())


