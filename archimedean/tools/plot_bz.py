"""
plot_bz.py
----------
Plot the first Brillouin zone with high-symmetry points and a k-path.

Usage
-----
Call `plot_bz(a1, a2, path)` with

    a1, a2  : array-like of length 2 or 3 – real-space primitive lattice vectors
              (z-component ignored)
    path    : list of HSP entries, each of the form [label, (fx, fy, fz)]  # noqa
              where (fx, fy, fz) are fractional coordinates w.r.t. b1, b2
              e.g. [r'$\Gamma$', (0, 0, 0)]
              The path is drawn in the order given; the first and last points
              are connected if they are the same (closed path).

Optional keyword arguments
--------------------------
    ax              : matplotlib Axes to draw into; if None a new figure is created
    figsize         : tuple  – figure size when creating a new figure (default: (5, 5))
    bz_color        : str    – colour of the BZ boundary (default: '#1a1a2e')
    bz_lw           : float  – BZ boundary line width (default: 1.5)
    bz_fill_color   : str    – BZ fill colour (default: '#f0f4ff')
    bz_fill_alpha   : float  – BZ fill alpha (default: 0.5)
    path_color      : str    – colour of the k-path line (default: 'red')
    path_lw         : float  – k-path line width (default: 1.5)
    hsp_color       : str    – colour of HSP markers (default: '#1a1a2e')
    hsp_size        : float  – HSP marker size (default: 40)
    label_fontsize  : float  – HSP label font size (default: 11)
    label_offset    : float  – label offset from point in data units (default: auto)
    title           : str    – axes title (default: none)
    show_axes       : bool   – show axis ticks (default: False)
    save_path       : str    – if given, save figure to this path

Returns
-------
    fig, ax  – matplotlib Figure and Axes (fig is None if ax was passed in)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_xy(v):
    v = np.asarray(v, dtype=float).ravel()
    return v[:2]


def reciprocal_vectors(a1, a2):
    """Compute 2D reciprocal lattice vectors from real-space primitive vectors."""
    a1, a2 = _to_xy(a1), _to_xy(a2)
    area = a1[0] * a2[1] - a1[1] * a2[0]
    b1 = (2 * np.pi / area) * np.array([ a2[1], -a2[0]])
    b2 = (2 * np.pi / area) * np.array([-a1[1],  a1[0]])
    return b1, b2


def _bz_polygon(b1, b2, n_shells=2):
    """
    Return the vertices of the first BZ as an ordered (M, 2) array,
    computed as the Voronoi cell of the origin in the reciprocal lattice.
    """
    # generate a patch of reciprocal lattice points around the origin
    r = range(-n_shells, n_shells + 1)
    pts = []
    for n1 in r:
        for n2 in r:
            pts.append(n1 * b1 + n2 * b2)
    pts = np.array(pts)

    vor = Voronoi(pts)

    # find the index of the origin in pts
    origin_idx = np.argmin(np.linalg.norm(pts, axis=1))

    # collect Voronoi vertices belonging to the cell of the origin
    region_idx = vor.point_region[origin_idx]
    vertex_indices = vor.regions[region_idx]
    vertices = vor.vertices[vertex_indices]

    # sort vertices by angle for a clean polygon
    angles = np.arctan2(vertices[:, 1], vertices[:, 0])
    vertices = vertices[np.argsort(angles)]

    return vertices


def _frac_to_cart(frac, b1, b2):
    """Convert fractional reciprocal coords (fx, fy, [fz]) to 2D Cartesian."""
    frac = np.asarray(frac, dtype=float).ravel()
    return frac[0] * b1 + frac[1] * b2


# ---------------------------------------------------------------------------
# main function
# ---------------------------------------------------------------------------

def plot_bz(
    a1,
    a2,
    path,
    *,
    ax=None,
    figsize=(5, 5),
    bz_color='#1a1a2e',
    bz_lw=1.5,
    bz_fill_color='#e8eef8',
    bz_fill_alpha=0.5,
    path_color='red',
    path_lw=1.5,
    hsp_color='#1a1a2e',
    hsp_size=40,
    label_fontsize=11,
    label_offset=None,
    title=None,
    show_axes=False,
    save_path=None,
):
    """Plot the first Brillouin zone with HSPs and k-path. See module docstring."""

    # --- reciprocal vectors -------------------------------------------------
    b1, b2 = reciprocal_vectors(a1, a2)

    # --- BZ polygon ---------------------------------------------------------
    bz_verts = _bz_polygon(b1, b2)

    # --- figure / axes setup ------------------------------------------------
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    ax.set_facecolor('white')
    ax.set_aspect('equal')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color('#333333')
    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # --- draw BZ ------------------------------------------------------------
    poly = plt.Polygon(bz_verts, closed=True,
                       facecolor=bz_fill_color, alpha=bz_fill_alpha,
                       edgecolor=bz_color, linewidth=bz_lw, zorder=1)
    ax.add_patch(poly)

    # --- convert HSP fractional coords to Cartesian -------------------------
    hsp_cart = {label: _frac_to_cart(frac, b1, b2)
                for label, frac in path}

    # deduplicate while preserving order (for the path line)
    seen = set()
    unique_path = []
    for label, _ in path:
        if label not in seen:
            seen.add(label)
            unique_path.append(label)

    # --- draw k-path --------------------------------------------------------
    path_pts = np.array([hsp_cart[label] for label, _ in path])
    ax.plot(path_pts[:, 0], path_pts[:, 1],
            color=path_color, lw=path_lw,
            zorder=2, solid_capstyle='round', solid_joinstyle='round')

    # --- draw HSP markers ---------------------------------------------------
    for label in unique_path:
        pt = hsp_cart[label]
        ax.scatter(*pt, s=hsp_size, color=hsp_color, zorder=4, linewidths=0)

    # --- labels -------------------------------------------------------------
    bz_scale = np.linalg.norm(bz_verts, axis=1).max()
    if label_offset is None:
        label_offset = 0.08 * bz_scale

    for label in unique_path:
        pt = hsp_cart[label]
        norm = np.linalg.norm(pt)
        if norm < 1e-10:
            # Gamma: offset diagonally up-left
            offset = np.array([-1, 1]) * label_offset
        else:
            # offset radially outward
            offset = (pt / norm) * label_offset
        ax.text(pt[0] + offset[0], pt[1] + offset[1],
                label, fontsize=label_fontsize,
                ha='center', va='center',
                fontfamily='serif', color=hsp_color, zorder=5)

    # --- viewport -----------------------------------------------------------
    pad = 0.25 * bz_scale
    ax.set_xlim(-bz_scale - pad, bz_scale + pad)
    ax.set_ylim(-bz_scale - pad, bz_scale + pad)

    if title:
        ax.set_title(title, fontsize=13, fontweight='bold',
                     pad=10, fontfamily='serif', color='#1a1a2e')

    if fig is not None:
        fig.tight_layout()

    if save_path:
        out = fig if fig is not None else ax.get_figure()
        out.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")

    return fig, ax


# ---------------------------------------------------------------------------
# example: kagome / triangular BZ with Gamma, M, K path
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from numpy import sqrt

    a = 1.0
    a1 = (a, 0)
    a2 = (a/2, (1+sqrt(3)/2)*a)

    # HSP entries: [label, (f1, f2, f3)]
    G = [r'$\Gamma$', (0,   0,   0)]
    Ms = [r'$M$',      (1/2, 1/2,   0)]
    Mh = [r'$M$',      (1/2, 0,   0)]
    X = [r'$X$',      (1/2, 0, 0)]
    Y = [r'$Y$',      (0, 1/2, 0)]
    Y1 = [r'$Y$',      (0, -1/2, 0)]
    K = [r'$K$',      (1/3, -1/3,   0)]
    C = [r'$C$',      (1/2, 1/2,   0)]
    H = [r'$H$',      (0.4641016, 0.7320508,   0)]
    H1 = [r'$H1$',      (0.5358984, 0.2679492,   0)]

    path = [G, Y, H, C, G, X, H1, G, H]

    fig, ax = plot_bz(a1, a2, path)
    plt.show()