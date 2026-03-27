"""
plot_lattice.py
---------------
Visualise 2-D Bravais lattices with a basis.
 
Usage
-----
Call `plot_lattice(a1, a2, basis)` with
 
    a1, a2  : array-like of length 2 or 3 – primitive lattice vectors
              (the z-component is ignored)
    basis   : list of array-like of length 2 or 3 – sublattice positions
              expressed in *fractional* coordinates of a1 and a2
              e.g. [0.5, 0.0] means 0.5*a1 + 0.0*a2
 
Optional keyword arguments
--------------------------
    title           : str   – figure title (default: none)
    bond_cutoff     : float – max distance for drawing a bond;
                              set to 0 or None to disable bonds
                              (default: auto = 1.05 * nearest-neighbour distance)
    n_shells        : int   – number of surrounding unit-cell layers to draw;
                              1 → 3×3 block, 2 → 5×5 block, etc. (default: 1)
    center          : (float, float) – Cartesian coordinates of the plot centre;
                              defaults to the centroid of the basis sites so that
                              the unit cell parallelogram is centred on the basis.
                              Pass an explicit value to override.
    cell_origin     : (float, float) – fractional coordinates of the corner of the
                              red parallelogram; defaults to None which places the
                              corner such that the cell is centred on the basis
                              centroid. Pass e.g. (-0.5, -0.5) to fine-tune.
    view_size       : float – half-width of the square viewport in data units;
                              if None, fitted automatically (default: None)
    figsize         : tuple – matplotlib figsize (default: (7, 7))
    site_color      : str   – colour for all lattice sites (default: '#1a1a2e')
    site_size       : float – scatter marker size (default: 100)
    bond_color      : str   – colour of bond lines (default: '#555555')
    bond_lw         : float – bond line width (default: 1.5)
    cell_color      : str   – colour of the unit-cell parallelogram (default: 'red')
    cell_lw         : float – unit-cell edge line width (default: 2.0)
    cell_alpha      : float – fill alpha of the unit cell (default: 0.08)
    show_vects      : bool  – draw lattice vector arrows from the basis centroid
                              (default: True)
    vect_color      : str   – colour of the lattice vector arrows (default: 'royalblue')
    vect_lw         : float – arrow line width (default: 2.0)
    vect_head_width : float – arrowhead width; auto-scaled if None (default: None)
    vect_head_length: float – arrowhead length; auto-scaled if None (default: None)
    vect_label_offset: float – perpendicular offset of the label from the arrow tip;
                              auto-scaled if None; pass a negative value to flip side
                              (default: None)
    show_axes       : bool  – show x/y axis ticks and labels (default: False)
    save_path       : str   – if given, save the figure to this path
 
Returns
-------
    fig, ax   – the matplotlib Figure and Axes objects
"""
 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from itertools import product as iproduct

# get lmodern font from LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{lmodern}"
})
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

cmap = plt.get_cmap('coolwarm')
cmap2 = plt.get_cmap('managua')
cmap3 = plt.get_cmap('vanimo')
cnorm = Normalize(vmin=0, vmax=1)
red = cmap(cnorm(0.95))
blue = cmap(cnorm(0.05))
yellow = cmap2(cnorm(0.15))
green = cmap3(cnorm(0.85))
 
 
# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
 
def _to_xy(v):
    """Return a 1-D numpy array of length 2 from a 2- or 3-component vector."""
    v = np.asarray(v, dtype=float).ravel()
    return v[:2]
 
 
def _build_sites(a1, a2, basis, n_shells=1):
    """
    Generate all lattice sites in the (2*n_shells+1)^2 supercell.
 
    Returns
    -------
    sites       : (N, 2) array of Cartesian positions
    which_basis : (N,) int array – index into `basis` for each site
    n1n2        : (N, 2) int array – (n1, n2) supercell indices
    """
    r = range(-n_shells, n_shells + 1)
    sites, which_basis, n1n2 = [], [], []
    for n1, n2 in iproduct(r, r):
        origin = n1 * a1 + n2 * a2
        for b_idx, frac in enumerate(basis):
            pos = origin + frac[0] * a1 + frac[1] * a2
            sites.append(pos)
            which_basis.append(b_idx)
            n1n2.append((n1, n2))
    return np.array(sites), np.array(which_basis), np.array(n1n2)
 
 
def _nearest_neighbour_dist(sites):
    """Return the smallest non-zero distance between any two sites."""
    diffs = sites[:, None, :] - sites[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    dists[dists < 1e-10] = np.inf
    return dists.min()
 
 
def _draw_unit_cell(ax, a1, a2, origin, color='red', lw=2.0, alpha=0.08):
    """Draw the unit-cell parallelogram with its corner at `origin`."""
    corners = np.array([
        origin,
        origin + a1,
        origin + a1 + a2,
        origin + a2,
        origin,       # close the loop
    ])
    ax.fill(corners[:-1, 0], corners[:-1, 1],
            color=color, alpha=alpha, zorder=1)
    ax.plot(corners[:, 0], corners[:, 1],
            color=color, lw=lw, zorder=3,
            solid_capstyle='round', solid_joinstyle='round')
 
 
# ---------------------------------------------------------------------------
# main function
# ---------------------------------------------------------------------------
 
def plot_lattice(
    a1,
    a2,
    basis,
    *,
    title=None,
    bond_cutoff='auto',
    n_shells=1,
    center=None,
    cell_origin=None,
    view_size=None,
    figsize=(7, 7),
    site_color='#1a1a2e',
    site_size=100,
    bond_color='#555555',
    bond_lw=1.8,
    show_cell=True,
    cell_color=red,
    cell_lw=3.0,
    cell_alpha=0.08,
    show_vects=True,
    vect_color=blue,
    vect_lw=3.0,
    vect_head_width=None,
    vect_head_length=None,
    vect_label_offset_a1=None,
    vect_label_offset_a2=None,
    show_axes=False,
    save_path=None,
    highlights=None,          # list of dicts, each with 'sites', optional 'color' and 'alpha'
    highlight_color='orange', # default color if not specified per group
    highlight_alpha=0.5,      # default alpha if not specified per group
    highlight_labels=True,
):
    """Plot a 2-D lattice. See module docstring for full parameter docs."""
 
    # --- normalise inputs ---------------------------------------------------
    a1    = _to_xy(a1)
    a2    = _to_xy(a2)
    basis = [_to_xy(b) for b in basis]   # fractional coords, length-2 each
 
    # --- centroid of basis sites --------------------------------------------
    # average in fractional space first, then convert to Cartesian —
    # this is correct for any lattice geometry including non-orthogonal
    basis_frac_mean = np.array(basis).mean(axis=0)
    centroid = basis_frac_mean[0] * a1 + basis_frac_mean[1] * a2
 
    # --- cell origin (corner of the red parallelogram) ----------------------
    if cell_origin is None:
        # place corner so the parallelogram centre lands on the centroid
        cell_origin_cart = centroid - (a1 + a2) / 2
    else:
        cell_origin = np.asarray(cell_origin, dtype=float)
        cell_origin_cart = cell_origin[0] * a1 + cell_origin[1] * a2
 
    # --- plot centre --------------------------------------------------------
    if center is None:
        center = centroid.copy()
    center = np.asarray(center, dtype=float)
 
    # --- build supercell ----------------------------------------------------
    sites, which_basis, n1n2 = _build_sites(a1, a2, basis, n_shells=n_shells)
 
    # --- bond cutoff --------------------------------------------------------
    nn_dist = _nearest_neighbour_dist(sites)
    if bond_cutoff == 'auto':
        bond_cutoff = 1.05 * nn_dist
    draw_bonds = bond_cutoff is not None and bond_cutoff > 0
 
    # --- figure setup -------------------------------------------------------
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
    else:
        ax.tick_params(length=3, labelsize=9)
 
    # --- draw bonds ---------------------------------------------------------
    if draw_bonds:
        N = len(sites)
        diffs = sites[:, None, :] - sites[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        for i in range(N):
            for j in range(i + 1, N):
                if dists[i, j] <= bond_cutoff:
                    ax.plot(
                        [sites[i, 0], sites[j, 0]],
                        [sites[i, 1], sites[j, 1]],
                        color=bond_color, lw=bond_lw,
                        zorder=2, solid_capstyle='round',
                    )

    # --- highlight edges ---------------------------------------------------

    if highlights is not None:
        for group in highlights:
            pts = np.array(group['sites'], dtype=float)
            color = group.get('color', highlight_color)
            alpha = group.get('alpha', highlight_alpha)
            ax.plot(
                pts[:, 0], pts[:, 1],
                color=color,
                alpha=alpha,
                lw=group.get('lw', site_size ** 0.5),  # auto-scaled if not specified
                zorder=3,
                solid_capstyle='round',
                solid_joinstyle='round',
            )
            if highlight_labels:
                label_name = group['label']
                label_pos = group['label_pos']
                ax.text(label_pos[0], label_pos[1], s=label_name,
                    color=color, fontsize=23,
                    ha='center', va='center', zorder=6)
 
    # --- draw sites ---------------------------------------------------------
    ax.scatter(
        sites[:, 0], sites[:, 1],
        s=site_size,
        color=site_color,
        zorder=4,
        linewidths=0,
    )
 
    # --- draw unit cell parallelogram ---------------------------------------
    if show_cell:
        _draw_unit_cell(ax, a1, a2,
                        origin=cell_origin_cart,
                        color=cell_color,
                        lw=cell_lw,
                        alpha=cell_alpha)
 
    # --- draw lattice vector arrows -----------------------------------------
    if show_vects:
        head_w  = vect_head_width    if vect_head_width   is not None else 0.08 * nn_dist
        head_l  = vect_head_length   if vect_head_length  is not None else 0.12 * nn_dist
        offsets = [vect_label_offset_a1, vect_label_offset_a2] 
 
        for vec, label, manual_off in zip([a1, a2], [r'$\bm{a}_1$', r'$\bm{a}_2$'], offsets):
            # arrow starts at centroid
            ax.arrow(centroid[0], centroid[1],
                     0.98 * vec[0], 0.98 * vec[1],
                     head_width=head_w, head_length=head_l,
                     lw=vect_lw, fc=vect_color, ec=vect_color,
                     length_includes_head=True, zorder=5)
            # label offset perpendicularly from the tip
            tip  = centroid + vec
            perp = np.array([-vec[1], vec[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-12)
            if manual_off is None:
                perp = np.array([-vec[1], vec[0]])
                perp = perp / (np.linalg.norm(perp) + 1e-12)
                lpos = tip + 0.18 * nn_dist * perp
            else:
                lpos = tip + np.asarray(manual_off, dtype=float)
            ax.text(lpos[0], lpos[1], label,
                    color=vect_color, fontsize=40,
                    ha='center', va='center', zorder=6)
 
    # --- title --------------------------------------------------------------
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold',
                     pad=12, fontfamily='serif', color='#1a1a2e')
 
    # --- viewport -----------------------------------------------------------
    if view_size is not None:
        half = float(view_size)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
    else:
        all_x, all_y = sites[:, 0], sites[:, 1]
        pad  = 0.6 * nn_dist
        cx, cy = center
        x_half = max(abs(all_x - cx).max() + pad, pad * 2)
        y_half = max(abs(all_y - cy).max() + pad, pad * 2)
        half   = max(x_half, y_half)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
 
    fig.tight_layout()
 
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
 
    return fig, ax, sites

# ---------------------------------------------------------------------------
# lattices
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from numpy import sqrt

    a = 1.0

    # triangular ------------------------------------------------------------
    # a1 = (a, 0)
    # a2 = (a/2, sqrt(3) * a / 2)

    # basis = [
    #     (0.0, 0.0),
    # ]

    # fig, ax = plot_lattice(
    #     a1, a2, basis,
    #     # cell_origin=(-0.5, -0.5),
    #     n_shells=3,
    #     view_size=1.2,
    #     vect_label_offset_a1 = (-0.3, -0.1),
    #     vect_label_offset_a2 = (-0.3, -0.25),
    #     save_path='archimedean/tools/unit_cells/triangular.pdf',
    # )
    # plt.show()

    # square ----------------------------------------------------------------
    # a1 = (a, 0)
    # a2 = (0, a)

    # basis = [
    #     (0.0, 0.0),
    # ]

    # fig, ax = plot_lattice(
    #     a1, a2, basis,
    #     # cell_origin=(-0.5, -0.5),
    #     n_shells=4,
    #     view_size=1.3,
    #     vect_label_offset_a1 = (-0.25, -0.15),
    #     vect_label_offset_a2 = (-0.15, -0.25),
    #     save_path='archimedean/tools/unit_cells/square.pdf',
    # )
    # plt.show()

    # honeycomb -------------------------------------------------------------
    # a1 = (a, 0.0)
    # a2 = (0.5*a, sqrt(3)/2*a)

    # basis = [
    #     (0.0, 0.0),
    #     (1/3, 1/3),
    # ]

    # fig, ax = plot_lattice(
    #     a1, a2, basis,
    #     n_shells=4,
    #     view_size=1.1,
    #     vect_label_offset_a1 = (-0.25, +0.1),
    #     vect_label_offset_a2 = (-0.05, -0.3),
    #     save_path='archimedean/tools/unit_cells/hexagonal.pdf',
    # )
    # plt.show()

    # kagome ----------------------------------------------------------------
    # a1 = (2 * a, 0)
    # a2 = (a, sqrt(3) * a)

    # basis = [
    #     (0.0, 0.0),
    #     (0.5, 0.0),
    #     (0.0, 0.5),
    # ]

    # fig, ax = plot_lattice(
    #     a1, a2, basis,
    #     # cell_origin=(-1/3, -1/3),
    #     n_shells=3,
    #     view_size=2.4,
    #     vect_label_offset_a1 = (-0.5, +0.25),
    #     vect_label_offset_a2 = (+0.05, -0.45),
    #     save_path='archimedean/tools/unit_cells/kagome.pdf',
    # )
    # plt.show()

    # kagome armchair -------------------------------------------------------
    # a1 = (2 * sqrt(3) * a, 0)
    # a2 = (0, 2 * a)

    # basis = [
    #     (0.0, 0.0),
    #     (0.0, 0.5),
    #     (0.25, 0.25),
    #     (0.5, 0.0),
    #     (0.5, 0.5),
    #     (0.75, 0.75),
    # ]

    # fig, ax = plot_lattice(
    #     a1, a2, basis,
    #     # cell_origin=(-1/3, -1/3),
    #     n_shells=4,
    #     view_size=4,
    #     vect_label_offset_a1 = (-0.15, 0.4),
    #     vect_label_offset_a2 = (-0.1, +0.45),
    #     save_path='archimedean/tools/unit_cells/kagome_armchair.pdf',
    # )
    # plt.show()

    # kagome edges ----------------------------------------------------------

    # a1 = (2 * a, 0)
    # a2 = (a, sqrt(3) * a)

    # basis = [
    #     (0.0, 0.0),
    #     (0.5, 0.0),
    #     (0.0, 0.5),
    # ]

    # fig, ax, sites = plot_lattice(
    #     a1, a2, basis,
    #     n_shells=3,
    #     view_size=3.3,
    #     show_cell=False,
    #     show_vects=False,
    #     # save_path='archimedean/tools/unit_cells/kagome_edges.pdf',
    # )

    # # print(sites)

    # def find_site(sites, coord, tol=1e-6):
    #     return np.where(np.linalg.norm(sites - np.array(coord), axis=1) < tol)[0]
    
    # # print(f'site id: {find_site(sites, (-1.5, 1.73205081))}')

    # flat_coords = [
    #     (-2, -1.73205081),
    #     (-1, -1.73205081),
    #     (0, -1.73205081),
    #     (1, -1.73205081),
    #     (2, -1.73205081),
    #     (3, -1.73205081),
    #     ]
    # flat_list = []
    # for coords in flat_coords:
    #     id = find_site(sites, (coords[0], coords[1]))[0]
    #     flat_list.append(id)

    # cove_coords = [
    #     # (-2, -1.73205081),
    #     (-1, -1.73205081),
    #     (-0.5, -0.8660254),
    #     (-1, 0),
    #     (-2, 0),
    #     (-1.5, 0.8660254),
    #     (-2, 1.73205081),
    #     (-1, 1.73205081),
    #     (-0.5, 2.59807621),
    #     (-1, 3.46410162),
    #     (-2, 3.46410162),
    #     (-1.5, 4.33012702),
    #     ]
    # cove_list = []
    # for coords in cove_coords:
    #     id = find_site(sites, (coords[0], coords[1]))[0]
    #     cove_list.append(id)

    # zig_coords = [
    #     (3, -1.73205081),
    #     (3.5, -0.8660254),
    #     (3, 0),
    #     (2.5, 0.8660254),
    #     (3, 1.73205081),
    #     (3.5, 2.59807621),
    #     (3, 3.46410162),
    #     (2.5, 4.33012702),
    #     ]
    # zig_list = []
    # for coords in zig_coords:
    #     id = find_site(sites, (coords[0], coords[1]))[0]
    #     zig_list.append(id)

    # arm_coords = [
    #     (2.5, 4.33012702),
    #     (2, 3.46410162),
    #     (1, 3.46410162),
    #     (0.5, 4.33012702),
    #     (0, 3.46410162),
    #     (-1, 3.46410162),
    #     (-1.5, 4.33012702),
    #     ]
    # arm_list = []
    # for coords in arm_coords:
    #     id = find_site(sites, (coords[0], coords[1]))[0]
    #     arm_list.append(id)

    # # [49, 69, 70, 90, 91, 111]
    # # [49, 69, 71, 52, 51, 53]
    # highlights = [
    #     {'sites': [sites[i] for i in flat_list], 'color': red, 'alpha': 0.7, 'label': r'\textbf{flat}', 'label_pos': (1.5,-2.25)},
    #     {'sites': [sites[i] for i in cove_list], 'color': blue,  'alpha': 0.7, 'label': r'\textbf{cove}', 'label_pos': (-2.5,0.8660254)},
    #     {'sites': [sites[i] for i in zig_list], 'color': green, 'alpha': 0.7, 'label': r'\textbf{zigzag}', 'label_pos': (3.5,0.8660254)},
    #     {'sites': [sites[i] for i in arm_list], 'color': yellow,  'alpha': 0.7, 'label': r'\textbf{armchair}', 'label_pos': (1.5,4.5)},
    # ]

    # fig, ax, sites = plot_lattice(
    #     a1, a2, basis,
    #     center = (0.5, 1),
    #     n_shells=4,
    #     view_size=4,
    #     show_cell=False,
    #     show_vects=False,
    #     highlights=highlights,
    #     save_path='archimedean/tools/unit_cells/kagome_edges.pdf',
    # )
    
    # plt.show()

    # rhombitrihexagonal ----------------------------------------------------
    # a1 = ((sqrt(3)+1)*a, 0)
    # a2 = ((sqrt(3)+1)/2*a, (sqrt(3)+3)/2*a)

    # basis = [
    #     (0.0, 0.0),
    #     (0.3660254, 0),
    #     (0.57735027, 0.21132487),
    #     (0.57735027, 0.57735027),
    #     (0.21132487, 0.57735027),
    #     (0, 0.3660254),
    # ]

    # fig, ax = plot_lattice(
    #     a1, a2, basis,
    #     # cell_origin=(-1/5, -1/5),
    #     n_shells=2,
    #     view_size=3.2,
    #     vect_label_offset_a1 = (0, -0.2),
    #     vect_label_offset_a2 = (-0.1, 0.21),
    #     save_path='archimedean/tools/unit_cells/rhombitrihexagonal.pdf',
    # )
    # plt.show()

    # truncated square ------------------------------------------------------
    # a1 = ((sqrt(2)+1)*a, 0)
    # a2 = (0, (sqrt(2)+1)*a)

    # basis = [
    #     (0.29289322, 0),
    #     (0, 0.29289322),
    #     (-0.29289322, 0),
    #     (0, -0.29289322),
    # ]

    # fig, ax = plot_lattice(
    #     a1, a2, basis,
    #     # cell_origin=(-1/5, -1/5),
    #     n_shells=3,
    #     view_size=3.0,
    #     vect_label_offset_a1 = (0.1, 0.25),
    #     vect_label_offset_a2 = (0.25, 0.1),
    #     save_path='archimedean/tools/unit_cells/truncated_square.pdf',
    # )
    # plt.show()

    # truncated hexagonal ---------------------------------------------------
    # a1 = ((sqrt(3) + 2)*a, 0)
    # a2 = ((sqrt(3)/2 + 1)*a, (sqrt(3) + 1.5)*a)

    # basis = [
    #     (0.40824829, 0.14942925),
    #     (0.40824829, 0.41737844),
    #     (0.14029910, 0.41737844),
    #     (0.56294883, 0.57207898),
    #     (0.83089802, 0.57207898),
    #     (0.56294883, 0.84002817),
    # ]

    # fig, ax = plot_lattice(
    #     a1, a2, basis,
    #     n_shells=3,
    #     view_size=4.3,
    #     vect_label_offset_a1 = (-0.75, +0.45),
    #     vect_label_offset_a2 = (0.35, -0.45),
    #     save_path='archimedean/tools/unit_cells/truncated_hexagonal.pdf',
    # )
    # plt.show()

    # truncated trihexagonal ------------------------------------------------
    # a1 = ((sqrt(3)+3)*a, 0)
    # a2 = ((sqrt(3)+3)/2*a, 1.5*(sqrt(3)+1)*a)

    # basis = [
    #     (0.33333333, 0.12200847),
    #     (0.5446582, 0.12200847),
    #     (0.5446582, 0.33333333),
    #     (0.33333333, 0.5446582),
    #     (0.12200847, 0.5446582),
    #     (0.12200847, 0.33333333),
    #     (0.66666667, 0.4553418),
    #     (0.87799153, 0.4553418),
    #     (0.87799153, 0.66666667),
    #     (0.66666667, 0.87799153),
    #     (0.4553418, 0.87799153),
    #     (0.4553418, 0.66666667),
    # ]

    # fig, ax = plot_lattice(
    #     a1, a2, basis,
    #     n_shells=3,
    #     view_size=5.2,
    #     vect_label_offset_a1 = (-1, -0.5),
    #     vect_label_offset_a2 = (0.1, -1.2),
    #     save_path='archimedean/tools/unit_cells/truncated_trihexagonal.pdf',
    # )
    # plt.show()

    # snub square -----------------------------------------------------------
    # a1 = (sqrt(2 + sqrt(3))*a, 0)
    # a2 = (0, sqrt(2 + sqrt(3))*a)

    # basis = [
    #     (0.0, 0.0),
    #     (0.5, 0.1339746),
    #     (0.3660254, 0.6339746),
    #     (-0.1339746, 0.5)
    # ]

    # fig, ax = plot_lattice(
    #     a1, a2, basis,
    #     # cell_origin=(-1/16, -1/6),
    #     n_shells=3,
    #     view_size=2.5,
    #     vect_label_offset_a1 = (-0.25, 0.3),
    #     vect_label_offset_a2 = (-0.25, -0.1),
    #     save_path='archimedean/tools/unit_cells/snub_square.pdf',
    # )
    # plt.show()

    # snub trihexagonal -----------------------------------------------------
    # a1 = (sqrt(7)*a, 0)
    # a2 = (sqrt(7)/2*a, sqrt(21)/2*a)

    # basis = [
    #     (0.28571429, 0.14285714),
    #     (0.57142857, 0.28571429),
    #     (0.85714286, 0.42857143),
    #     (0.71428571, 0.85714286),
    #     (0.42857143, 0.71428571),
    #     (0.14285714, 0.57142857),
    # ]

    # fig, ax = plot_lattice(
    #     a1, a2, basis,
    #     n_shells=3,
    #     view_size=3.2,
    #     vect_label_offset_a1 = (-0.55, -0.3),
    #     vect_label_offset_a2 = (+0.3, 0.05),
    #     save_path='archimedean/tools/unit_cells/snub_trihexagonal.pdf',
    # )
    # plt.show()

    # elongated triangular --------------------------------------------------
    # a1 = (a, 0)
    # a2 = (a/2, (1+sqrt(3)/2)*a)

    # basis = [
    #     (0.0, 0.0),
    #     (0.26794919, 0.46410162)
    # ]

    # fig, ax = plot_lattice(
    #     a1, a2, basis,
    #     # cell_origin=(-1/3, -1/4),
    #     n_shells=4,
    #     view_size=2.1,
    #     vect_label_offset_a1 = (-0.2, 0.25),
    #     vect_label_offset_a2 = (0.25, -0.1),
    #     save_path='archimedean/tools/unit_cells/elongated_triangular.pdf',
    # )
    # plt.show()

    # testing ----------------------------------------------------------------
    a1 = [ 3*a, 0 ]
    a2 = [ 1.5*a, 0.5*sqrt(3)*a ]

    basis = [
    [0, 0],
    [1/3, 0],
    ]

    fig, ax, _ = plot_lattice(
        a1, a2, basis,
        # cell_origin=(-1/3, -1/3),
        n_shells=5,
        view_size=3,
        # vect_label_offset_a1 = (-0.5, +0.25),
        # vect_label_offset_a2 = (+0.05, -0.45),
        # save_path='archimedean/tools/unit_cells/test.pdf',
    )
    plt.show()