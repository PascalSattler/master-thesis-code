"""
Archimedean Tiling Generator
============================

Generates and visualises all 11 Archimedean tilings from a vertex
configuration tuple, e.g. (3, 6, 3, 6) for the Kagome lattice.

Algorithm
---------
The tiling is grown by BFS, placing a complete VERTEX FIGURE at each step.
A vertex figure is one central vertex surrounded by its full ring of polygons
(as defined by the config); every vertex carries the same figure shape, just
rotated.

BFS step for vertex W
  1. Gather registered vertices within bond_length of W (known neighbours).
  2. Build all candidate rotations that map some spoke[k] onto each known
     neighbour, then keep only those where ALL neighbours lie on some spoke.
     Deduplicate by spoke-tip fingerprint so that symmetric configs like
     (6,6,6) correctly yield only one candidate.
  3. Exactly one candidate  →  place figure.
     Ambiguous              →  apply lookahead: eliminate candidates whose
                               figure would place a new vertex adjacent to a
                               committed figure centre but NOT on that
                               centre's spokes.
     Still ambiguous        →  defer (re-queue) up to MAX_DEFER times, then
                               force-commit to first surviving candidate.

Performance
-----------
The rotation-candidate search is fully vectorised with broadcasting:
  - R_all  (K*n,)    : all candidate rotations at once
  - tips   (K*n,n,2) : all spoke-tip tensors at once
  - dist2  (K*n,K,n) : squared distance from every neighbour to every
                        candidate spoke, as a single numpy operation
  - valid  (K*n,)    : one boolean per candidate

The processed-figure array for lookahead is maintained incrementally
(append-only lists → numpy array) so it is never rebuilt from scratch.

Budget
------
num_figures controls how many vertex figures to place.  Vertex count will
be roughly 3–8x num_figures depending on the config (e.g. kagome with
num_figures=200 gives ~650 vertices and ~350 polygons).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from collections import deque

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

COORD_PREC  = 7
DIST_TOL    = 1e-4
SPOKE_ROUND = 3

POLYGON_COLORS = {
    3 : '#f4a261',
    4 : '#e9c46a',
    6 : '#2a9d8f',
    8 : '#457b9d',
    12: '#9b5de5',
}

ARCHIMEDEAN = {
    'triangular'            : (3,3,3,3,3,3),
    'square'                : (4,4,4,4),
    'hexagonal'             : (6,6,6),
    'kagome'                : (3,6,3,6),
    'rhombitrihexagonal'    : (3,4,6,4),
    'truncated_square'      : (4,8,8),
    'truncated_hexagonal'   : (3,12,12),
    'truncated_trihexagonal': (4,6,12),
    'snub_square'           : (3,3,4,3,4),
    'snub_trihexagonal'     : (3,3,3,3,6),
    'elongated_triangular'  : (3,3,3,4,4),
}


# --------------------------------------------------------------------------- #
# Geometry helpers                                                            #
# --------------------------------------------------------------------------- #

def _int_angle(n):
    return (n - 2) * np.pi / n

def _circumradius(n, side):
    return side / (2 * np.sin(np.pi / n))

def _spoke_offsets(config):
    offsets = [0.0]
    for n in config[:-1]:
        offsets.append(offsets[-1] + _int_angle(n))
    return np.array(offsets)

def _generate_figure(center, rotation, config, bond_length):
    """
    Returns (poly_list, outer_verts, edge_set).

    edge_set is a set of frozensets of coord_key pairs for all bond-length
    edges touching the center vertex.  These are the only TRUE bond neighbours
    of center (spoke edges) plus polygon-side edges between outer vertices.
    Using this instead of a distance search avoids false neighbours in tilings
    like (4,6,12) where non-adjacent vertices happen to be exactly bond_length
    apart across a polygon interior.
    """
    center      = np.asarray(center, dtype=float)
    ck          = _key(center)
    poly_list   = []
    current_ang = float(rotation)
    for n in config:
        alpha    = _int_angle(n)
        bisector = current_ang + alpha / 2.0
        R        = _circumradius(n, bond_length)
        poly_c   = center + R * np.array([np.cos(bisector), np.sin(bisector)])
        poly_rot = bisector + np.pi
        angles   = np.linspace(0, 2*np.pi, n, endpoint=False) + poly_rot
        verts    = poly_c + R * np.column_stack([np.cos(angles), np.sin(angles)])
        poly_list.append((n, np.round(verts, COORD_PREC)))
        current_ang += alpha
    all_v = np.round(np.vstack([v for _, v in poly_list]), COORD_PREC)
    c_r   = np.round(center, COORD_PREC)
    outer = all_v[np.linalg.norm(all_v - c_r, axis=1) > DIST_TOL]
    _, keep = np.unique(outer, axis=0, return_index=True)
    outer = outer[np.sort(keep)]

    # Build edge set: spoke edges (center→outer at bond_length) and
    # polygon-side edges between consecutive outer vertices of each polygon
    edge_set = set()
    for v in outer:
        if abs(np.linalg.norm(v - center) - bond_length) < DIST_TOL:
            edge_set.add(frozenset([ck, _key(v)]))
    for _, verts in poly_list:
        for i in range(len(verts)):
            va, vb = verts[i], verts[(i+1) % len(verts)]
            dab = np.linalg.norm(va - vb)
            da  = np.linalg.norm(va - center)
            db  = np.linalg.norm(vb - center)
            if abs(dab - bond_length) < DIST_TOL and da > DIST_TOL and db > DIST_TOL:
                edge_set.add(frozenset([_key(va), _key(vb)]))

    return poly_list, outer, edge_set


# --------------------------------------------------------------------------- #
# Vectorised rotation resolution                                              #
# --------------------------------------------------------------------------- #

def _consistent_rotations(center_pos, neighbors_arr, offsets, bond_length):
    """
    Return geometrically distinct rotations consistent with all neighbours.

    Fully vectorised distance check via (K*n, K, n) broadcasting.
    Deduplicates by spoke-tip frozenset fingerprint.
    """
    K = len(neighbors_arr)
    n = len(offsets)

    # All K*n candidate R values
    diffs  = neighbors_arr - center_pos                         # (K, 2)
    dirs   = np.arctan2(diffs[:, 1], diffs[:, 0])              # (K,)
    R_all  = (dirs[:, None] - offsets[None, :]).ravel()        # (K*n,)

    # Spoke-tip x,y for all candidates: (K*n, n)
    ang_all = R_all[:, None] + offsets[None, :]                 # (K*n, n)
    tips_x  = center_pos[0] + bond_length * np.cos(ang_all)    # (K*n, n)
    tips_y  = center_pos[1] + bond_length * np.sin(ang_all)    # (K*n, n)

    # Squared dist from every neighbour to every spoke tip
    # Shape: (K*n, K, n)
    nbr_x  = neighbors_arr[:, 0]                               # (K,)
    nbr_y  = neighbors_arr[:, 1]
    dx2    = (nbr_x[None, :, None] - tips_x[:, None, :]) ** 2  # (K*n, K, n)
    dy2    = (nbr_y[None, :, None] - tips_y[:, None, :]) ** 2
    min_d2 = (dx2 + dy2).min(axis=2)                           # (K*n, K)
    valid  = np.all(min_d2 < DIST_TOL**2, axis=1)              # (K*n,)

    valid_R = R_all[valid]
    if len(valid_R) == 0:
        return []

    # Deduplicate by spoke-tip frozenset fingerprint
    seen, result = set(), []
    for R in valid_R:
        a   = R + offsets
        tx  = np.round(center_pos[0] + bond_length * np.cos(a), SPOKE_ROUND)
        ty  = np.round(center_pos[1] + bond_length * np.sin(a), SPOKE_ROUND)
        fp  = frozenset(zip(tx.tolist(), ty.tolist()))
        if fp not in seen:
            seen.add(fp)
            result.append(float(R % (2 * np.pi)))
    return result


def _lookahead_filter(candidates, center_pos, config, offsets, bond_length,
                      proc_c, proc_r):
    """
    Eliminate candidates whose figure contradicts already-placed figures.
    proc_c : (M,2), proc_r : (M,)
    """
    if len(proc_c) == 0:
        return candidates

    valid = []
    for R in candidates:
        _, outer_c, _ = _generate_figure(center_pos, R, config, bond_length)

        # (m, M) distance matrix
        dists       = np.linalg.norm(
            outer_c[:, None, :] - proc_c[None, :, :], axis=2)
        on_boundary = np.abs(dists - bond_length) < DIST_TOL

        ok = True
        for vi, pi in zip(*np.where(on_boundary)):
            a  = proc_r[pi] + offsets
            sx = proc_c[pi, 0] + bond_length * np.cos(a)
            sy = proc_c[pi, 1] + bond_length * np.sin(a)
            if np.hypot(sx - outer_c[vi, 0], sy - outer_c[vi, 1]).min() > DIST_TOL:
                ok = False
                break
        if ok:
            valid.append(R)
    return valid


# --------------------------------------------------------------------------- #
# Key helper                                                                  #
# --------------------------------------------------------------------------- #

def _key(pos):
    p = np.asarray(pos, dtype=float)
    return (round(float(p[0]), COORD_PREC), round(float(p[1]), COORD_PREC))


# --------------------------------------------------------------------------- #
# Main class                                                                  #
# --------------------------------------------------------------------------- #

class ArchimedeanTiling:
    """
    Generates an Archimedean tiling from a vertex configuration.

    Parameters
    ----------
    config          : tuple of ints
    bond_length     : float (default 1)
    global_rotation : float, degrees (default 0)
    """

    def __init__(self, config, bond_length=1.0, global_rotation=0.0):
        self.config      = tuple(config)
        self.n           = len(config)
        self.bond_length = bond_length
        self.global_rot  = np.radians(global_rotation)

        angle_sum = sum(_int_angle(k) for k in config)
        if not np.isclose(angle_sum, 2*np.pi, atol=1e-9):
            raise ValueError(
                f"Angles sum to {np.degrees(angle_sum):.4f}°, not 360°.")

        self._offsets = _spoke_offsets(config)

        self.vertex_coords = []
        self._v_registry   = {}
        self._c_registry   = {}
        self._raw_polys    = []
        self._edge_set     = set()   # frozensets of key-pairs, for drawing
        self._adj          = {}      # key → set of neighbour keys (O(1) lookup)

        # Incremental proc cache: lists appended per commit, numpy built lazily
        self._pc_list  = []   # [(x,y), ...]
        self._pr_list  = []   # [rotation, ...]
        self._pc_arr   = None
        self._pr_arr   = None

        self._tree = None

    # ----------------------------------------------------------------------- #
    # Internals                                                               #
    # ----------------------------------------------------------------------- #

    def _key(self, pos):
        return _key(pos)

    def _register(self, pos):
        k = self._key(pos)
        if k not in self._v_registry:
            self._v_registry[k] = len(self.vertex_coords)
            self.vertex_coords.append(np.asarray(pos, dtype=float))
            self._tree = None
        return k

    def _nbrs(self, pos):
        """True bond neighbours via adjacency dict — no false positives."""
        k  = _key(pos)
        nb = [self.vertex_coords[self._v_registry[nk]]
              for nk in self._adj.get(k, set())
              if nk in self._v_registry]
        return np.array(nb) if nb else np.empty((0, 2))

    def _commit(self, key, rot, polys, outer, edge_set=None):
        self._c_registry[key] = rot
        self._raw_polys.extend(polys)
        self._pc_list.append(key)
        self._pr_list.append(rot)
        self._pc_arr = self._pr_arr = None   # invalidate cache
        if edge_set:
            self._edge_set.update(edge_set)
            for ek in edge_set:
                keys = list(ek)
                if len(keys) == 2:
                    ka, kb = keys
                    self._adj.setdefault(ka, set()).add(kb)
                    self._adj.setdefault(kb, set()).add(ka)

    def _proc(self):
        """Cached (M,2) array of committed centres and (M,) rotations."""
        if self._pc_arr is None:
            self._pc_arr = np.array(self._pc_list, dtype=float) \
                if self._pc_list else np.empty((0, 2))
            self._pr_arr = np.array(self._pr_list, dtype=float) \
                if self._pr_list else np.empty(0)
        return self._pc_arr, self._pr_arr

    # ----------------------------------------------------------------------- #
    # BFS                                                                     #
    # ----------------------------------------------------------------------- #

    def generate(self, num_figures=200, bbox=None):
            """
            Grow the tiling by BFS of vertex figures.

            num_figures : number of figures to place (vertex count ≈ 3–8× this).
            bbox        : (xmin,xmax,ymin,ymax) centroid bounding box; overrides
                        num_figures when given.
            """
            self.vertex_coords = []
            self._v_registry   = {}
            self._c_registry   = {}
            self._raw_polys    = []
            self._edge_set     = set()
            self._adj          = {}
            self._pc_list      = []
            self._pr_list      = []
            self._pc_arr       = self._pr_arr = None
            self._tree         = None

            def inb(pos):
                if bbox is None: return True
                x,y = float(pos[0]), float(pos[1])
                return bbox[0]<=x<=bbox[1] and bbox[2]<=y<=bbox[3]

            def budget():
                return bbox is not None or len(self._c_registry) < num_figures

            _, _seed_outer, _ = _generate_figure(
                np.array([0., 0.]), 0.0, self.config, self.bond_length)
            MAX_DEFER = max(self.n * 6, len(_seed_outer) * 4)

            # Seed
            ok     = np.array([0.0, 0.0])
            ok_key = self._key(ok)
            self._register(ok)
            polys, outer, eseed = _generate_figure(ok, self.global_rot, self.config, self.bond_length)
            self._commit(ok_key, self.global_rot, polys, outer, eseed)

            queue  = deque()
            queued = {}   # key -> current defer count

            for pos in outer:
                k = self._register(pos)
                if k != ok_key:
                    queue.append((k, 0))
                    queued[k] = 0

            while queue:
                key, dc = queue.popleft()
                queued.pop(key, None)

                if key in self._c_registry:
                    continue

                cp = np.asarray(key, dtype=float)
                if not inb(cp):
                    continue

                nb = self._nbrs(cp)

                if len(nb) == 0:
                    if dc < MAX_DEFER and budget():
                        queue.append((key, dc + 1))
                        queued[key] = dc + 1
                    continue

                cands = _consistent_rotations(cp, nb, self._offsets, self.bond_length)

                if len(cands) != 1:
                    pc, pr = self._proc()
                    cands = _lookahead_filter(
                        cands, cp, self.config, self._offsets, self.bond_length, pc, pr)

                if len(cands) == 1:
                    rot = cands[0]

                elif len(cands) == 0:
                    if dc < MAX_DEFER and budget():
                        queue.append((key, dc + 1))
                        queued[key] = dc + 1
                    continue

                else:
                    if dc < MAX_DEFER and budget():
                        queue.append((key, dc + 1))
                        queued[key] = dc + 1
                        continue
                    rot = cands[0]

                if not budget():
                    continue

                # ---- Place the figure ----------------------------------------
                polys, outer, eset = _generate_figure(cp, rot, self.config, self.bond_length)
                self._commit(key, rot, polys, outer, eset)

                for pos in outer:
                    k = self._register(pos)
                    if k not in self._c_registry and k not in queued:
                        if budget() and inb(np.asarray(k)):
                            queue.append((k, 0))
                            queued[k] = 0

            return self

    # ----------------------------------------------------------------------- #
    # Unique polygons / plot                                                  #
    # ----------------------------------------------------------------------- #

    def _unique_polygons(self):
        seen, result = set(), []
        for n, verts in self._raw_polys:
            ck = tuple(np.round(verts.mean(axis=0), 3))
            if ck not in seen:
                seen.add(ck); result.append((n, verts))
        return result

    def plot(self, show_edges=True, show_vertices=True, color_faces=True,
             mark_origin=False, figsize=(8,8), title=None,
             save_path=None, display_bbox=None):
        if not self.vertex_coords:
            raise RuntimeError("Call generate() first.")

        coords = np.array(self.vertex_coords)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal'); ax.axis('off')

        if color_faces:
            patches, fc = [], []
            for n, v in self._unique_polygons():
                patches.append(MplPolygon(v, closed=True))
                fc.append(POLYGON_COLORS.get(n, '#aaaaaa'))
            ax.add_collection(PatchCollection(
                patches, facecolor=fc, edgecolor='none', alpha=0.85, zorder=1))

        if show_edges:
            drawn = set()
            for ek in self._edge_set:
                keys = list(ek)
                if len(keys) != 2: continue
                ka, kb = keys
                if (ka,kb) in drawn or (kb,ka) in drawn: continue
                if ka not in self._v_registry or kb not in self._v_registry: continue
                ia, ib = self._v_registry[ka], self._v_registry[kb]
                ax.plot([coords[ia,0],coords[ib,0]],[coords[ia,1],coords[ib,1]],
                        color='black', lw=0.7, zorder=2)
                drawn.add((ka,kb))

        if show_vertices:
            ax.scatter(coords[:,0], coords[:,1], c='black', s=18, zorder=3)
        if mark_origin:
            cmap = plt.get_cmap('coolwarm')
            cnorm = Normalize(vmin=0, vmax=1)
            red= cmap(cnorm(0.95))

            ax.scatter([0.],[0.], c=red, s=60, zorder=4)

        if display_bbox is not None:
            xmin,xmax,ymin,ymax = display_bbox
            ax.add_patch(plt.Rectangle(
                (xmin,ymin),xmax-xmin,ymax-ymin,
                fill=False, edgecolor='#333333', lw=1.5, zorder=6))
            ax.set_xlim(xmin-.5, xmax+.5); ax.set_ylim(ymin-.5, ymax+.5)

        if color_faces and self._raw_polys:
            un = sorted({n for n,_ in self._raw_polys})
            ax.legend(handles=[
                mpatches.Patch(facecolor=POLYGON_COLORS.get(n,'#aaaaaa'),
                               edgecolor='black', label=f'{n}-gon')
                for n in un], loc='upper right', fontsize=9)

        ax.set_title(title or f"Archimedean tiling  {self.config}",
                     fontsize=13, pad=10)
        ax.autoscale_view(); plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Saved: {save_path}")
        # plt.show()

    def __repr__(self):
        return (f"ArchimedeanTiling(config={self.config}, "
                f"vertices={len(self.vertex_coords)}, "
                f"figures={len(self._c_registry)})")
    

# --------------------------------------------------------------------------- #
# Lattice vector and basis extraction                                         #
# --------------------------------------------------------------------------- #    

def find_lattice_vectors(c_registry: dict, tol: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the two primitive lattice vectors from a placed tiling.

    Strategy: figures with the same rotation are related by pure lattice
    translations. The shortest two linearly independent displacement vectors
    between same-rotation figures are the primitive lattice vectors.

    Parameters
    ----------
    c_registry : dict of coord_key -> rotation (from ArchimedeanTiling._c_registry)
    tol        : tolerance for grouping rotations and comparing vector lengths

    Returns
    -------
    a, b : two (2,) lattice vectors, ordered so that a is the shorter one
    """
    # Group figure centres by rotation
    centres   = np.array(list(c_registry.keys()), dtype=float)
    rotations = np.array(list(c_registry.values()))

    # Find unique rotations
    unique_rots = []
    for r in rotations:
        if not any(abs(r - ur) < tol for ur in unique_rots):
            unique_rots.append(r)

    # Collect all displacement vectors between same-rotation figures
    candidates = []
    for ur in unique_rots:
        mask  = np.abs(rotations - ur) < tol
        group = centres[mask]
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                d = group[j] - group[i]
                if np.linalg.norm(d) > tol:
                    candidates.append(d)

    # Sort by length
    candidates.sort(key=lambda v: np.linalg.norm(v))

    # Pick shortest vector as a, then find shortest independent vector as b
    a = candidates[0]
    b = None
    for v in candidates[1:]:
        # Check linear independence via cross product magnitude
        cross = abs(a[0] * v[1] - a[1] * v[0])
        if cross > tol:
            b = v
            break

    if b is None:
        raise ValueError("Could not find two independent lattice vectors.")

    # Enforce convention: first nonzero component positive
    def make_positive(v):
        for c in v:
            if abs(c) > tol:
                return v if c > 0 else -v
        return v

    a = make_positive(a)
    b = make_positive(b)

    # Enforce angle <= 90°
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    if cos_angle < -tol:
        b = -b

    # Order so a is shorter (or same length)
    if np.linalg.norm(b) < np.linalg.norm(a) - tol:
        a, b = b, a

    return a, b

def find_basis(vertices: list,
               a: np.ndarray, b: np.ndarray,
               tol: float = 1e-3) -> np.ndarray:
    """
    Find the fractional coordinates of all inequivalent vertices within
    one primitive unit cell.

    Parameters
    ----------
    c_registry : dict of coord_key -> rotation (from ArchimedeanTiling._c_registry)
    vertices   : list of (2,) arrays (ArchimedeanTiling.vertex_coords)
    a, b       : primitive lattice vectors as (2,) arrays
    tol        : tolerance for deduplication

    Returns
    -------
    basis : (N, 2) array of fractional coordinates in [0, 1)
    """
    # Matrix to convert Cartesian to fractional: columns are a and b
    M     = np.column_stack([a, b])
    M_inv = np.linalg.inv(M)

    # Convert all vertices to fractional coordinates
    coords = np.array(vertices)                        # (V, 2)
    frac   = (M_inv @ coords.T).T                      # (V, 2)

    # Reduce to [0, 1) and clean up near-integer values
    frac = frac % 1.0
    frac[np.abs(frac) < tol]       = 0.0
    frac[np.abs(frac - 1.0) < tol] = 0.0

    # Deduplicate
    unique = []
    for f in frac:
        if not any(np.linalg.norm(f - u) < tol for u in unique):
            unique.append(f)

    basis = np.array(sorted(unique, key=lambda x: (round(x[0], 3), round(x[1], 3))))
    return basis


# --------------------------------------------------------------------------- #
# Convenience wrapper                                                         #
# --------------------------------------------------------------------------- #

def generate(config, num_figures=200, global_rotation=0.0,
             bbox=None, save_path=None, **kw):
    """
    One-liner: create, grow, and plot a tiling.
    config may be a name string or tuple.

    Examples
    --------
    >>> generate('kagome', num_figures=300)
    >>> generate((3,3,4,3,4), num_figures=200, global_rotation=15)
    >>> generate('hexagonal', bbox=(-8,8,-8,8))
    """
    if isinstance(config, str):
        config = ARCHIMEDEAN[config]
    t = ArchimedeanTiling(config, global_rotation=global_rotation)
    t.generate(num_figures=num_figures, bbox=bbox)
    u = t._unique_polygons()
    print(f"Config {config}: {len(t.vertex_coords)} vertices, "
          f"{len(t._c_registry)} figures, {len(u)} polygons")
    t.plot(title=f"Archimedean tiling  {config}", save_path=save_path, **kw)
    return t

def plot_tiling_ax(ax, tiling, display_bbox, title=None):
    """
    Plot a tiling onto an existing axes, clipped to display_bbox.

    Parameters
    ----------
    ax           : matplotlib axes
    tiling       : ArchimedeanTiling (already generated)
    display_bbox : (xmin, xmax, ymin, ymax) clip region
    title        : axes title string
    """
    coords = np.array(tiling.vertex_coords)
    xmin, xmax, ymin, ymax = display_bbox

    if tiling._unique_polygons():
        patches, fc = [], []
        for n, v in tiling._unique_polygons():
            patches.append(MplPolygon(v, closed=True))
            fc.append(POLYGON_COLORS.get(n, '#aaaaaa'))
        ax.add_collection(PatchCollection(
            patches, facecolor=fc, edgecolor='none', alpha=0.85, zorder=1))

    drawn = set()
    for ek in tiling._edge_set:
        keys = list(ek)
        if len(keys) != 2: continue
        ka, kb = keys
        if (ka,kb) in drawn or (kb,ka) in drawn: continue
        if ka not in tiling._v_registry or kb not in tiling._v_registry: continue
        ia, ib = tiling._v_registry[ka], tiling._v_registry[kb]
        ax.plot([coords[ia,0], coords[ib,0]],
                [coords[ia,1], coords[ib,1]],
                color='black', lw=0.7, zorder=2)
        drawn.add((ka,kb))

    ax.scatter(coords[:,0], coords[:,1], c='black', s=10, zorder=3)
    # ax.scatter([0.], [0.], c='red', s=40, zorder=4)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=28, pad=8)


# --------------------------------------------------------------------------- #
# Smoke-test                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    # import matplotlib, time
    # matplotlib.use('Agg')

    # print(f"\n{'Name':<28} {'Verts':>6} {'Figs':>5} {'Polys':>6}"
    #       f"  {'Time':>5}  Sizes       Status")
    # print('-'*76)

    # for name, config in ARCHIMEDEAN.items():
    #     try:
    #         t0 = time.time()
    #         t  = ArchimedeanTiling(config)
    #         t.generate(num_figures=200)
    #         dt = time.time()-t0

    #         u        = t._unique_polygons()
    #         found    = sorted({n for n,_ in u})
    #         expected = sorted(set(config))
    #         ok       = (found == expected)

    #         print(f"{name:<28} {len(t.vertex_coords):>6} "
    #               f"{len(t._c_registry):>5} {len(u):>6}"
    #               f"  {dt:>4.1f}s  {str(found):<11}"
    #               f"  {'OK' if ok else 'FAIL '+str(expected)}")

    #         t.plot(save_path=f"archimedean/tools/tiling_generation/tilings/{name}.pdf",
    #                title=f"{name}  {config}")
    #         plt.close('all')

    #     except Exception as e:
    #         import traceback
    #         print(f"{name:<28}   ERROR: {e}")
    #         traceback.print_exc()

    import matplotlib.pyplot as plt

    plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Latin Modern Roman'],
    })

    # CONFIGS = [
    #     dict(
    #         name       = 'triangular',
    #         config     = (3,3,3,3,3,3),
    #         num_figures= 200,
    #         bbox       = (-5, 5, -5, 5),   # generation bbox
    #         display    = (-2.8, 2.8, -2.8, 2.8),   # display clip
    #         title      = 'Triangular',
    #     ),
    #     dict(
    #         name       = 'square',
    #         config     = (4,4,4,4),
    #         num_figures= 200,
    #         bbox       = (-5, 5, -5, 5),
    #         display    = (-2.8, 2.8, -2.8, 2.8),
    #         title      = 'Square',
    #     ),
    #     dict(
    #         name       = 'hexagonal',
    #         config     = (6,6,6),
    #         num_figures= 200,
    #         bbox       = (-5, 5, -5, 5),
    #         display    = (-3.8, 3.8, -3.8, 3.8),
    #         title      = 'Honeycomb',
    #     ),
    # ]


    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # # fig.suptitle('Regular Tilings', fontsize=16, y=1.02)

    # for ax, cfg in zip(axes, CONFIGS):
    #     t = ArchimedeanTiling(cfg['config'])
    #     t.generate(num_figures=cfg['num_figures'], bbox=cfg['bbox'])
    #     plot_tiling_ax(ax, t, cfg['display'], title=cfg['title'])

    # plt.tight_layout()
    # plt.savefig('archimedean/tools/tiling_generation/tilings/regular_tilings.pdf', dpi=600, bbox_inches='tight')
    # plt.show()

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    # fig.suptitle('Archimedean Tilings', fontsize=16, y=1.02)
    plt.subplots_adjust(hspace=0.4)

    CONFIGS_8 = [
        dict(config=(3,6,3,6),     num_figures=200, bbox=None, display=(-4,4,-4,4), title=r'Kagome'),
        dict(config=(3,4,6,4),     num_figures=200, bbox=None, display=(-4,4,-4,4), title=r'Ruby'),
        dict(config=(4,8,8),       num_figures=200, bbox=None, display=(-4,4,-4,4), title=r'CaVO'),
        dict(config=(3,12,12),     num_figures=200, bbox=None, display=(-5,5,-5,5), title=r'Star'),
        dict(config=(4,6,12),      num_figures=200, bbox=None, display=(-5,5,-5,5), title=r'SHD'),
        dict(config=(3,3,4,3,4),   num_figures=200, bbox=None, display=(-4,4,-4,4), title=r'SSL'),
        dict(config=(3,3,3,3,6),   num_figures=200, bbox=None, display=(-4,4,-4,4), title=r'Maple-Leaf'),
        dict(config=(3,3,3,4,4),   num_figures=200, bbox=None, display=(-4,4,-4,4), title=r'Trellis'),
    ]

    for ax, cfg in zip(axes.flat, CONFIGS_8):
        t = ArchimedeanTiling(cfg['config'], global_rotation=cfg.get('rotation', 0))
        t.generate(num_figures=cfg['num_figures'], bbox=cfg['bbox'])
        plot_tiling_ax(ax, t, cfg['display'], title=cfg['title'])

    plt.tight_layout()
    plt.savefig('archimedean/tools/tiling_generation/tilings/semiregular_tilings.pdf', dpi=600, bbox_inches='tight')
    plt.show()

    # ARCHIMEDEAN = {
    #     'triangular'            : (3,3,3,3,3,3),
    #     'square'                : (4,4,4,4),
    #     'hexagonal'             : (6,6,6),
    #     'kagome'                : (3,6,3,6),
    #     'rhombitrihexagonal'    : (3,4,6,4),
    #     'truncated_square'      : (4,8,8),
    #     'truncated_hexagonal'   : (3,12,12),
    #     'truncated_trihexagonal': (4,6,12),
    #     'snub_square'           : (3,3,4,3,4),
    #     'snub_trihexagonal'     : (3,3,3,3,6),
    #     'elongated_triangular'  : (3,3,3,4,4),
    # }

    # for name, config in ARCHIMEDEAN.items():
    #     t = ArchimedeanTiling(config)
    #     t.generate(num_figures=100)
    #     a, b = find_lattice_vectors(t._c_registry)
    #     print(f"{name:<24}  a={np.round(a,4)}  b={np.round(b,4)}"
    #           f"  |a|={np.linalg.norm(a):.4f}  |b|={np.linalg.norm(b):.4f}"
    #           f"  angle={np.degrees(np.arccos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))):.1f}°")
        
    # for name, config in ARCHIMEDEAN.items():
    #     t = ArchimedeanTiling(config)
    #     t.generate(num_figures=100)
    #     a, b = find_lattice_vectors(t._c_registry)
    #     basis = find_basis(t.vertex_coords, a, b)
    #     print(f"\n{name}")
    #     print(f"  a={np.round(a,4)}  b={np.round(b,4)}")
    #     print(f"  basis ({len(basis)} site(s)):")
    #     for f in basis:
    #         print(f"    {np.round(f, 4)}")