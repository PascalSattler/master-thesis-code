"""
bz_config.py

Brillouin zone types and their high-symmetry point definitions.

Contains static BZ libraries and dynamic functions for determining
high-symmetry points based on lattice geometry.
"""

import numpy as np
from numpy.linalg import norm
from magnon_solver.utils import angle_between_vectors



# =============================================================================
# Static Brillouin Zone Library
# =============================================================================

BZ_LIBRARY = {
    'cubic': {
        'G': ([0, 0, 0], r'$\Gamma$'),
        'M': ([0.5, 0.5, 0], 'M'),
        'R': ([0.5, 0.5, 0.5], 'R'),
        'X': ([0.5, 0, 0], 'X'),
    },
    'hexagonal': {
        # NOTE: K and K' positions are angle-dependent
        # use get_hexagonal_hsp() for correct positions
        'G': ([0, 0, 0], r'$\Gamma$'),
        'M': ([0.5, 0, 0], 'M'),
        'A': ([0, 0, 0.5], 'A'),
        'L': ([0.5, 0, 0.5], 'L'),
        'H': ([1/3, 1/3, 0.5], 'H'),
    },
    'orthorhombic': {
        'G': ([0, 0, 0], r'$\Gamma$'),
        'X': ([0.5, 0, 0], 'X'),
        'Y': ([0, 0.5, 0], 'Y'),
        'Z': ([0, 0, 0.5], 'Z'),
        'S': ([0.5, 0.5, 0], 'S'),
        'U': ([0.5, 0, 0.5], 'U'),
        'T': ([0, 0.5, 0.5], 'T'),
        'R': ([0.5, 0.5, 0.5], 'R'),
        'Xp': ([-0.5, 0, 0], r"X$'$"),
        'Yp': ([0, -0.5, 0], r"Y$'$"),
        'Zp': ([0, 0, -0.5], r"Z$'$"),
        'Sp': ([-0.5, -0.5, 0], r"S$'$"),
        'Up': ([-0.5, 0, -0.5], r"U$'$"),
        'Tp': ([0, -0.5, -0.5], r"T$'$"),
        'Rp': ([-0.5, -0.5, -0.5], r"R$'$"),
    },
    'oblique': {
        'G': ([0, 0, 0], r'$\Gamma$'),
        'Y': ([0.5, 0, 0], 'Y'),
        'H': ([0.53589838, -0.26794919, 0], 'H'),
        'C': ([0.5, -0.5, 0], 'C'),
        'H1': ([0.46410162, -0.73205081, 0], 'H1'),
        'X': ([0, -0.5, 0], 'X'),
    },
    'oblique2': {
        'G': ([0, 0, 0], r'$\Gamma$'),
        'Y': ([0, 0.5, 0], 'Y'),
        'A1': ([0.46410162, 0.26794919, 0], 'A1'),
        'X': ([0.53589838, -0.26794919, 0], 'X'),
    },
    'obliqueAFLOW': {
        'G': ([0, 0, 0], r'$\Gamma$'),
        'X': ([0.5, 0, 0], 'X'),
        'Y': ([0, 0.5, 0], 'Y'),
        'Y1': ([0, -0.5, 0], 'Y1'),
        'Z': ([0, 0.5, 0], 'Z'),
        'C': ([0.5, 0.5, 0], 'C'),
        # NOTE: H, H1, H2 positions are angle-dependent
        # use get_oblique_hsp() for correct positions
    },
}

# =============================================================================
# Brillouin Zone Boundary Vertices
# =============================================================================

# Brillouin zone boundary vertices (in reciprocal lattice coordinates)
# Vertices define the polygon boundary in the kx-ky plane

BZ_VERTICES = {
    'cubic': [
        [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]
    ],
    'orthorhombic': [
        [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]
    ],
    'hexagonal': [  # regular hexagon
        [0.5, 0.0], [0.25, 0.433], [-0.25, 0.433],
        [-0.5, 0.0], [-0.25, -0.433], [0.25, -0.433]
    ],
}

# =============================================================================
# Dynamic High-Symmetry Point Functions
# =============================================================================

def get_hexagonal_hsp(reciprocal_vectors: np.ndarray) -> dict:
    """
    Determine high-symmetry points for hexagonal BZ based on reciprocal lattice angle.

    The positions of K and K' points depend on the angle between the reciprocal
    lattice vectors b1 and b2:
    - 60° angle -> K = (1/3, 1/3, 0), K' = (-1/3, -1/3, 0)
    - 120° angle -> K = (1/3, -1/3, 0), K' = (-1/3, 1/3, 0)

    Parameters
    ----------
    reciprocal_vectors : np.ndarray of shape (3, 3)
        Reciprocal lattice vectors [b1, b2, b3] as rows.

    Returns
    -------
    hsp : dict
        High-symmetry points with correct K and K' positions.
        Format: {label: ([kx, ky, kz], display_label)}

    Raises
    ------
    ValueError
        If the angle between b1 and b2 is not 60° or 120°.

    Examples
    --------
    >>> b1 = np.array([1, 0, 0])
    >>> b2 = np.array([0.5, np.sqrt(3)/2, 0])  # 60° angle
    >>> b3 = np.array([0, 0, 1])
    >>> hsp = get_hexagonal_hsp(np.array([b1, b2, b3]))
    >>> hsp['K']
    ([0.333..., 0.333..., 0], 'K')
    """
    b1, b2, b3 = reciprocal_vectors

    # compute angle between b1 and b2
    angle = angle_between_vectors(b1, b2)

    # start with base hexagonal points
    hsp = {
        'G': ([0, 0, 0], r'$\Gamma$'),
        'M': ([0.5, 0, 0], 'M'),
        'A': ([0, 0, 0.5], 'A'),
        'L': ([0.5, 0, 0.5], 'L'),
        'H': ([1/3, 1/3, 0.5], 'H'),
    }

    # Determine K and K' based on angle
    angle_deg = angle * 180 / np.pi

    if abs(angle - np.pi/3) < 1e-3:  # 60 degrees
        hsp['K'] = ([1/3, 1/3, 0], 'K')
        hsp['Kp'] = ([-1/3, -1/3, 0], r"K$'$")
    elif abs(angle - 2*np.pi/3) < 1e-3:  # 120 degrees
        hsp['K'] = ([1/3, -1/3, 0], 'K')
        hsp['Kp'] = ([-1/3, 1/3, 0], r"K$'$")
    else:
        raise ValueError(
            f"Hexagonal lattice: angle between b1 and b2 is {angle_deg:.1f}°. "
            f"Expected 60° or 120° for standard hexagonal symmetry."
        )

    return hsp


def get_oblique_hsp(reciprocal_vectors: np.ndarray) -> dict:
    """
    Determine high-symmetry points for oblique (in-plane part of monoclinic) BZ 
    based on reciprocal lattice angle. Follows the AFLOW standard.
    (doi.org/10.1016/j.commatsci.2010.05.010)

    The positions of H, H1, H2 points depend on the angle between the real space
    lattice vectors a1 and a2, hence they are calculated from the reciprocal 
    vectors.

    Parameters
    ----------
    reciprocal_vectors : np.ndarray of shape (3, 3)
        Reciprocal lattice vectors [b1, b2, b3] as rows.

    Returns
    -------
    hsp : dict
        High-symmetry points with correct H, H1, H2 positions.
        Format: {label: ([kx, ky, kz], display_label)}
    """
    b1, b2, b3 = reciprocal_vectors

    # AFLOW uses real space vects
    vol = np.dot(b1, np.cross(b2, b3))

    a1 = 2 * np.pi * np.cross(b2, b3) / vol
    a2 = 2 * np.pi * np.cross(b3, b1) / vol
    a3 = 2 * np.pi * np.cross(b1, b2) / vol

    # compute angle between a1 and a2
    angle = angle_between_vectors(a1, a2) # 75° or 5pi/12 for elongated triangular
    b = norm(a1)    # 1
    c = norm(a2)    # sqrt(2 + sqrt(3))

    eta = (1 - (b / c) * np.cos(angle)) / (2 * (np.sin(angle))**2) # 2*sqrt(3) - 3 ~ 0.4641016
    nu = 1/2 - eta * (c / b) * np.cos(angle)                       # 2 - sqrt(3) ~ 0.2679492

    hsp = {
        'G': ([0, 0, 0], r'$\Gamma$'),
        'X': ([0.5, 0, 0], 'X'),
        'Y': ([0, 0.5, 0], 'Y'),
        'Y1': ([0, -0.5, 0], 'Y1'),
        'Z': ([0, 0.5, 0], 'Z'),
        'C': ([0.5, 0.5, 0], 'C'),
        'H': ([eta, 1 - nu, 0], 'H'),   # (0.4641016, 0.7320508)
        'H1': ([1 - eta, nu, 0], 'H1'), # (0.5358984, 0.2679492)
        'H2': ([eta, - nu, 0], 'H2'),   # (0.4641016, - 0.2679492)
    }
    
    return hsp


def get_bz_hsp(bz_type: str, reciprocal_vectors: np.ndarray = None) -> dict:
    """
    Get high-symmetry points for a given Brillouin zone type.

    For most BZ types, returns static definitions from BZ_LIBRARY.
    For hexagonal, computes K and K' positions dynamically based on
    reciprocal lattice geometry.

    Parameters
    ----------
    bz_type : str
        Brillouin zone type (e.g., 'cubic', 'hexagonal', 'orthorhombic').
    reciprocal_vectors : np.ndarray of shape (3, 3), optional
        Reciprocal lattice vectors [b1, b2, b3]. Required for 'hexagonal'.

    Returns
    -------
    hsp : dict
        High-symmetry points dictionary.

    Raises
    ------
    ValueError
        If bz_type is unknown or reciprocal_vectors not provided for hexagonal.
    """
    if bz_type not in BZ_LIBRARY:
        raise ValueError(
            f"Unknown BZ type '{bz_type}'. "
            f"Available types: {list(BZ_LIBRARY.keys())}"
        )

    if bz_type == 'hexagonal':
        if reciprocal_vectors is None:
            raise ValueError(
                "reciprocal_vectors must be provided for hexagonal BZ type"
            )
        return get_hexagonal_hsp(reciprocal_vectors)
    elif bz_type == 'obliqueAFLOW':
        if reciprocal_vectors is None:
            raise ValueError(
                "reciprocal_vectors must be provided for oblique BZ type"
            )
        return get_oblique_hsp(reciprocal_vectors)
    else:
        return BZ_LIBRARY[bz_type].copy()