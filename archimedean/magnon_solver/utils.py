"""
utils.py

General mathematical and computational utilities for the magnon solver.
"""

import numpy as np
from numpy.linalg import norm



def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the angle between two vectors in radians.

    Parameters
    ----------
    v1, v2 : np.ndarray
        Input vectors (any dimension).

    Returns
    -------
    angle : float
        Angle between vectors in radians [0, π].

    Examples
    --------
    >>> v1 = np.array([1, 0, 0])
    >>> v2 = np.array([0, 1, 0])
    >>> angle_between_vectors(v1, v2)
    1.5707963267948966  # π/2
    """
    # normalize vectors
    v1_norm = v1 / norm(v1)
    v2_norm = v2 / norm(v2)

    # compute cosine of angle
    cos_angle = np.dot(v1_norm, v2_norm)

    # clip to handle numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # return angle in radians
    return np.arccos(cos_angle)


def reciprocal_lattice(a1: np.ndarray, a2: np.ndarray, a3: np.ndarray) -> np.ndarray:
    """
    Compute reciprocal lattice vectors from real-space lattice vectors.

    The reciprocal lattice vectors satisfy:
        b_i · a_j = 2π δ_ij

    Parameters
    ----------
    a1, a2, a3 : np.ndarray of shape (3,)
        Real-space lattice vectors.

    Returns
    -------
    reciprocal_vectors : np.ndarray of shape (3, 3)
        Reciprocal lattice vectors as rows [b1, b2, b3].

    Examples
    --------
    >>> a1 = np.array([1, 0, 0])
    >>> a2 = np.array([0, 1, 0])
    >>> a3 = np.array([0, 0, 1])
    >>> b = reciprocal_lattice(a1, a2, a3)
    >>> np.allclose(b, 2*np.pi*np.eye(3))
    True
    """
    # volume of unit cell
    volume = np.abs(np.dot(a1, np.cross(a2, a3)))

    if volume < 1e-10:
        raise ValueError("Lattice vectors are coplanar (volume = 0)")

    # Reciprocal lattice vectors
    b1 = 2 * np.pi * np.cross(a2, a3) / volume
    b2 = 2 * np.pi * np.cross(a3, a1) / volume
    b3 = 2 * np.pi * np.cross(a1, a2) / volume

    return np.array([b1, b2, b3])


def rotation_matrix_3d(axis: str, angle: float) -> np.ndarray:
    """
    Create a 3D rotation matrix for rotation about a coordinate axis.

    Parameters
    ----------
    axis : {'x', 'y', 'z'}
        Axis of rotation.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    R : np.ndarray of shape (3, 3)
        Rotation matrix.

    Examples
    --------
    >>> R = rotation_matrix_3d('z', np.pi/2)
    >>> v = np.array([1, 0, 0])
    >>> np.allclose(R @ v, [0, 1, 0])
    True
    """
    c = np.cos(angle)
    s = np.sin(angle)

    if axis.lower() == 'x':
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis.lower() == 'y':
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    elif axis.lower() == 'z':
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Axis must be 'x', 'y', or 'z', got '{axis}'")


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Parameters
    ----------
    v : np.ndarray
        Input vector.

    Returns
    -------
    v_norm : np.ndarray
        Normalized vector.

    Raises
    ------
    ValueError
        If vector has zero length.
    """
    norm = np.linalg.norm(v)
    if norm < 1e-14:
        raise ValueError("Cannot normalize zero vector")
    return v / norm