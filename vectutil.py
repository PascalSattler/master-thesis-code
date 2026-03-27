from numpy.linalg import norm
from numpy import dot, arccos


def angle_between(v1, v2):
    """Compute angle between 0 and pi between `v1` and `v2`."""
    c = dot(v1, v2) / (norm(v1) * norm(v2))
    return arccos(c)

