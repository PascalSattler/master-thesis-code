import numpy as np
from numpy import meshgrid, sqrt, sin, cos, array, zeros
from numpy.linalg import norm

hexedgetol = 1e-3


def hexgrid(nums, hexlen, rotangle=0, offset=(0, 0)):
    """
    Generate a hexagonal grid.

    Parameters
    ----------
    nums : int or tuple of two ints
        Numbers of points used in the underlying rectangular mesh.
    hexlen : float
        Distance between two adjacent hexagon corners.
    rotangle : float, optional
        Rotation angle in radiants. If None, two edges are parallel to the
        x axis. Default is None.
    offset : tuple of float, optional
        Shift of the hexagon with respect to the origin. If zero, the hexagon
        is centered about the origin. Default is zero.
    """
    def local_hwidth(y):
        res = hwidth - abs(y)/hheight * hwidth/2
        return res

    def rotate(x, y):
        if rotangle == 0:
            return x, y
        res_x = x*cos(rotangle) - y*sin(rotangle)
        res_y = x*sin(rotangle) + y*cos(rotangle)
        return res_x, res_y

    if not isinstance(nums, (tuple, list)):
        nums = 2 * [nums]
    abstol = hexlen * hexedgetol
    hwidth = hexlen
    hheight = sqrt(3)/2 * hwidth
    xs = linspace(-hwidth, hwidth, num=nums[0])
    ys = linspace(-hheight, hheight, num=nums[1])
    xms, yms = meshgrid(xs, ys)
    hxms = []
    hyms = []
    for j in range(nums[1]):
        lborder = rborder = False
        hxms.append([])
        hyms.append([])
        y = yms[j, 0]
        lhw = local_hwidth(y)
        for i in range(nums[0]):
            if rborder:
                break
            x = xms[j, i]
            lhw = local_hwidth(y)
            # left of hexagon
            if x < -lhw and lborder:
                continue
            if not lborder:
                lborder = True
                hxm = -lhw
            # right of hexagon
            elif x > lhw:
                rborder = True
                hxm = lhw
            # inner hexagon
            else:
                # skip points close to the edge
                if abs(x - lhw) < abstol or abs(x + lhw) < abstol:
                    continue
                hxm = x
            hxm, y = rotate(hxm, y)
            hxms[j].append(hxm)
            hyms[j].append(y)
    hxms, hyms = array(hxms), array(hyms)
    return hxms, hyms


def parallelogram_grid(v1, v2, nums, startpoint=True, endpoint=True, offset=0,
                       gtype='center'):
    """
    Generate a parallelogram grid spanned by two vectors.

    Parameters:
    -----------
    v1 : np.ndarray
        First vector spanning the parallelogram.
        entries.
    v2 : np.ndarray
        Second vector spanning the parallelogram.
    num : int or tuple
        Numbers of subdivisions along each input vector. If it is a tuple, two
        entries are expected which subdivide along the two axes. If it is an
        integer, the number specifies the subdivisions of the shorter axis
        while the longer axis is subdivided such that the spacing of the grid
        points is approximately the same along both axes (i. e. the
        parallelogram is subdivided into approximate rhombi).
    startpoint : bool, optional
        If True, the parallelogram edges, where one coefficient of v1 or v2 is
        zero are sampled, otherwise they are omitted. Default is True.
    endpoint : bool, optional
        If True, the parallelogram edges, where one coefficient of v1 or v2 is
        one are sampled, otherwise they are omitted. Default is True.
    offset : float or iterable of floats, optional
        Offset of the grid in (v1, v2) basis. Default is (0, 0) such that the
        center of the sampled parallelogram lies at (0.5, 0.5). If it is given
        as float, it will be applied to each axis while a tuple represents the
        shift along the two axis. For a parallelogram centered at (0, 0), use
        `offset=-0.5`. Default is 0.
    gtype : str, optional
        Type of grid that can be "center" for a center grid in which the points
        are distributed equally along the two axes and the number of points
        along each axis is given by `nums`. In that case, the points do not
        sample other points than within the parallelogram spanned by v1 and v2.
        If "corner", the point for the center-type grid will be used as centers
        for the mesh such that the corner-type mesh contains sub-parallelograms
        which surround the vertices of the center-type mesh. In that case,
        points outsite of the parallelogram may be sampled and the number of
        points is increased by 1 for each axis compared to `nums` parameter.

    """
    assert gtype in ['center', 'corner']
    v1norm = norm(v1)
    v2norm = norm(v2)
    if isinstance(nums, int):
        # if nums is int, take it for smaller axis and compute nums for larger
        # axis such that step size is similar
        cen_nis = nintervals(nums, startpoint, endpoint, gtype='center')
        if v1norm < v2norm:
            num1 = nums
            nis2 = int(round(cen_nis * v2norm / v1norm))
            num2 = npoints(nis2, startpoint, endpoint, gtype='center')
        else:
            nis1 = int(round(cen_nis * v1norm / v2norm))
            num1 = npoints(nis1, startpoint, endpoint, gtype='center')
            num2 = nums
        nums = [num1, num2]
    nums = np.array(nums)
    try:
        offset[0]
    except TypeError:
        offset = [offset, offset]
    start1 = start2 = 0
    end1 = end2 = 1
    if gtype == 'corner':
        # number of intervals for center-type mesh
        cen_nis1 = nintervals(nums[0], startpoint, endpoint, gtype='center')
        cen_nis2 = nintervals(nums[1], startpoint, endpoint, gtype='center')
        # length of intervals (for center-type mesh) in (v1, v2) basis
        len1 = 1 / cen_nis1
        len2 = 1 / cen_nis2
        # shift whole grid to the "left" to surround center-type mesh
        offset[0] -= len1 / 2
        offset[1] -= len2 / 2
        # extend mesh by double corner-type interval length
        end1 += len1
        end2 += len2
        # add singular point for both axes for corner-type mesh
        nums += 1
    xs = linspace(start1, end1, nums[0], startpoint, endpoint) + offset[0]
    ys = linspace(start2, end2, nums[1], startpoint, endpoint) + offset[1]
    xms, yms = meshgrid(xs, ys)
    res = zeros((len(v1), nums[1], nums[0]))
    for i in range(nums[1]):
        for j in range(nums[0]):
            x, y = xms[i, j], yms[i, j]
            v = x * v1 + y * v2
            res[:, i, j] = v
    return res


def nintervals(npoints, startpoint, endpoint, gtype):
    """Return number of intervals for given number of points."""
    nintervals = npoints - 1
    if not startpoint:
        nintervals += 1
    if not endpoint:
        nintervals += 1
    if gtype == 'corner':
        nintervals += 1
    return nintervals


def npoints(nintervals, startpoint, endpoint, gtype):
    """Return number of points for given number of intervals."""
    npoints = nintervals + 1
    if not startpoint:
        npoints -= 1
    if not endpoint:
        npoints -= 1
    if gtype == 'corner':
        npoints -= 1
    return npoints


def linspace(start, stop, num=50, startpoint=True, endpoint=True,
             retstep=False, dtype=None, axis=0):
    """
    Return evenly spaced numbers over a specified interval.

    Parameters:
    -----------
    start: array_like
        The starting value of the sequence, unless `startpoint` is set to
        False. In that case, the sequence consists of all but the first of
        `num + 1` evenly spaced samples, so that `start` is excluded. Note that
        the step size changes when `startpoint` is False.
    stop: array_like
        The end value of the sequence, unless `endpoint` is set to False. In
        that case, the sequence consists of all but the last of `num + 1`
        evenly spaced samples, so that `start` is excluded. Note that the step
        size changes when `endpoint` is False.
    num: int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    startpoint: bool, optional
        If True, `start` is the first sample. Otherwise, it is not included.
        Default is True.
    endpoint: bool, optional
        If True, `end` is the first sample. Otherwise, it is not included.
        Default is True.
    retstep: bool, optional
        If True, return (samples, step), where step is the spacing between
        samples.
    dtype: dtype, optinal
        The type of the output array. if `dtype` is not given, the data type is
        inferred from `start` and `stop`. The inferred dtype will never be an
        integer; `float` is chosen even if the arguments would produce an array
        of integers.
    axis: int, optional
        The axis in the result to store the samples. Relevant only if start or
        stop are array-like. By default (0), the samples will be along a new
        axis inserted at the beginning. Use -1 to get an axis at the end.

    Returns:
    --------
    samples: ndarray
        There are `num` equally spaced samples in the closed interval
        [start, stop] or the open interval (start, stop) or the half-open
        intervals (start, stop] or [start, stop) (depeinding on whether
        `startpoint` and `endpoint` are True or False).
    step: float, optional
        Only returned if `retstep` if True. Size of spacing between samples.

    """
    if startpoint:
        sind = 0
    else:
        sind = 1
        num += 1
    res = np.linspace(start, stop, num, endpoint, retstep, dtype, axis)
    if retstep:
        samples, step = res
        samples = samples[sind:]
        return samples, step
    samples = res
    samples = samples[sind:]
    return samples


def area_quadrilateral(v1, v2, v3, v4):
    """Compute area of quadrilateral with vertices v1, v2, v3, and v4."""
    v12 = v2 - v1
    v14 = v4 - v1
    v23 = v3 - v2
    v43 = v3 - v4
    area1 = norm(np.cross(v12, v14)) / 2
    area2 = norm(np.cross(v23, v43)) / 2
    area = area1 + area2
    return area


def areagrid_quadrilateral(grid):
    """Compute 2D area grid of quadrilaterals."""
    ny, nx = grid.shape[1:]
    nx -= 1
    ny -= 1
    areagrid = np.zeros((ny, nx), dtype=float)
    for ix in range(nx):
        for iy in range(ny):
            v1 = grid[:, iy, ix]
            v2 = grid[:, iy, ix + 1]
            v3 = grid[:, iy + 1, ix + 1]
            v4 = grid[:, iy + 1, ix]
            area = area_quadrilateral(v1, v2, v3, v4)
            areagrid[iy, ix] = area
    return area
