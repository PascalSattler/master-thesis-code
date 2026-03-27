#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 09:35:43 2020

@author: robin
"""

import numpy as np
import matplotlib.pyplot as plt
from filemanaging import get_unique_fname
from vectutil import angle_between
from numpy import pi


class PseudoObject:
    pass


class OneDimDispersion:
    """Define set of auxiliary methods for plotting 1D bandstructures."""

    @staticmethod
    def get_pathlabels(*points):
        """Return list of strings with name of high-symmetry points."""
        return [point[0] for point in points]

    @staticmethod
    def get_pathcoords(*points):
        """Return list of tuples with coordinates of high-symmetry points."""
        return [point[1] for point in points]

    def __init__(self, spinw, system):
        self.spinw = spinw
        self.system = system

    def get_dispersion(self, *points, num=100, isometric=True, basis='rcpr'):
        """
        Calculate the energies along the path connecting all points.

        Parameters
        ----------
        *points : sequence of high-symmetry points (see class Cubic).
            The points to be connected by straight lines along which the
            energies are calculated.
        num : int, optional
            The number of points sampled on the path. On each sampled point
            the energies are obtained.
        isometric : bool, optional
            If True, the subpaths connecting two subsequent points are equally
            long irrespective of the real distance. If False, the subpaths
            scale as the interpoint distances.

        Returns
        -------
        dispersion : dict
            ts : list of float
                The one-dimensional parametrization of the path given as
                floating point numbers between 0 and 1 (including both ends).
            ks : list of tuples
                The list that contains the three-dimensional k points that were
                sampled. The points are given in order.
            point_maps : list of float
                Specifies the location of the given high-symmetry points in
                parameter space, i. e. as numbers between 0 and 1.
            energies : list of lists
                The magnon energies at the sampled k points. The energies
                corresponding to one k point are arranged in the inner lists
                in descending order and the outer list has the same order as
                'ts' and 'ks'.
            pathlabels : list of str
                Names of the high-symmetry points.
            pathcoords : list of tuples
                Sampled points in k space.
        """
        pathcoords = self.get_pathcoords(*points)
        pathlabels = self.get_pathlabels(*points)
        ts, ks, point_maps, energies = self.spinw.energy_path(
            *pathcoords, num=num, isometric=isometric, basis=basis)
        dispersion = dict(
            points=points,
            pathcoords=pathcoords,
            pathlabels=pathlabels,
            ts=ts,
            ks=ks,
            energies=energies,
            point_maps=point_maps)
        return dispersion

    def save_dispersion(self, dispersion, fname=None, overwrite=False):
        """
        Save the dispersion relation.

        Parameters
        ----------
        dispersion : dict
            See documentation of 'get_dispersion'.
        fname : str
            Name of the file.
        overwrite : bool, optional
            If True, a possibly existing previous save file is overwritten
            else a new name is generated.
        """
        if fname is None:
            fname = f'./results/kagome/disp_{self.system}.dat'
        energies = dispersion['energies']
        wshape = (energies.shape[0], energies.shape[1] + 1)
        write = np.zeros(wshape)
        write[:, 0] = dispersion['ts']
        write[:, 1:] = energies
        if not overwrite:
            fname = get_unique_fname(fname)
        np.savetxt(fname, write, fmt='%f')

    @classmethod
    def plot_dispersion(cls, dispersion, bands=None, pltopts=dict()):
        """
        Return pyplot.figure on which the energies are plotted.

        Parameters
        ----------
        dispersion : dict
            See documentation of 'get_dispersion'.
        bands : list, optional
            Indices of the bands to be plotted. If `None`, all bands are
            plotted. Default is `None`.
        pltopts : dict, optional
            Optional keyword arguments passed to the plot calls along with the
            magnon band data.
        """
        point_maps, pathlabels, energies, ts = [
            dispersion[x] for x in ['point_maps', 'pathlabels', 'energies',
                                    'ts']]
        fig = plt.gcf()
        plt.ylabel(r'Energy $\varepsilon(\mathbf{k})$ (a. u.)')
        plt.xlabel(r'Wave vector $\mathbf{k}$')
        plt.xticks(ticks=point_maps, labels=pathlabels)
        plt.grid()
        if bands is None:
            bands = range(energies.shape[-1])
        for band in bands:
            plt.plot(ts, energies[:, band], **pltopts)
        plt.tight_layout()
        return fig

    def save_dispersion_plot(self, fig, fname=None, overwrite=False):
        """
        Save pyplot.figure on which the energies are plotted.

        Parameters
        ----------
        axis : mathplotlib.pyplot.figure
            The plot that shall be saved.
        fname : str, optional
            File name of the plot. If None, a default name is chosen. Default
            is None.
        overwrite : bool, optional
            If True, an existing file with the given or default file name will
            be overwritten else a unique name is generated.
        """
        if fname is None:
            fname = f'./results/kagome/images/disp_{self.system}.pdf'
        if not overwrite:
            fname = get_unique_fname(fname)
        fig.savefig(fname)


class Generic(OneDimDispersion):
    """Describe an arbitray symmetry class."""
    hsp = PseudoObject()
    hsp.g = [r'$\Gamma$', (0, 0, 0)]
    hsp.a = [r'A', (1/2, 0, 0)]
    hsp.ap = [r'A$^\prime$', (-1/2, 0, 0)]
    hsp.b = [r'B', (0, 1/2, 0)]
    hsp.bp = [r'B$^\prime$', (0, -1/2, 0)]
    hsp.c = [r'C', (0, 0, 1/2)]
    hsp.cp = [r'C$^\prime$', (0, 0, -1/2)]
    hsp.l = [r'L', (1/2, 1/2, 0)]


class Cubic(OneDimDispersion):
    """Feature high-symmetry points for cubic (reciprocal) lattices."""

    def __init__(self, spinw, system):
        hsp = PseudoObject()
        # high-symmetry points
        hsp.g = [r'$\Gamma$', (0, 0, 0)]
        hsp.m = [r'M', (1/2, 1/2, 0)]
        hsp.r = [r'R', (1/2, 1/2, 1/2)]
        hsp.x = [r'X', (1/2, 0, 0)]
        self.hsp = hsp
        super().__init__(spinw, system)


class Hexagonal(OneDimDispersion):
    """Feature high-symmetry points for hexagonal (reciprocal) lattices."""

    def __init__(self, spinw, system):
        b1, b2, b3 = spinw.lat.num_rcpr_vects
        a12 = angle_between(b1, b2)
        eps = 1e-3
        hsp = PseudoObject()
        # high-symmetry points
        hsp.g = [r'$\Gamma$', (0, 0, 0)]
        hsp.m = [r'M', (1/2, 0, 0)]
        if abs(a12 - pi/3) < eps:
            hsp.k = [r'K', (1/3, 1/3, 0)]
            hsp.kp = [r'K$^\prime$', (-1/3, -1/3, 0)]
        elif abs(a12 - 2*pi/3) < eps:
            hsp.k = [r'K', (1/3, -1/3, 0)]
            hsp.kp = [r'K$^\prime$', (-1/3, 1/3, 0)]
        else:
            raise ValueError(f'angle between first two reciprocal vector '
                             f'needs to be 60° or 120° not {a12/pi*180}°')
        hsp.a = [r'A', (0, 0, 1/2)]
        hsp.l = [r'L', (1/2, 0, 1/2)]
        hsp.h = [r'H', (1/3, 1/3, 1/2)]
        self.hsp =hsp
        super().__init__(spinw, system)


class Orthorhombic(OneDimDispersion):
    """Feature high-symmetry points for cubic (reciprocal) lattices."""

    def __init__(self, spinw, system):
        hsp = PseudoObject()
        # high-symmetry points
        hsp.g = [r'$\Gamma$', (0, 0, 0)]
        hsp.x = [r'X', (1/2, 0, 0)]
        hsp.y = [r'Y', (0, 1/2, 0)]
        hsp.z = [r'Z', (0, 0, 1/2)]
        hsp.s = [r'S', (1/2, 1/2, 0)]
        hsp.u = [r'U', (1/2, 0, 1/2)]
        hsp.t = [r'T', (0, 1/2, 1/2)]
        hsp.r = [r'R', (1/2, 1/2, 1/2)]
        hsp.xp = [r'X$^\prime$', (-1/2, 0, 0)]
        hsp.yp = [r'Y$^\prime$', (0, -1/2, 0)]
        hsp.zp = [r'Z$^\prime$', (0, 0, -1/2)]
        hsp.sp = [r'S$^\prime$', (-1/2, -1/2, 0)]
        hsp.up = [r'U$^\prime$', (-1/2, 0, -1/2)]
        hsp.tp = [r'T$^\prime$', (0, -1/2, -1/2)]
        hsp.rp = [r'R$^\prime$', (-1/2, -1/2, -1/2)]
        self.hsp = hsp
        super().__init__(spinw, system)

class Oblique(OneDimDispersion):
    """Feature high-symmetry points for oblique (reciprocal) lattices."""

    def __init__(self, spinw, system):
        hsp = PseudoObject()
        # high-symmetry points
        hsp.g = [r'$\Gamma$', (0, 0, 0)]
        hsp.y = [r'Y', (1/2, 0, 0)]
        hsp.h = [r'H', (0.53589838, -0.26794919, 0)]
        hsp.c = [r'C', (1/2, -1/2, 0)]
        hsp.h1 = [r'H1', (0.46410162, -0.73205081, 0)]
        hsp.x = [r'X', (0, -1/2, 0)]
        
        self.hsp = hsp
        super().__init__(spinw, system)