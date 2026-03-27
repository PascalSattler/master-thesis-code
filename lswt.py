#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:34:23 2018

@author: robinneumann
"""

import sympy as sp
import numpy as np
from numpy import array
from numpy.linalg import cholesky, eigh, eig, inv, norm, LinAlgError
from math import floor
from itertools import product
from random import random
from scipy.integrate import nquad
from scipy.optimize import minimize
from scipy.special import spence
import copy
from sortedcontainers import SortedList
# import plotly.graph_objects as go
from mesh import parallelogram_grid, linspace, area_quadrilateral
from mesh import areagrid_quadrilateral


class MagneticLattice:
    """
    Represent the magnetic lattice and contains all sublattices.

    Informations like translational vectors, reciprocal lattice vectors the
    basis is stored in the instances of this class.

    """

    def __init__(self, vects, sublats, num_parameters=None):
        self.vects = vects.copy()
        self.sublats = sublats.copy()
        self.sublats_count = len(sublats)
        self.nsublats = self.sublats_count
        self.update_rcpr_vects()
        self.num_vects = None
        self.num_rcpr_vects = None
        self.num_parameters = num_parameters
        if num_parameters is not None:
            self.parameterize()

    def copy(self):
        """Return a copy of the current instance."""
        obj_copy = copy.deepcopy(self)
        return obj_copy

    def transf_to_cart(self, vect):
        """Transform vector from lattice vector to cartesian basis."""
        res = self.vects.T * sp.Matrix(vect)
        return res

    def transf_to_num_cart(self, vect):
        """Transform vector from lattice vector to cartesian basis."""
        res = self.num_vects.T.dot(vect)
        return res

    def transf_to_latt(self, vect):
        """Transform vector from cartesian to lattice vector basis."""
        res = self.rcpr_vects * sp.Matrix(vect) / (2 * sp.pi)
        return res

    def transf_to_num_latt(self, vect):
        """Transform numeric vector from cartesian to lattice vector basis."""
        res = self.num_rcpr_vects.dot(np.array(vect)) / (2 * np.pi)
        return res

    def transf_to_num_rcpr_cart(self, vect):
        """Transform vector from reciprocal lattice to cartesian basis."""
        res = self.num_rcpr_vects.T.dot(vect)
        return res

    def transf_to_num_rcpr_latt(self, vect):
        """Transform vector from reciprocal cartesian to lattice basis."""
        res = self.num_vects.dot(vect) / (2 * np.pi)
        return res

    def downfold_num_rcpr_latt(self, vect):
        """Map reciprocal lattice vector to first Brillouin zone."""
        return self.ws_downfold_num_rcpr_latt(vect)

    def downfold_num_rcpr_cart(self, vect):
        """Map reciprocal cartesian vector to first Brillouin zone."""
        return self.ws_downfold_num_rcpr_cart(vect)

    def ws_downfold_num_rcpr_latt(self, vect):
        """Map reciprocal lattice vector to first Brillouin zone."""
        remainder = np.array(vect) % 1
        shifts = np.array(list(product([0, 1], repeat=3)))
        latt_points = remainder - shifts
        transf = self.transf_to_num_rcpr_cart
        cart_points = np.array([transf(v) for v in latt_points])
        norms = norm(cart_points, axis=1)
        min_ind = norms.argmin()
        res = latt_points[min_ind]
        return res

    def ws_downfold_num_rcpr_cart(self, vect):
        """Map reciprocal cartesian vector to first Brillouin zone."""
        lattvect = self.transf_to_num_rcpr_latt(vect)
        foldlattvect = self.downfold_num_rcpr_latt(lattvect)
        foldcartvect = self.transf_to_num_rcpr_cart(foldlattvect)
        return foldcartvect

    def pe_downfold_num_rcpr_latt(self, vect):
        """Map reciprocal lattice vector to origin-centered parallelepiped."""
        res = (vect + 0.5) % 1 - 0.5
        return res

    def pe_downfold_num_rcpr_cart(self, vect):
        """Map reciprocal cart. vector to origin-centered parallelepiped."""
        lattvect = self.transf_to_num_rcpr_latt(vect)
        foldlattvect = self.pe_downfold_num_rcpr_latt(lattvect)
        foldcartvect = self.transf_to_num_rcpr_cart(foldlattvect)
        return foldcartvect

    def parameterize(self):
        """Create numerical copies of all symbolic attributes."""
        if self.num_parameters is None:
            raise ValueError('Parameters have not been specified.')
        self.num_vects = np.array(
            self.vects.subs(self.num_parameters)).astype(np.float64)
        self.num_rcpr_vects = np.array(
            self.rcpr_vects.subs(self.num_parameters)).astype(np.float64)

    def update_rcpr_vects(self):
        """Generate new reciprocal vectors based on real lattice vectors."""
        r1 = self.vects[0, :]
        r2 = self.vects[1, :]
        r3 = self.vects[2, :]
        vol = r1.dot(r2.cross(r3))
        g1 = 2 * sp.pi / vol * r2.cross(r3)
        g2 = 2 * sp.pi / vol * r3.cross(r1)
        g3 = 2 * sp.pi / vol * r1.cross(r2)
        self.rcpr_vects = sp.Matrix([g1, g2, g3])

    def update_num_parameters(self, newparams):
        """Replace numerical parameters and update all numerical attributes."""
        if self.num_parameters is None:
            self.num_parameters = newparams
        else:
            self.num_parameters.update(newparams)
        self.parameterize()
        for sublat in self.sublats:
            sublat.update_num_parameters(newparams)

    @classmethod
    def from_str(cls, rowstrs, num_parameters=None):
        """Instantiate by formatted input file."""
        sublats_count = len(rowstrs) - 3
        # tranlation vectors
        # ------------------
        vects = []
        for rowstr in rowstrs[:3]:
            # structure of a row:
            # [a*sin(30), cos(30), 0]
            vects.append(sp.sympify(rowstr))
        vects = sp.Matrix(vects)
        # sublattice parameters
        # ---------------------
        sublats = [None for i in range(sublats_count)]
        for rowstr in rowstrs[3:]:
            # structure of a row:
            # sublattice | basisvect | spinlen | gyro | gsdir xor unitvectors
            # 1 | [1/2, 0, 0] | S | [[1, 2, 3], ...] | [1, 1, 1] xor [[..], ..]

            # extract the number of sublattice (has to be an int)
            cells = rowstr.split('|')
            num = int(cells[0]) - 1
            if sublats[num] is not None:  # check if sublattice exists already
                raise ValueError('Sublattice has been defined more than once.')
            # extract the basisvect (potentially symbolic)
            basisvect = sp.Matrix(sp.sympify(cells[1]))
            if basisvect.shape != (3, 1):
                raise ValueError('Too many or too few components given for the'
                                 ' basisvector.')
            basisvect = vects.T * basisvect
            # extract the spinlen
            spinlen = sp.sympify(cells[2])
            # extract the gyromagnetic matrix
            gyromatr = sp.Matrix(sp.sympify(cells[3]))
            if gyromatr.shape != (3, 3):
                raise ValueError('Gyromagnetic matrix has the wrong shape.')
            # groundstate direction xor optional: local coordinate system
            unitvects = sp.Matrix(sp.sympify(cells[4]))
            if unitvects.shape != (3, 3):
                if unitvects.shape != (3, 1):
                    raise ValueError('Ground state direction xor local '
                                     'coordinate system were not given in the'
                                     ' right form.')
                grdstatedir = unitvects
                unitvects = None

            # print(f"basisvect = {basisvect}, spinlen = {spinlen}, grdstatedir = {grdstatedir}")

            # create sublattice instance
            if not unitvects: # NOTE if this variable is None (when ground state is given as a unit vec) this statement is true -> code executes
                sublattice = MagneticSublattice.from_grdstatedir(
                    basisvect, spinlen, gyromatr, grdstatedir, num_parameters)  # NOTE here a local coord sys is created form the unit vector given as the ground state first and the the object is instantiated
            else:
                sublattice = MagneticSublattice(basisvect, spinlen, gyromatr,
                                                unitvects, num_parameters) # NOTE when the ground state is already given as a tripod it can directly be instantiated.
            sublats[num] = sublattice

        if None in sublats:
            raise ValueError('Less sublattices specified than expected.')
        return cls(vects, sublats, num_parameters)

    def change_grdstatedirs(self, newdirs):
        """Replace old ground state spin directions with new ones."""
        oldsublats = self.sublats
        newsublats = []
        for i, oldsublat in enumerate(oldsublats):
            newsublats.append(
                MagneticSublattice.from_grdstatedir(
                    oldsublat.basisvect, oldsublat.spinlen, oldsublat.gyromatr,
                    newdirs[i], num_parameters=self.num_parameters))
        self.sublats = newsublats

    def get_grdstatedirs(self):
        """Assemble ground state directions of all sublattices."""
        dirs = []
        for sublat in self.sublats:
            dirs.append(sublat.unitvects.row(-1).T)
        return dirs

    def get_num_grdstatedirs(self):
        """Assemble numerical ground state directions of all sublattices."""
        dirs = []
        for sublat in self.sublats:
            dirs.append(sublat.num_unitvects[-1])
        return dirs

    def get_shell(self, shellcount, ucs=None, dim=3, eps=1e-6):
        """
        Identify the next neighbours based on lattice and basis vectors.

        The neighbours are organized into shell where only the n-th next
        neighbours are included for each atom in the basis. The value of n
        is bounded by the parameter 'shellcount'. All shells up to this
        boundary will be considered.

        Parameters
        ----------
        shellcount : int
            The number of shells that should be identified. It bounds the
            variable n from above. Must be positiv.
        ucs : list, optional
            A list of the coordinates (in units of lattice vectors) that
            restrict the volume in which the shells are located. If None, it
            will assume a 'cube' of length 2*shellcount (+1), the dimension of
            which is given by `dim`. It is assumed that the coordinates for the
            actual primitive translation vectors are the first ones.
            Default is None.
        dim : int, optional
            Dimension of the lattice. Only relevant if `ucs` is None.
            Default is 3.
        eps : float, optional
            The maximal distance for which atoms are considered to be within
            the same shell. The default is 1e-6.

        Returns
        -------
        dists : 2D list
            The distances of a basis atom (first index) to the shells up to
            shellcount (second index: shell).
        unitcells : 3D list
            The unit cells of the atoms (third index) within a shell (second
            index) for all basis atoms (first index) in the unit cell
            [0, 0, 0] in units of lattice vectors.
        basisnums : 3D list
            The basis or sublattice numbers of the atoms (third index) in a
            shell (second index) for all basis atoms (first index) in the same
            order as unitcells.

        Note
        ----
        The n-th next neighbours can be identified by their unit cell u and
        sublattice-specifying number i. For the basis atom j in the unit cell
        [0, 0, 0] these neighbours are given by u in unitcells[j][n][:] and
        i in basisnums[j][n][:].

        """
        # initialize set of neighbour unitcells
        if ucs is None:
            prdarg = dim * [range(-shellcount, shellcount + 1)]
            prdarg += (3 - dim) * [[0]]
            ucs = list(product(*prdarg))

        nsublats = self.nsublats
        vects = self.num_vects
        dists = [SortedList() for i in range(nsublats)]
        unitcells = [[] for i in range(nsublats)]
        basisnums = [[] for i in range(nsublats)]

        for i in range(nsublats):
            cenpos = self.sublats[i].num_basisvect
            for uc in ucs:
                ucpos = np.array(uc).dot(vects)
                for j in range(nsublats):
                    # skip same spin where the distant is always 0
                    if i == j and norm(ucpos) < eps:
                        continue
                    othpos = ucpos + self.sublats[j].num_basisvect
                    dist = norm(othpos - cenpos)
                    # check if distance exists already
                    odist = dists[i].irange(max(0, dist - eps), dist + eps)
                    lodist = list(odist)
                    if lodist == []:
                        dists[i].add(dist)
                        index = dists[i].index(dist)
                        unitcells[i].insert(index, [uc])
                        basisnums[i].insert(index, [j])
                    else:
                        index = dists[i].index(lodist[0])
                        unitcells[i][index].append(uc)
                        basisnums[i][index].append(j)

        dists = [dists[i][:shellcount] for i in range(nsublats)]
        basisnums = [basisnums[i][:shellcount] for i in range(nsublats)]
        unitcells = [unitcells[i][:shellcount] for i in range(nsublats)]
        return dists, unitcells, basisnums

    def get_nns(self, ucs=None, dim=3):
        """
        Identify nearest neighbors for each site in first unit cell.

        Parameters
        ----------
        ucs : list, optional
            A list of the coordinates (in units of lattice vectors) that
            restrict the volume in which the shells are located. If None, it
            will assume a 'cube' of length 2*shellcount (+1), the dimension of
            which is given by `dim`. It is assumed that the coordinates for the
            actual primitive translation vectors are the first ones.
            Default is None.
        dim : int, optional
            Dimension of the lattice. Only relevant if `ucs` is None.
            Default is 3.

        Returns
        -------
        nns : list of list of list of int and list
            contains for each sublattice (first dimension) a list of nearest
            neighbors (second dimension) that are specified by their sublattice
            index (third dimension; index 0) and a list of 3 integers
            specifying their unit cells in the basis of the primitive
            translation vectors (third dimension; index 0).

        """
        dists, ucs, isls = self.get_shell(shellcount=1, ucs=ucs, dim=dim)
        nns = [
            [
                [isls[i][0][j], ucs[i][0][j]] for j in range(len(ucs[i][0]))
            ]
            for i in range(len(ucs))]
        return nns

    def draw_lattice(self, ucells, offset=None):
        """Draw a 3D representation of the lattice for the given unit cells."""
        if offset is None:
            offset = np.array([0, 0, 0])
        xs = []
        ys = []
        zs = []
        us = []
        vs = []
        ws = []
        for ucell in ucells:
            ucpos = self.transf_to_num_cart(ucell)
            for sl in self.sublats:
                slpos = sl.num_basisvect
                pos = ucpos + slpos + offset
                spin = sl.num_unitvects[-1]
                xs.append(pos[0])
                ys.append(pos[1])
                zs.append(pos[2])
                us.append(spin[0])
                vs.append(spin[1])
                ws.append(spin[2])
        cone = go.Cone(x=xs, y=ys, z=zs, u=us, v=vs, w=ws)
        return cone

    def to_suplat_nns(self, nns):
        """
        Compute nearest neighbors for superlattice.

        Parameters:
        -----------
        nns : list of lists of lists
            Nearest neighbors of all atoms in the primitive unit cell. First
            index represents the atom in the unit cell, second index runs over
            the nearest neighbors each of which is represented by (i) the
            sublattice number (integer) and (ii) its unit cell given in the
            basis of the translation vectors (list of 3 integers).

        Return:
        -------
        super_nns : list of lists of lists
            Same as `nns` for the larger unit cell.

        """
        raise NotImplementedError('this code is currently not working')
        if not hasattr(self, 'newsublats') or not hasattr(self, 'newvects'):
            raise AttributeError('lattice has not been generated as '
                                 'superlattice; primitive cell unkown')
        newvects, newsublats = self.newvects, self.newsublats
        super_nns = []
        for sublat, uc in zip(newsublats, newvects):
            csuper_nns = []
            cnns = nns[sublat]
            for othersublat, otheruc in nns:
                new_othercoords = self.to_suplat_coords(othersublat, otheruc)
                cnns.append(new_othercoords)
            super_nns.append(csuper_nns)

    def to_suplat_coords(self, old_sublat, old_uc):
        """Convert sublattice and unit cell to superlattice coordinates."""
        raise NotImplementedError('this code is currently not working')
        if not hasattr(self, 'newsublats') or not hasattr(self, 'newvects'):
            raise AttributeError('lattice has not been generated as '
                                 'superlattice; primitive cell unkown')
        newvects, newsublats = self.newvects, self.newsublats
        for sl, uc in newsublats:
            bv = self.sublats[sl].basisvect

    def slab_lattice(self, dim, num):
        """Instantiate `MagneticLattice` for slab."""
        sublats = self.slab_sublattices(dim, num)
        old_vects = self.vects
        num_parameters = self.num_parameters.copy()
        lat = MagneticSlabLattice(old_vects, sublats, dim, num, num_parameters)
        return lat

    def slab_sublattices(self, dim, num):
        """Instantiate `MagneticLattice` for slab."""
        num_parameters = self.num_parameters.copy()
        vects = self.vects
        sublats = []
        for old_uc in range(num):
            for old_sindex, old_sublat in enumerate(self.sublats):
                old_basisvect = old_sublat.basisvect
                basisvect = old_basisvect + old_uc*vects.row(dim).T
                spinlen = old_sublat.spinlen
                gyromatr = old_sublat.gyromatr
                unitvects = old_sublat.unitvects
                old_uc3d = np.array([old_uc if i == dim else 0
                                     for i in range(3)])
                sublat = MagneticSlabSublattice(basisvect, spinlen, gyromatr,
                                                unitvects, old_uc3d,
                                                old_sindex, num_parameters)
                sublats.append(sublat)
        return sublats


class MagneticSlabLattice(MagneticLattice):
    """Track the topology of the infinite system."""

    def __init__(self, old_vects, sublats, dim, num, num_parameters=None):
        self.old_vects = old_vects
        self.dim = dim
        self.num = num
        vects = old_vects.copy()
        i1, i2 = (dim + 1) % 3, (dim + 2) % 3
        v1, v2 = vects[i1, :], vects[i2, :]
        v3 = v1.cross(v2)
        if i1 < i2:
            vects[0, :] = v1
            vects[1, :] = v2
            vects[2, :] = v3
            self.indexmap_to_slab = [-1, -1, -1]
            self.indexmap_to_slab[i1] = 0
            self.indexmap_to_slab[i2] = 1
            self.indexmap_to_slab[dim] = 2
        else:
            vects[0, :] = v2
            vects[1, :] = v1
            vects[2, :] = -v3
            self.indexmap_to_slab = [-1, -1, -1]
            self.indexmap_to_slab[i1] = 1
            self.indexmap_to_slab[i2] = 0
            self.indexmap_to_slab[dim] = 2
        # placement of obsolte translation vector linked to conversion methods
        # vects[dim, :] = vects[dim, :].norm() * v3/v3.norm()
        super().__init__(vects, sublats, num_parameters)

    def to_slab_coords(self, old_uc, old_sindex):
        """Transform position in extended system to slab coordinates."""
        if not self.is_in_slab(old_uc):
            raise ValueError('given unit cell is outside of slab')
        # rearrange unit cell coordinates and set finite direction to zero
        im = self.indexmap_to_slab
        uc = np.array([-1, -1, -1])
        for i in range(3):
            uc[im[i]] = old_uc[i]
        uc[-1] = 0
        # iterate over all slab sublattices until a matching
        old_uccomp = old_uc[self.dim]
        for i, sublat in enumerate(self.sublats):
            cold_uccomp = sublat.old_uc[self.dim]
            if cold_uccomp == old_uccomp and sublat.old_sindex == old_sindex:
                return uc, i
        raise RuntimeError(f'site ({old_uc}, {old_sindex}) not found')

    def to_extended_coords(self, uc, sindex):
        """Transform position in slab to extended system's coordinates."""
        sublat = self.sublats[sindex]
        imap = self.indexmap_to_slab
        old_uc = np.array([uc[imap[i]] + j for i, j in
                           enumerate(sublat.old_uc)])
        old_sindex = sublat.old_sindex
        return old_uc, old_sindex

    def is_in_slab(self, old_uc):
        """Check whether specified unit cell is within the slab."""
        res = 0 <= old_uc[self.dim] < self.num
        return res

    def to_slab_nns(self, nns):
        """Compute next neighbors (NNs) in slab system based on bulk NNs."""
        sublats = self.sublats
        slab_nns = []
        for sublat in sublats:
            old_sind = sublat.old_sindex
            old_uc = sublat.old_uc
            slab_nns.append([])
            for nn in nns[old_sind]:
                old_other_sind, old_other_uc = nn
                old_other_uc = np.array(old_other_uc) + np.array(old_uc)
                if not self.is_in_slab(old_other_uc):
                    continue
                new_nn = self.to_slab_coords(old_other_uc, old_other_sind)
                new_other_uc, new_other_sind = new_nn
                new_other_uc = [int(comp) for comp in new_other_uc]
                slab_nns[-1].append([new_other_sind, new_other_uc])
        return slab_nns


class MagneticSublattice:
    """Represents a sublattice and is contained within a MagneticLattice."""

    def __init__(self, basisvect, spinlen, gyromatr, unitvects,
                 num_parameters=None):
        self.basisvect = basisvect.copy()
        self.spinlen = spinlen
        self.gyromatr = gyromatr.copy()
        self.unitvects = unitvects.copy()
        self.u = unitvects.row(0).T + sp.I * unitvects.row(1).T
        self.v = unitvects.row(2).T
        self.num_parameters = num_parameters
        self.num_gyromatr = None
        self.num_unitvects = None
        self.num_spinlen = None
        self.num_basisvect = None
        self.num_u = None
        self.num_v = None
        if num_parameters is not None:
            self.parameterize()

    def parameterize(self):
        """Create numerical copy of symbolic attributes."""
        num_parameters = self.num_parameters
        if num_parameters is None:
            raise ValueError('parameters have not been specified')
        self.num_gyromatr = np.array(
            self.gyromatr.subs(num_parameters)).astype(np.float64)
        self.num_unitvects = np.array(
            self.unitvects.subs(num_parameters)).astype(np.float64)
        self.num_spinlen = np.array(
            self.spinlen.subs(num_parameters)).astype(np.float64)
        self.num_basisvect = np.array(
            self.basisvect.T.subs(num_parameters)).astype(np.float64)[0]
        self.num_u = np.array(
            self.u.T.subs(num_parameters)).astype(np.complex128)[0]
        self.num_v = np.array(
            self.v.T.subs(num_parameters)).astype(np.complex128)[0]

    def update_num_parameters(self, newparams):
        """Replace old parameters and update numerical attributes."""
        if self.num_parameters is None:
            self.num_parameters = newparams
        else:
            self.num_parameters.update(newparams)
        self.parameterize()

    @classmethod
    def from_grdstatedir(cls, basisvect, spinlen, gyromatr, grdstatedir,
                         num_parameters=None):
        """Instantiate sublattice and local axes by ground state direction."""
        grdstatedir = sp.Matrix(grdstatedir)
        v_norm = grdstatedir.norm()

        if v_norm != 1:  # v_norm =/= 0
            grdstatedir /= v_norm
        if grdstatedir[0] != 0:
            xdir = sp.Matrix([grdstatedir[1], -grdstatedir[0], 0])
        elif grdstatedir[1] != 0:
            xdir = sp.Matrix([0, grdstatedir[2], -grdstatedir[1]])
        else:  # grdstatedir[2] =/= 0
            xdir = sp.Matrix([grdstatedir[2], 0, -grdstatedir[0]])
        ydir = grdstatedir.cross(xdir)
        xdir = xdir / xdir.norm()
        ydir = ydir / ydir.norm()
        unitvects = []
        unitvects.append(xdir.T)
        unitvects.append(ydir.T)
        unitvects.append(grdstatedir.T)
        unitvects = sp.Matrix(unitvects)
        # print(unitvects)
        return cls(basisvect, spinlen, gyromatr, unitvects, num_parameters)


class MagneticSlabSublattice(MagneticSublattice):
    """Keep track of the topology with respect to the infinte system."""

    def __init__(self, basisvect, spinlen, gyromatr, unitvects, old_uc,
                 old_sindex, num_parameters=None):
        self.old_uc = old_uc
        self.old_sindex = old_sindex
        super().__init__(basisvect, spinlen, gyromatr, unitvects,
                         num_parameters)


class MainBasis:
    """Store the interactions between atoms."""

    def __init__(self, lat):
        self.lat = lat
        atoms = [0 for i in range(lat.sublats_count)]
        for i in range(lat.sublats_count):
            atom = BasisAtom(lat, i)
            atoms[i] = atom
        self.atoms = atoms

    def copy(self):
        """Return a copy of the current instance."""
        obj_copy = copy.deepcopy(self)
        return obj_copy

    def add_interactions_from_str(self, rowstrs, symmetrize=True):
        """
        Add interaction from formatted string.

        Parameters
        ----------
        rowstrs : list of strings
            The rows containing the necessary information to describe the
            bilinear spin-spin interaction. It is assumed to have the shape
            'main basis|other basis|difference lattice vector|interaction'.
        symmetrize : bool, optional
            If true, for each specified interaction between a ordered pair of
            atoms (i, j) a second equivalent between the reverse-ordered pair
            (j, i) is added. The default is True.

        """
        for rowstr in rowstrs:
            # general structure of a row
            # main basis|other basis|difference lattice vector|interaction
            # 1         | 2         | [0, 0, 0]               | [[...], ...]
            cells = rowstr.split('|')
            sublat = int(cells[0]) - 1
            othersublat = int(cells[1]) - 1
            othervect = self.lat.transf_to_cart(sp.sympify(cells[2]))
            intermatr = sp.Matrix(sp.sympify(cells[3]))

            self.atoms[sublat].add_interaction(othersublat, othervect,
                                               intermatr)
            if symmetrize:
                self.atoms[othersublat].add_interaction(sublat, -othervect,
                                                        intermatr.T)

    def parameterize(self):
        """Replace old parameters and update numerical attributes."""
        for a in self.atoms:
            a.parameterize()

    def slab_basis(self, lat, dim, num):
        """Instantiate `MainBasis` of the slab."""
        basis = MainBasis(lat)
        old_basis = self
        for atom in basis.atoms:
            old_uc, old_sublat = lat.to_extended_coords((0, 0, 0), atom.sublat)
            old_inters = [el
                          for l in old_basis.atoms[old_sublat].interactions
                          for el in l]
            for old_inter in old_inters:
                old_othersublat = old_inter.othersublat
                old_othervect = old_inter.othervect
                old_duc = array(self.lat.transf_to_latt(old_othervect).T)[0]
                old_otheruc = old_duc + old_uc
                intermatr = old_inter.intermatr
                try:
                    otheruc, othersublat = lat.to_slab_coords(old_otheruc,
                                                              old_othersublat)
                except ValueError:
                    # interaction partner is outside the slab
                    continue
                othervect = lat.transf_to_cart(otheruc)
                # othervect = old_othervect
                atom.add_interaction(othersublat, othervect, intermatr)
        return basis


class BasisAtom:
    """Stores the interaction of one atom with the others."""

    def __init__(self, lat, sublat):
        self.sublat = sublat
        self.lat = lat
        sublat_count = lat.sublats_count
        self.interactions = [[] for i in range(sublat_count)]

    def add_interaction(self, othersublat, othervect, intermatr):
        """Add bilinear interaction with one atom."""
        interaction = Interaction(self.lat, self.sublat, othersublat,
                                  othervect, intermatr)
        self.interactions[othersublat].append(interaction)

    def sum_phased_intermatr(self, k, othersublat, minus=False):
        """
        Return Fourier-transformed interaction matrix.

        The symbolic expression reads sum_r e^{i k r} J(r). r only iterates
        over the atom position belongig on the sublattice given by
        'othersublat'.

        """
        res = sp.zeros(3, 3)
        for i in self.interactions[othersublat]:
            res += i.phased_intermatr(k, minus)
        return res

    def sum_num_phased_intermatr(self, k, othersublat, minus=False):
        """
        Return Fourier-transformed interaction matrix.

        The symbolic expression reads sum_r e^{i k r} J(r). r only iterates
        over the atom position belongig on the sublattice given by
        'othersublat'.

        """
        res = sp.zeros(3, 3)                        # NOTE should be np.zeros ?
        for i in self.interactions[othersublat]:
            res += i.num_phased_intermatr(k, minus)
        return res

    def sum_intermatr(self, othersublat):
        """
        Return interaction matrix sum with atoms of specific sublattice.

        This is identical with the expression given in the docstring of
        'sum_phased_intermatr' if k is set to zero.

        """
        return self.sum_phased_intermatr(sp.Matrix([0, 0, 0]), othersublat)

    def sum_num_intermatr(self, othersublat):
        """Do the same as 'sum_intermatr' with the numerical matrices."""
        res = np.zeros((3, 3), dtype='complex')
        for i in self.interactions[othersublat]:
            res += i.num_intermatr
        return res

    def parameterize(self):
        """Replace old parameters and update numerical attributes."""
        for ilist in self.interactions:
            for i in ilist:
                i.parameterize()


class Interaction:
    """Contains all informations of a bilinear spin-spin interaction."""

    def __init__(self, lat, sublat, othersublat, othervect, intermatr):
        self.lat = lat
        self.sublat = sublat
        self.othervect = othervect.copy()
        self.othersublat = othersublat
        self.intermatr = intermatr.copy()
        self.num_intermatr = None
        self.num_othervect = None
        if self.lat.num_parameters is not None:
            self.parameterize()

    def parameterize(self):
        """Create numerical copies of interaction matrices."""
        num_parameters = self.lat.num_parameters
        if num_parameters is None:
            raise ValueError('parameters have not been specified')
        self.num_intermatr = np.array(
            self.intermatr.subs(num_parameters)).astype(np.float64)
        self.num_othervect = np.array(
            self.othervect.T.subs(num_parameters)).astype(np.float64)[0]
        # self.num_intermatr = sp.lambdify(
        #     '', self.intermatr.subs(num_parameters), 'numpy')()
        # self.num_othervect = sp.lambdify(
        #     '', self.othervect.T.subs(num_parameters), 'numpy')()[0]

    def phased_intermatr(self, k, minus=False):
        """Return interaction matrix multiplied with phase factor."""
        sign = sp.Integer(1) if not minus else sp.Integer(-1)
        return sp.exp(sign * sp.I * k.dot(self.othervect)) * self.intermatr

    def num_phased_intermatr(self, k, minus=False):
        """Return interaction matrix multiplied with phase factor."""
        sign = 1 if not minus else -1
        intermatr = self.num_intermatr
        return np.exp(sign * 1j * k.dot(self.num_othervect)) * intermatr


class LinearSpinWave:
    """The main class for description of non-interacting magnons."""

    k_x, k_y, k_z = sp.symbols('k_x k_y k_z', real=True)
    k = sp.Matrix([k_x, k_y, k_z])
    B_x, B_y, B_z = sp.symbols('B_x B_y B_z', real=True)
    B = sp.Matrix([B_x, B_y, B_z])
    kb = 0.0862  # in meV / K
    hbar = 0.6582  # in meV * ps
    c = 1  # speed of light

    @classmethod
    def bose(cls, energ, temp):
        """Return the Bose-Einstein distribution function."""
        # overflow = np.log(1e18 + 1)
        x = energ / (LinearSpinWave.kb * temp)
        # if x >= overflow:
        #     return 0
        return 1 / (np.exp(x) - 1)

    def __init__(self, lat, mainbasis, num_parameters=None):
        self.lat = lat
        self.mainbasis = mainbasis
        self.nbands = lat.sublats_count
        self.bilinear_hamil = None
        self.num_parameters = num_parameters
        self.rparam_subsdict = None
        self.num_bilinear_hamil = None
        bands = lat.sublats_count
        self.metric = np.kron([[1, 0], [0, -1]], np.identity(bands))
        self.identity = np.eye(2 * bands)
        self.sp_metric = sp.diag(sp.eye(bands), -sp.eye(bands))
        self.ndmetric = np.kron([[0, 1], [-1, 0]], np.identity(bands))
        self.paulix = np.kron([[0, 1], [1, 0]], np.identity(bands))
        self.num_B = None
        if num_parameters is not None:
            self.parameterize()

        # for debugging
        # self.debug1 = None
        # self.debug2 = None
        # self.debug3 = None
        # self.debug4 = None

    def copy(self):
        """Return a copy of the current instance."""
        obj_copy = copy.deepcopy(self)
        return obj_copy

    def parameterize(self):
        """Initialize numerical variables."""
        if self.num_parameters is None:
            raise ValueError('Parameters have not been specified.')
        num_parameters = self.num_parameters
        B = self.__class__.B
        Bx, By, Bz = (num_parameters[x] for x in ['Bx', 'By', 'Bz'])
        subs = {B[0]: Bx, B[1]: By, B[2]: Bz}
        self.num_B = np.array(B.T.subs(subs)).astype(np.float64)[0]

    def assume_real_params(self):
        """
        Assume that all parameters are real.

        Returns
        -------
        rparam_subsdict: dict
            Contains variable names as keys and sympy symbols assumed to be
            real as items. Can be used to refine sympy expressions by
            substitution.

        """
        if self.num_parameters is None:
            raise ValueError('variable names unknown')
        variables = self.num_parameters.keys()
        varsymbols = sp.symbols(' '.join(variables), real=True)
        rparam_subsdict = dict([[v, s] for v, s in zip(variables, varsymbols)])
        self.rparam_subsdict = rparam_subsdict
        return rparam_subsdict

    def holstein_primakoff(self, symbolic=True, basisphase=True,
                           simplify=False):
        """Do Holstein-Primakoff transformation and return Hamilton kernel."""
        # print(f"holstein_primakoff called, symbolic={symbolic}")
        n = self.nbands
        if symbolic:
            k = self.__class__.k
            inv_k = {k[0]: -k[0], k[1]: -k[1], k[2]: -k[2]}
            amatr = self.amatr(basisphase, simplify)
            amatr2 = amatr.subs(inv_k).conjugate()
            bmatr = self.bmatr(basisphase, simplify)
            bmatr2 = bmatr.T.conjugate()
            cmatr = self.cmatr(simplify)
            dmatr = self.dmatr(simplify)
            hmatr = sp.zeros(2*n, 2*n)
            self.insert_hsubmatr(hmatr, amatr, amatr2, bmatr, bmatr2, cmatr,
                                 dmatr)
            if self.rparam_subsdict is not None:
                hmatr = hmatr.subs(self.rparam_subsdict)
        else:
            # print("About to call auxilliary function")
            def hmatr(*k):
                # print(f"input k={k}")
                k = np.array(k)
                # print(f"array k={k}")
                amatr = self.num_amatr(k, basisphase)
                amatr2 = self.num_amatr(-k, basisphase).conjugate()
                bmatr = self.num_bmatr(k, basisphase)
                bmatr2 = bmatr.T.conj()
                cmatr = self.num_cmatr()
                dmatr = self.num_dmatr()
                hmatr = np.zeros((2*n, 2*n), dtype='complex')
                # print(hmatr)
                # print("About to call insert_hsubmatr")
                self.insert_hsubmatr(hmatr, amatr, amatr2, bmatr, bmatr2,
                                     cmatr, dmatr)
                # print(hmatr)
                return hmatr
        # print(f"Auxilliary function returned: {hmatr}, type: {type(hmatr)}")
        self.bilinear_hamil = hmatr
        if not symbolic:
            self.num_bilinear_hamil = hmatr
            # print(self.num_bilinear_hamil)
        return hmatr

    @staticmethod
    def insert_hsubmatr(hmatr, amatr, amatr2, bmatr, bmatr2, cmatr, dmatr):
        n = hmatr.shape[0] // 2
        hmatr[:n, :n] = amatr - cmatr - dmatr
        hmatr[:n, n:2*n] = bmatr
        hmatr[n:2*n, :n] = bmatr2
        hmatr[n:2*n, n:2*n] = amatr2 - cmatr - dmatr

    def amatr(self, basisphase=True, simplify=False):
        n = self.nbands
        k = self.__class__.k
        amatr = sp.zeros(n, n)
        sublats = self.lat.sublats
        atoms = self.mainbasis.atoms
        for i in range(n):
            for j in range(i, n):
                # spin lengths
                si, sj = sublats[i].spinlen, sublats[j].spinlen
                # basis vectors
                bi, bj = sublats[i].basisvect, sublats[j].basisvect
                # phase factor
                pf = sp.exp(sp.I * k.dot(bj - bi))
                # local z directions
                ui, uj = sublats[i].u, sublats[j].u
                # Fourier-transformed interaction matrices
                jmatrij = atoms[i].sum_phased_intermatr(k, j)
                jmatrji = atoms[j].sum_phased_intermatr(k, i, True)
                tmp = sp.sqrt(si * sj) / sp.Integer(4)
                if basisphase:
                    tmp *= pf
                tmp2 = ui.T * jmatrij * uj.conjugate()
                tmp3 = uj.conjugate().T * jmatrji * ui
                res = tmp * (tmp2 + tmp3)
                if simplify:
                    res = sp.simplify(res)
                amatr[i, j] = res
                amatr[j, i] = res.conjugate()
        return amatr

    def num_amatr(self, k, basisphase=True):
        n = self.nbands
        amatr = np.zeros((n, n), dtype='complex')
        sublats = self.lat.sublats
        atoms = self.mainbasis.atoms
        k = np.array(k)
        for i in range(n):
            for j in range(i, n):
                # spin lengths
                si, sj = sublats[i].num_spinlen, sublats[j].num_spinlen
                # basis vectors
                bi, bj = sublats[i].num_basisvect, sublats[j].num_basisvect
                # phase factor
                pf = np.exp(1j * k.dot(bj - bi))
                # local z directions
                ui, uj = sublats[i].num_u, sublats[j].num_u
                # Fourier-transformed interaction matrices
                jmatrij = atoms[i].sum_num_phased_intermatr(k, j)
                jmatrji = atoms[j].sum_num_phased_intermatr(k, i, True)
                tmp = np.sqrt(si * sj) / 4
                if basisphase:
                    tmp *= pf
                tmp2 = ui.dot(jmatrij).dot(uj.conjugate())
                tmp3 = uj.conjugate().dot(jmatrji).dot(ui)
                amatr[i, j] = tmp * (tmp2 + tmp3)
                amatr[j, i] = amatr[i, j].conjugate()
        return amatr

    def bmatr(self, basisphase=True, simplify=False):
        k = self.__class__.k
        n = self.nbands
        sublats = self.lat.sublats
        atoms = self.mainbasis.atoms
        bmatr = sp.zeros(n, n)
        # construct B
        for i in range(n):
            for j in range(n):
                # spin lengths
                si, sj = sublats[i].spinlen, sublats[j].spinlen
                # basis vectors
                bi, bj = sublats[i].basisvect, sublats[j].basisvect
                # phase factor
                pf = sp.exp(sp.I * k.dot(bj - bi))
                # local z directions
                ui, uj = sublats[i].u, sublats[j].u
                # Fourier-transformed interaction matrices
                jmatrij = atoms[i].sum_phased_intermatr(k, j)
                tmp = sp.sqrt(si * sj) / sp.Integer(2)
                if basisphase:
                    tmp *= pf
                tmp2 = ui.T * jmatrij * uj
                res = tmp * tmp2
                if simplify:
                    res = sp.simplify(res)
                bmatr[i, j] = res
        return bmatr

    def num_bmatr(self, k, basisphase=True):
        n = self.nbands
        bmatr = np.zeros((n, n), dtype='complex')
        sublats = self.lat.sublats
        atoms = self.mainbasis.atoms
        # construct B
        for i in range(n):
            for j in range(n):
                # spin lengths
                si, sj = sublats[i].num_spinlen, sublats[j].num_spinlen
                # basis vectors
                bi, bj = sublats[i].num_basisvect, sublats[j].num_basisvect
                # phase factor
                pf = np.exp(1j * k.dot(bj - bi))
                # local z directions
                ui, uj = sublats[i].num_u, sublats[j].num_u
                # Fourier-transformed interaction matrices
                jmatrij = atoms[i].sum_num_phased_intermatr(k, j)
                tmp = np.sqrt(si * sj) / 2
                if basisphase:
                    tmp *= pf
                tmp2 = ui.dot(jmatrij).dot(uj)
                bmatr[i, j] = tmp * tmp2
        return bmatr

    def cmatr(self, simplify=False):
        n = self.nbands
        sublats = self.lat.sublats
        atoms = self.mainbasis.atoms
        cmatr = sp.zeros(n, n)
        # construct C
        for i in range(n):
            for l in range(n):
                # spin lengths
                sl = sublats[l].spinlen
                # local xy directions
                vi, vl = sublats[i].v, sublats[l].v
                # Fourier-transformed interaction matrices
                jmatril = atoms[i].sum_intermatr(l)
                jmatrli = atoms[l].sum_intermatr(i)
                tmp = sl / sp.Integer(2)
                tmp2 = vi.T * jmatril * vl
                tmp3 = vl.T * jmatrli * vi
                res = tmp * (tmp2 + tmp3)[0]
                if simplify:
                    res = sp.simplify(res)
                cmatr[i, i] += res
        return cmatr

    def num_cmatr(self):
        n = self.nbands
        sublats = self.lat.sublats
        atoms = self.mainbasis.atoms
        cmatr = np.zeros((n, n), dtype='complex')
        # construct C
        for i in range(n):
            for l in range(n):
                # spin lengths
                sl = sublats[l].num_spinlen
                # local xy directions
                vi, vl = sublats[i].num_v, sublats[l].num_v
                # Fourier-transformed interaction matrices
                jmatril = atoms[i].sum_num_intermatr(l)
                jmatrli = atoms[l].sum_num_intermatr(i)
                tmp = sl / 2
                tmp2 = vi.dot(jmatril).dot(vl)
                tmp3 = vl.dot(jmatrli).dot(vi)
                cmatr[i, i] += tmp * (tmp2 + tmp3)
        return cmatr

    def dmatr(self, simplify=False):
        n = self.nbands
        sublats = self.lat.sublats
        dmatr = sp.zeros(n, n)
        B = self.__class__.B
        # construct D
        for i in range(n):
            res = B.T * sublats[i].gyromatr * sublats[i].v
            if simplify:
                res = sp.simplify(res)
            dmatr[i, i] = res
        return dmatr

    def num_dmatr(self):
        n = self.nbands
        num_B = self.num_B
        sublats = self.lat.sublats
        dmatr = np.zeros((n, n), dtype='complex')
        # construct D
        for i in range(n):
            vi = sublats[i].num_v
            gyromatr = sublats[i].num_gyromatr
            dmatr[i, i] = num_B.dot(gyromatr).dot(vi)
        return dmatr

    def bandstructure(self):
        """Return the analytical dispersion relation."""
        gh = self.sp_metric * self.bilinear_hamil
        return gh.eigenvals()

    def parameterize_hamil(self, chempot=None):
        """
        Replace symbols by numerical values and add chemical potential.

        A small diagonal matrix chempt * diag(1, 1, ...) is added to the
        matrix to gap the spectrum which is necessary in order to make the
        Hamilton kernel positive definite and thus decomposible by Cholesky's
        algorithm. If None, the chemical potential is chosen to be as large as
        needed to make the Hamilton kernel positive definite at k = 0 or, if
        it is specified in the field num_parameters, this is used instead.

        """
        if self.num_parameters is None:
            raise ValueError('Parameters have not been specified')
        num_params = self.num_parameters
        bands = self.nbands
        try:
            tmp = self.bilinear_hamil.subs({
                self.B_x: num_params['Bx'],
                self.B_y: num_params['By'],
                self.B_z: num_params['Bz']})
            # self.debug1 = tmp
            num_hamil_no_cp = sp.lambdify(
                self.__class__.k,
                tmp.subs(self.num_parameters),
                'numpy')
            # self.debug2 = num_hamil_no_cp
        except AttributeError:
            num_hamil_no_cp = self.bilinear_hamil
        # self.debug3 = num_hamil_no_cp

        def add_chem_potential(cp):
            self.num_bilinear_hamil = lambda *k: num_hamil_no_cp(*k) \
                + cp*np.eye(2*bands)
        if chempot is not None:
            add_chem_potential(chempot)
            self.num_parameters['chempot'] = chempot
            return chempot
        if chempot is None and 'chempot' in num_params:
            chempot = num_params['chempot']
            add_chem_potential(chempot)
            return chempot
        # find 'chemical element'
        print('Automatically finding chemical potential ...')
        chempot = 0
        add_chem_potential(chempot)
        while True:
            try:
                self.bog_trafo(0, 0, 0)
                print(f'Using {chempot} as chemical potential.')
                num_params['chempot'] = chempot
                return chempot
            except LinAlgError:
                chempot = 1e-10 if chempot == 0 else chempot * 10
                add_chem_potential(chempot)

    def parameterize_dhamil(self):
        """Parameterize k-derivative of Hamilton kernel."""
        if self.num_parameters is None:
            raise ValueError('Parameters have not been specified.')
        num_params = self.num_parameters
        subs = {
            self.B_x: num_params['Bx'],
            self.B_y: num_params['By'],
            self.B_z: num_params['Bz']}
        subs.update(self.num_parameters)
        k = self.k
        hamil = self.bilinear_hamil
        dhamil = [
            sp.lambdify(k, hamil.diff(self.k_x).subs(subs), 'numpy'),
            sp.lambdify(k, hamil.diff(self.k_y).subs(subs), 'numpy'),
            sp.lambdify(k, hamil.diff(self.k_z).subs(subs), 'numpy')]
        self.num_bilinear_dhamil = dhamil

    def get_grddirmatr(self, comp):
        """
        Compose diagonal matrix of sublattice ground state directions.

        Return diag(v_1, ..., v_N, v1, ..., v_N) where vi is a component of the
        spin direction of the i-th sublattice.
        """
        n = self.lat.sublats_count
        sls = self.lat.sublats
        grdstatedirs = [sls[i].num_unitvects[-1][comp] for i in range(n)]
        res = np.diag([0.0 for i in range(2 * n)])
        for i in range(n):
            res[i + n, i + n] = res[i, i] = grdstatedirs[i]
        return res

    def get_velmatr_hp(self, kx, ky, kz, comp):
        """Return velocity matrix elements."""
        dh = self.num_bilinear_dhamil[comp](kx, ky, kz)
        return dh / self.hbar

    def get_velmatr(self, kx, ky, kz, tinv, tinvadj, comp):
        """Return velocity matrix elements."""
        vmatr_hp = self.get_velmatr_hp(kx, ky, kz, comp)
        return tinvadj.dot(vmatr_hp).dot(tinv)

    def get_curmatr_hp(self, kx, ky, kz, vdir, sdir):
        """Return current matrix elements."""
        vmatr = self.get_velmatr_hp(kx, ky, kz, vdir)
        smatr = self.get_spinmatr_hp(kx, ky, kz, sdir)
        gmatr = self.metric
        return (vmatr.dot(gmatr).dot(smatr) + smatr.dot(gmatr).dot(vmatr)) / 2

    def get_curmatr(self, kx, ky, kz, tinv, tinvadj, vdir, sdir):
        """Return current matrix elements."""
        vmatr = self.get_velmatr(kx, ky, kz, tinv, tinvadj, vdir)
        smatr = self.get_smmmatr(tinv, tinvadj, sdir)
        gmatr = self.metric
        return (vmatr.dot(gmatr).dot(smatr) + smatr.dot(gmatr).dot(vmatr)) / 2

    def init_smm(self):
        """Initialize analy. linear, biliner spin magnetic moment operators."""
        n = self.nbands
        initvect = 3 * [2 * n * [sp.Integer(0)]]
        initmatr = 3 * [2 * [2 * n * [sp.Integer(0)]]]
        smm_vect = sp.MutableDenseNDimArray(initvect)
        smm_matr = sp.MutableDenseNDimArray(initmatr)
        for i, sublat in enumerate(self.lat.sublats):
            sli = sublat.spinlen
            epi = sublat.u
            emi = sublat.u.conjugate()
            ezi = sublat.v
            gyromatr = sublat.gyromatr
            gepi = gyromatr @ epi
            gemi = gyromatr @ emi
            gezi = gyromatr @ ezi
            smm_vect[:, i] = -sli * gemi / sp.sqrt(2)
            smm_vect[:, i + n] = -sli * gepi / sp.sqrt(2)
            smm_matr[:, i, i] = gezi
            smm_matr[:, i + n, i + n] = gezi
        self.smm_vect = smm_vect
        self.smm_matr = smm_matr

    def init_num_smm(self):
        """Initalize numeric linear, biliner spin magnetic moment operators."""
        n = self.nbands
        num_smm_vect = np.zeros((3, 2 * n), dtype='complex')
        num_smm_matr = np.zeros((3, 2 * n, 2 * n), dtype='float')
        for i, sublat in enumerate(self.lat.sublats):
            sli = sublat.num_spinlen
            epi = sublat.num_u
            emi = sublat.num_u.conjugate()
            ezi = sublat.num_v
            gyromatr = sublat.num_gyromatr
            gepi = gyromatr.dot(epi)
            gemi = gyromatr.dot(emi)
            gezi = gyromatr.dot(ezi)
            num_smm_vect[:, i] = -sli * gemi / sp.sqrt(2)
            num_smm_vect[:, i + n] = -sli * gepi / sp.sqrt(2)
            num_smm_matr[:, i, i] = gezi
            num_smm_matr[:, i + n, i + n] = gezi
        self.num_smm_vect = num_smm_vect
        self.num_smm_matr = num_smm_matr

    def bog_smm_vect(self, tinv):
        """Transform linear spin magnetic moment operator into eigenbasis."""
        hp_smm_vect = self.num_smm_vect
        bog_smm_vect = np.einsum('ij,jk->ik', hp_smm_vect, tinv)
        return bog_smm_vect

    def bog_smm_matr(self, tinv, tinvadj):
        """Transform bilinear spin magnetic moment operator into eigenbasis."""
        hp_smm_matr = self.num_smm_matr
        bog_smm_matr = np.einsum('ij,kjm,mn->kin', tinvadj, hp_smm_matr, tinv)
        return bog_smm_matr

    def get_smmmatr_hp(self, comp):
        """
        Compose spin magnetic moment matrix (in HP basis).

        Return diag(v_1, ..., v_N, v1, ..., v_N) where vi is a component of the
        atomic spin magnetic moment direction of the i-th sublattice.
        """
        n = self.nbands
        sls = self.lat.sublats
        momdirs = [sl.num_gyromatr.dot(sl.num_unitvects[-1]) for sl in sls]
        res = np.diag([0.0 for i in range(2 * n)])
        for i in range(n):
            res[i + n, i + n] = res[i, i] = np.real(momdirs[i][comp])
        return res

    def get_smmmatr(self, tinv, tinvadj, comp):
        smm_hp = self.get_smmmatr_hp(comp)
        res = tinvadj.dot(smm_hp).dot(tinv)
        return res

    def smagn_moment_from_Tinv(self, tinvmat):
        """Calculate magnetic moments from eigenvectors."""
        n = self.nbands
        res = np.zeros((n, 3))
        for i in range(n):
            tmp = 0
            for j in range(n):
                cur_sublat = self.lat.sublats[j]
                num_unitv = cur_sublat.num_unitvects[-1, :]
                cur_dir = cur_sublat.num_gyromatr.dot(num_unitv)
                cur_mult = np.abs(tinvmat[j, i]) ** 2 \
                    + np.abs(tinvmat[j + n, i]) ** 2
                tmp += cur_dir * cur_mult
            res[i, :] = tmp
        return res

    def smagn_moment_grid(self, grid):
        """
        Calculate spin magnetic moments in two-dimensional grid in k space.

        Parameters
        ----------
        grid : 4-dimensional numpy.ndarray
            Contains the diagonalizing matrices for a 2D grid in k space. First
            two indices reference the k points, and third, fourth index
            represent the rows, columns of the matrices, respectively.

        Returns
        -------
        smagn_moments : 4D numpy.ndarray
            Spin magnetic moments of all bands starting from the highest one in
            energy. First index corresponds to the band, second to the
            cartisian component, and third and fourth indices represent
            different k points in the 2D grid.

        """
        bands = self.nbands
        m, n = grid.shape[:2]
        smagn_moments = np.zeros((bands, 3, m, n))
        for i in range(m):
            for j in range(n):
                t = grid[i, j]
                mm = self.smagn_moment_from_Tinv(t)
                smagn_moments[:, :, i, j] = mm
        return smagn_moments

    def smagn_moment_path(self, *points, num=100, isometric=False,
                          basis='rcpr'):
        """Calculate magnetic moment along polygonal chain in k-space."""
        ts = np.linspace(0, 1, num=num)
        k_of_t, point_maps = self.path_from_points(
            *points, isometric=isometric, basis=basis)
        ks = [k_of_t(t) for t in ts]
        magn_moments = np.array(
            [self.smagn_moment_from_Tinv(self.bog_trafo(*k)[1]) for k in ks])
        return ts, ks, point_maps, magn_moments

    def spin_quantfluc_lswt_from_tinv(self, tinv):
        """
        Return reciprocal-space contribution to spin quantum fluctions in LSWT.

        Parameters
        ----------
        tinv : np.ndarray
            Eigenvectors that diagonalize Hamiltonian.

        Returns
        -------
        spinflucs : np.ndarray
            Fluctuations of the spins on all sublattices (first index) that
            reduces the spin length of that sublattice.

        """
        abssqtinv = np.abs(tinv) ** 2
        nbands = self.nbands
        spinflucs = abssqtinv[:nbands, nbands:].sum(axis=-1)
        return spinflucs

    def loc_spin_quantfluc_lswt(self, kx, ky, kz):
        """
        Return reciprocal-space contribution to spin quantum fluctions in LSWT.

        Parameters
        ----------
        kx : float
            First component of Bloch vector.
        ky : float
            Second component of Bloch vector.
        kz : float
            Third component of Bloch vector.

        Returns
        -------
        spinflucs : np.ndarray
            Fluctuations of the spins on all sublattices (first index) that
            reduces the spin length of that sublattice.

        """
        es, tinv = self.bog_trafo(kx, ky, kz)
        spinflucs = self.spin_quantfluc_lswt_from_tinv(tinv)
        return spinflucs

    def spin_quantumfluc_lswt_grid(self, grid):
        """
        Calculate LSWT spin quantum fluctuations in 2D grid in k space.

        Parameters
        ----------
        grid : 4-dimensional numpy.ndarray
            Contains the diagonalizing matrices for a 2D grid in k space. First
            two indices reference the k points, and third, fourth index
            represent the rows, columns of the matrices, respectively.

        Returns
        -------
        spinflucgrid : 3D numpy.ndarray
            Fluctuations of the sublattice spins (first index) on a 2D mesh in
            reciprocal space (second, third indices).

        """
        nsublats = self.lat.sublats_count
        m, n = grid.shape[:2]
        spinflucgrid = np.zeros((nsublats, m, n))
        for i in range(m):
            for j in range(n):
                tinv = grid[i, j]
                spinflucs = self.spin_quantfluc_lswt_from_tinv(tinv)
                spinflucgrid[:, i, j] = spinflucs
        return spinflucgrid

    def spin_fluc_lswt_from_eig(self, es, tinv, temp):
        """
        Return reciprocal-space contribution to spin quantum fluctions in LSWT.

        Parameters
        ----------
        es : np.ndarray
            Eigenvalues of the Hamiltonian.
        tinv : np.ndarray
            Eigenvectors that diagonalize Hamiltonian.
        temp : float
            Temperature at which quantum and thermal fluctuations are
            calculated. If zero, only quantum flucutations are considered.

        Returns
        -------
        spinflucs : np.ndarray
            Fluctuations of the spins on all sublattices (first index) that
            reduces the spin length of that sublattice.

        """
        nbands = self.nbands
        astinv = np.abs(tinv) ** 2
        astinv_pp = astinv[:nbands, :nbands]
        astinv_ph = astinv[:nbands, nbands:]
        astinv_hp = astinv[nbands:, :nbands]
        spinflucs = astinv_ph.sum(axis=-1)
        if temp > 0:
            bvals = self.bose(es[:nbands], temp)
            spinflucs += np.einsum('i,ji->j', bvals, astinv_pp + astinv_hp)
        return spinflucs

    def loc_spin_fluc_lswt(self, kx, ky, kz, temp):
        """
        Return reciprocal-space contribution to spin quantum fluctions in LSWT.

        Parameters
        ----------
        kx : float
            First component of Bloch vector.
        ky : float
            Second component of Bloch vector.
        kz : float
            Third component of Bloch vector.
        temp : float
            Temperature at which quantum and thermal fluctuations are
            calculated. If zero, only quantum flucutations are considered.

        Returns
        -------
        spinflucs : np.ndarray
            Fluctuations of the spins on all sublattices (first index) that
            reduces the spin length of that sublattice.

        """
        es, tinv = self.bog_trafo(kx, ky, kz)
        spinflucs = self.spin_fluc_lswt_from_tinv(es, tinv, temp)
        return spinflucs

    def spin_fluc_lswt_grid(self, egrid, tgrid, temp):
        """
        Calculate LSWT spin quantum fluctuations in 2D grid in k space.

        Parameters
        ----------
        egrid : 4-dimensional numpy.ndarray
            Contains eigenenergies of the Hamiltonian at 2D k grid points.
            First and second indices reference the k points, third index
            labels the bands.
        tgrid : 4-dimensional numpy.ndarray
            Contains the diagonalizing matrices for a 2D grid in k space. First
            two indices reference the k points, and third, fourth index
            represent the rows, columns of the matrices, respectively.
        temp : float
            Temperature at which quantum and thermal fluctuations are
            calculated. If zero, only quantum flucutations are considered.

        Returns
        -------
        spinflucgrid : 3D numpy.ndarray
            Fluctuations of the sublattice spins (first index) on a 2D mesh in
            reciprocal space (second, third indices).

        """
        nsublats = self.lat.sublats_count
        m, n = tgrid.shape[:2]
        spinflucgrid = np.zeros((nsublats, m, n))
        for i in range(m):
            for j in range(n):
                es = egrid[i, j]
                tinv = tgrid[i, j]
                spinflucs = self.spin_fluc_lswt_from_eig(es, tinv, temp)
                spinflucgrid[:, i, j] = spinflucs
        return spinflucgrid

    def get_dipolematr_hp(self, kx, ky, kz, comp):
        gmatr = self.metric
        res = np.zeros(gmatr.shape, dtype='complex')
        c1, c2 = (comp + 1) % 3, (comp + 2) % 3
        velmatr1 = self.get_velmatr_hp(kx, ky, kz, c1)
        velmatr2 = self.get_velmatr_hp(kx, ky, kz, c2)
        smmmatr1 = self.get_smmmatr_hp(c1)
        smmmatr2 = self.get_smmmatr_hp(c2)
        res = np.zeros(gmatr.shape, dtype='complex')
        res += velmatr1.dot(gmatr).dot(smmmatr2)
        res += smmmatr2.dot(gmatr).dot(velmatr1)
        res -= velmatr2.dot(gmatr).dot(smmmatr1)
        res -= smmmatr1.dot(gmatr).dot(velmatr2)
        res /= 2 * self.c ** 2
        return res

    def get_dipolematr(self, kx, ky, kz, tinv, tinvadj, comp):
        res = self.get_dipolematr_hp(kx, ky, kz, comp)
        return tinvadj.dot(res).dot(tinv)

    def dipole_moment_grid(self, kgrid, grid):
        """
        Calculate electrical moments in two-dimensional grid in k space.

        Parameters
        ----------
        kgrid : 3-dimensional numpy.ndarray
            First index represents the component in k space, second and third
            indices represent the axes of the two-dimensional grid.
        grid : 4-dimensional numpy.ndarray
            Contains the diagonalizing matrices for a 2D grid in k space. First
            two indices reference the k points, and third, fourth index
            represent the rows, columns of the matrices, respectively.

        Returns
        -------
        dipole_moments : 4D numpy.ndarray
            Electrical moments of all bands starting from the highest one in
            energy. First index corresponds to the band, second to the
            cartisian component, and third and fourth indices represent
            different k points in the 2D grid.

        """
        bands = self.nbands
        m, n = grid.shape[:2]
        dipole_moments = np.zeros((bands, 3, m, n))
        for i in range(m):
            for j in range(n):
                k = kgrid[:, i, j]
                t = grid[i, j]
                tadj = t.transpose().conjugate()
                for comp in range(3):
                    d = self.get_dipolematr(*k, t, tadj, comp)
                    dipole_moments[:, comp, i, j] = np.diag(d[:bands])
        return dipole_moments

    def init_knb(self, nns, basisphase=True):
        """
        Build the matrices for the Katsura-Nagaosa-Balatsky operator.

        Parameters
        ----------
        nns : list of lists of lists
            Nearest neighbors of all atoms in the unit cell. First index
            represents the atom in the unit cell, second index runs over the
            nearest neighbors each of which is represented by (i) the
            sublattice number (integer) and (ii) its unit cell given in the
            basis of the translation vectors (list of 3 integers).
        """
        n = self.nbands
        sublats = self.lat.sublats
        k = self.k
        inv_k = {k[0]: -k[0], k[1]: -k[1], k[2]: -k[2]}
        # initialize matrices
        initvect = 3 * [n * [sp.Integer(0)]]
        initmatr = 3 * [n * [n * [sp.Integer(0)]]]
        rvect = sp.MutableDenseNDimArray(initvect)
        xmatr = sp.MutableDenseNDimArray(initmatr)
        ymatr = sp.MutableDenseNDimArray(initmatr)
        lmatr = sp.MutableDenseNDimArray(initmatr)
        # iterate over spins in unit cell
        for i in range(n):
            cnns = nns[i]
            sli = sublats[i]
            si = sli.spinlen
            bi = sli.basisvect
            epi = sli.u
            emi = sli.u.conjugate()
            ezi = sli.v
            # iterate over nearest neighbors
            for nn in cnns:
                j, uc = nn
                slj = sublats[j]
                sj = slj.spinlen
                bj = slj.basisvect
                epj = slj.u
                emj = slj.u.conjugate()
                ezj = slj.v
                delta = self.lat.transf_to_cart(uc)
                diff = delta + bj - bi
                if basisphase:
                    phase = sp.exp(sp.I * k.dot(diff))
                else:
                    phase = sp.exp(sp.I * k.dot(delta))
                sqrt = sp.sqrt(si * sj)
                # build matrices
                rel = sp.sqrt(2 * si * sj ** 2) * diff.cross(emi.cross(ezj))
                xel = sqrt * diff.cross(epi.cross(emj)) * phase
                yel = sqrt * diff.cross(epi.cross(epj)) * phase
                lel = -sj * diff.cross(ezi.cross(ezj))
                rvect[:, i] += sp.MutableDenseNDimArray(list(rel))
                xmatr[:, i, j] += sp.MutableDenseNDimArray(list(xel))
                ymatr[:, i, j] += sp.MutableDenseNDimArray(list(yel))
                lmatr[:, i, i] += sp.MutableDenseNDimArray(list(lel))
        rconjvect = rvect.applyfunc(lambda x: x.conjugate())
        xtrpmatr = sp.MutableDenseNDimArray([m.transpose() for m in xmatr])
        xtrpinvkmatr = xtrpmatr.applyfunc(lambda x: x.subs(inv_k))
        ytrpmatr = sp.MutableDenseNDimArray([m.transpose() for m in ymatr])
        yadjmatr = sp.conjugate(ytrpmatr)
        knb_vect = sp.MutableDenseNDimArray(np.zeros((3, 2 * n)))
        knb_matr = sp.MutableDenseNDimArray(np.zeros((3, 2 * n, 2 * n)))
        knb_vect[:, :n] = rvect
        knb_vect[:, n:] = rconjvect
        knb_matr[:, :n, :n] = xmatr + lmatr
        knb_matr[:, :n, n:] = ymatr
        knb_matr[:, n:, :n] = yadjmatr
        knb_matr[:, n:, n:] = xtrpinvkmatr + lmatr
        self.knb_vect = knb_vect
        self.knb_rvect = rvect
        self.knb_matr = knb_matr
        self.knb_xmatr = xmatr
        self.knb_ymatr = ymatr
        self.knb_lmatr = lmatr

    def parameterize_knb(self):
        """Initialize numeric matrix Katsura-Nagaosa-Balatsky operator."""
        vect = self.knb_vect.applyfunc(lambda x: x.subs(self.num_parameters))
        rvect = self.knb_rvect.applyfunc(lambda x: x.subs(self.num_parameters))
        matr = self.knb_matr.applyfunc(lambda x: x.subs(self.num_parameters))
        xmatr = self.knb_xmatr.applyfunc(lambda x: x.subs(self.num_parameters))
        ymatr = self.knb_ymatr.applyfunc(lambda x: x.subs(self.num_parameters))
        lmatr = self.knb_lmatr.applyfunc(lambda x: x.subs(self.num_parameters))
        vect = vect.tolist()
        rvect = rvect.tolist()
        matr = matr.tolist()
        xmatr = xmatr.tolist()
        ymatr = ymatr.tolist()
        lmatr = lmatr.tolist()
        num_knb_vect = np.array(vect, dtype=complex)
        num_knb_rvect = np.array(rvect, dtype=complex)
        num_knb_matr = sp.lambdify(self.__class__.k, matr, 'numpy')
        num_knb_xmatr = sp.lambdify(self.__class__.k, xmatr, 'numpy')
        num_knb_ymatr = sp.lambdify(self.__class__.k, ymatr, 'numpy')
        num_knb_lmatr = np.array(lmatr, dtype=float)
        self.num_knb_vect = num_knb_vect
        self.num_knb_rvect = num_knb_rvect
        self.num_knb_matr = lambda *k: np.array(num_knb_matr(*k))
        self.num_knb_xmatr = lambda *k: np.array(num_knb_xmatr(*k))
        self.num_knb_ymatr = lambda *k: np.array(num_knb_ymatr(*k))
        self.num_knb_lmatr = num_knb_lmatr

    def bog_knb_vect(self, tinv):
        """Transform linear KNB operator into eigenbasis."""
        hp_knb_vect = self.num_knb_vect
        bog_knb_vect = np.einsum('ij,jl->il', hp_knb_vect, tinv)
        return bog_knb_vect

    def slres_bog_knb_vect(self, tinv):
        """
        Transform linear KNB operator into eigenbasis with sublat. resolution.

        Parameters
        ----------
        tinv : np.array
            Left eigenvectors of Hamiltonian at a specific Bloch vector k.

        Returns
        -------
        slres_bog_knb_vect : 3-dimensional np.ndarray
            Electric dipole operator where first index represents polarization
            direction, second represents sublattice, and third represents
            the magnon normal modes.

        """
        hp_knb_vect = self.num_knb_vect
        slres_bog_knb_vect = hp_knb_vect[:, :, None] * tinv[None, :, :]
        return slres_bog_knb_vect

    def bog_knb_matr(self, kx, ky, kz, tinv, tinvadj):
        """Transform the Katsura-Nagaosa-Balatsky operator into eigenbasis."""
        hp_knb_matr = self.num_knb_matr(kx, ky, kz)
        bog_knb_matr = np.einsum('ij,ljm,mn->lin', tinvadj, hp_knb_matr, tinv)
        return bog_knb_matr

    def bog_knb_lmatr(self, tinv, tinvadj):
        """Transform Lambda submatrix of the KNB operator into eigenbasis."""
        hp_lmatr = self.num_knb_lmatr
        bog_lmatr = np.einsum('ij,ljm,mn->lin', tinvadj, hp_lmatr, tinv)
        return bog_lmatr

    def knb_grid(self, kgrid, grid):
        """
        Calculate KNB electrical moments in two-dimensional grid in k space.

        Parameters
        ----------
        kgrid : 3-dimensional numpy.ndarray
            First index represents the component in k space, second and third
            indices represent the axes of the two-dimensional grid.
        grid : 4-dimensional numpy.ndarray
            Contains the diagonalizing matrices for a 2D grid in k space. First
            two indices reference the k points, and third, fourth index
            represent the rows, columns of the matrices, respectively.

        Returns
        -------
        dipole_moments : 4D numpy.ndarray
            Electrical moments of all bands starting from the highest one in
            energy. First index corresponds to the band, second to the
            cartisian component, and third and fourth indices represent
            different k points in the 2D grid.

        """
        bands = self.nbands
        m, n = grid.shape[:2]
        dipole_moments = np.zeros((bands, 3, m, n), dtype=complex)
        for i in range(m):
            for j in range(n):
                k = kgrid[:, i, j]
                t = grid[i, j]
                tadj = t.transpose().conjugate()
                d = self.bog_knb_matr(*k, t, tadj)
                diag = np.diagonal(d[:, :bands, :bands], axis1=1, axis2=2)
                dipole_moments[:, :, i, j] = np.transpose(diag)
        return dipole_moments

    def magnetothermal_cond(self, temp, relax_time, dim, reg_func, jac_det):
        """Initialize (extrinsic) transport tensor calculation routine."""
        return MagnetothermalTransport(temp, relax_time, dim, reg_func,
                                       jac_det, self)

    def num_grdstate_energy(self, dirs, params=None):
        """Return classical energy for given spin configuration and B-field."""
        if params is not None:
            old_params = self.num_parameters.copy()
            self.update_num_parameters(params)
        n = self.lat.sublats_count
        sublats = self.lat.sublats
        atoms = self.mainbasis.atoms
        e_zeeman = 0
        e_pair = 0

        magn_field = [self.num_parameters[c] for c in ('Bx', 'By', 'Bz')]
        tmp = np.array([0 for i in range(3)], dtype=float)
        for i in range(n):
            sl = sublats[i].num_spinlen
            tmp += sl * np.dot(sublats[i].num_gyromatr, dirs[i])
        e_zeeman = np.dot(magn_field, tmp)

        for i in range(n):
            jmatsum = np.zeros((3, 3), dtype=float)
            for j in range(n):
                jmatsum = atoms[i].sum_num_intermatr(j)
                e_pair += sublats[i].num_spinlen * sublats[j].num_spinlen \
                    * np.dot(dirs[i], np.dot(jmatsum, dirs[j]))
        e_pair /= 2
        if params is not None:
            self.update_num_parameters(old_params)
        return np.real(e_pair + e_zeeman)

    def find_grdstate(self, magn_field, tol=1e-12, method=None,
                      startangles=None):
        """Find ground state by numerical minimzation of Hamilton function."""
        n = self.lat.sublats_count
        params = dict(Bx=magn_field[0],
                      By=magn_field[1],
                      Bz=magn_field[2])
        # function R^(2n) --> R that should be minimized:

        def minfunc(angles):
            newdirs = self.__class__.to_dirs(angles)
            return self.num_grdstate_energy(newdirs, params)
        if startangles is None:
            startangles = np.zeros((2 * n,))
        # pair of angles: first azimuth, second polar
        bounds = [[0, 2 * np.pi], [0, np.pi]] * n
        minresult = minimize(minfunc, startangles, tol=tol, method=method,
                             bounds=bounds)
        print('ground state minimization result', minresult)
        minangles = minresult.x
        return minangles, self.__class__.to_dirs(minangles)

    @classmethod
    def to_dirs(cls, angles):
        """Convert list of spherical coordinates to cartesian coordinates."""
        n = len(angles) // 2
        dirs = np.zeros((n, 3))
        for i in range(n):
            dirs[i, :] = PolarCoord.to_cartesian(1, angles[2*i],
                                                 angles[2*i + 1])
        return dirs

    def refine_grdstate(self, magn_field, steps=1000):
        n = self.lat.sublats_count
        params = dict(Bx=magn_field[0],
                      By=magn_field[1],
                      Bz=magn_field[2])
        new_dirs = [self.lat.sublats[i].num_unitvects[-1] for i in range(n)]
        new_energ = self.num_grdstate_energy(new_dirs, params)
        temp = 0.11
        for step in range(steps + 100):
            if step < steps:
                temp -= 0.1 / steps
            for i in range(n):
                old_dirs = new_dirs.copy()
                old_energ = new_energ
                new_dirs[i] = self.random_grdstate_dir()
                new_energ = self.num_grdstate_energy(new_dirs, params)
                if new_energ > old_energ:
                    r = random()
                    if r > np.exp((old_energ - new_energ) / temp):
                        new_dirs = old_dirs.copy()
                        new_energ = old_energ
        return new_energ, new_dirs

    def get_hamilfun(self):
        """Return HamiltonFunction object for instantiating Monte Carlo."""
        nvars = self.lat.sublats_count
        vardims = nvars * [3]
        normflags = nvars * [True]
        def hfun(*x): return self.num_grdstate_energy(np.array(x))
        hfmngr = HamiltonFunction(hfun, vardims, normflags)
        return hfmngr

    def trsflag(self):
        """Determine whether magnetic field is present."""
        bx = self.num_parameters['Bx']
        by = self.num_parameters['By']
        bz = self.num_parameters['Bz']
        trsflag = all(np.isclose([bx, by, bz], 0))
        return trsflag

    def bog_trafo(self, kx, ky, kz, global_gauge=True, decimals=5,
                  force_phs=True):
        """
        Perform Bogoliubov transformation for bosonic diagonalization.

        The eigenproblem is solved such that the bosonic commutation relations
        and the particle-hole symmetry are preserved. The latter is achieved
        by diagonalizing H(-k) if the first nonzero component of k is negative
        and employing particle-hole symmetry to obain the eigenvalues and
        eigenvectors at k. This algorithm is robust with respect to
        degeneracies. This behavior can be controlled (see below).

        Parameters
        ----------
        kx : float
            First component of the Bloch vector.
        ky : float
            Second component of the Bloch vector.
        kz : float
            Third component of the Bloch vector.
        global_gauge : bool, optional
            If True, a phase factor is multiplied to the eigenvectors such that
            the entry with the largest absolute value is positive (and real).
            This enforces particle-hole symmetry except in case of
            degeneracies. Default is True.
        decimals : int, optional
            Number of decimals to which the absolute values of the eigenvector
            entries are rounded before they are compared. The purpose is to
            avoid instable, inconsistent behavior of the gauging when multiple
            elements have the same magnitude. Default is 5.
        force_phs : bool, optional
            If True, particle-hole symmetry is enforced by restricting the
            Bloch vector to a half space of the reciprocal space (see above).
            Default is True.

        Returns
        -------
        ldiag : np.ndarray
            One-dimensional numpy array containing quasiparticle energies at k
            in descending order and negative quasiparticles at -k in ascending
            order.
        tinvmat : np.ndarray
            Two dimensional numpy array containing the eigenvectors of all
            bands. The first index corresponds to the sublattice, while the
            second one is a band index.

        """
        metric = self.metric
        dim = self.nbands
        hmat = self.num_bilinear_hamil
        k = np.array([kx, ky, kz])
        # self.debug4 = hmat

        # ensure particle-hole symmetry by mapping k to -k if necessary
        k_is_zeros = np.isclose(k, 0)
        k_is_zero = all(k_is_zeros)
        if force_phs and not k_is_zero:
            # make first nonzero component positive
            sign = np.sign(k[~k_is_zeros][0])
            k *= sign
            kx, ky, kz = k

        # 1: cholesky's decomposition
        # ---------------------------
        # lower triangular matrix
        try:
            kadjmat = cholesky(hmat(kx, ky, kz))
        except LinAlgError:
            raise LinAlgError('Matrix is for k = ({}, {}, {}) not positive'
                              ' definite.'.format(kx, ky, kz))
        kmat = kadjmat.transpose().conj()

        # print(kmat, kadjmat, sep='\n')

        # 2: unitary diagonalization of K G K^+
        # -------------------------------------
        ldiag, umat = eigh(np.dot(kmat, np.dot(metric, kadjmat)))
        # print(f"ldiag=\n{ldiag}, \n umat=\n{umat}")
        # rearrange eigenvalues and eigenvectors
        order = list(range(2 * dim))
        # order.sort(key=ldiag.__getitem__, reverse=True)
        tmp = order[dim:]
        tmp.reverse()
        order = tmp + order[:dim]
        # print(f"order = {order}")
        umat = np.array(list(map(lambda i: umat[:, i], order))).transpose()
        ldiag = np.array(list(map(ldiag.__getitem__, order)))
        # print(f"ordered ldiag=\n{ldiag}, \n ordered umat=\n{umat}")
        # wrap ldiag with zeros
        lmat = np.array([[ldiag[i] if i == j else 0 for i in range(2 * dim)]
                         for j in range(2 * dim)])
        emat = np.dot(metric, lmat)
        # print(f"full L=\n{lmat}, \n Emat=\n{emat}")

        # 3: solve U E^(1/2) = K T^(-1)
        # -----------------------------
        ehalfmat = np.sqrt(emat)
        tinvmat = np.dot(inv(kmat), np.dot(umat, ehalfmat))

        # print(f"sqrt of Emat=\n{ehalfmat}, \n eigenvects T^-1=\n{tinvmat}")

        # 4: global gauge
        # ---------------
        k_is_zero = all(np.isclose(0, [kx, ky, kz]))
        if global_gauge:
            absmatr1 = np.abs(tinvmat[:dim, :dim]).round(decimals) # particles
            absmatr2 = np.abs(tinvmat[dim:, dim:]).round(decimals) # holes
            # for each column, find the row with the largest magnitude
            maxrows1 = absmatr1.argmax(axis=0)
            maxrows2 = absmatr2.argmax(axis=0)
            phases = np.zeros((2 * dim, 2 * dim), dtype=complex)
            for col, (maxrow1, maxrow2) in enumerate(zip(maxrows1, maxrows2)):
                # for each row, find the element with the largest magnitude
                maxel1 = tinvmat[maxrow1, col]
                maxel2 = tinvmat[maxrow2 + dim, col + dim]
                # extract the phase: x/|x| = e^(i*arg(x)), store in diagonal matrix
                phases[col, col] = maxel1 / np.abs(maxel1)
                phases[col + dim, col + dim] = maxel2 / np.abs(maxel2)
            # phases = np.diag(np.concatenate((
            #     tinvmat[0, :dim] / np.abs(tinvmat[0, :dim]),
            #     tinvmat[dim, dim:] / np.abs(tinvmat[dim, dim:])
            # )))
            # multiply eigenvectors with conjugate phase -> element with largest abs is positive and real
            tinvmat = tinvmat.dot(phases.conj())

        # enforce particle-hole symmetry for k = 0
        if force_phs:
            if k_is_zero:
                tinvmat[dim:, dim:] = tinvmat[:dim, :dim].conjugate()
                tinvmat[:dim, dim:] = tinvmat[dim:, :dim].conjugate()
            # revert back -k to k if applicable
            elif sign < 0:
                ldiag[:dim], ldiag[dim:] = -ldiag[dim:], -ldiag[:dim]
                paulix = self.paulix
                tinvmat = paulix.dot(tinvmat.conjugate()).dot(paulix)

        return ldiag, tinvmat

    def bog_trafo2(self, kx, ky, kz):
        """Perform unitary Bogoliubov transformation of nonhermitean matrix."""
        raise NotImplementedError('particle-hole symmetrization missing')
        # diagonalize nonhermitean effective Hamiltonian
        metric = self.metric
        eff_hmatr = metric.dot(self.num_bilinear_hamil(kx, ky, kz))
        es, tinv = eig(eff_hmatr)
        # order eigenvalues and eigenvectors by energy
        nbands = self.nbands
        order = list(range(2 * nbands))
        order.sort(key=lambda i: es[i].real, reverse=True)
        porder = order[:nbands]
        horder = order[:nbands-1:-1]
        ses = es.copy()
        ses[:nbands] = es[porder]
        ses[nbands:] = es[horder]
        stinv = tinv.copy()
        stinv[:, :nbands] = tinv[:, porder]
        stinv[:, nbands:] = tinv[:, horder]
        # compute paranorms
        stinvadj = stinv.transpose().conjugate()
        pnorms = np.diag(np.abs(stinvadj.dot(metric).dot(stinv)))
        # compute phases of largest entries of all eigenvectors
        maxrows1 = np.abs(tinv[:, :nbands]).argmax(axis=0)
        maxrows2 = np.abs(tinv[:, nbands:]).argmax(axis=0)
        maxels1 = tinv[maxrows1, range(nbands)]
        maxels2 = tinv[maxrows2, range(nbands, 2*nbands)]
        phases = np.zeros((2 * nbands,), dtype=complex)
        phases[:nbands] = maxels1 / np.abs(maxels1)
        phases[nbands:] = maxels2 / np.abs(maxels2)
        # gauge eigenvectors
        stinv *= phases.conj() / np.sqrt(pnorms)
        return ses, stinv

    def path_from_points(self, *points, isometric=False, basis='rcpr'):
        """
        Interpolate polygon chain between given points.

        Parameters
        ----------
        *points: numpy.ndarray
            Coordinates of the points specifying the polygon.
        isometric: bool, optional
            If False, the links between the points are all of the same length.
            If True, the length of the links are proportional to the distances
            between the points. This is similar to an arc length
            parametrization in mathematics. Default is False.
        basis: str
            Name of the basis in which the coordinates are given. For more
            details see `LinearSpinWave.transform_basis`.

        Returns
        -------
        path: function
            Maps [0, 1] to the polygon chain between the defined points.
        point_maps: list
            Inverse image of the given points.
        """
        n = len(points) - 1
        points = self.transform_basis(*points, basis=basis)

        if not isometric:
            def path(t):
                m = floor(t * n)
                if m == n:
                    m -= 1
                dt = n * t - m
                p1 = points[m]
                p2 = points[m + 1]
                return dt * (p2 - p1) + p1
            point_maps = [i / n for i in range(n + 1)]
        else:
            # distances to previous point -> length of each segment
            dists = [0]
            # distances to start along all previous segments (cumulative)
            pts_dist = [0]
            # total distance, equal to pts_dist[-1]
            tot_dist = 0
            for i in range(n):
                p1 = points[i]
                p2 = points[i + 1]
                dist = norm(p2 - p1)
                tot_dist += dist
                dists.append(dist)
                pts_dist.append(tot_dist)

            # binary search for sement at a relative position along the whole path, e.g. 50% along the way -> on which segment is this point?
            def path(t):
                # bounds for the segment id, min 0, at max one less then number of points
                lower = 0
                upper = n
                while upper - lower > 1: # while bounds are not equal, continue search
                    mid = floor((upper - lower) / 2) + lower # find segment near middle of bounds
                    q = pts_dist[mid] / tot_dist # cumulative distance up until this segment mapped to relative position, i.e. normalized to be between 0 and 1
                    if q < t: # distance is lower than desired relative position
                        lower = mid # adjust lower bound
                    else: # distance is higher
                        upper = mid # adjust upper bound

                # linear interpolation on the segment
                # tot_dist * t -> target distance along the path
                # pts_dist[lower] -> distance to start of current segment
                # dists[upper] -> length of current segment
                dt = (tot_dist * t - pts_dist[lower]) / dists[upper] # relative distance along current segment
                p1 = points[lower]
                p2 = points[upper]
                return dt * (p2 - p1) + p1 # interpolation formula
            
            #--------
            # example
            #--------

            # points = [A, B, C, D]
            # Distances:
            # A→B: 3 units
            # B→C: 5 units  
            # C→D: 2 units

            # dists = [0, 3, 5, 2]
            # pts_dist = [0, 3, 8, 10]  Cumulative: [0, 0+3, 3+5, 8+2]
            # tot_dist = 10

            # If t = 0.65 (65% along the path)
            # Total distance = 10
            # Target distance = 0.65 * 10 = 6.5 units

            # Binary search finds:
            # pts_dist[2] = 8 > 6.5  ✓ upper bound
            # pts_dist[1] = 3 < 6.5  ✓ lower bound
            # So the point is in segment B→C

            # Interpolation
            # Found segment B→C (starts at 3, length 5)
            # dt = (6.5 - 3) / 5 = 0.7  # 70% along the B→C segment

            point_maps = np.array(pts_dist) / tot_dist # normalize distances to [0,1]
        return path, point_maps

    def transform_basis(self, *points, basis='rcpr'):
        """Transform vector(s) into other basis."""
        if basis == 'rcpr':
            points = np.array(list(map(
                lambda p: np.dot(p, self.lat.num_rcpr_vects), points)))
        else:
            points = list(map(np.array, points))
        return points

    def energy_path(self, *points, num=100, isometric=False, basis='rcpr'):
        """Calculate energies along polygonal chain in k-space."""
        ts = np.linspace(0, 1, num=num)
        k_of_t, point_maps = self.path_from_points(
            *points, isometric=isometric, basis=basis)
        ks = [k_of_t(t) for t in ts]
        energs = np.array([self.bog_trafo(*k)[0][:self.nbands]
                           for k in ks])
        return ts, ks, point_maps, energs

    def set_num_magn_field(self, bx, by, bz):
        """Redefine numerical magnetic field."""
        if self.num_parameters is None:
            raise ValueError('Parameters have not been specified.')
        self.num_parameters['Bx'] = bx
        self.num_parameters['By'] = by
        self.num_parameters['Bz'] = bz

    def get_num_magn_field(self):
        """Return the numeric magnetic field or raise an error otherwise."""
        params = self.num_parameters
        try:
            bx, by, bz = params['Bx'], params['By'], params['Bz']
        except KeyError:
            raise ValueError('Numeric magnetic field not yet specified.')
        return bx, by, bz

    def change_magn_field(self, bx, by, bz, dirs=None):
        """Redefine numerical magnetic field (and ground state directions)."""
        self.set_num_magn_field(bx, by, bz)
        if dirs is not None:
            self.lat.change_grdstatedirs(dirs)

    def update_num_parameters(self, newparams):
        """Replace old parameters and update numerical attributes."""
        if self.num_parameters is None:
            self.num_parameters = newparams
        else:
            self.num_parameters.update(newparams)
        self.lat.update_num_parameters(newparams)
        self.mainbasis.parameterize()
        self.parameterize()

    def num_bphase_cancel_matr(self, *k):
        """Unitary and paraunitary matrix removing the basis phase factor."""
        num = self.nbands
        m = np.zeros((2*num, 2*num), dtype='complex')
        for i in range(num):
            b = self.lat.sublats[i].num_basisvect
            entry = np.exp(-1j * np.dot(k, b))
            m[i, i] = entry
            m[i+num, i+num] = entry
        return m

    def evgrid(self, kgrid, startpoint=True, endpoint=True, offset=0,
               gtype='center', global_gauge=True, decimals=5):
        """
        Calculate eigenvalues and -vectors on a grid of k points.

        Parameters
        ----------
        kgrid : int or tuple or numpy.ndarray
            Numbers of subdivisions along each reciprocal lattice vector. If it
            is a tuple, two entries are expected which subdivide along the two
            axes. If it is an integer, the number specifies the subdivisions of
            the shorter axis while the longer axis is subdivided such that the
            spacing of the grid points is approximately the same along both
            axes (i. e. the unitcell is subdivided into approximate rhombi).
            If it is a numpy array, the first index should give the components
            in k space, the second and thrid indices represent the rows and
            columns, respectively.
        startpoint : bool, optional
            It only has an effect if `kgrid` is not a numpy.ndarray. If True,
            the generated grid in k space does include the outer left border of
            the Brillouin zone. If the eigenvectors are used for calculating
            the Chern number, it needs to be True. Default is True.
        endpoint : bool, optional
            It only has an effect if `kgrid` is not a numpy.ndarray. If True,
            the generated grid in k space does include the outer right border
            of the Brillouin zone. If the eigenvectors are used for calculating
            the Chern number, it needs to be True. Default is True.
        offset : float or iterable of floats, optional
            See `parallelogram_grid`. Only has an effect if `kgrid` is not a
            numpy.ndarray. Default is 0.
        gtype : str, optional
            See `parallelogram_grid`. Only has an effect if `kgrid` is not a
            numpy.ndarray. Default is "center".
        global_gauge : bool, optional
            Whether the eigenvectors should have a defined gauge. This is
            necessary if eigenvectors should behave analytically, e.g., when
            computing Berry curvatures. Default is True.
        decimals : int, optional
            Number of decimals that should be taken to round during the gauge
            process. It only has an effect if `global_gauge` is set to True.
            Default is 5.

        Returns
        -------
        es : np.ndarray
            Energies of all bands (last index) for all points in k grid (first
            and second indices).
        ts : np.ndarray
            Eigenvectors (columns of the matrix defined by the third and fourth
            indices) for every point in the k grid (first and second indices).
            They are gauged such that at every k point the first element of all
            eigenvectors are real.
        """
        bands = self.nbands
        if isinstance(kgrid, (int, tuple)):
            v1, v2 = self.lat.num_rcpr_vects[:2]
            kxs, kys, kzs = parallelogram_grid(v1, v2, kgrid, startpoint,
                                               endpoint, offset, gtype)
        elif isinstance(kgrid, np.ndarray):
            kxs, kys, kzs = kgrid
        else:
            raise ValueError('invalid type for parameter `kgrid`')
        m, n = kxs.shape
        es = np.zeros((m, n, 2 * bands))
        ts = np.zeros((m, n, 2 * bands, 2 * bands), dtype='complex')
        for i in range(m):
            for j in range(n):
                kx, ky, kz = kxs[i, j], kys[i, j], kzs[i, j]
                e, t = self.bog_trafo(kx, ky, kz, global_gauge, decimals)
                es[i, j, :] = e
                ts[i, j, :, :] = t
        return es, ts

    def linkphases(self, ev1, ev2, cutoff=1e-4):
        """
        Calculate phaseshift between neighboring eigenfunctions for each band.

        Correspond to U_µ(k) in J. Phys. Soc. Jpn. 74, 1674 eq. (7).

        Parameters
        ----------
        ev1 : np.ndarray
            Eigenvector matrix at k; corresponds to the 'bra' state.
        ev2 : np.ndarray
            Eigenvector matrix at k + µ; corresponds to the 'ket' state.
        cutoff : float, optional
            Lower bound for the norm of the overlap below which the overlap is
            set to one, which corresponds to a phase change of zero. Then the
            states do not contribute to the Berry curvature.

        Returns:
        --------
        phases : np.ndarray
            Complex one-dimensional array corresponding to U(1) gauge field
            (phase factors of the overlap between two neighboring eigenstates
            in reciprocal space) for all bands.

        """
        metric = self.metric
        overlap = metric @ ev1.conj().T @ metric @ ev2
        diagol = np.diag(overlap)
        absol = np.abs(diagol)
        mask = absol > cutoff
        phases = np.ones((2 * self.nbands,), dtype=complex)
        phases[mask] = diagol[mask] / absol[mask]
        return phases

    def flux(self, link1, link2, link3, link4):
        """Calculate (Berry) flux through plaquette surrounded by links."""
        flux = -np.log(link1 * link2 / (link3 * link4))
        flux = flux.imag
        return flux

    def fluxdens(self, link1, link2, link3, link4, k1, k2, k3, k4):
        """
        Calculate (Berry) flux density through plaquette surrounded by links.

        Parameters:
        -----------
        link1 : np.ndarray
            Complex U(1) overlap between eigenfunctions at k1 and k2.
        link2 : np.ndarray
            Complex U(1) overlap between eigenfunctions at k2 and k3.
        link3 : np.ndarray
            Complex U(1) overlap between eigenfunctions at k4 and k3.
        link4 : np.ndarray
            Complex U(1) overlap between eigenfunctions at k1 and k4.
        k1 : np.ndarray
            Bloch vector k1 in reciprocal space.
        k2 : np.ndarray
            Bloch vector k2 in reciprocal space.
        k3 : np.ndarray
            Bloch vector k3 in reciprocal space.
        k4 : np.ndarray
            Bloch vector k4 in reciprocal space.

        Returns:
        --------
        fluxdens : np.ndarray
            Berry curvatures / flux densities of all bands.

        """
        flux = self.flux(link1, link2, link3, link4)
        area = area_quadrilateral(k1, k2, k3, k4)
        fluxdens = flux / area
        return fluxdens

    def flux_from_eigvects(self, ev1, ev2, ev3, ev4, olcutoff=1e-04):
        """
        Compute Berry flux from eigenvectors through quad.

        Parameters:
        -----------
        ev1 : np.ndarray
            Complex eigenvectors of Hamiltonian at k1.
        ev2: np.ndarray
            Complex eigenvectors of Hamiltonian at k2.
        ev3: np.ndarray
            Complex eigenvectors of Hamiltonian at k3.
        ev4: np.ndarray
            Complex eigenvectors of Hamiltonian at k4.
        olcutoff : float, optional
            Lower bound for the norm of the overlap below which the overlap is
            set to one, which corresponds to a phase change of zero. Then the
            states do not contribute to the Berry curvature. Default is 1e-04.

        Returns:
        --------
        flux : np.ndarray
            Array of Berry fluxes though plaquette with verices k1, k2, k3, and
            k4 for all bands.

        """
        link1 = self.linkphases(ev1, ev2, olcutoff)  # <n(k)|n(k+x)>
        link2 = self.linkphases(ev2, ev3, olcutoff)  # <n(k+x)|n(k+x+y)>
        link3 = self.linkphases(ev4, ev3, olcutoff)  # <n(k+y)|n(k+x+y)>
        link4 = self.linkphases(ev1, ev4, olcutoff)  # <n(k)|n(k+y)>
        flux = self.flux(link1, link2, link3, link4)
        return flux

    def fluxdens_from_eigvects(self, ev1, ev2, ev3, ev4, k1, k2, k3, k4,
                               olcutoff):
        """
        Calculate (Berry) flux density through plaquette surrounded by links.

        Parameters:
        -----------
        ev1 : np.ndarray
            Complex eigenvectors of Hamiltonian at k1.
        ev2: np.ndarray
            Complex eigenvectors of Hamiltonian at k2.
        ev3: np.ndarray
            Complex eigenvectors of Hamiltonian at k3.
        ev4: np.ndarray
            Complex eigenvectors of Hamiltonian at k4.
        k1 : np.ndarray
            Bloch vector k1 in reciprocal space.
        k2 : np.ndarray
            Bloch vector k2 in reciprocal space.
        k3 : np.ndarray
            Bloch vector k3 in reciprocal space.
        k4 : np.ndarray
            Bloch vector k4 in reciprocal space.
        olcutoff : float, optional
            Lower bound for the norm of the overlap below which the overlap is
            set to one, which corresponds to a phase change of zero. Then the
            states do not contribute to the Berry curvature. Default is 1e-04.

        Returns:
        --------
        fluxdens : np.ndarray
            Berry curvatures / flux densities of all bands.

        """
        flux = self.flux_from_eigvects(ev1, ev2, ev3, ev4, olcutoff)
        area = area_quadrilateral(k1, k2, k3, k4)
        fluxdens = flux / area
        return fluxdens

    def flux_from_nondeg_eigvects(self, e1, e2, e3, e4, ev1, ev2, ev3, ev4,
                                  olcutoff=1e-04, cutoff=1e-04, repby=0):
        """
        Compute Berry flux from eigenvectors through quad.

        Parameters:
        -----------
        e1 : np.ndarray
            Eigenvalues of Hamiltonian at k1.
        e2 : np.ndarray
            Eigenvalues of Hamiltonian at k2.
        e3 : np.ndarray
            Eigenvalues of Hamiltonian at k3.
        e4 : np.ndarray
            Eigenvalues of Hamiltonian at k4.
        ev1 : np.ndarray
            Complex eigenvectors of Hamiltonian at k1.
        ev2 : np.ndarray
            Complex eigenvectors of Hamiltonian at k2.
        ev3 : np.ndarray
            Complex eigenvectors of Hamiltonian at k3.
        ev4 : np.ndarray
            Complex eigenvectors of Hamiltonian at k4.
        olcutoff : float, optional
            Lower bound for the norm of the overlap below which the overlap is
            set to one, which corresponds to a phase change of zero. Then the
            states do not contribute to the Berry curvature. Default is 1e-04.
        cutoff : float, optional
            Absolute energy cutoff for the splitting of the eigenvalues below
            which the flux is set to zero. The band splittings are computed at
            all four Bloch vectors and it is sufficient for one to undercut the
            cutoff in order for the flux to be artificially set to zero. Note
            that the Berry flux is not well-defined for degenerate states.
            The splittings are only computed between particle-like bands. If
            None, degeneracies are not checked and noise may appear in case of
            degeneracies. Default is 1e-04.
        repby : float, optional
            Value of the Berry curvature at degenerate points. Default is 0.

        Returns:
        --------
        flux : np.ndarray
            Array of Berry fluxes though plaquette with verices k1, k2, k3, and
            k4 for all bands.

        """
        if cutoff is None:
            flux = self.flux_from_eigvects(ev1, ev2, ev3, ev4, olcutoff)
            return flux
        isdeg1 = self.count_degeneracies(e1, cutoff) > 0
        isdeg2 = self.count_degeneracies(e2, cutoff) > 0
        isdeg3 = self.count_degeneracies(e3, cutoff) > 0
        isdeg4 = self.count_degeneracies(e4, cutoff) > 0
        if isdeg1 or isdeg2 or isdeg3 or isdeg4:
            nbands = self.nbands
            flux = np.full((2 * nbands,), repby, dtype=float)
        else:
            flux = self.flux_from_eigvects(ev1, ev2, ev3, ev4, olcutoff)
        return flux

    def fluxdens_from_nondeg_eigvects(self, e1, e2, e3, e4, ev1, ev2, ev3, ev4,
                                      k1, k2, k3, k4, olcutoff=1e-04,
                                      cutoff=1e-04, repby=0):
        """
        Compute Berry flux from eigenvectors through quad.

        Parameters:
        -----------
        e1 : np.ndarray
            Eigenvalues of Hamiltonian at k1.
        e2 : np.ndarray
            Eigenvalues of Hamiltonian at k2.
        e3 : np.ndarray
            Eigenvalues of Hamiltonian at k3.
        e4 : np.ndarray
            Eigenvalues of Hamiltonian at k4.
        ev1 : np.ndarray
            Complex eigenvectors of Hamiltonian at k1.
        ev2 : np.ndarray
            Complex eigenvectors of Hamiltonian at k2.
        ev3 : np.ndarray
            Complex eigenvectors of Hamiltonian at k3.
        ev4 : np.ndarray
            Complex eigenvectors of Hamiltonian at k4.
        k1 : np.ndarray
            Bloch vector k1 in reciprocal space.
        k2 : np.ndarray
            Bloch vector k2 in reciprocal space.
        k3 : np.ndarray
            Bloch vector k3 in reciprocal space.
        k4 : np.ndarray
            Bloch vector k4 in reciprocal space.
        olcutoff : float, optional
            Lower bound for the norm of the overlap below which the overlap is
            set to one, which corresponds to a phase change of zero. Then the
            states do not contribute to the Berry curvature. Default is 1e-04.
        cutoff : float, optional
            Absolute energy cutoff for the splitting of the eigenvalues below
            which the flux is set to zero. The band splittings are computed at
            all four Bloch vectors and it is sufficient for one to undercut the
            cutoff in order for the flux to be artificially set to zero. Note
            that the Berry flux is not well-defined for degenerate states.
            The splittings are only computed between particle-like bands. If
            None, degeneracies are not checked and noise may appear in case of
            degeneracies. Default is 1e-04.
        repby : float, optional
            Value of the Berry curvature at degenerate points. Default is 0.

        Returns:
        --------
        fluxdens : np.ndarray
            Berry curvatures / flux densities of all bands.

        """
        args = (ev1, ev2, ev3, ev4, k1, k2, k3, k4, olcutoff)
        if cutoff is None:
            fluxdens = self.fluxdens_from_eigvects(*args)
            return fluxdens
        isdeg1 = self.count_degeneracies(e1, cutoff) > 0
        isdeg2 = self.count_degeneracies(e2, cutoff) > 0
        isdeg3 = self.count_degeneracies(e3, cutoff) > 0
        isdeg4 = self.count_degeneracies(e4, cutoff) > 0
        if isdeg1 or isdeg2 or isdeg3 or isdeg4:
            nbands = self.nbands
            fluxdens = np.full((2 * nbands,), repby, dtype=float)
        else:
            fluxdens = self.fluxdens_from_eigvects(*args)
        return fluxdens

    def count_degeneracies(self, es, cutoff=1e-04):
        """
        Count degeneracies between particle bands for a given set of energies.

        Parameters:
        -----------
        es : np.ndarray
            Eigenvalues of the Hamiltonian at an arbitrary Bloch vector.
        cutoff : float, optional
            Absolute energy cutoff for the splitting of the eigenvalues below
            which a degeneracy is counted. Default is 1e-04.

        Returns:
        --------
        ndegs : int
            Number of degeneracies between given bands.

        """
        nbands = self.nbands
        des = -np.diff(es[:nbands])
        isdeg = des < cutoff
        ndegs = isdeg.sum()
        return ndegs

    def ndeggrid(self, egrid, cutoff=1e-04):
        """
        Count degeneracies between particle bands for a given energy mesh.

        Parameters:
        -----------
        egrid : np.ndarray
            Eigenvalues of the Hamiltonian at an arbitrary Bloch vector.
        cutoff : float, optional
            Absolute energy cutoff for the splitting of the eigenvalues below
            which a degeneracy is counted. Default is 1e-04.

        Returns:
        --------
        ndeggrid : 2D np.ndarray
            Two-dimensional integer numpy array of degeneracy number.

        """
        ny, nx = egrid.shape[:-1]
        ndeggrid = np.zeros((ny, nx), dtype=int)
        for ix in range(nx):
            for iy in range(ny):
                es = egrid[iy, ix]
                ndegs = self.count_degeneracies(es, cutoff)
                ndeggrid[iy, ix] = ndegs
        return ndeggrid

    def has_global_bandgap(self, egrid, cutoff):
        """
        Evaluate if there are global gaps between the bands in Brillouin zone.

        Parameters:
        -----------
        egrid : np.ndarray
            3D array where first two indices correspond to the grid axes of the
            k mesh and the last index corresponds to the energies of all bands
            at the k grid points.
        cutoff : float
            Energy below which the bands are considered degenerate.

        Returns:
        --------
        has_deg : np.ndarray of type bool
            1D array that specifies whether the band n and n+1 have at least
            one degeneracy starting with the highest-energy pair.

        """
        minsplits = self.global_bandsplit(egrid)
        has_deg = minsplits < cutoff
        return has_deg

    def global_bandgap(self, egrid):
        """
        Compute minimal energy splitting between all bands sampled on a k grid.

        Parameters:
        -----------
        egrid : np.ndarray
            3D array where first two indices correspond to the grid axes of the
            k mesh and the last index corresponds to the energies of all bands
            at the k grid points.

        Returns:
        --------
        gaps : np.ndarray
            1D array that contains the minimal splitting between all pairs of
            adjacent bands starting with the highest in energy.

        """
        nbands = self.nbands
        egrid2 = np.roll(egrid, shift=-1, axis=-1)
        degrid = egrid - egrid2
        degrid = degrid[..., :nbands-1]
        gaps = degrid.min(axis=(0, 1))
        return gaps

    def curvature_grid(self, grid, kgrid=None, egrid=None, olcutoff=1e-04,
                       cutoff=None, repby=0):
        """
        Calculate curvature for eigenfunction in discrete parameter space.

        Method adopted from J. Phys. Soc. Jpn. 74, 1674 (for 2D).

        Parameters
        ----------
        grid : 4-dimensional numpy.ndarray
            Contains the diagonalizing matrices for a 2D grid in k space. First
            two indices reference the k points, and third, fourth index
            represent the rows, columns of the matrices, respectively.
        kgrid : 3-dimensional numpy.ndarray, optional
            Contains the points in k space at which the eigenvectors were
            calculated. First index corresponds to the component of k, second,
            and third index give the values of that component on each grid
            point. This is used to compute the Berry flux density, i. e., the
            real curvature instead of the Berry flux.
        egrid : 3-dimensional numpy.ndarray, optional
            Energies of all bands (last index) on two-dimensional grid in
            k-space. This is only needed if cutoff is defined and degeneracies
            are checked otherwise a ValueError will be raised.
        olcutoff : float, optional
            Lower bound for the norm of the overlap below which the overlap is
            set to one, which corresponds to a phase change of zero. Then the
            states do not contribute to the Berry curvature. Default is 1e-04.
        cutoff : float, optional
            Absolute energy cutoff for the splitting of the eigenvalues below
            which the Berry curvature is set to zero. The band splittings are
            computed at all four Bloch vectors and it is sufficient for one to
            undercut the cutoff in order for the flux to be artificially set to
            zero. Note that the Berry flux is not well-defined for degenerate
            states.  The splittings are only computed between particle-like
            bands. If None, degeneracies are not checked and noise may appear
            in case of degeneracies. Note that egrid needs to be set if cutoff
            is set. Default is None.
        repby : float, optional
            Value of the Berry curvature at degenerate points. Default is 0.

        Returns
        -------
        curvs : 3D numpy.ndarray
            Curvatures of all bands starting from the highest one in energy.
            First index corresponds to the band, second, and third indices
            represent different k points in the 2D grid.

        """
        if (egrid is None) and (cutoff is not None):
            msg = 'cutoff has been set, but no egrid given'
            raise ValueError(msg)
        if (egrid is not None) and (cutoff is None):
            msg = 'egrid has been set, but no cutoff defined'
            raise ValueError(msg)
        checkdegs = cutoff is not None
        m, n = grid.shape[:2]
        bands = grid.shape[-1]
        curvs = np.zeros((bands, m - 1, n - 1))
        for i in range(m - 1):
            for j in range(n - 1):
                ev1 = grid[i, j]  # |n(k)>
                ev2 = grid[i, j + 1]  # |n(k+x)>
                ev3 = grid[i + 1, j + 1]  # |n(k+x+y)>
                ev4 = grid[i + 1, j]  # |n(k+y)>
                if checkdegs:
                    e1 = egrid[i, j]  # e(k)
                    e2 = egrid[i, j + 1]  # e(k+x)
                    e3 = egrid[i + 1, j + 1]  # e(k+x+y)
                    e4 = egrid[i, j + 1]  # e(k+y)
                    args = (e1, e2, e3, e4, ev1, ev2, ev3, ev4, olcutoff,
                            cutoff, repby)
                    curvs[:, i, j] = self.flux_from_nondeg_eigvects(*args)
                else:
                    args = (ev1, ev2, ev3, ev4, olcutoff)
                    curvs[:, i, j] = self.flux_from_eigvects(*args)
                if kgrid is not None:
                    k1 = kgrid[:, i, j]
                    k2 = kgrid[:, i, j + 1]
                    k3 = kgrid[:, i + 1, j + 1]
                    k4 = kgrid[:, i + 1, j]
                    area = area_quadrilateral(k1, k2, k3, k4)
                    curvs[:, i, j] /= area
        return curvs

    def curvature(self, kx, ky, kz, dk=1e-5, unit='latt', olcutoff=1e-04,
                  cutoff=None, repby=0):
        """
        Compute Berry curvature by overlap method from eigenvectors.

        Parameters:
        -----------
        kx : float
            X component of the Bloch vector.
        ky : float
            Y component of the Bloch vector.
        kz : float
            Z component of the Bloch vector.
        dk : float, optional
            Size of finite step width in reciprocal space in measured in given
            units. Default is 1e-5.
        unit : str, optional
            Unit of the step size in reciprocal space. Options are `latt` for
            the length of the smaller reciprocal lattice vectors or `cart` for
            absolute units in 1 / length with length being the unit of length
            used to represent the lattice parameters.
        olcutoff : float, optional
            Lower bound for the norm of the overlap below which the overlap is
            set to one, which corresponds to a phase change of zero. Then the
            states do not contribute to the Berry curvature. Default is 1e-04.
        cutoff : float, optional
            Absolute energy cutoff for the splitting of the eigenvalues below
            which the flux is set to the value of repby. The band splittings
            are computed at all four Bloch vectors and it is sufficient for one
            to undercut the cutoff in order for the flux to be artificially set
            to repby. Note that the Berry flux is not well-defined for
            degenerate states. The splittings are only computed between
            particle-like bands. If None, degeneracies are not checked and
            noise may appear in case of degeneracies. Default is None.
        repby : float, optional
            Value of the Berry curvature at degenerate points. Default is 0.

        Returns:
        --------
        curvs : numpy.ndarray
            One-dimensional array with Berry curvatures of each band.

        """
        if unit == 'latt':
            v1, v2 = self.lat.num_rcpr_vects[:-1]
            minnorm = min(norm(v1), norm(v2))
            dk *= minnorm
        ex = np.array([dk, 0, 0])
        ey = np.array([0, dk, 0])
        k = np.array([kx, ky, kz])
        k1 = k - 0.5 * ex - 0.5 * ey
        k2 = k + 0.5 * ex - 0.5 * ey
        k3 = k + 0.5 * ex + 0.5 * ey
        k4 = k - 0.5 * ex + 0.5 * ey
        e1, ev1 = self.bog_trafo(*k1)  # |n(k)>
        e2, ev2 = self.bog_trafo(*k2)  # |n(k+x)>
        e3, ev3 = self.bog_trafo(*k3)  # |n(k+x+y)>
        e4, ev4 = self.bog_trafo(*k4)  # |n(k+y)>
        if cutoff is None:
            flux = self.flux_from_eigvects(ev1, ev2, ev3, ev4, olcutoff)
        else:
            args = (e1, e2, e3, e4, ev1, ev2, ev3, ev4, olcutoff, cutoff,
                    repby)
            flux = self.flux_from_nondeg_eigvects(*args)
        curvs = flux / dk ** 2
        return curvs

    def chern(self, grid):
        """
        Calculate Chern numbers for eigenfunction in discrete parameter space.

        Method adopted from J. Phys. Soc. Jpn. 74, 1674 (for 2D).

        Parameters
        ----------
        grid : 4-dimensional numpy.ndarray
            Contains the diagonalizing matrices for a 2D grid in k space. First
            two indices reference the k points, and third, fourth index
            represent the rows, columns of the matrices, respectively.

        Returns
        -------
        chern : 1D numpy.ndarray
            Chern numbers of all bands starting from the highest one in energy.

        """
        curvs = self.curvature_grid(grid)
        chern = -curvs.sum(axis=-1).sum(axis=-1) / (2 * np.pi)
        return chern

    def _berry(self, kx, ky, kz, gematr, tinv, tinvadj, vdir, tdir):
        n = self.nbands
        metric = self.metric
        old_hbar, self.hbar = self.hbar, 1
        vdirvmatr = self.get_velmatr(kx, ky, kz, tinv, tinvadj, vdir)
        tdirvmatr = self.get_velmatr(kx, ky, kz, tinv, tinvadj, tdir)
        self.hbar = old_hbar
        vdmatr = metric.dot(vdirvmatr)
        tdmatr = metric.dot(tdirvmatr)

        res = np.zeros((2 * n,))
        for i in range(2 * n):
            for j in range(2 * n):
                if i == j:
                    continue
                denom = (gematr[i, i] - gematr[j, j]) ** 2
                if denom < 1e-10:
                    # print(f'Warning: degeneracy at k = ({kx}, {ky}, {kz})')
                    continue
                res[i] += -2 * np.imag(vdmatr[i, j] * tdmatr[j, i]) / denom
        return res

    def berry(self, kx, ky, kz, mu, nu):
        es, tinv = self.bog_trafo(kx, ky, kz)
        tinvadj = tinv.transpose().conjugate()
        gematr = np.diag(es)
        return self._berry(kx, ky, kz, gematr, tinv, tinvadj, mu, nu)

    def num_per_sl(self, tinv):
        """
        Compute average number of spin flips per sublattice.

        Parameters:
        -----------
        tinv : np.ndarray
            Two-dimensional array of eigenvectors.

        Returns:
        --------
        res : np.ndarray
            Two-dimensional array where first index represents the band index
            and second index represents the sublattice index.

        """
        n = len(tinv) // 2
        res = np.zeros((n, n))
        res = np.abs(tinv[:n, :n]) ** 2 + np.abs(tinv[n:, :n]) ** 2
        res = np.transpose(res)
        return res

    def superlattice(self, new_sublats, new_vects):
        """
        Construct super lattice from this lattice.

        Parameters:
        -----------
        new_sublats : list of lists
            Contains the sublattices of the new super lattice. First index
            represents the number of the new sublattice, inner list contains
            number of sublattice and unit cell in old lattice.
        new_vects : list of lists
            Coefficients for the generation of new translation vectors. First
            index represents the new lattice vector and second indices
            represents the old lattice vectors.

        Return:
        -------
        spinw : LinearSpinWave
            Instance with the updated lattice vectors
        """
        lat = self.suplat_lattice(new_sublats, new_vects)
        basis = self.suplat_basis(lat, new_sublats, new_vects)
        spinw = LinearSpinWave(lat, basis, self.num_parameters)
        return spinw

    def suplat_sublats(self, new_sublats):
        num_parameters = self.num_parameters
        vects = self.lat.vects
        sublats = []
        old_sublats = self.lat.sublats
        for new_sublat in new_sublats:
            old_sublat_id, old_uc = new_sublat
            old_sublat = old_sublats[old_sublat_id]
            old_basisvect = old_sublat.basisvect
            basisvect = old_basisvect + vects.T @ sp.Matrix(old_uc)
            spinlen = old_sublat.spinlen
            gyromatr = old_sublat.gyromatr
            unitvects = old_sublat.unitvects
            sublat = MagneticSublattice(basisvect, spinlen, gyromatr,
                                        unitvects, num_parameters)
            sublats.append(sublat)
        return sublats

    def suplat_lattice(self, new_sublats, new_vects):
        """Instantiate `MagneticLattice` for slab."""
        sublats = self.suplat_sublats(new_sublats)
        old_vects = self.lat.vects
        vects = sp.Matrix(new_vects) @ old_vects
        num_parameters = self.num_parameters
        lat = MagneticLattice(vects, sublats, num_parameters)
        return lat

    def suplat_basis(self, lat, new_sublats, new_vects):
        """Instantiate `MainBasis` of the slab."""
        basis = MainBasis(lat)
        old_basis = self.mainbasis
        old_sublats = self.lat.sublats
        new_vects_c = sp.Matrix(new_vects) @ self.lat.vects
        convert = new_vects_c.transpose().inv()
        assert len(basis.atoms) == len(new_sublats)
        for atom, new_sublat in zip(basis.atoms, new_sublats):
            old_sublat, old_uc = new_sublat
            old_inters = [el
                          for l in old_basis.atoms[old_sublat].interactions
                          for el in l]
            for old_inter in old_inters:
                old_othersublat = old_inter.othersublat
                old_othervect = old_inter.othervect
                intermatr = old_inter.intermatr
                old_vect = self.lat.vects.T @ sp.Matrix(old_uc)
                otherpos = old_sublats[old_othersublat].basisvect
                otherpos += old_othervect + old_vect
                for i, onew_sublat in enumerate(lat.sublats):
                    # bv = old_sublats[onew_sublat[0]].basisvect
                    bv = onew_sublat.basisvect
                    diff = convert @ (otherpos - bv)
                    ii = all(map(lambda f: float(f).is_integer(), diff))
                    if ii:
                        othersublat = i
                        othervect = otherpos - bv
                        break
                else:
                    print("Interaction partner not identified")
                    #raise ValueError('Interaction partner not identified')
                atom.add_interaction(othersublat, othervect, intermatr)
        return basis

    def slab(self, dim, num):
        """
        Instantiate system with reduced periodicity.

        Parameters
        ----------
        dim : int
            Direction in which open boundary conditions are imposed. Should be
            0, 1, 2 specifying the translation vector along which the final
            system is finite.
        num : int
            Number of unit cell in the finite direction.

        Returns
        -------
        spinw : LinearSpinWave
            Finite system.

        """
        lat = self.lat.slab_lattice(dim, num)
        basis = self.mainbasis.slab_basis(lat, dim, num)
        spinw = LinearSpinWave(lat, basis, self.num_parameters.copy())
        return spinw


class InputProcessing:
    """Utility class for text file input."""

    comments = '#'
    splits = '->'
    metacomments = '$'
    metacomm_parse = dict(symmetrize=bool)

    @classmethod
    def init_nns_from_txtfile(cls, fname):
        with open(fname, 'r') as file:
            nns = file.readlines()
        nns = [eval(el) for el in nns]
        return nns

    @classmethod
    def init_from_str(cls, string, symmetrize='infer'):
        """Alias for 'cls.initialize_from_str'."""
        return cls.initialize_from_str(string, symmetrize)

    @classmethod
    def initialize_from_str(cls, string, symmetrize='infer'):
        """
        Parse input file and instantiate LinearSpinWave object.

        Parameters
        ----------
        string : str or list of lines
            The formatted input string or a list of its lines.
        symmetrize : bool or 'infer', optional
            If True, each interaction line adds a equivalent interaction with
            the reversed order of interacting spins
            (see MainBasis.add_interactions_from_str). If 'infer', the argument
            is set according the meta comments. The default is 'infer' or, if
            no meta comment exists, True.

        """
        # Group lines, parse metacomments, extract parameter values
        lines = cls.str_to_lines(string)
        lines, metacoms = cls.parse_metacomments(lines)
        if symmetrize == 'infer':
            symmetrize = cls.infer_symmetrize(metacoms)
        lat_lines, int_lines, params = cls.group_lines(lines)
        # print(f'lat_lines={lat_lines}', f'int_lines={int_lines}', f'params={params}', sep='\n')
        # Instantiate objects with information
        lat = MagneticLattice.from_str(lat_lines, params)
        basis = MainBasis(lat)
        basis.add_interactions_from_str(int_lines, symmetrize=symmetrize)
        spinw = LinearSpinWave(lat, basis, params)
        return spinw, lat, basis

    @classmethod
    def init_lat_from_str(cls, string):
        """
        Parse input file and instantiate MagneticLattice object.

        Parameters
        ----------
        string : str or list of lines
            The formatted input string or a list of its lines.

        """
        # Group lines, parse metacomments, extract parameter values
        lines = cls.str_to_lines(string)
        lines, metacoms = cls.parse_metacomments(lines)
        lat_lines, int_lines, params = cls.group_lines(lines)
        # Instantiate objects with information
        lat = MagneticLattice.from_str(lat_lines, params)
        return lat

    @classmethod
    def group_lines(cls, lines):
        """
        Group lines into list of lines used for building individual objects.

        Parameters
        ----------
        lines : list of lines
            List of lines generated from input file.

        Returns
        -------
        lat_lines : list of str
            List of lines used for initializing magnetic lattice.
        int_lines : list of str
            List of lines used for initializing spin interactions.
        params : dict
            Numerical parameter values used for numerical computations. If not
            given, None is returned.

        """
        # Remove comment lines
        lines = cls.remove_comlines(lines)
        # Identify lattice lines
        ind_lattice = lines.index('')
        try:
            # Identify parameterization lines
            ind_inter = lines.index('', ind_lattice + 1)
        except ValueError:
            # No parametrization given
            ind_inter = len(lines)
            params = None
        else:
            # Convert parameter lines to dict
            paramlines = lines[ind_inter+1:]
            params = {}
            for paramline in paramlines:
                if len(paramline) == 0:
                    continue
                symb, val = paramline.split(cls.splits)
                symb = symb.strip()
                val = float(sp.sympify(val))
                params[symb] = val
        lat_lines = lines[:ind_lattice]
        int_lines = lines[ind_lattice+1:ind_inter]
        return lat_lines, int_lines, params

    @classmethod
    def str_to_lines(cls, string):
        """Convert input string to lines."""
        # Convert `string` to lines
        if isinstance(string, str):
            if string[-1] == '\n':
                # Remove last new line character
                string = string[:-1]
            lines = string.split('\n')
        else:
            lines = string
        return lines

    @classmethod
    def remove_comlines(cls, lines):
        """Remove lines that start with comment character."""
        # Remove comment lines
        li = 0
        while li < len(lines):
            line = lines[li]
            if len(line) >= 1 and line[0] in cls.comments:
                del lines[li]
            else:
                li += 1
        return lines

    @classmethod
    def infer_symmetrize(cls, metacomm):
        """Infer `symmetrize` argument from metacomments."""
        try:
            symmetrize = metacomm['symmetrize']
        except KeyError:
            symmetrize = True
        return symmetrize

    @classmethod
    def parse_metacomments(cls, lines):
        """
        Extract informations from metacomments.

        Known metacomments are symmetrize (True/False).

        Parameters
        ----------
        lines : list of str
            The lines that do not need to contain only meta-data. Lines that
            do not begin with a meta-comment character (without whitespace) are
            ignored.

        Return
        ------
        lines_new : list of str
            A copy of the input argument without metacomments.
        metacomm : dict
            A dict containing the meta-data. The fields are:
            symmetrize : bool

        """
        lines = list(map(lambda line: line.strip(), lines))
        lines_new = []
        metacomm = {}
        for line in lines:
            # Look for metacomments
            if len(line) > 0 and line[0] in cls.metacomments:
                # Find key, val pair
                cols = line.split(cls.splits)
                if len(cols) != 2:
                    raise ValueError(f'Metacomments have the form key '
                                     f'{cls.splits} value.')
                    key, value = cols[0].lower().strip(), cols[0].strip()
                    if key not in cls.metacomments:
                        raise ValueError('Metacomment is unknown.')
                    metacomm[key] = cls.metacomm_parse(value)
            else:
                lines_new.append(line)
        return lines_new, metacomm

    @classmethod
    def init_from_txtfile(cls, filename, symmetrize='infer'):
        """Alias for 'cls.initialize_from_txtfile'."""
        return cls.initialize_from_txtfile(filename, symmetrize)

    @classmethod
    def initialize_from_txtfile(cls, filename, symmetrize='infer'):
        """
        Parse input file and Instantiate LinearSpinWave object.

        Parameters
        ----------
        filename : str
            The path to the input file.
        symmetrize : bool, optional
            If True, each interaction line adds a equivalent interaction with
            the reversed order of interacting spins (see
            MainBasis.add_interactions_from_str). If 'infer', the value is set
            according to the meta comments. The default is 'infer' or, if no
            metacomment exists, True.

        """
        with open(filename, 'r') as file:
            string = file.read()
            return cls.initialize_from_str(string, symmetrize)

    @classmethod
    def init_lat_from_txtfile(cls, filename):
        """
        Parse input file and Instantiate LinearSpinWave object.

        Parameters
        ----------
        filename : str
            The path to the input file.

        """
        with open(filename, 'r') as file:
            string = file.read()
            return cls.init_lat_from_str(string)

    @classmethod
    def save_as_str(cls, spinw, symmetrize=True):
        """
        Convert a LinearSpinWave object to a string.

        The generated string can be saved to a file and read in again with the
        method 'initialize_from_txtfile' of this class. The file is not created
        automatically.

        Parameters
        ----------
        spinw : LinearSpinWave
            The object that should be converted to a formatted string.
        symmetrize : bool, optional
            If True, equivalent interactions from identical pairs with
            reveresed order are filtered out
            (see MainBasis.add_interactions_from_str).

        """
        c = cls.comments[0]
        m = cls.metacomments[0]
        out = ''
        # Write metacomments
        out += '# metacomments\n'
        out += '# ------------\n'
        out += f'{m} symmetrize -> {symmetrize}\n'
        # Write translation vectors
        out += f'{c} translation vectors\n'
        out += f'{c} -------------------\n'
        out += cls._vects_as_str(spinw.lat)
        # Write sublattices
        out += f'{c} lattice\t|\tbasisvector\t|\tspinlength\t|\tgyromagnetic' \
               f' matrix\t\t\t\t|\tground state direction xor local ' \
               f'coordinate system\n'
        out += cls._sublats_as_str(spinw.lat)
        # Write interactions
        out += f'\n{c} interactions\n'
        out += f'{c} ------------\n'
        out += '# main lattice\t|\tother lattice\t|\tdifference vector\t|\t' \
               'interaction matrix\n'
        out += cls._interactions_as_str(spinw.mainbasis, symmetrize)
        # Write parameters (if available)
        if spinw.num_parameters is not None:
            out += f'\n{c} numerical values of symbols\n'
            out += f'{c} ---------------------------\n'
            out += cls._parameters_as_str(spinw.num_parameters)
        return out

    @classmethod
    def _vects_as_str(cls, lat):
        out = ''
        vects = (list(lat.vects.row(i)) for i in range(3))
        for vect in vects:
            out += f'{vect}\n'
        return out

    @classmethod
    def _sublats_as_str(cls, lat):
        out = ''
        for i, subl in enumerate(lat.sublats):
            # lattice
            out += f'{i + 1}\t\t|\t'
            # basisvector
            out += f'{list(lat.transf_to_latt(subl.basisvect))}\t|\t'
            # spin length
            out += f'{subl.spinlen}\t\t|\t'
            # gyromagnetic matrix
            out += f'{subl.gyromatr}\t|\t'
            # local coordinate system
            out += f'{list(subl.v)}\n'
        return out

    @classmethod
    def _interactions_as_str(cls, basis, symmetrize=True):
        lat = basis.lat
        skip = []
        out = ''
        for i, atom in enumerate(basis.atoms):
            others = (oatom for osubl in atom.interactions for oatom in osubl)
            for other in others:
                mlat = atom.sublat + 1
                olat = other.othersublat + 1
                diffv = lat.transf_to_latt(other.othervect)
                imatr = other.intermatr

                save_inter = True
                if symmetrize:
                    idict = dict(mlat=mlat, olat=olat, diffv=diffv,
                                 imatr=imatr, skipped=False)
                    try:
                        # Check if equivalent interaction was already captured
                        ind = skip.index(idict)
                    except ValueError:
                        # Interaction not yet captured or already skipped
                        odict = dict(mlat=olat, olat=mlat, diffv=-diffv,
                                     imatr=imatr.T, skipped=False)
                        skip.append(odict)
                    else:
                        # Interaction already captured and not yet skipped
                        # Update skipped flag
                        skip[ind]['skipped'] = True
                        save_inter = False
                if save_inter:
                    out += f'{mlat}\t\t|\t'
                    out += f'{olat}\t\t|\t'
                    out += f'{list(diffv)}\t\t|\t'
                    out += f'{imatr}\n'
        return out

    @classmethod
    def _parameters_as_str(cls, params):
        out = ''
        splits = cls.splits
        for key, val in params.items():
            out += f'{key} {splits} {val}\n'
        return out


class PolarCoord:
    """Utility methods for transformations involving spherical coordinates."""

    @classmethod
    def to_cartesian(cls, r, azimuth, polar):
        """Transform polar coordinates to cartisian coordinates."""
        return r * np.array([np.sin(polar) * np.cos(azimuth),
                             np.sin(polar) * np.sin(azimuth),
                             np.cos(polar)])

    @classmethod
    def to_polar(cls, x, y, z):
        """Transform cartisian to polar coordinates."""
        r = np.sqrt(x**2 + y**2 + z**2)
        if r == 0:
            return r, 0, 0
        phi = np.arctan2(y, x)
        theta = np.arccos(z / r)
        return r, phi, theta