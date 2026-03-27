from __future__ import annotations
from typing import Self, Any, Callable
from copy import deepcopy
import pandas as pd
import os
from itertools import chain
import numpy as np
import numpy.typing as npt
from numpy.linalg import cholesky, eigh, eig, inv, norm, det, LinAlgError
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import SymLogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from numba import jit, njit, prange

from tools.plot_lattice import _to_xy, _build_sites, _nearest_neighbour_dist, _draw_unit_cell


class Initialize:
    '''
    Utility class for initializing the calculation by instantiating a lattice, its atomic basis and its Hamiltonian from a setup file.
    '''
    
    @classmethod
    def from_csv(cls, csv_file: str) -> tuple[LinearSpinWave, Lattice, Basis]:
        '''
        Initialize by reading an input file in .csv format.
        
        Arguments:
        ----------
        csv_file: str
            Path to input file.

        Returns:
        --------
        lat: Lattice
            Instance of the Lattice class.
        basis: Basis
            Instance of the Basis class
        lsw: LinearSpinWave
            Instane of the LinearSpinWave class.
        '''
        data = pd.read_csv(csv_file, dtype=str)

        vects = cls.get_vects(data)
        spins = cls.get_spins(data)
        interactions = cls.get_interactions(data)
        params = cls.get_params(data)

        # print(f"transl_vects = {vects}", 
        #       f"spins = {spins}", 
        #       f"interactions = {interactions}", 
        #       f"params = {params}", 
        #       sep='\n')

        lat = Lattice.create(vects, spins, params)
        basis = Basis(lat)
        basis.set_interactions(interactions)
        lsw = LinearSpinWave(lat, basis, params)

        return lsw, lat, basis

    'TODO reading from other types of input files' 
    @classmethod
    def from_json(cls, json_file: str):
        '''
        Initialize by reading an input file in .json format.
        
        Arguments:
        ----------
        json_file: str
            Path to input file.
        '''
        data = pd.read_json(json_file)

        raise NotImplementedError('Will be added later.')

    @classmethod
    def from_txt(cls, txt_file: str):
        raise NotImplementedError('Will be added later.')

    @classmethod
    def from_dict(cls, dictionary: dict):
        raise NotImplementedError('Will be added later.')

    @classmethod
    def show_setup(cls):
        # print out the input file as a pandas dataframe
        # maybe showsetup=False as default argument of from_xyz and then if statement that prints data if True
        raise NotImplementedError('Will be added later.')

    @classmethod
    def get_vects(cls, data: pd.DataFrame) -> list[str]:
        '''
        Collects data of the translation vectors of a lattice.
        
        Arguments:
        ----------
        data: pd.DataFrame
            pandas DataFrame object containing all data given by the setup file.

        Return:
        -------
        list[str]
            List of translation vector arrays in string format.
        '''
        return data['#translation vectors'].dropna().to_list() # maybe convert to numpy arrays of float32 dtype to increase performance 

    @classmethod
    def get_spins(cls, data: pd.DataFrame) -> list[dict[str, float | str]]:
        '''
        Collects data of all spins inside the unit cell of a lattice and stores it in a list. Each spin has information about its sublattice index, basis vector, ground state direction and spin length, which is stored in a dictionary.
        
        Arguments:
        ----------
        data: pd.DataFrame
            pandas DataFrame object containing all data given by the setup file.

        Return:
        -------
        list[dict[str, float | str]]
            List of dictionaries corresponding to each spin, containing key:value pairs of their data.
        '''
        return data[['sublattice', 
                      'basis vector', 
                      'ground state direction', 
                      'spin length', 
                      'gyromagnetic matrix']].dropna().to_dict('records')

    @classmethod
    def get_interactions(cls, data: pd.DataFrame) -> list[dict[str, float | str]]:
        '''
        Collects data of all interactions between spins on sublattice sites and stores it in a list. Each interaction has information about the a sublattice inside the reference unit cell, a neighboring sublattice inside the same or an adjacent unit cell, the difference vector between their unit cells (zero vector if both inside the reference cell) and the interaction matrix. The information is stored in a dictionary.
        
        Arguments:
        ----------
        data: pd.DataFrame
            pandas DataFrame object containing all data given by the setup file.

        Return:
        -------
        list[dict[str, float | str]]
            List of dictionaries corresponding to each interaction, containing key:value pairs of their data.
        '''
        # TODO filter for interaction types by finding their starting index (i.e. grouping them)
        return data[['reference sublat', 
                      'neighbor sublat', 
                      'difference vector', 
                      'interaction matrix']].dropna().to_dict('records')

    @classmethod
    def get_params(cls, data: pd.DataFrame) -> dict[str, float]:
        '''
        Collects data of all paramteters.
        
        Arguments:
        ----------
        data: pd.DataFrame
            pandas DataFrame object containing all data given by the setup file.

        Return:
        -------
        dict[str, float]
            Dictionary with the parameter key strings and their values as floats.
        '''
        return data[['#parameters', 'value']].dropna().set_index('#parameters')['value'].to_dict()



class Lattice:
    '''
    Instantiates a lattice object that contains all information regarding the lattice translation vectors, spins inside the unit cell etc. and calculates related quantities, e.g. reciprocal lattice vectors.
    '''
    def __init__(self, 
                 transl_vects: sp.Matrix, 
                 sublats: list[Sublattice], 
                 parameters: dict[str, str] = None) -> None:
        '''
        Arguments:
        ----------
        transl_vects: sp.Matrix
            sympy Matrix object representing the translation vectors of the lattice.
        sublats: list[Sublattice]
            List of Sublattice objects.
        parameters: dict[str, float]
            Dictionary of parameter:value pairs extracted from the setup file.
        '''
        # declare class attributes as copies to avoid overriding the original input data
        self.vects = transl_vects.copy()
        self.sublats = sublats.copy()
        self.n_sublats = len(sublats)

        # calculate reciprocal lattice vectors
        self.rcpr_vects = None
        self.get_rcpr_vects()

        # extract numerical parameters from their symbolic representation TODO maybe improve this system?
        self.parameters = parameters
        self.num_vects = None
        self.num_rcpr_vects = None
        self.parameterize()

    @classmethod
    def create(cls, 
              transl_vects: list[str], 
              spin_data: list[dict[str, float | str]], 
              parameters: dict[str, str] = None) -> Self:
        '''
        Instantiates a lattice object from the data extracted from the setup file.
        
        Arguments:
        ----------
        transl_vects: list[str]
            List of translation vector arrays in string format.
        spin_data: list[dict[str, float | str]]
            List of dictionaries corresponding to each spin, containing key:value pairs of their data.
        parameters: dict[str, float]
            Dictionary of parameter:value pairs extracted from the setup file.

        Return:
        -------
        cls(...): Self
            Lattice class instance.
        '''
        # translation vectors: list of row vectors -> sympy Matrix object
        vects = [[sp.sympify(x) for x in vec.strip('[]').split(',')] 
                     for vec in transl_vects]
        vects = sp.Matrix(vects)

        # spin data -> sublattice objects
        n_sublats = len(spin_data)
        sublats = [None] * n_sublats

        for id, spin in enumerate(spin_data):
            if id != int(spin['sublattice'])-1: 
                raise ValueError(f"Index {id} does not match {int(spin['sublattice'])-1}!") #TODO remove if everything works

            # check if the sublattice already exists
            if sublats[id] is not None:
                raise ValueError(f"Sublattice (id={int(spin['sublattice'])}) has been defined more than once.")
            
            # extract the basis vector and check for correct dimensions
            basisvect = sp.Matrix(sp.sympify(spin['basis vector']))
            if basisvect.shape != (3, 1):
                raise ValueError(f'Basis vector (id={int(spin['sublattice'])}) is not 3-dimensional.')
            basisvect = vects.T * basisvect

            # extract the ground state
            grdstatedir = sp.Matrix(sp.sympify(spin['ground state direction']))
            if grdstatedir.shape not in [(3,3), (3,1)]:
                raise ValueError(f"Ground state direction must be given either as a unit vector (shape (3,1)) in the global coordinate system of all spins, or as a tripod (shape (3,3)) of a local coordinate system of the spin specific to the sublattice (id={int(spin['sublattice'])}).")
            
            # extract the spin length
            spinlen = sp.sympify(spin['spin length']) # TODO build in something such that it works if the parameter is just called S -> problem with sympy

            # extract gyromagnetic factor
            gyromat = sp.Matrix(sp.sympify(spin['gyromagnetic matrix']))
            if gyromat.shape != (3,3):
                raise ValueError(f"Gyromagnetic matrix of sublattice {int(spin['sublattice'])} is not of shape (3,3)")

            # create a sublattice instance
            sublats[id] = Sublattice.create(basisvect, grdstatedir, spinlen, gyromat, parameters)

        if None in sublats:
            raise ValueError(f"Less sublattices specified than expected ({n_sublats}).")
        
        return cls(vects, sublats, parameters)

    def copy(self) -> Self:
        '''
        Return a copy of the lattice instance. More specifically a deepcopy that not only copies the outer object (lattice), but also creates a copy of the inner objects with their own unique adresses in memory, preventing the originals from getting overwritten. This is necessary since a lattice object is a nested data structure containing mutable objects.
        
        Return:
        -------
        deepcopy(self): Self
            Lattice class instance.
        '''
        return deepcopy(self)
    
    def parameterize(self) -> None:
        '''
        Creates numerical copies of the class attributes that are given in symbolic form.
        '''
        params = self.parameters
        if params is None:
            raise ValueError('Parameters have not been specified.')
        
        self.num_vects = np.array(self.vects.subs(params), dtype=np.float64)
        self.num_rcpr_vects = np.array(self.rcpr_vects.subs(params), dtype=np.float64)

    def update_parameters(self, newparams: dict[str, float]) -> None:
        '''
        Updates and replaces the old parameter set with a new one and reparameterizes the numerical class attributes as well as the attributes of the sublattice objects.
        
        Arguments:
        ----------
        newparams: dict[str, float]
            Dictionary of parameter:value pairs from a new set of parameters.
        '''
        self.parameters.update(newparams)
        self.parameterize()

        for sublat in self.sublats:
            sublat.update_parameters(newparams)
        
    def get_rcpr_vects(self) -> None:
        '''
        Calculate the reciprocal lattice vectors from the real space lattice vectors.
        '''
        # extract translation vectors from matrix
        t1 = self.vects[0,:]
        t2 = self.vects[1,:]
        t3 = self.vects[2,:]

        # unit cell volume
        vol = t1.dot(t2.cross(t3)) 

        # calculate rcpr_vects
        const = 2 * sp.pi / vol
        g1 = const * t2.cross(t3)
        g2 = const * t3.cross(t1)
        g3 = const * t1.cross(t2)
        self.rcpr_vects = sp.Matrix([g1, g2, g3])

    def to_cartesian_basis(self, vect: sp.Matrix) -> sp.Matrix:
        '''
        Transform a vector given in the lattice vector basis to the cartesian basis.
        
        Arguments:
        ----------
        vect: list[float] NOTE check for correct type
            Vector in lattice vector basis.

        Return:
        -------
        sp.Matrix
            Vector in cartesian basis.
        '''
        return self.vects.T * sp.Matrix(vect)
    
    def to_lattice_basis(self, vect: sp.Matrix) -> sp.Matrix:
        '''
        Transform a vector given in the cartesian basis to the lattice vector basis.
        
        Arguments:
        ----------
        vect: sp.Matrix
            Vector in cartesian basis.

        Return:
        -------
        sp.Matrix
            Vector in lattice basis.
        '''
        return self.rcpr_vects * sp.Matrix(vect) / (2 * sp.pi)
    
    def slab_lattice(self, dim: int, num: int) -> SlabLattice:
        '''
        TODO
        '''
        slab_sublats: list[SlabSublattice] = self.slab_sublattices(dim, num)
        slab_translvects = self.vects
        params = self.parameters.copy()
        slab_lat = SlabLattice(slab_translvects, slab_sublats, dim, num, params)
        return slab_lat

    def slab_sublattices(self, dim: int, num: int) -> list[SlabSublattice]:
        '''
        TODO
        '''
        params = self.parameters.copy()
        slab_vects = self.vects
        slab_sublats = []

        for unit_cell_id in range(num):
            for bulk_site_id, bulk_sublat in enumerate(self.sublats):
                bulk_basisvect = bulk_sublat.basisvect
                slab_basisvect = bulk_basisvect + unit_cell_id * slab_vects.row(dim).T
                slab_local_tripod = bulk_sublat.local_tripod
                slab_spinlen = bulk_sublat.spinlen
                slab_gyromat = bulk_sublat.gyromat

                unit_cell_3Dpos = np.array([unit_cell_id if i == dim else 0 for i in range(3)])

                slab_sublat = SlabSublattice(slab_basisvect, 
                                        slab_local_tripod, 
                                        slab_spinlen,
                                        slab_gyromat,
                                        unit_cell_3Dpos, 
                                        bulk_site_id, 
                                        params)
                
                slab_sublats.append(slab_sublat)

        return slab_sublats
                


class Sublattice:
    '''
    Instantiates a sublattice object as part of a lattice object.
    '''
    def __init__(self, 
                 basisvect: sp.Matrix, 
                 local_tripod: sp.Matrix, 
                 spinlen: sp.Symbol,
                 gyromat: sp.Matrix, 
                 parameters: dict[str, str] = None) -> None:
        '''
        Arguments:
        ----------
        basisvect: sp.Matrix
            sympy Matrix object representing the column vector that points to the position of the sublattice in the unit cell (basis).
        grdstatedir: sp.Matrix
            sympy Matrix object representing the ground state direction as a column vector.
        spinlen: sp.Symbol
            sympy symbolic representation of the spin length parameter.
        gyromat: sp.Matrix
            Gyromagnetic matrix of the spin on this sublattice.
        parameters: dict[str, float]
            Dictionary of parameter:value pairs extracted from the setup file.
        '''
        # declare class attributes as copies to avoid overriding the original input data
        self.basisvect = basisvect.copy()
        self.local_tripod = local_tripod.copy()
        self.spinlen = spinlen # TODO check if a .copy() is needed here, too
        self.gyromat = gyromat.copy()

        # create two auxilliary vectors from the spin direction
        self.u_vec = local_tripod.row(0).T + sp.I * local_tripod.row(1).T   # sp.I is the imaginary unit
        self.v_vec = local_tripod.row(2).T

        # extract numerical parameters from their symbolic representation TODO maybe improve this system?
        self.parameters = parameters
        self.num_basisvect = None
        self.num_tripod = None
        self.num_spinlen = None
        self.num_gyromat = None
        self.num_u = None
        self.num_v = None
        self.parameterize()

    @classmethod
    def create(cls, 
              basisvect: sp.Matrix, 
              grdstatedir: sp.Matrix, 
              spinlen: sp.Symbol,
              gyromat: sp.Matrix, 
              parameters: dict[str, str] = None) -> Self:
        '''
        Instantiates a sublattice object to be used as part of the lattice object. Has information about the basis vector, local coordinate system, spinlength and parameters.
        
        Arguments:
        ----------
        basisvect: sp.Matrix
            sympy Matrix object representing the column vector that points to the position of the sublattice in the unit cell (basis).
        grdstatedir: sp.Matrix
            sympy Matrix object representing the ground state direction as a column vector.
        spinlen: sp.Symbol
            sympy symbolic representation of the spin length parameter.
        gyromat: sp.Matrix
            Gyromagnetic matrix of the spin on this sublattice.
        parameters: dict[str, float]
            Dictionary of parameter:value pairs extracted from the setup file.

        Return:
        -------
        cls(...): Self
            Sublattice class instance.
        '''
        #TODO if local_tripod is given only as a 1D grdstatedir, construct a local tripod
        if grdstatedir.shape == (3,1):
            local_tripod = cls.construct_local_tripod(grdstatedir)
        else:
            local_tripod = grdstatedir

        return cls(basisvect, local_tripod, spinlen, gyromat, parameters)

    @staticmethod
    def construct_local_tripod(grdstatedir: sp.Matrix) -> sp.Matrix:
        '''
        Constructs a tripod of a local coordinate system for a sublattice if the spins ground state direction is given as a single vector in the global coordinate system of the lattice.
        
        Arguments:
        ----------
        grdstatedir: sp.Matrix
            sympy Matrix object representing the ground state direction as a column vector.

        Return:
        -------
        sp.Matrix
            sympy Matrix object representing the local tripod with the x,y,z-axes as the rows of the matrix.
        '''
        norm = grdstatedir.norm()
        if norm != 1: grdstatedir /= norm
        if grdstatedir[0] != 0:
            xdir = sp.Matrix([grdstatedir[1], -grdstatedir[0], 0]).normalized()
        elif grdstatedir[1] != 0:
            xdir = sp.Matrix([0, grdstatedir[2], -grdstatedir[1]]).normalized()
        else:  # elif grdstatedir[2] != 0
            xdir = sp.Matrix([grdstatedir[2], 0, -grdstatedir[0]]).normalized()
        ydir = grdstatedir.cross(xdir).normalized()

        return sp.Matrix([xdir.T, ydir.T, grdstatedir.T])

    def parameterize(self) -> None:
        '''
        Creates numerical copies of the class attributes that are given in symbolic form.
        '''
        params = self.parameters
        if params is None:
            raise ValueError('Parameters have not been specified.')
        
        self.num_basisvect = np.array(self.basisvect.T.subs(params), dtype=np.float64)[0]
        self.num_tripod = np.array(self.local_tripod.T.subs(params), dtype=np.float64)
        self.num_spinlen = np.array(self.spinlen.subs(params), dtype=np.float64)
        self.num_gyromat = np.array(self.gyromat.subs(params), dtype=np.float64)
        self.num_u = np.array(self.u_vec.T.subs(params), dtype=np.complex128)[0]
        self.num_v = np.array(self.v_vec.T.subs(params), dtype=np.complex128)[0]

    def update_parameters(self, newparams: dict[str, str]) -> None:
        '''
        TODO reconsider datatype of values in parameter dictionary: reading from dataframe defaults to str, but this could be annoying when updating params

        Updates and replaces the old parameter set with a new one and reparameterizes the numerical class attributes.
        
        Arguments:
        ----------
        newparams: dict[str, float]
            Dictionary of parameter:value pairs from a new set of parameters.
        '''
        self.parameters.update(newparams)
        self.parameterize()



class SlabLattice(Lattice):
    '''
    TODO
    '''
    def __init__(self, 
                 transl_vects: sp.Matrix, 
                 slab_sublats: list[SlabSublattice], 
                 dim: int, 
                 num: int, 
                 parameters: dict[str, str] = None) -> None:
        '''
        TODO
        '''
        self.bulk_vects = transl_vects
        self.dim = dim
        self.num = num

        vects = transl_vects.copy()
        id1, id2 = (dim + 1) % 3, (dim + 2) % 3
        v1, v2 = vects[id1, :], vects[id2, :]
        v3 = v1.cross(v2)
        if id1 < id2:
            vects[0, :] = v1
            vects[1, :] = v2
            vects[2, :] = v3
            self.indexmap_to_slab = [-1, -1, -1]
            self.indexmap_to_slab[id1] = 0
            self.indexmap_to_slab[id2] = 1
            self.indexmap_to_slab[dim] = 2
        else:
            vects[0, :] = v2
            vects[1, :] = v1
            vects[2, :] = - v3
            self.indexmap_to_slab = [-1, -1, -1]
            self.indexmap_to_slab[id1] = 1
            self.indexmap_to_slab[id2] = 0
            self.indexmap_to_slab[dim] = 2

        super().__init__(vects, slab_sublats, parameters)

    def is_in_slab(self, bulk_unit_cell) -> bool:
        '''
        TODO
        '''
        is_in = (0 <= bulk_unit_cell[self.dim] < self.num)
        return is_in

    def to_slab_coords(self, bulk_unit_cell, bulk_site_id: int) -> tuple[npt.NDArray, int]:
        '''
        TODO
        '''
        if not self.is_in_slab(bulk_unit_cell):
            raise ValueError("Given unit cell is outside of slab.")

        # rearrange unit cell coords and set finite direction to zero
        idmap = self.indexmap_to_slab
        slab_unit_cell = np.array([-1, -1, -1])
        for i in range(3):
            slab_unit_cell[idmap[i]] = bulk_unit_cell[i]
        slab_unit_cell[-1] = 0

        # iterate over all slab sublats until match is found
        bulk_uc_comp = bulk_unit_cell[self.dim]
        for i, sublat in enumerate(self.sublats):
            match = sublat.unit_cell_3Dpos[self.dim]
            if match == bulk_uc_comp and sublat.bulk_site_id == bulk_site_id:
                return slab_unit_cell, i
        
        raise RuntimeError(f'Site ({bulk_unit_cell}, {bulk_site_id}) not found.')
    
    def to_extended_coords(self, unit_cell_3Dpos_in: tuple | npt.NDArray, slab_site_id: int) -> tuple[npt.NDArray, int]:
        '''
        TODO
        '''
        slab_sublat = self.sublats[slab_site_id]
        idmap = self.indexmap_to_slab
        bulk_unit_cell = np.array([unit_cell_3Dpos_in[idmap[i]] + j 
                                   for i, j in enumerate(slab_sublat.unit_cell_3Dpos)])
        bulk_site_id = slab_sublat.bulk_site_id
        
        return bulk_unit_cell, bulk_site_id



class SlabSublattice(Sublattice):
    '''
    TODO
    '''
    def __init__(self, 
                 basisvect: sp.Matrix, 
                 local_tripod: sp.Matrix, 
                 spinlen: sp.Symbol,
                 gyromat: sp.Matrix,
                 unit_cell_3Dpos: npt.NDArray, 
                 bulk_site_id: int,
                 parameters: dict[str, str] = None) -> None:
        '''
        TODO
        '''
        self.unit_cell_3Dpos = unit_cell_3Dpos
        self.bulk_site_id = bulk_site_id

        super().__init__(basisvect, local_tripod, spinlen, gyromat, parameters)



class Basis:
    '''
    Stores the interactions between atoms on the lattice.
    '''
    def __init__(self, lat: Lattice) -> None:
        '''
        Arguments:
        ----------
        lat: Lattice
            Instance of the Lattice class.
        '''
        self.lat = lat
        atoms = [None] * lat.n_sublats
        for id in range(lat.n_sublats):
            atoms[id] = Atom(lat, id)
        self.atoms: list[Atom] = atoms

    def copy(self) -> Self:
        '''
        Returns a copy of the current Basis instance.
        
        Return:
        -------
        deepcopy(self): Self
            Basis class instance.
        '''
        return deepcopy(self)
    
    def set_interactions(self, interactions: list[dict[str, float | str]], symmetrize: bool = True) -> None:
        '''
        Set up the interactions between atoms.

        Arguments:
        ----------
        interactions: list[dict[str, float | str]]
            List of dictionaries containing the information that describes the spin-spin interaction between two atoms.
        symmetrize: bool
            If true, for each specified interaction between a ordered pair of
            atoms (i, j) a second equivalent between the reverse-ordered pair
            (j, i) is added. The default is True.
        '''
        for inter in interactions:
            ref_sublat = int(inter['reference sublat']) - 1
            nbr_sublat = int(inter['neighbor sublat']) - 1
            diffvect = self.lat.to_cartesian_basis(sp.sympify(inter['difference vector']))
            intmat = sp.Matrix(sp.sympify(inter['interaction matrix']))

            self.atoms[ref_sublat].add_interaction(nbr_sublat, diffvect, intmat)

            if symmetrize:
                self.atoms[nbr_sublat].add_interaction(ref_sublat, -diffvect, intmat.T)

    def slab_basis(self, slab_lat: SlabLattice, dim: int, num: int) -> Basis:
        '''
        TODO
        '''
        slab_basis = Basis(slab_lat)
        bulk_basis = self

        for atom in slab_basis.atoms:
            bulk_unit_cell, bulk_site_id = slab_lat.to_extended_coords((0,0,0), atom.ref_sublat)
            bulk_inter = [el for l in bulk_basis.atoms[bulk_site_id].interactions for el in l] # list(chain.from_iterable(bulk_basis.atoms[bulk_site_id].interactions)) 

            for inter in bulk_inter:
                bulk_nbr_sublat = inter.nbr_sublat
                bulk_diffvect = inter.diffvect
                bulk_dist_to_uc = np.array(self.lat.to_lattice_basis(bulk_diffvect).T)[0]
                bulk_nbr_uc = bulk_dist_to_uc + bulk_unit_cell
                intmat = inter.intmat
                try:
                    slab_nbr_uc, slab_nbr_sublat = slab_lat.to_slab_coords(bulk_nbr_uc, bulk_nbr_sublat)
                except ValueError:
                    # interaction partner is outside the slab
                    continue
                slab_diffvect = slab_lat.to_cartesian_basis(slab_nbr_uc) # slab_diffvect = bulk_diffvect
                atom.add_interaction(slab_nbr_sublat, slab_diffvect, intmat)

        return slab_basis



class Atom:
    '''
    Stores the interaction one specific atom has with any amount of the other atoms.
    '''
    def __init__(self, lat: Lattice, sublat_id: int):
        '''
        Arguments:
        ----------
        lat: Lattice
            Instance from the Lattice class.
        sublat_id: int
            Index of a sublattice inside the unit cell.
        '''
        self.lat = lat
        self.ref_sublat = sublat_id
        n_sublats = lat.n_sublats
        self.interactions: list[list[Interaction]] = [[] for _ in range(n_sublats)]

    def add_interaction(self, nbr_sublat: int, diffvect: sp.Matrix, intmat: sp.Matrix) -> None:
        '''
        Set the interaction the reference atom has with some neighboring atom (or itself).
        
        Arguments:
        ----------
        nbr_sublat: int
            Index of the neighbor atom.
        diffvect: sp.Matrix
            Difference vector  between reference and neighbor sublattices in cartesian coordinates.
        intmat: sp.Matrix
            Interaction matrix of the specific interaction.
        '''
        inter = Interaction(self.lat, self.ref_sublat, nbr_sublat, diffvect, intmat)
        self.interactions[nbr_sublat].append(inter)

    def sum_intmat(self, 
                   k: sp.Matrix, 
                   nbr_sublat: int,
                   inv_phase:bool = False) -> sp.Matrix:
        '''
        Returns the Fourier-transformed interaction matrix between two atoms in symbolic form: sum_r e^{i k r} I(r). r iterates over the atom positins of the sublattice corresponding to 'nbr_sublat'

        Arguments:
        ----------
        k: sp.Matrix
            k-point at which the interaction matrix will be evaluated.
        nbr_sublat: int
            Index of the sublattice that the reference sublattice interacts with. Can be a neighbor or the reference itself.
        inv_phase: bool
            Sign of the phase. If True, the phase will be negative.

        Return:
        -------
        sp.Matrix
            Fourier-transformed interaction matrix.
        '''
        res = sp.zeros(3, 3)
        for inter in self.interactions[nbr_sublat]:
            res += inter.get_intmat(k, inv_phase)
        return res
    
    def sum_num_intmat(self,
                       k: npt.NDArray[np.float64],
                       nbr_sublat: int,
                       inv_phase: bool = False) -> npt.NDArray[np.complex128]:
        '''
        Returns the Fourier-transformed interaction matrix between two atoms in numeric form.

        Arguments:
        ----------
        k: npt.NDArray[np.float64]
            k-point at which the interaction matrix will be evaluated.
        nbr_sublat: int
            Index of the sublattice that the reference sublattice interacts with. Can be a neighbor or the reference itself.
        inv_phase: bool
            Sign of the phase. If True, the phase will be negative.

        Return:
        -------
        sp.Matrix
            Fourier-transformed interaction matrix.
        '''
        res = np.zeros((3, 3), dtype='complex')
        for inter in self.interactions[nbr_sublat]:
            res += inter.get_num_intmat(k, inv_phase)
        return res
    
    def sum_intmat_k_zero(self, nbr_sublat: int) -> sp.Matrix:
        '''
        Returns the interaction matrix in the case that k is the zero-vector.

        Arguments:
        ----------
        nbr_sublat: int
            Index of the sublattice that the reference sublattice interacts with. Can be a neighbor or the reference itself.

        Return:
        -------
        sp.Matrix
            Fourier-transformed interaction matrix.
        '''
        return self.sum_intmat(sp.Matrix([0,0,0]), nbr_sublat)
    
    def sum_num_intmat_k_zero(self, nbr_sublat: int) -> npt.NDArray[np.complex128]: 
        '''
        Returns the interaction matrix in the case that k is the zero-vector.

        Arguments:
        ----------
        nbr_sublat: int
            Index of the sublattice that the reference sublattice interacts with. Can be a neighbor or the reference itself.

        Return:
        -------
        sp.Matrix
            Fourier-transformed interaction matrix.
        '''
        return self.sum_num_intmat(np.array([0,0,0]), nbr_sublat)



class Interaction:
    '''
    Stores information about a bilinear spin-spin interaction.
    '''
    def __init__(self, 
                 lat: Lattice, 
                 ref_sublat: int, 
                 nbr_sublat: int, 
                 diffvect: sp.Matrix, 
                 intmat: sp.Matrix) -> None:
        '''
        Arguments:
        ----------
        lat: Lattice
            Instance from the Lattice class.
        ref_sublat: int
            Index of the reference sublattice.
        nbr_sublat: int
            Index of the neighbor sublattice.
        diffvect: sp.Matrix
            Difference vector between reference and neighbor sublattices in cartesian coordinates.
        intmat: sp.Matrix
            Interaction matrix of the specific interaction.
        '''
        self.lat = lat
        self.ref_sublat = ref_sublat
        self.nbr_sublat = nbr_sublat
        self.diffvect = diffvect.copy()
        self.intmat = intmat.copy()
        self.num_diffvect = None
        self.num_intmat = None
        self.parameterize()

    def parameterize(self) -> None:
        '''
        Creates numerical copies of the class attributes that are given in symbolic form.
        '''
        params = self.lat.parameters
        if params is None:
            raise ValueError('Parameters have not been specified.')
        
        self.num_diffvect = np.array(self.diffvect.T.subs(params), dtype=np.float64)[0]
        self.num_intmat = np.array(self.intmat.subs(params), dtype=np.float64)

    def get_intmat(self, k: sp.Matrix, inv_phase: bool = False) -> sp.Matrix:
        '''
        Returns the interaction matrix between two atoms multiplied with a phase factor in symbolic form.

        Arguments:
        ----------
        k: sp.Matrix
            k-point at which the interaction matrix will be evaluated.
        inv_phase: bool
            Sign of the phase. If True, the phase will be negative.

        Return:
        -------
        sp.Matrix
            Interaction matrix multiplied with a phase factor.
        '''
        sign = sp.Integer(1) if not inv_phase else sp.Integer(-1)
        return sp.exp(sign * sp.I * k.dot(self.diffvect)) * self.intmat
    
    def get_num_intmat(self, k: npt.NDArray[np.float64], inv_phase: bool = False) -> npt.NDArray[np.complex128]:
        '''
        Returns the interaction matrix between two atoms multiplied with a phase factor in numerical form.

        Arguments:
        ----------
        k: npt.NDArray[np.float64]
            k-point at which the interaction matrix will be evaluated.
        inv_phase: bool
            Sign of the phase. If True, the phase will be negative.

        Return:
        -------
        sp.Matrix
            Interaction matrix multiplied with a phase factor.
        '''
        sign = 1 if not inv_phase else -1
        return np.exp(sign * 1j * k.dot(self.num_diffvect)) * self.num_intmat



class LinearSpinWave:
    '''
    Instantiates a linear spin wave Hamiltonian (non-interacting magnons) and handles everthing regarding their description and analysis.
    '''
    # initialize common vectors
    kx, ky, kz = sp.symbols('k_x k_y k_z', real=True)
    k = sp.Matrix([kx, ky, kz])
    Bx, By, Bz = sp.symbols('B_x B_y B_z', real=True)
    B = sp.Matrix([Bx, By, Bz])

    # constants
    kb = 0.0862  # in meV / K
    hbar = 0.6582  # in meV * ps
    c = 1  # speed of light

    def __init__(self, 
                 lattice: Lattice, 
                 basis: Basis, 
                 parameters: dict[str, str] = None) -> None:
        '''
        Arguments:
        ----------
        lat: Lattice
            Instance of the Lattice class.
        basis: Basis
            Instance of the Basis class.
        parameters: dict[str, float]
            Dictionary of parameter:value pairs extracted from the setup file.
        '''
        # input objects
        self.lat = lattice
        self.basis = basis
        self.parameters = parameters

        # Hamiltonian matrix setup
        self.n_bands = lattice.n_sublats
        self.bihamil = None
        self.num_bihamil = None

        dim = lattice.n_sublats
        self.idmat = np.eye(2*dim)
        self.metric = np.diag(np.concatenate([np.ones(dim), -np.ones(dim)]))
        self.ndmetric = np.block([[np.zeros((dim, dim)), np.eye(dim)], [-np.eye(dim), np.zeros((dim, dim))]])
        self.symbolic_metric = sp.diag(sp.eye(dim), -sp.eye(dim))

        self.paulix = np.block([[np.zeros((dim, dim)), np.eye(dim)], [np.eye(dim), np.zeros((dim, dim))]])
        self.num_B = None

        self.real_params = None

        self.parameterize_magfield()

    def copy(self) -> Self:
        '''
        Returns a copy of the current LinearSpinWave instance.
        
        Return:
        -------
        deepcopy(self): Self
            LinearSpinWave class instance.
        ''' 
        return deepcopy(self)
    
    def parameterize_magfield(self) -> None:
        '''
        Parameterizes the external magnetic field vector's symbolic expression extracted from the setup file.
        '''
        params = self.lat.parameters
        if params is None:
            raise ValueError('Parameters have not been specified.')
        
        # TODO maybe find a better/clearer method here?
        B = LinearSpinWave.B
        Bx, By, Bz = pd.eval(params['B'])
        symbs = {B[0]: Bx, B[1]: By, B[2]: Bz}
        self.num_B = np.array(B.T.subs(symbs), dtype=np.float64)[0]

    def HP_trafo(self,
                 symbolic: bool = True, 
                 basisphase: bool = True, 
                 simplify: bool = True) -> sp.Matrix | Callable:
        '''
        Perform the Holstein-Primakoff transformation and return the Hamiltonian matrix in second quantization.
        
        Arguments:
        ----------
        symbolic: bool = True
            If True, the Hamiltonian is evaluated in symbolic from and its analytic result can be displayed. Else it is evaluated purely numerically. 
        basisphase: bool = True
            If True, elements of the Hamiltonian are multiplied by a phase facter resulting from the difference of basisvectors between sublattices of different unit cells.
        simplify: bool = True
            If True, the symbolic result for the Hamiltonian is simplified as far as possible.

        Return:
        -------
        Hmat: sp.Matrix | Callable
            If symbolic, returns the analytical form of the Hamiltonian matrix. Else, returns a function that numerically calculates the Hamiltonian matrix for given values.

        '''
        if symbolic:
            Hmat = self.HP_symbolic(basisphase, simplify)
            self.bihamil = Hmat
            return Hmat
        else:
            # if not symbolic, then HP_trafo does not return the matrix directly but instead sets self.bihamil and self.num_bihamil as well as HP_trafo to be the HP_numeric function, that then later can be executed if provided an argument
            Hmat = self.HP_numeric
            self.bihamil = Hmat
            self.num_bihamil = Hmat
            return Hmat

    def HP_symbolic(self, 
                    basisphase: bool = True, 
                    simplify: bool = True) -> sp.Matrix:
        '''
        Constructs the Hamiltonian matrix from its block matrices given in symbolic form.
        
        Arguments:
        ----------
        basisphase: bool = True
            If True, elements of the Hamiltonian are multiplied by a phase facter resulting from the difference of basisvectors between sublattices of different unit cells.
        simplify: bool = True
            If True, the symbolic result for the Hamiltonian is simplified as far as possible.
        
        Return:
        -------
        Hmat: sp.Matrix
            Returns the analytical form of the Hamiltonian matrix.
        '''
        n = self.n_bands
        k = LinearSpinWave.k
        k_inv = {k[0]: -k[0], k[1]: -k[1], k[2]: -k[2]}
        
        # Hamiltonian matrix from block matrices
        Amat11 = self.Amat(basisphase, simplify)
        Amat22 = Amat11.subs(k_inv).conjugate()

        Bmat12 = self.Bmat(basisphase, simplify)
        Bmat21 = Bmat12.T.conjugate()

        Cmat = self.Cmat(simplify)
        Dmat = self.Dmat(simplify)

        Hmat = LinearSpinWave.construct_symbolic_Hmat(n, Amat11, Amat22, Bmat12, Bmat21, Cmat, Dmat)

        if self.real_params is not None:
            Hmat = Hmat.subs(self.real_params)
        # if assume_real:
        #     old_symbols = Hmat.free_symbols
        #     map = {s: sp.Symbol(s.name, real=True) for s in old_symbols}
        #     Hmat = Hmat.subs(map)

        return Hmat

    @staticmethod
    def construct_symbolic_Hmat(n: int,
                                Amat11: sp.Matrix, Amat22: sp.Matrix,
                                Bmat12: sp.Matrix, Bmat21: sp.Matrix,
                                Cmat: sp.Matrix, Dmat: sp.Matrix) -> sp.Matrix:
        '''
        Auxilliary function that matches entries of the Hamiltonian matrix with the block matrices.
        '''
        Hmat = sp.zeros(2*n, 2*n)
        Hmat[:n, :n] = Amat11 - Cmat - Dmat
        Hmat[:n, n:2*n] = Bmat12
        Hmat[n:2*n, :n] = Bmat21
        Hmat[n:2*n, n:2*n] = Amat22 - Cmat - Dmat

        return Hmat

    def HP_numeric(self, 
                   *k: np.float64,
                   basisphase: bool = True) -> npt.NDArray[np.complex128]:
        '''
        Constructs the Hamiltonian matrix from its block matrices given in numeric form.
        
        Arguments:
        ----------
        k: np.float64
            Points at which the Hamiltonian is evaluated at.
        basisphase: bool = True
            If True, elements of the Hamiltonian are multiplied by a phase facter resulting from the difference of basisvectors between sublattices of different unit cells.
                
        Return:
        -------
        Hmat: Callable
            Returns the numerical form of the Hamiltonian matrix as a function that can be called.
        '''
        n = self.n_bands
        k = np.array(k)

        # Hamiltonian matrix from block matrices
        Amat11 = self.num_Amat(k, basisphase)
        Amat22 = self.num_Amat(-k, basisphase).conjugate()

        Bmat12 = self.num_Bmat(k, basisphase)
        Bmat21 = Bmat12.T.conjugate()

        Cmat = self.num_Cmat()
        Dmat = self.num_Dmat()

        Hmat = LinearSpinWave.construct_numeric_Hmat(n, Amat11, Amat22, Bmat12, Bmat21, Cmat, Dmat)

        return Hmat

    @staticmethod
    def construct_numeric_Hmat(n: int,
                               Amat11: npt.NDArray[np.complex128], 
                               Amat22: npt.NDArray[np.complex128],
                               Bmat12: npt.NDArray[np.complex128], 
                               Bmat21: npt.NDArray[np.complex128],
                               Cmat: npt.NDArray[np.complex128], 
                               Dmat: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        '''
        Auxilliary function that matches entries of the Hamiltonian matrix with the block matrices.
        '''
        Hmat = np.zeros((2*n, 2*n), dtype='complex')
        Hmat[:n, :n] = Amat11 - Cmat - Dmat
        Hmat[:n, n:2*n] = Bmat12
        Hmat[n:2*n, :n] = Bmat21
        Hmat[n:2*n, n:2*n] = Amat22 - Cmat - Dmat

        return Hmat 

    def Amat(self, basisphase: bool = True, simplify: bool = False) -> sp.Matrix:
        '''
        TODO
        
        Arguments:
        ----------
        basisphase: bool = True

        simplify: bool = False

        Return:
        -------
        '''
        n = self.n_bands
        k = LinearSpinWave.k
        sublats = self.lat.sublats
        atoms = self.basis.atoms
        Amat = sp.zeros(n,n)

        for i in range(n):
            for j in range(i,n):
                # spin lengths
                Si, Sj = sublats[i].spinlen, sublats[j].spinlen

                # basis vectors
                ti, tj = sublats[i].basisvect, sublats[j].basisvect

                # local z directions
                ui, uj = sublats[i].u_vec, sublats[j].u_vec

                # Fourier-transformed interaction matrices
                intmat_ij = atoms[i].sum_intmat(k, j)
                intmat_ji = atoms[j].sum_intmat(k, i, inv_phase = True)

                # pre-factor
                pre = sp.sqrt(Si * Sj) / sp.Integer(4)

                # phase factor
                if basisphase:
                    phase_factor = sp.exp(sp.I * k.dot(tj - ti))
                    pre *= phase_factor

                # result
                tmp1 = ui.T * intmat_ij * uj.conjugate()
                tmp2 = uj.conjugate().T * intmat_ji * ui
                res = pre * (tmp1 + tmp2)

                if simplify:
                    res = sp.simplify(res)
                
                Amat[i,j] = res
                Amat[j,i] = res.conjugate()
            
        return Amat
    
    def num_Amat(self, 
                 k: tuple[float] | list[float] | npt.NDArray[np.float64], 
                 basisphase:bool = True) -> npt.NDArray[np.complex128]:
        '''
        TODO
        
        Arguments:
        ----------
        k: tuple[float] | list[float] | npt.NDArray[np.float64]

        basisphase: bool = True

        Return:
        -------
        '''
        n = self.n_bands
        sublats = self.lat.sublats
        atoms = self.basis.atoms
        k = np.array(k)
        Amat = np.zeros((n,n), dtype='complex')

        for i in range(n):
            for j in range(i,n):
                # spin lengths
                Si, Sj = sublats[i].num_spinlen, sublats[j].num_spinlen

                # basis vectors
                ti, tj = sublats[i].num_basisvect, sublats[j].num_basisvect

                # local z directions
                ui, uj = sublats[i].num_u, sublats[j].num_u

                # Fourier-transformed interaction matrices
                intmat_ij = atoms[i].sum_num_intmat(k, j)
                intmat_ji = atoms[j].sum_num_intmat(k, i, inv_phase = True)

                # pre-factor
                pre = np.sqrt(Si * Sj) / 4

                # phase factor
                if basisphase:
                    phase_factor = np.exp(1j * k.dot(tj - ti))
                    pre *= phase_factor

                # result
                tmp1 = ui.dot(intmat_ij).dot(uj.conjugate())
                tmp2 = uj.conjugate().dot(intmat_ji).dot(ui)

                Amat[i, j] = pre * (tmp1 + tmp2)
                Amat[j, i] = Amat[i, j].conjugate()

            return Amat

    def Bmat(self, basisphase: bool = True, simplify: bool = False) -> sp.Matrix:
        '''
        TODO
        
        Arguments:
        ----------
        basisphase: bool = True

        simplify: bool = False

        Return:
        -------
        '''
        n = self.n_bands
        k = LinearSpinWave.k
        sublats = self.lat.sublats
        atoms = self.basis.atoms
        Bmat = sp.zeros(n,n)

        for i in range(n):
            for j in range(n):
                # spin lengths
                Si, Sj = sublats[i].spinlen, sublats[j].spinlen

                # basis vectors
                ti, tj = sublats[i].basisvect, sublats[j].basisvect

                # local z directions
                ui, uj = sublats[i].u_vec, sublats[j].u_vec

                # Fourier-transformed interaction matrices
                intmat_ij = atoms[i].sum_intmat(k, j)

                # pre-factor
                pre = sp.sqrt(Si * Sj) / sp.Integer(2)

                # phase factor
                if basisphase:
                    phase_factor = sp.exp(sp.I * k.dot(tj - ti))
                    pre *= phase_factor

                # result
                tmp = ui.T * intmat_ij * uj
                res = pre * tmp

                if simplify:
                    res = sp.simplify(res)
                
                Bmat[i, j] = res
            
        return Bmat
    
    def num_Bmat(self, 
                 k: tuple[float] | list[float] | npt.NDArray[np.float64], 
                 basisphase:bool = True) -> npt.NDArray[np.complex128]:
        '''
        TODO
        
        Arguments:
        ----------
        k: tuple[float] | list[float] | npt.NDArray[np.float64]

        basisphase: bool = True

        Return:
        -------
        '''
        n = self.n_bands
        sublats = self.lat.sublats
        atoms = self.basis.atoms
        k = np.array(k)
        Bmat = np.zeros((n,n), dtype='complex')

        for i in range(n):
            for j in range(n):
                # spin lengths
                Si, Sj = sublats[i].num_spinlen, sublats[j].num_spinlen

                # basis vectors
                ti, tj = sublats[i].num_basisvect, sublats[j].num_basisvect

                # local z directions
                ui, uj = sublats[i].num_u, sublats[j].num_u

                # Fourier-transformed interaction matrices
                intmat_ij = atoms[i].sum_num_intmat(k, j)

                # pre-factor
                pre = np.sqrt(Si * Sj) / 2

                # phase factor
                if basisphase:
                    phase_factor = np.exp(1j * k.dot(tj - ti))
                    pre *= phase_factor

                # result
                tmp = ui.dot(intmat_ij).dot(uj)

                Bmat[i, j] = pre * tmp

            return Bmat

    def Cmat(self, simplify: bool = False) -> sp.Matrix:
        '''
        TODO
        
        Arguments:
        ----------
        simplify: bool = False

        Return:
        -------
        '''
        n = self.n_bands
        sublats = self.lat.sublats
        atoms = self.basis.atoms
        Cmat = sp.zeros(n,n)

        for i in range(n):
            for l in range(n):
                # spin lengths
                Sl = sublats[l].spinlen

                # local x, y-directions
                vi, vl = sublats[i].v_vec, sublats[l].v_vec

                # Fourier-transformed interaction matrices
                intmat_il = atoms[i].sum_intmat_k_zero(l)
                intmat_li = atoms[l].sum_intmat_k_zero(i)

                # pre-factor
                pre = Sl / sp.Integer(2)

                # result
                tmp1 = vi.T * intmat_il * vl
                tmp2 = vl.T * intmat_li * vi
                res = pre * (tmp1 + tmp2)[0]

                if simplify:
                    res = sp.simplify(res)

                Cmat[i, i] += res

        return Cmat
    
    def num_Cmat(self) -> npt.NDArray[np.complex128]:
        '''
        TODO

        Return:
        -------
        '''
        n = self.n_bands
        sublats = self.lat.sublats
        atoms = self.basis.atoms
        Cmat = np.zeros((n,n), dtype='complex')

        for i in range(n):
            for l in range(n):
                # spin lengths
                Sl = sublats[l].num_spinlen

                # local x, y-directions
                vi, vl = sublats[i].num_v, sublats[l].num_v

                # Fourier-transformed interaction matrices
                intmat_il = atoms[i].sum_num_intmat_k_zero(l)
                intmat_li = atoms[l].sum_num_intmat_k_zero(i)

                # pre-factor
                pre = Sl / 2

                # result
                tmp1 = vi.dot(intmat_il).dot(vl)
                tmp2 = vl.dot(intmat_li).dot(vi)

                Cmat[i, i] += pre * (tmp1 + tmp2)

            return Cmat

    def Dmat(self, simplify: bool = False) -> sp.Matrix:
        '''
        TODO
        
        Arguments:
        ----------
        simplify: bool = False

        Return:
        -------
        '''
        n = self.n_bands
        sublats = self.lat.sublats
        B = LinearSpinWave.B
        Dmat = sp.zeros(n, n)

        for i in range(n):
            res = B.T * sublats[i].gyromat * sublats[i].v_vec

            if simplify:
                res = sp.simplify(res)

            Dmat[i, i] = res

        return Dmat
    
    def num_Dmat(self) -> npt.NDArray[np.complex128]:
        '''
        TODO

        Return:
        -------
        '''
        n = self.n_bands
        sublats = self.lat.sublats
        num_B = self.num_B
        Dmat = np.zeros((n,n), dtype='complex')

        for i in range(n):
            vi = sublats[i].num_v
            gyromat = sublats[i].num_gyromat

            Dmat[i, i] = num_B.dot(gyromat).dot(vi)

        return Dmat
    
    def dispersion(self, warn: bool = True) -> dict[float, int]:
        '''
        TODO
        '''
        disp: sp.Matrix = self.symbolic_metric * self.bihamil

        n = disp.rows
        half = n // 2

        # check if block diagonal
        top_right = disp[:half, half:]
        bottom_left = disp[half:, :half]

        if top_right.is_zero_matrix and bottom_left.is_zero_matrix:
            # solve blocks separately
            top_left = disp[:half, :half]
            bottom_right = disp[half:, half:]
        
        eigvals_1: dict = top_left.eigenvals()
        eigvals_2: dict = bottom_right.eigenvals()
        
        # merge eigvals
        all_eigvals = eigvals_1.copy()
        for eigval, mult in eigvals_2.items():
            all_eigvals[eigval] = all_eigvals.get(eigval, 0) + mult
            return all_eigvals
        else:
            if warn:
                response = input(f"Hamiltonian is NOT block diagonal. Computing eigenvalues of {n}×{n} symbolic matrix may take very long. Continue? (y/n): ")
                if response.lower() != 'y':
                    raise RuntimeError("Eigenvalue computation cancelled by user")
            print("Computing eigenvalues (may take a while)...")
            return disp.eigvals()
        
    def assume_real_parameters(self) -> dict:
        '''
        TODO
        '''
        params = self.lat.parameters
        if params is None:
            raise ValueError('Parameters have not been specified.')
        
        vars = params.keys()
        symbs = sp.symbols(' '.join(vars), real = True)
        real_params = dict([v,s] for v,s in zip(vars, symbs))
        self.real_params = real_params

        return real_params # FIXME Bog trafo stops working when this is executed before HP trafo
    
    def parameterize_hamil(self, chempot: np.float64 = None) -> np.float64:
        '''
        TODO
        '''
        params = self.parameters
        if params is None:
            raise ValueError('Parameters have not been specified.')
        
        n = self.n_bands
        Bx, By, Bz = pd.eval(params['B'])

        try:
            tmp = self.bihamil.subs({LinearSpinWave.Bx: Bx,
                                     LinearSpinWave.By: By,
                                     LinearSpinWave.Bz: Bz})
            num_hamil_no_chempot = sp.lambdify(LinearSpinWave.k, tmp.subs(params), 'numpy')
        except AttributeError:
            num_hamil_no_chempot = self.bihamil

        def add_chempot(chempot: np.float64) -> None:
            self.num_bihamil = lambda *k: num_hamil_no_chempot(*k) + chempot * np.eye(2 * n)

        if chempot is not None:
            add_chempot(chempot)
            self.parameters['chempot'] = chempot
            return chempot
        if chempot is None and 'chempot' in params:
            chempot = params['chempot']
            add_chempot(chempot)
            return chempot
        
        print('Automatically finding sufficient chemical potential...')
        chempot = 0
        add_chempot(chempot)
        while True:
            try:
                # check if Hmat is positive definite: if not, Bogoliubov_trafo will throw LinAlgError
                self.Bogoliubov_trafo(0, 0, 0)
                print(f'Using {chempot} as chemical potential.')
                params['chempot'] = chempot
                return chempot
            except LinAlgError:
                # if Hmat is not positive definite, increase chempot
                chempot = 1e-10 if chempot == 0 else chempot * 10
                add_chempot(chempot)

    def update_parameters(self, new: dict[str, float], chempot: float = None) -> None:
        '''
        TODO
        '''
        for key in new.keys():
            self.parameters[key] = new[key]
        if chempot is None:
            new_chempot = self.parameterize_hamil()
        else:
            new_chempot = self.parameterize_hamil(chempot)
        print(f'Updated parameters and now using {new_chempot} as chemical potential.')

    def Bogoliubov_trafo(self,
                         kx: np.float64, ky: np.float64, kz: np.float64,
                         decimals: int = 5,
                         global_gauge: bool = True,
                         force_phsymmetry: bool = True) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
        '''
        TODO
        '''
        eta = self.metric
        dim = self.n_bands
        Hmat = self.num_bihamil
        k = np.array([kx, ky, kz])

        # check if k is the zero vector
        components_zero = np.isclose(k, 0)
        k_is_zero = all(components_zero)
                        
        # ensure particle-hole symmetry by mapping k to -k if necessary
        if force_phsymmetry and not k_is_zero:
            # ensure first non-zero component is positive
            sign = np.sign(k[~components_zero][0])
            k *= sign
            kx, ky, kz = k

        # COLPA's algorithm
        # -------------------------
        # 1. Cholesky decomposition
        # -------------------------

        try:
            # calculate lower triangular matrix
            Kmat_adj = cholesky(Hmat(kx, ky, kz))
        except LinAlgError:
            raise LinAlgError(f"Hamiltonian matrix is not positive definite for k=({kx},{ky},{kz})")
        
        # calculate upper triangular matrix
        Kmat = Kmat_adj.transpose().conjugate()

        # ---------------------------------------
        # 2. Unitary diagonalization of K η K†
        # ---------------------------------------

        # 1D array of eigenvalues and 2D array of eigenvectors (columns) 
        Ldiag, Umat = eigh(np.dot(Kmat, np.dot(eta, Kmat_adj)))

        # reorder eigenvalues/vectors: first dim diagonal elements positive, second dim negative
        order = np.concatenate([np.arange(dim, 2 * dim)[::-1], np.arange(dim)])
        Umat = Umat[:, order]
        Ldiag = Ldiag[order]
        
        # build matrix with eigenvalues on its diagonal and calculate the eigenenergy matrix
        Lmat = np.diag(Ldiag)
        Emat = np.dot(eta, Lmat)

        # ---------------------------------------------------------
        # 3. Solve for the transformation matrix T = K^-1 U E^(1/2)
        # ---------------------------------------------------------

        sqrt_Emat = np.sqrt(Emat)
        Tmat = np.dot(inv(Kmat), np.dot(Umat, sqrt_Emat))

        # ---------------
        # 4. Global gauge
        # ---------------

        k_is_zero = np.allclose(0, [kx, ky, kz])
        if global_gauge:
            absmat1 = np.abs(Tmat[:dim, :dim]).round(decimals) # particles
            absmat2 = np.abs(Tmat[dim:, dim:]).round(decimals) # holes

            # for each column, find the row with the largest magnitude
            maxrows1 = absmat1.argmax(axis=0)
            maxrows2 = absmat2.argmax(axis=0)

            # for each column, find the row that contains the element with the largest magnitude
            maxel1 = Tmat[maxrows1, np.arange(dim)]
            maxel2 = Tmat[maxrows2 + dim, np.arange(dim) + dim]
            
            # extract the phase: x/|x| = e^(i*arg(x)), store in diagonal matrix
            phases = np.zeros((2 * dim, 2 * dim), dtype='complex')
            phases[np.arange(dim), np.arange(dim)] = maxel1 / np.abs(maxel1)
            phases[np.arange(dim) + dim, np.arange(dim) + dim] = maxel2 / np.abs(maxel2)

            # multiply eigenvectors with conjugate phase -> element with largest absval is positive and real
            Tmat = Tmat.dot(phases.conjugate())

        # enforce particle-hole symmetry for k = 0
        if force_phsymmetry:
            if k_is_zero:
                Tmat[dim:, dim:] = Tmat[:dim, :dim].conjugate()
                Tmat[:dim, dim:] = Tmat[dim:, :dim].conjugate()
            # revert back -k to k if applicable
            elif sign < 0:
                Ldiag[:dim], Ldiag[dim:] = -Ldiag[dim:], -Ldiag[:dim]
                paulix = self.paulix
                Tmat = paulix.dot(Tmat.conjugate()).dot(paulix)

        return Ldiag, Tmat
    
    def transform_basis(self, 
                        points: npt.NDArray[np.float64], 
                        basis: str = 'rcpr', 
                        new_basisvects: npt.NDArray[np.float64] = None) -> npt.NDArray[np.float64]:
        '''
        Transforms a vector in its current representation to a new basis.

        Arguments:
        ----------
        points: npt.NDArray[np.float64]
            Array of point coordinates stacked as its rows.
        basis: str = 'rcpr'
            Type of the new basis. By default 'rcpr' transforms to the reciprocal lattice vector basis.
        new_basisvects: npt.NDArray[np.float64]
            Set of new basis vectors that the basis is transformed to if basis = 'other' is chosen. By default it is None.

        Return:
        -------
        '''
        if basis == 'rcpr':
            return points @ self.lat.num_rcpr_vects
        elif basis == 'other':
            assert new_basisvects is not None, "A set of new basis vectors must be specified"
            return points @ new_basisvects

    @staticmethod 
    def arc_length_parameterization(points: npt.NDArray[np.float64]) -> tuple[Callable, npt.NDArray[np.float64]]:
        '''
        TODO
        '''
        n = points.shape[0] - 1

        segment_lens = np.array([norm(points[i+1] - points[i]) for i in range(n)])
        cumulative_dist = np.concatenate([[0], np.cumsum(segment_lens)])
        total_dist = cumulative_dist[-1]
        normalized_dist = cumulative_dist / total_dist
        
        def interpolate_path(pt: np.float64) -> npt.NDArray:
            '''
            TODO
            '''
            # find the segment corresponding to relative fraction of total path length
            id = np.searchsorted(normalized_dist, pt) - 1
            id = np.clip(id, 0, n-1) # same as max(0, min(id, n - 1)) or id = 0 if id<0 else (n-1) if id>(n-1)

            # segment boundaries
            start = normalized_dist[id]
            end = normalized_dist[id + 1]

            # position within segment
            # avoid division by zero (for two consecutive points with dist=0, should not occur)
            if end == start:
                dt = 0
            else:
                dt = (pt - start) / (end - start)

            # interpolation
            return points[id] + dt * (points[id + 1] - points[id])
        
        return interpolate_path, normalized_dist
    
    @staticmethod
    def uniform_path_interpolation(points: npt.NDArray[np.float64]) -> tuple[Callable, npt.NDArray[np.float64]]:
        '''
        TODO
        '''
        n = points.shape[0] - 1

        point_maps = np.arange(n + 1) / n

        def interpolate_path(pt: np.float64) -> npt.NDArray:
            '''
            TODO
            '''
            # find the segment corresponding to relative fraction of total point count
            id = int(pt * n) # int suffices because pt is always between 0 and 1
            id = np.clip(id, 0, n-1)

            # position within segment
            dt = n * pt - id

            # interpolation
            return points[id] + dt * (points[id + 1] - points[id])
        
        return interpolate_path, point_maps
            
    def path_from_points(self, 
                        points: npt.NDArray[np.float64],
                        isometric: bool = True,
                        basis: str = 'rcpr',
                        new_basisvects: npt.NDArray[np.float64] = None) -> tuple[Callable, npt.NDArray[np.float64]]:
        '''
        TODO
        path through the BZ: first all points along the path must be expressed in terms of the reciprocal lattice vectors -> transform basis
        '''
        points = self.transform_basis(points, basis = basis, new_basisvects = new_basisvects)

        if isometric:
            return self.arc_length_parameterization(points)
        else:
            return self.uniform_path_interpolation(points)

    def energy_along_path(self, 
                          points: npt.NDArray[np.float64],
                          num: int = 100,
                          isometric: bool = True,
                          basis: str = 'rcpr',
                          new_basisvects: npt.NDArray[np.float64] = None) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        '''
        TODO
        '''
        path_func, point_maps = self.path_from_points(points, 
                                                      isometric=isometric, 
                                                      basis=basis, 
                                                      new_basisvects=new_basisvects)
        
        pts = np.linspace(0, 1, num=num, dtype=np.float64)
        ks = [path_func(pt) for pt in pts]

        energies = np.array([self.Bogoliubov_trafo(*k)[0][:self.n_bands] for k in ks])

        return point_maps, pts, ks, energies
    
    def evgrid(self,
               kgrid: int | tuple[int, int] | npt.NDArray[np.float64],
               startpoint: bool = True,
               endpoint: bool = True,
               offset: int = 0,
               gtype: str = 'center',
               global_gauge: bool = True,
               decimals: int = 5) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        '''
        TODO
        '''
        bands = self.n_bands
        if isinstance(kgrid, (int, tuple)):
            v1, v2 = self.lat.num_rcpr_vects[:2]
            kxs, kys, kzs = Mesh.parallelogram_grid(v1, v2, kgrid, startpoint, endpoint, offset, gtype)
        elif isinstance(kgrid, npt.NDArray):
            kxs, kys, kzs = kgrid
        else:
            raise ValueError("Invalid type for parameter 'kgrid'.")
        
        q, p = kxs.shape
        es = np.zeros((q, p, 2 * bands))
        ts = np.zeros((q, p, 2 * bands, 2 * bands), dtype='complex')

        for i in range(q):
            for j in range(p):
                kx, ky, kz = kxs[i,j], kys[i,j], kzs[i,j]
                e, t = self.Bogoliubov_trafo(kx, ky, kz, global_gauge=global_gauge, decimals=decimals)
                es[i, j, :] = e
                ts[i, j, :, :] = t

        return es, ts
    
    def link_phases(self, 
                    ev1: npt.NDArray[np.complex128], 
                    ev2: npt.NDArray[np.complex128], 
                    cutoff: float = 1e-04) -> npt.NDArray:
        '''
        TODO eq. (7)
        '''
        # calculate the inner product of the eigenvectors w.r.t. the metric space i.e. η ev1† η ev2
        # diagonal elements -> normalization, off-diagonal elements -> similarity of eigenstates
        metric = self.metric
        overlap = metric @ ev1.conjugate().T @ metric @ ev2
        diag_overlap = np.diag(overlap)
        norm_overlap = np.abs(diag_overlap)

        # create a mask to only select values above a certain threshhold
        mask = norm_overlap > cutoff
        phases = np.ones(2 * self.n_bands, dtype='complex')
        phases[mask] = diag_overlap[mask] / norm_overlap[mask]
        return phases
    
    def flux(self, 
             link1: npt.NDArray[np.complex128], 
             link2: npt.NDArray[np.complex128], 
             link3: npt.NDArray[np.complex128], 
             link4: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
        '''
        TODO eq. (8)
        '''
        flux = -np.log(link1 * link2 / (link3 * link4))
        flux = flux.imag
        return flux
    
    def flux_from_eigenvects(self, 
                             ev1: npt.NDArray[np.complex128], 
                             ev2: npt.NDArray[np.complex128], 
                             ev3: npt.NDArray[np.complex128], 
                             ev4: npt.NDArray[np.complex128], 
                             cutoff: float = 1e-04) -> npt.NDArray[np.float64]:
        '''
        TODO
        '''
        link1 = self.link_phases(ev1, ev2, cutoff)  # <n(k)|n(k+x)>
        link2 = self.link_phases(ev2, ev3, cutoff)  # <n(k+x)|n(k+x+y)>
        link3 = self.link_phases(ev4, ev3, cutoff)  # <n(k+y)|n(k+x+y)>
        link4 = self.link_phases(ev1, ev4, cutoff)  # <n(k)|n(k+y)>
        flux = self.flux(link1, link2, link3, link4)
        return flux

    def count_degeneracies(self, eigvals: npt.NDArray[np.float64], cutoff: float = 1e-04) -> int:
        '''
        TODO
        '''
        bands = self.n_bands
        gaps = - np.diff(eigvals[:bands])
        is_degen = gaps < cutoff
        n_degens = is_degen.sum()
        return n_degens

    def flux_from_nondegenerate_eigenvects(self,
                                           eigvals1: npt.NDArray[np.float64],
                                           eigvals2: npt.NDArray[np.float64],
                                           eigvals3: npt.NDArray[np.float64],
                                           eigvals4: npt.NDArray[np.float64],
                                           eigvects1: npt.NDArray[np.complex128],
                                           eigvects2: npt.NDArray[np.complex128],
                                           eigvects3: npt.NDArray[np.complex128],
                                           eigvects4: npt.NDArray[np.complex128],
                                           overlap_cutoff: float = 1e-04,
                                           energy_cutoff:float = 1e-04,
                                           repby: float = 0) -> npt.NDArray[np.float64]:
        '''
        TODO repby -> replace by?
        '''
        if energy_cutoff is None:
            flux = self.flux_from_eigenvects(eigvects1, eigvects2, eigvects3, eigvects4, overlap_cutoff)
            return flux
        
        is_degen1 = self.count_degeneracies(eigvals1, energy_cutoff) > 0
        is_degen2 = self.count_degeneracies(eigvals2, energy_cutoff) > 0
        is_degen3 = self.count_degeneracies(eigvals3, energy_cutoff) > 0
        is_degen4 = self.count_degeneracies(eigvals4, energy_cutoff) > 0

        if any((is_degen1, is_degen2, is_degen3, is_degen4)):
            flux = np.full(2 * self.n_bands, repby, dtype=float)
        else:
            flux = self.flux_from_eigenvects(eigvects1, eigvects2, eigvects3, eigvects4, overlap_cutoff)

        return flux

    def curvature_grid(self, 
                       grid: npt.NDArray[np.float64], 
                       kgrid: npt.NDArray[np.float64] = None, 
                       egrid: npt.NDArray[np.float64] = None, 
                       overlap_cutoff: float = 1e-04,
                       energy_cutoff: float = None,
                       repby: float = 0) -> npt.NDArray[np.float64]:
        '''
        TODO
        '''
        if (egrid is None) and (energy_cutoff is not None):
            raise ValueError("cutoff has been set, but no egrid was given")
        if (egrid is not None) and (energy_cutoff is None):
            raise ValueError("egrid has been set, but no cutoff was defined")
        
        check_for_degens = energy_cutoff is not None

        q, p = grid.shape[:2]
        bands = grid.shape[-1]
        curvs = np.zeros((bands, q - 1, p - 1))
        for i in range(q - 1):
            for j in range(p - 1):
                ev1 = grid[i, j]            # |n(k)>
                ev2 = grid[i, j + 1]        # |n(k+x)>
                ev3 = grid[i + 1, j + 1]    # |n(k+x+y)>
                ev4 = grid[i + 1, j]        # |n(k+y)>
                if check_for_degens:
                    e1 = egrid[i, j]            # e(k)
                    e2 = egrid[i, j + 1]        # e(k+x)
                    e3 = egrid[i + 1, j + 1]    # e(k+x+y)
                    e4 = egrid[i, j + 1]        # e(k+y)
                    args = (e1, e2, e3, e4, ev1, ev2, ev3, ev4, overlap_cutoff, energy_cutoff, repby)
                    curvs[:, i, j] = self.flux_from_nondegenerate_eigenvects(*args)
                else:
                    args = (ev1, ev2, ev3, ev4, overlap_cutoff)
                    curvs[:, i, j] = self.flux_from_eigenvects(*args)
                if kgrid is not None:
                    k1 = kgrid[:, i, j]
                    k2 = kgrid[:, i, j + 1]
                    k3 = kgrid[:, i + 1, j + 1]
                    k4 = kgrid[:, i + 1, j]
                    area = Mesh.area_quadrilateral(k1, k2, k3, k4)
                    curvs[:, i, j] /= area

        return curvs

    def chern(self, grid: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        '''
        TODO
        '''
        curv = self.curvature_grid(grid)
        chern = -curv.sum(axis=-1).sum(axis=-1) / (2 * np.pi)
        return chern
    
    def abelian_chern(self, N1: int, N2: int = None, rounding: bool = True) -> list[float]:
        '''
        TODO
        '''
        if N2 is None:
            N2 = N1
        n1_range = np.arange(N1)
        n2_range = np.arange(N2)

        n1_grid, n2_grid = np.meshgrid(n1_range, n2_range, indexing = 'ij')

        k1_frac = n1_grid / N1
        k2_frac = n2_grid / N2

        r1, r2 = self.lat.num_rcpr_vects[:2]

        k_grid = k1_frac[..., np.newaxis] * r1 + k2_frac[..., np.newaxis] * r2

        testE, testV = self.Bogoliubov_trafo(*k_grid[0,0])
        n_bands = testE.shape[0]
        dim = testV.shape[0]

        egrid = np.zeros((N1, N2, n_bands))
        evgrid = np.zeros((N1, N2, dim, n_bands), dtype=np.complex128)

        print(f"Calculating eigenvectors on {N1}x{N2} grid...")
        for i in range(N1):
            for j in range(N2):
                k_curr = k_grid[i, j]
                vals, vecs = self.Bogoliubov_trafo(*k_curr)
                egrid[i, j] = vals
                evgrid[i, j] = vecs
                
        metric = self.metric

        ev1 = np.roll(evgrid, -1, axis=0)
        ev2 = np.roll(evgrid, -1, axis=1)

        metric_ev1 = np.einsum('pq,nmqr->nmpr', metric, ev1)
        metric_ev2 = np.einsum('pq,nmqr->nmpr', metric, ev2)

        link1 = np.sum(evgrid.conj() * metric_ev1, axis=2)
        link2 = np.sum(evgrid.conj() * metric_ev2, axis=2)

        link1 /= np.abs(link1)
        link2 /= np.abs(link2)

        link2_to_1 = np.roll(link2, -1, axis=0) # U_2(k + r1)
        link1_to_2 = np.roll(link1, -1, axis=1) # U_1(k + r2)

        plaquette = link1 * link2_to_1 * np.conj(link1_to_2) * np.conj(link2)

        flux = np.imag(np.log(plaquette)) # F_12

        chern = np.sum(flux, axis=(0, 1)) / (2 * np.pi)

        if rounding:
            chern.round(2, out=chern)

        return chern, flux, egrid
    
    def non_abelian_chern(self, evs: npt.NDArray[np.float64], bands: list[int], round = True) -> np.float64:
        '''
        TODO link phases according to eq. (16)
        '''
        # get (points, EVs) array of selected bands
        psi = evs[:, :, :, bands]

        # shift to neighboring grid points (k_1 + 1, k_2) and (k_1, k_2 + 1)
        psi_1 = np.roll(psi, -1, axis=0)
        psi_2 = np.roll(psi, -1, axis=1)

        # calculate link variables from multiplet states ψ
        # ------------------------------------------------
        # 1. pre-apply metric to neighbor EVs using einsum
        # ij for metric entries, pqjk for psi (k1, k2, 2*dim, band) -> pqik is metric applied to EV components (hole entries aquire minus sign)
        psi_1 = np.einsum('ij,pqjk->pqik', self.metric, psi_1)
        psi_2 = np.einsum('ij,pqjk->pqik', self.metric, psi_2)

        # 2. calcualte overlap η T†(k) η T(k+μ)
        # pre-apply metric and transpose reference EVs (last two axes of psi)
        # psi = np.einsum('ij,pqjk->pqik', self.metric, psi)  # NOTE not sure if necessary, ik -> ki for transpose
        psi_dag = psi.conj().swapaxes(-1, -2)

        # overlap matrices
        link1 = np.matmul(psi_dag, psi_1)
        link2 = np.matmul(psi_dag, psi_2)

        # determinants
        link1 = np.linalg.det(link1)
        link2 = np.linalg.det(link2)

        # normalize to obtain link phases
        link1 /= np.abs(link1)
        link2 /= np.abs(link2)

        # 3. calcualte flux through plaquette U_1 * U_2(k_1 + 1) * U_1(k_2 + 2)† * U_2†
        # get neighbor quantities
        link1_to_2 = np.roll(link1, -1, axis=1) # U_1(k_2 + 2)
        link2_to_1 = np.roll(link2, -1, axis=0) # U_2(k_1 + 1)

        # get product of phases (for complex numbers of magnitude 1 conjugate is the same as inverse)
        plaquette = link1 * link2_to_1 * np.conj(link1_to_2) * np.conj(link2)

        # principal branch of logarithm yields Berry flux
        flux = np.imag(np.log(plaquette))

        # 4. sum and normalize for chern number
        chern = np.sum(flux) / (2 * np.pi)

        if round:
            chern = np.round(chern, 5)

        return chern
    
    def berry_curvature(self, flux: npt.NDArray, band: int):
        '''
        TODO
        '''
        x, y, z = flux[:,0,0], flux[1], flux[:,:,band]

        r1, r2 = self.lat.num_rcpr_vects[:2]
        plaq_area = np.abs(r1[0]*r2[1] - r1[1]*r2[0]) / (len(x) * len(y))
        z /= plaq_area

        fig, ax = plt.subplots()
        im = ax.pcolormesh(x,y,z, cmap='viridis', shading='auto')
        plt.colorbar(im, ax=ax, label='Value')
        plt.show()
            
    def sublat_localization(self, Tmat: npt.NDArray) -> npt.NDArray:
        '''
        TODO
        '''
        n = Tmat.shape[0] // 2
        slloc = (np.abs(Tmat[:n, :n])**2 + np.abs(Tmat[n:, :n])**2).T
        return slloc
    
    def create_slab(self, dim: int, num: int) -> LinearSpinWave:
        '''
        TODO
        '''
        slab_lat = self.lat.slab_lattice(dim, num)
        slab_basis = self.basis.slab_basis(slab_lat, dim, num)
        slab_lsw = LinearSpinWave(slab_lat, slab_basis, self.parameters.copy())
        return slab_lsw



class Mesh:
    '''
    Utility class for creating mesh grids. Contains a collection of independent methods.
    '''
    @staticmethod
    def parallelogram_grid(v1: npt.NDArray, v2: npt.NDArray, 
                           nums: int | tuple[int, int],
                           startpoint: bool = True,
                           endpoint: bool = True,
                           offset: float | list[float] = 0,
                           gtype: str = 'center') -> npt.NDArray:
        '''
        TODO
        '''
        assert gtype in ['center', 'corner'], "gtype can only be center or corner."

        norm1, norm2 = norm(v1), norm(v2)

        if isinstance(nums, int):
            center_noi = Mesh.n_intervals(nums, startpoint, endpoint, gtype='center')
            if norm1 < norm2:
                num1 = nums
                noi2 = int(round(center_noi * norm2 / norm1))
                num2 = Mesh.n_points(noi2, startpoint, endpoint, gtype='center')
            else:
                num2 = nums
                noi1 = int(round(center_noi * norm1 / norm2))
                num1 = Mesh.n_points(noi1, startpoint, endpoint, gtype='center')
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
            center_noi1 = Mesh.n_intervals(nums[0], startpoint, endpoint, gtype='center')
            center_noi2 = Mesh.n_intervals(nums[1], startpoint, endpoint, gtype='center')

            # length of intervals in (v1, v2) basis
            len1 = 1 / center_noi1
            len2 = 1 / center_noi2

            # shift whole grid to the left to surround center type mesh
            offset[0] -= len1/2
            offset[1] -= len2/2

            # extend mesh by double corner-type interval length
            end1 += len1
            end2 += len2

            # add singular point for both axes for corner-type mesh
            nums += 1

        xs = Mesh.evenly_spaced(start1, end1, nums[0], startpoint, endpoint) + offset[0]
        ys = Mesh.evenly_spaced(start2, end2, nums[1], startpoint, endpoint) + offset[1]

        mx, my = np.meshgrid(xs, ys)

        res = np.zeros((len(v1), nums[1], nums[0]))

        for i in range(nums[1]):
            for j in range(nums[0]):
                x, y = mx[i,j], my[i,j]
                v = x * v1 + y * v2
                res[:, i, j] = v

        return res
                
    @staticmethod
    def n_intervals(n_points: int, startpoint: bool, endpoint: bool, gtype: str) -> int:
        '''
        TODO
        '''
        n_intervals = n_points - 1
        if not startpoint:
            n_intervals += 1
        if not endpoint:
            n_intervals += 1
        if gtype == 'corner':
            n_intervals += 1
        return n_intervals

    @staticmethod
    def n_points(n_intervals: int, startpoint: bool, endpoint: bool, gtype: str) -> int:
        '''
        TODO
        '''
        n_points = n_intervals + 1
        if not startpoint:
            n_points -= 1
        if not endpoint:
            n_points -= 1
        if gtype == 'corner':
            n_points -= 1
        return n_points
    
    @staticmethod
    def evenly_spaced(start: npt.ArrayLike,
                      stop: npt.ArrayLike,
                      num: int = 50,
                      startpoint: bool = True,
                      endpoint: bool = True,
                      retstep: bool = False,
                      dtype: str = None,
                      axis: int = 0) -> npt.NDArray | tuple[npt.NDArray, float]:
        '''
        TODO
        '''
        if startpoint:
            start_id = 0
        else:
            start_id = 1
            num += 1
        
        res = np.linspace(start, stop, num, endpoint, retstep, dtype, axis)

        if retstep:
            samples, step = res
            samples = samples[start_id:]
            return samples, step
        else:
            samples = res
            samples = samples[start_id:]
            return samples
        
    @staticmethod
    def area_quadrilateral(v1: npt.NDArray, v2: npt.NDArray, v3: npt.NDArray, v4: npt.NDArray) -> np.float64:
        '''
        Compute the area of a quadrilateral (four-sided polygon) with vertices v1, v2, v3, and v4.
        
        Arguments:
        ----------
        vi: npt.NDArray
            Vertices of the quadrilateral give as vectors stored in an array.

        Return:
        -------
        area: np.float64
            Area of the quadrilateral.
        '''
        v12 = v2 - v1
        v14 = v4 - v1
        v23 = v3 - v2
        v43 = v3 - v4
        area1 = norm(np.cross(v12, v14))
        area2 = norm(np.cross(v23, v43))
        area =(area1 + area2) / 2
        return area
    
    @staticmethod
    def area_quadrilateral_alt(v1: npt.NDArray, v2: npt.NDArray, v3: npt.NDArray, v4: npt.NDArray) -> np.float64:
        '''
        Compute the area of a quadrilateral (four-sided polygon) with vertices v1, v2, v3, and v4. The vertices must be given in cyclic order.
        
        Arguments:
        ----------
        vi: npt.NDArray
            Vertices of the quadrilateral give as vectors stored in an array.

        Return:
        -------
        area: np.float64
            Area of the quadrilateral.
        '''
        v12 = v2 - v1
        v13 = v3 - v1
        v14 = v4 - v1

        area1 = norm(np.cross(v12, v13))
        area2 = norm(np.cross(v13, v14))
        area =(area1 + area2) / 2
        return area
        


class BandStructure1D:
    '''
    Set of methods used for plotting 1D band structures (TODO and managing files).
    '''
    def __init__(self, lsw: LinearSpinWave, system: str) -> None:
        '''
        Arguments:
        ----------
        lsw: LinearSpinWave
            Instance of the LinearSpinWave class.
        system: str
            System name for file managing.
        '''
        self.lsw = lsw
        self.system = system

    def from_path(path: str):
        '''
        TODO specify HSP path as a string
        '''
        # example for path string: 'G, M, K, G'
        hsp = path.split(',')
        raise NotImplementedError("Not implemented yet.")

    @staticmethod
    def get_pathlabels(*points: list[str, str, npt.NDArray[np.float64]]) -> list[str]:
        '''
        Returns a list of strings with names of the high-symmetry points.
        
        Arguments:
        ----------

        Return:
        -------
        '''
        return [point[0] for point in points]

    @staticmethod
    def get_pathcoords(*points: list[str, str, npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        '''
        Returns an array of high-symmetry point coordinates stacked as its rows.
        
        Arguments:
        ----------

        Return:
        -------
        '''
        return np.array([point[1] for point in points])
    
    def get_dispersion(self, 
                       *points: list[str, str, npt.NDArray[np.float64]], 
                       num: int = 100, 
                       isometric: bool = True, 
                       basis: str = 'rcpr',
                       new_basisvects: npt.NDArray[np.float64] = None) -> dict[str, tuple[list] | list[str] | npt.NDArray[np.float64] | list[npt.NDArray[np.float64]]]:
        '''
        TODO Calculate the energies along the path connecting all points.
        
        Arguments:
        ----------
        *points: sequence of high-symmetry points (see e.g. class Cubic).
            The points to be connected by straight lines along which the energies are calculated.
        num: int
            The number of points sampled on the path. On each sampled point the energies are obtained.
        isometric: bool
            If True, the subpaths connecting two subsequent points are equally long irrespective of the real distance. If False, the subpaths scale as the interpoint distances.

        Returns:
        --------
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
        '''
        pathcoords = self.get_pathcoords(*points)
        pathlabels = self.get_pathlabels(*points)

        assert pathcoords.shape[1] == 3, "Ensure that points are given as tuples of 3 float numbers."
        assert basis in ['rcpr', 'other'], "Basis can only be 'rcpr' or 'other'. In case of 'other' a new set of basisvects must be specified (ny default is None)."

        point_maps, pts, ks, energies = self.lsw.energy_along_path(pathcoords, 
                                                                   num=num, 
                                                                   isometric=isometric, 
                                                                   basis=basis, 
                                                                   new_basisvects=new_basisvects)
        dispersion = dict(points = points, 
                          pathcoords = pathcoords, 
                          pathlabels = pathlabels, 
                          pts = pts, 
                          ks = ks,
                          energies = energies, 
                          point_maps = point_maps)
        
        return dispersion
    
    def plot_bandstructure(self, 
                           dispersion: dict[str, tuple[list] | list[str] | npt.NDArray[np.float64] | list[npt.NDArray[np.float64]]], 
                           bands: list[int] = None,
                           show_params: bool = False,
                           show_chern: bool = False,
                           chern_gridsize: int = 20,
                           label_margin: float = 0.15, 
                           pltopts: dict = dict()) -> Figure:
        '''
        TODO
        '''
        pathlabels, point_maps, pts, energies = [dispersion[x] for x in ['pathlabels', 
                                                                         'point_maps', 
                                                                         'pts', 
                                                                         'energies']]
        
        # latex styling
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{lmodern}"})
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[(0.2298057, 0.298717966, 0.753683153, 0.8)])

        # figure setup
        fig = plt.gcf()
        plt.xlabel(r"path in $\bm{k}$-space", fontsize=14)
        plt.ylabel(r"$\varepsilon(\bm{k}) \, \, [\mathrm{meV}]$", fontsize = 14)
        plt.xticks(ticks = point_maps, labels = pathlabels, fontsize = 12)
        plt.yticks(fontsize = 14)
        plt.grid(which = 'both', linestyle = ':', alpha = 0.8)
        plt.tight_layout()
        plt.margins(x = 0, y = 0)
        plt.ylim(0, np.max(energies) + 0.5)
        
        # display system parameters in title
        if show_params:
            param_dict = self.lsw.parameters
            check_for_params = ['J', 'J1', 'J2', 'A', 'Dz', 'Dz1', 'Dz2', 'Dz3', 'chempot']
            title = '$' + ', '.join(f'{PlotManager.format_to_latex(key)}={abs(float(param_dict[key])) if key.startswith("Dz") or key == "D" else float(param_dict[key])}' 
                                for key in check_for_params 
                                if key in param_dict and float(param_dict[key]) != 0) + '$'
            plt.title(title, fontsize = 14)

        # plot selected bands only: specify a list of bands indexed from top to bottom
        if bands is None:
            bands = range(energies.shape[-1])

        # plotting bands
        for band in bands:
            plt.plot(pts, energies[:, band], **pltopts)

        # plotting chern numbers
        if show_chern:
            _, EVs = self.lsw.evgrid(chern_gridsize)

            chern_numbers = np.around(self.lsw.chern(EVs), 2)
            chern_numbers = chern_numbers[:len(chern_numbers) // 2]

            for band in bands:
                total = len(chern_numbers)
                number = chern_numbers[band]
            
                label_k, label_energy = PlotManager.find_label_position(pts, energies[:, band], energies, band, margin=label_margin)

                plt.text(label_k, label_energy, rf'$C_{{{total - band}}} = {int(number)}$', fontsize=14, ha='center', va='center')

        return fig
    
    def plot_bs_and_lat(self, 
                        dispersion: dict[str, tuple[list] | list[str] | npt.NDArray[np.float64] | list[npt.NDArray[np.float64]]],
                        a1,
                        a2,
                        basis,
                        BZ_type,
                        bond_cutoff='auto',
                        n_shells=1,
                        center=None,
                        view_size=None,
                        cell_origin=None,
                        vect_lw=2.0,
                        vect_head_width=None,
                        vect_head_length=None,
                        vect_label_offset_a1=None,
                        vect_label_offset_a2=None,
                        figsize = (12,5),
                        wratio = [2, 1],
                        hratio = [2, 1],
                        wspace = 0.05,
                        hspace = 0.1,
                        pltopts: dict = dict()) -> Figure:
        '''
        TODO
        '''
        pathlabels, point_maps, pts, energies = [dispersion[x] for x in ['pathlabels', 
                                                                         'point_maps', 
                                                                         'pts', 
                                                                         'energies']]
        
        # latex styling
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{lmodern}"})
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[(0.2298057, 0.298717966, 0.753683153, 0.8)])

        # figure setup
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, width_ratios=wratio, height_ratios=hratio,
                            wspace=wspace, hspace=hspace)

        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1])

        # bandstructure
        ax1.set_xlabel(r"path in $\bm{k}$-space", fontsize=14)
        ax1.set_ylabel(r"$\varepsilon(\bm{k}) \, \, [\mathrm{meV}]$", fontsize = 14)
        ax1.set_xticks(ticks = point_maps, labels = pathlabels, fontsize = 12)
        ax1.tick_params(axis='y', labelsize=14)
        ax1.grid(which = 'both', linestyle = ':', alpha = 0.8)
        ax1.margins(x = 0, y = 0)
        ax1.set_ylim(0, np.max(energies) + 0.5)

        ax1.plot(pts, energies[:, :], **pltopts)

        # lattice plot
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
        sites, _, _ = _build_sites(a1, a2, basis, n_shells=n_shells)

        # --- bond cutoff --------------------------------------------------------
        nn_dist = _nearest_neighbour_dist(sites)
        if bond_cutoff == 'auto':
            bond_cutoff = 1.05 * nn_dist
        draw_bonds = bond_cutoff is not None and bond_cutoff > 0

        # fig setup
        ax2.set_facecolor('white')
        ax2.set_aspect('equal')

        show_axes = False

        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_color('#333333')
        if not show_axes:
            ax2.set_xticks([])
            ax2.set_yticks([])
        else:
            ax2.tick_params(length=3, labelsize=9)

        # --- draw bonds ---------------------------------------------------------
        bond_color='#555555'
        bond_lw=1.5

        if draw_bonds:
            N = len(sites)
            diffs = sites[:, None, :] - sites[None, :, :]   # (N,N,2)
            dists = np.linalg.norm(diffs, axis=-1)           # (N,N)
            for i in range(N):
                for j in range(i + 1, N):
                    if dists[i, j] <= bond_cutoff:
                        ax2.plot(
                            [sites[i, 0], sites[j, 0]],
                            [sites[i, 1], sites[j, 1]],
                            color=bond_color, lw=bond_lw,
                            zorder=2, solid_capstyle='round',
                        )

        # --- draw sites (on top of bonds, no white edge gap) --------------------
        site_color='#1a1a2e'
        site_size=40

        ax2.scatter(
            sites[:, 0], sites[:, 1],
            s=site_size,
            color=site_color,
            zorder=4,
            linewidths=0,
        )

        # --- draw unit cell (n1=0, n2=0 → origin) ------------------------------
        cmap = plt.get_cmap('coolwarm')
        cnorm = Normalize(vmin=0, vmax=1)
        cell_color= cmap(cnorm(0.95))
        cell_lw=2.0
        cell_alpha=0.08
        _draw_unit_cell(ax2, a1, a2,
                        origin=cell_origin_cart,
                        color=cell_color,
                        lw=cell_lw,
                        alpha=cell_alpha)
        
        # --- draw lattice vector arrows -----------------------------------------
        vect_color= cmap(cnorm(0.05))

        head_w  = vect_head_width    if vect_head_width   is not None else 0.08 * nn_dist
        head_l  = vect_head_length   if vect_head_length  is not None else 0.12 * nn_dist
        offsets = [vect_label_offset_a1, vect_label_offset_a2] 
 
        for vec, label, manual_off in zip([a1, a2], [r'$\bm{a}_1$', r'$\bm{a}_2$'], offsets):
            # arrow starts at centroid
            ax2.arrow(centroid[0], centroid[1],
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
            ax2.text(lpos[0], lpos[1], label,
                    color=vect_color, fontsize=16,
                    ha='center', va='center', zorder=6)

        # --- viewport -----------------------------------------------------------
        if view_size is not None:
            half = float(view_size)
            ax2.set_xlim(center[0] - half, center[0] + half)
            ax2.set_ylim(center[1] - half, center[1] + half)
        else:
            all_x, all_y = sites[:, 0], sites[:, 1]
            pad = 0.6 * nn_dist
            cx, cy = center
            # centre the auto window on `center` while still showing all sites
            x_half = max(abs(all_x - cx).max() + pad, pad * 2)
            y_half = max(abs(all_y - cy).max() + pad, pad * 2)
            half = max(x_half, y_half)
            ax2.set_xlim(cx - half, cx + half)
            ax2.set_ylim(cy - half, cy + half)

        # BZ plot
        ax3.set_aspect('equal')
        ax3.axis('off')

        if BZ_type == 'square':
            scale = np.pi
            bz_verts = scale * np.array([[1,1],[-1,1],[-1,-1],[1,-1]])
            G = np.array([0, 0])
            X = scale * np.array([1, 0])
            M = scale * np.array([1, 1])

            path_pts = np.array([G, X, M, G])
            HSPs = [(r'$\Gamma$', G), (r'$\mathrm{M}$', M), (r'$\mathrm{X}$', X)]

        elif BZ_type == 'hexagonal':
            angles = np.radians(np.arange(6) * 60 + 90)
            scale = (4 * np.pi) / (3)
            bz_verts = scale * np.array([np.cos(angles), np.sin(angles)]).T
            G = np.array([0, 0])
            K = scale * np.array([np.cos(np.radians(30)), np.sin(np.radians(30))])
            M = (np.sqrt(3)/2 * scale, 0)

            path_pts = np.array([G, M, K, G])
            HSPs = [(r'$\Gamma$', G), (r'$\mathrm{M}$', M), (r'$\mathrm{K}$', K)]

        elif BZ_type == 'oblique':
            # reciprocal vectors for a1=(a,0), a2=(a/2, a*(1+sqrt(3)/2)), with a=1
            b1 = np.array([2*np.pi,       -2*np.pi*(2-np.sqrt(3))])
            b2 = np.array([0,          2*np.pi*2*(2-np.sqrt(3))])

            # convert fractional to Cartesian
            def frac_to_cart(f1, f2):
                return f1 * b1 + f2 * b2
            
            scale = 1.4

            G  = frac_to_cart(0, 0)
            X  = frac_to_cart(0.5, 0)
            Y  = frac_to_cart(0, 0.5)
            C  = frac_to_cart(0.5, 0.5)
            H  = frac_to_cart(0.4641016, 0.7320508)
            H1 = frac_to_cart(0.5358984, 0.2679492)

            bz_verts = np.array([Y, H, C, H1, X, H - 2*Y, -H, -H1, -H + 2*Y])
            # G = np.array([0, 0])
            # X = np.array([1/2, 0])
            # Y = np.array([0, 1/2])
            # C = np.array([1/2, 1/2])
            # H = np.array([0.4641016, 0.7320508])
            # H1 = np.array([0.5358984, 0.2679492])
            # G = [r'$\Gamma$', (0,   0,   0)]
            # X = [r'$X$',      (1/2, 0, 0)]
            # Y = [r'$Y$',      (0, 1/2, 0)]
            # C = [r'$C$',      (1/2, 1/2,   0)]
            # H = [r'$H$',      (0.4641016, 0.7320508,   0)]
            # H1 = [r'$H1$',      (0.5358984, 0.2679492,   0)]

            path_pts = np.array([G, Y, H, C, G, X, H1, G, H])
            HSPs = [(r'$\Gamma$', G), (r'$\mathrm{X}$', X), (r'$\mathrm{Y}$', Y), (r'$\mathrm{C}$', C), (r'$\mathrm{H}$', H), (' ' + r'$\mathrm{H1}$', H1)]

        ax3.plot(*np.append(bz_verts, [bz_verts[0]], axis=0).T, color='#1a1a2e', lw=1.5, zorder=2)
        
        ax3.plot(path_pts[:,0], path_pts[:,1], color=cell_color, lw=1.5, zorder=3)

        for label, pt in HSPs:
            ax3.scatter(*pt, color='#1a1a2e', s=30, zorder=4, linewidths=0)
            offset = 0.25 * scale * (pt / norm(pt) if norm(pt) > 1e-10 else np.array([-0.9,0]))
            ax3.text(*(pt + offset), label, fontsize=14, ha='center', va='center')

        # fig.tight_layout()

        return fig

    def plot_bs_and_chern(self, 
                          dispersion: dict[str, tuple[list] | list[str] | npt.NDArray[np.float64] | list[npt.NDArray[np.float64]]],
                          chern_gridsize: int = 20,
                          sigma = 0.05,
                          points = 1000,
                          figsize = (12,5),
                          wratio = [2, 1],
                          wspace = 0.05,
                          decimals = 0,
                          show_chern: bool = True,
                          label_margin: float = 0.15,
                          split = False,
                          loc='upper right',
                          loc2 = None,
                          pltopts: dict = dict()) -> Figure:
        '''
        TODO
        '''
        pathlabels, point_maps, pts, energies = [dispersion[x] for x in ['pathlabels', 
                                                                         'point_maps', 
                                                                         'pts', 
                                                                         'energies']]
        
        # latex styling
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{lmodern}"})
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[(0.2298057, 0.298717966, 0.753683153, 0.8)])

        # figure setup
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True, gridspec_kw={'width_ratios': wratio})
        fig.subplots_adjust(wspace=wspace)


        # bandstructure
        ax1.set_xlabel(r"path in $\bm{k}$-space", fontsize=14)
        ax1.set_ylabel(r"$\varepsilon(\bm{k}) \, \, [\mathrm{meV}]$", fontsize = 14)
        ax1.set_xticks(ticks = point_maps, labels = pathlabels, fontsize = 12)
        ax1.tick_params(axis='y', labelsize=14)
        ax1.grid(which = 'both', linestyle = ':', alpha = 0.8)
        ax1.margins(x = 0, y = 0)
        ax1.set_ylim(0, np.max(energies) + 0.5)


        ax1.plot(pts, energies[:, :], **pltopts)


        nbands = energies.shape[-1]

        # plotting chern numbers
        Es, EVs = self.lsw.evgrid(chern_gridsize)

        chern_numbers = np.around(self.lsw.chern(EVs), 2)
        chern_numbers = chern_numbers[:len(chern_numbers) // 2]

        if show_chern:
            for band in range(nbands):
                total = len(chern_numbers)
                number = chern_numbers[band]
            
                label_k, label_energy = PlotManager.find_label_position(pts, energies[:, band], energies, band, margin=label_margin)

                ax1.text(label_k, label_energy, rf'$C_{{{total - band}}} = {int(number)}$', fontsize=14, ha='center', va='center')

        # energy resolved chern number plot
        curv = self.lsw.curvature_grid(EVs)

        e_min, e_max = np.min(Es) - 3*sigma, np.max(Es) + 3*sigma
        energy_axis = np.linspace(e_min, e_max, points)
        chern_density = np.zeros((points, nbands))

        for b in range(nbands):
            nk1_flux, nk2_flux = curv.shape[1], curv.shape[2]
            
            energies_b = Es[:nk1_flux, :nk2_flux, b].ravel()
            flux_b = curv[b].ravel()
            
            diff = energy_axis[:, np.newaxis] - energies_b[np.newaxis, :]
            weights = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (diff / sigma)**2)
            
            chern_density[:, b] = - (weights @ flux_b) / (2 * np.pi)

        total_density = np.sum(chern_density, axis=1)

        if nbands == 2:
            colors = ['g', 'r']
        elif nbands == 3:
            colors = ['b', 'g', 'r']
        elif nbands == 4:
            colors = ['y', 'b', 'g', 'r']
        elif nbands == 6:
            colors = ['c', 'm', 'y', 'b', 'g', 'r']
        elif nbands == 12:
            colors = ['c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y', 'b', 'g', 'r']

        for b in range(nbands):
            c_val = - np.sum(curv[b]) / (2 * np.pi)
            if decimals == 0:
                val_str = rf'${c_val:.0f}$'
            elif decimals == 2:
                val_str = rf'${c_val:.2f}$'
            if val_str == r'$-0$' or val_str == r'$-0.00$':
                val_str = r'$ 0$'
            ax2.plot(chern_density[:, b], energy_axis, color=colors[b], label=rf'$C_{{{nbands - b}}} = \, \,$' + val_str)
            
        ax2.axvline(0, color='black', alpha=0.5)
        ax2.set_xlabel(r"$C(\varepsilon) \, [10^{-2}]$", fontsize=14)
        ax2.tick_params(axis='x', labelsize=12)
        ax2.tick_params(left=False)
        if split:
            handles, labels = ax2.get_legend_handles_labels()
            mid = len(handles) // 2

            leg1 = ax2.legend(handles[:mid], labels[:mid], frameon=False, loc=loc, handlelength=0.5)
            leg2 = ax2.legend(handles[mid:], labels[mid:], frameon=False, loc=loc2, handlelength=0.5)

            ax2.add_artist(leg1)
        else:
            ax2.legend(frameon=False, loc=loc)
        ax2.grid(which = 'both', linestyle = ':', alpha = 0.8)

        # fig.tight_layout()

        return fig
    
    def save_dispersion(self, 
                        dispersion:dict[str, tuple[list] | list[str] | npt.NDArray[np.float64] | list[npt.NDArray[np.float64]]], 
                        directory: str = None,
                        filename: str = None,
                        format: str = '.dat',
                        overwrite: bool = True) -> None:
        '''
        TODO
        '''
        if directory is None:
            directory = f'./results/{self.system}/dispersion_data/'
            if not os.path.isdir(directory):
                FileManager.create_unique_dir(directory)

        if filename is None:
            filename = f'disp_{self.system}'

            check_for_DMI = self.lsw.parameters
            if any(key.startswith('Dz') and float(check_for_DMI[key]) != 0 for key in check_for_DMI):
                filename += '_DMI'

        energies = dispersion['energies']
        shape = (energies.shape[0], energies.shape[1] + 1)
        write = np.zeros(shape)
        write[:, 0] = dispersion['pts']
        write[:, 1:] = energies

        assert format in ['.dat', '.txt'], "Choose .dat or .txt file format."

        filepath = directory + filename + format

        if not overwrite:
            filepath = FileManager.get_unique_fname(directory, filename, format)

        np.savetxt(filepath, write, fmt='%f')

    def save_plot(self, fig: Figure, directory: str = None, filename: str = None, format: str = '.pdf', overwrite: bool = False) -> None:
        '''
        TODO
        '''
        if directory is None:
            directory = f'./results/{self.system}/'
            if not os.path.isdir(directory):
                FileManager.create_unique_dir(directory)

        if filename is None:
            filename = f'bdstr_{self.system}'

            check_for = self.lsw.parameters
            if any(key.startswith('J2') and float(check_for[key]) != 0 for key in check_for):
                filename += '_NNN'
            if any(key.startswith('Dz') and float(check_for[key]) != 0 for key in check_for):
                filename += '_DMI'
            
        assert format in ['.pdf', '.png'], "Choose .pdf or .png image file format."

        filepath = directory + filename + format

        if not overwrite:
            filepath = FileManager.get_unique_fname(directory, filename, format)

        fig.savefig(filepath, facecolor='w', transparent=False, dpi = 300, bbox_inches='tight')

    def slab_bandplot(self,
                         n_unitcells: int,
                         k_space: str = 'full', 
                         mask_degen_bands: bool = True, 
                         edge_layers: int = 5,
                         states: list[tuple] = None) -> Figure:
        '''
        TODO
        '''
        assert k_space in ['full', 'half', 'positive'], "Choose k values in (-2pi, 2pi) -> 'full' / (-pi, pi) -> 'half' / (0, 2pi) -> positive."

        # latex styling
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{lmodern}"})
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

        if k_space == 'full':
            tick_positions = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1] # [-1, -0.5, 0, 0.5, 1]
            tick_labels = [r"$-2\pi$",r"",r"$-\pi$",r"", r"$0$", r"", r"$\pi$", r"", r"$2\pi$"] # [r"$-2\pi$",r"$-3\pi/2$",r"$-\pi$",r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
            ks = np.linspace(-1, 1, 1000)
        elif k_space == 'half':
            tick_positions = [-0.5, -0.25, 0, 0.25, 0.5]
            tick_labels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
            ks = np.linspace(-0.5, 0.5, 500)
        elif k_space == 'positive':
            tick_positions = [0, 0.25, 0.5, 0.75, 1]
            tick_labels = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
            ks = np.linspace(0, 1, 500)

        dir = self.lsw.lat.num_rcpr_vects[0]
        energies = np.empty((len(ks), 2 * self.lsw.lat.n_sublats))
        amps = np.empty((len(ks), self.lsw.lat.n_sublats, self.lsw.lat.n_sublats))
        # eigvects = np.empty((len(ks), 2 * self.lsw.lat.n_sublats, 2 * self.lsw.lat.n_sublats), dtype='complex')

        for i, j in enumerate(ks):
            k = j*dir # cut dir vector into len(ks)-1 pieces
            E, EV = self.lsw.Bogoliubov_trafo(k[0], k[1], k[2])
            amp = self.lsw.sublat_localization(EV)
            energies[i] = E
            amps[i] = amp
            # eigvects[i] = EV

        energies = energies[:, :len(energies[1]) // 2]

        loc_ids = edge_layers * int(self.lsw.lat.n_sublats / n_unitcells)

        loc = amps[:, :, -loc_ids:].sum(axis=-1) - amps[:, :, :loc_ids].sum(axis=-1)
        
        if mask_degen_bands:
            tol = 1e-5
            degen_mask = np.zeros_like(energies, dtype=bool)

            nk, nbands = energies.shape

            for i in range(nk):
                e_on_k = energies[i, :]

                for j in range(nbands):
                    if j > 0 and np.abs(e_on_k[j] - e_on_k[j-1]) < tol:
                        degen_mask[i, j] = True
                    if j < (nbands-1) and np.abs(e_on_k[j] - e_on_k[j+1]) < tol:
                        degen_mask[i, j] = True

            loc[degen_mask] = 0

        if states is not None:
            # complete figure setup
            fig = plt.figure(figsize=(14, 8), constrained_layout=True)
            gs_main = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.05)

            # bandstructure: left half
            ax_bs = fig.add_subplot(gs_main[0, 0])

            # spatial distribution: right half
            ax_sd = fig.add_subplot(gs_main[0, 1])

            # plot bands
            for i in range(nbands):
                bandplot = ax_bs.scatter(ks, energies[:,i], marker = '.', c = loc[:, i], edgecolor='none', cmap = 'coolwarm', vmin = -1.0, vmax = 1.0, s = 10, rasterized=True)

            # mark states
            for state in states:
                k_val = state[0]
                E_val = state[1]
                k_index = np.abs(ks - k_val).argmin()
                band_index = np.abs(energies[k_index] - E_val).argmin()

                ax_bs.scatter(ks[k_index], energies[k_index, band_index], facecolors = 'none', edgecolors= 'black', marker = 'o', linewidths = 1.5, s = 50, zorder = 3, rasterized=True)

                # label arrow angle and position
                angle = np.radians(state[4])
                label_x = ks[k_index] + state[3] * 0.05 * np.cos(angle)
                label_y = energies[k_index, band_index] + state[3] * 0.05 * np.sin(angle)

                # label options
                text_kwargs = {'fontsize': 14, 'ha': 'center', 'va': 'center'}
                arrow_kwargs = {'arrowstyle': '-', 'color': 'black', 'lw': 1.5, 'connectionstyle': 'arc3,rad=0'}
                ax_bs.annotate(state[2], 
                                xy=(ks[k_index], 
                                energies[k_index, band_index]), 
                                xytext=(label_x, label_y), 
                                arrowprops=arrow_kwargs, 
                                **text_kwargs)
                
            # customize plot
            ax_bs.set_xticks(tick_positions, tick_labels, fontsize = 16)
            ax_bs.set_xlabel(r'$k \, \, [a^{-1}]$', fontsize=16)
            ax_bs.tick_params(axis='y', labelsize=16)
            ax_bs.set_ylabel(r'$\varepsilon(k) \, \, [\mathrm{meV}]$', fontsize=16)
            ax_bs.margins(x=0, y=0)
            ax_bs.set_ylim(0, np.max(energies) + 0.2)

            # customize colorbar
            cbar_bs = fig.colorbar(bandplot, ax=ax_bs, pad = 0.01, orientation='horizontal', location='top', fraction=0.05, aspect=40)
            cbar_bs.set_label('edge localization', fontsize = 16)
            cbar_bs.set_ticks([-1, 0, 1])
            cbar_bs.set_ticklabels(['bottom', 'bulk', 'top'])
            cbar_bs.ax.tick_params(labelsize = 14)
            cbar_bs.ax.xaxis.set_ticks_position('bottom')
            cbar_bs.ax.xaxis.set_label_position('top')

            # get colors from colorbar for clear visualization
            cmap = plt.get_cmap('coolwarm')
            cnorm = Normalize(vmin=0, vmax=1)
            color_min = cmap(cnorm(0.05))
            color_max = cmap(cnorm(0.95))

            # plot spatial distribution
            sites = [i + 1 for i in range(self.lsw.lat.n_sublats)]

            ycoords = [self.lsw.lat.sublats[i].num_basisvect[1] for i in range(self.lsw.lat.n_sublats)]


            for state in states:
                k_val = state[0]
                E_val = state[1]
                k_index = np.abs(ks - k_val).argmin()
                band_index = np.abs(energies[k_index] - E_val).argmin()

                loc_amp = [amps[k_index, band_index, i] for i in range(self.lsw.lat.n_sublats)]

                ysort, locsort = zip(*sorted(zip(ycoords, loc_amp)))

                if np.average(loc_amp[:5]) > np.average(loc_amp[-5:]):
                    symbol = 'v:'
                    c_loc = color_min # idea: set color to norm of max amp in loc_amp
                else:
                    symbol = '^:'
                    c_loc = color_max

                ax_sd.plot(ysort, locsort, symbol, markersize=6, linewidth=2, color=c_loc, label='state ' + state[2], rasterized=True)

            # customize plot
            # first, last = 1, self.lsw.lat.n_sublats
            # ticks = ax_sd.get_xticks()
            # if ticks[0] != first:
            #     ticks = np.insert(ticks, 0,first)
            # if ticks[-1] != last:
            #     ticks = np.append(ticks, last)
            # ax_sd.set_xticks(ticks)
            ax_sd.tick_params(axis = 'x', labelsize=16)
            ax_sd.set_xlabel(r'sublattice $y$-coordinate $[a]$', fontsize=16)
            yticks = ax_sd.get_yticks()
            ax_sd.set_yticks(yticks[1::2])
            ax_sd.tick_params(axis = 'y', labelsize=16)
            ax_sd.set_ylabel(r'$\vert \mathbf{T}(q) \vert^2$ for selected $\bm{k}$ and $\varepsilon(k)$ on band $\lambda$', fontsize=16)
            ax_sd.margins(x=0.02, y=0.01)
            ax_sd.legend(
                loc='upper center',
                fontsize=16,
                frameon=True,
                framealpha=0.9,
                edgecolor='black',
                facecolor='white',)
            fig.text(0.775, 0.945, 'spatial distribution', ha='center', fontsize=18)

        else:
            fig, ax_bs = plt.subplots(figsize=(8, 6))

            # plot bands
            for i in range(nbands):
                bandplot = ax_bs.scatter(ks, energies[:,i], marker = '.', c = loc[:, i], edgecolor='none', cmap = 'coolwarm', vmin = -1.0, vmax = 1.0, s = 10, rasterized=True)

            # customize plot
            ax_bs.set_xticks(tick_positions, tick_labels, fontsize = 16)
            ax_bs.set_xlabel(r'$k \, \, [a^{-1}]$', fontsize=16)
            ax_bs.tick_params(axis='y', labelsize=16)
            ax_bs.set_ylabel(r'$\varepsilon(k) \, \, [\mathrm{meV}]$', fontsize=16)
            ax_bs.margins(x=0, y=0)
            ax_bs.set_ylim(0, np.max(energies) + 0.2)

            # customize colorbar
            cbar_bs = fig.colorbar(bandplot, ax=ax_bs, pad = 0.01, orientation='vertical', location='right', fraction=0.1)
            cbar_bs.set_label('edge localization', fontsize = 16)
            cbar_bs.set_ticks([-1, 0, 1])
            cbar_bs.set_ticklabels(['bottom', 'bulk', 'top'])
            cbar_bs.ax.tick_params(labelsize = 14)

        return fig
    
    def slab_latticeplot(self,
                         n_unitcells: int,
                         states: list[tuple],
                         direction = (1,0),
                         edge_layers: int = 5,
                         reps: int = 10,
                         gap_pad: float = 1.5,
                         gap_dot_scale: tuple[float] = (70, 5),
                         left_pad: float = 0,
                         right_pad: float = 0,
                         x_offset: float = 0.5,
                         fade_pad: float = 1.5,
                         fade_val: float = 0.3,
                         k_space: str = 'full') -> Figure:
        '''
        TODO
        '''
        assert k_space in ['full', 'half', 'positive'], "Choose k values in (-2pi, 2pi) -> 'full' / (-pi, pi) -> 'half' / (0, 2pi) -> positive."
        assert len(states) <= 2, "Choose at max. two states."

        # latex styling
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{lmodern}"})
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

        if k_space == 'full':
            ks = np.linspace(-1, 1, 1000)
        elif k_space == 'half':
            ks = np.linspace(-0.5, 0.5, 500)
        elif k_space == 'positive':
            ks = np.linspace(0, 1, 500)

        dir = direction[0] * self.lsw.lat.num_rcpr_vects[0] + direction[1] * self.lsw.lat.num_rcpr_vects[1]
        energies = np.empty((len(ks), 2 * self.lsw.lat.n_sublats))
        amps = np.empty((len(ks), self.lsw.lat.n_sublats, self.lsw.lat.n_sublats))

        for i, j in enumerate(ks):
            k = j*dir
            E, EV = self.lsw.Bogoliubov_trafo(k[0], k[1], k[2])
            amp = self.lsw.sublat_localization(EV)
            energies[i] = E
            amps[i] = amp

        energies = energies[:, :len(energies[1]) // 2]

        points = np.array([self.lsw.lat.sublats[i].num_basisvect[0:2] for i in range(self.lsw.lat.n_sublats)])

        k_val_A = states[0][0]
        E_val_A = states[0][1]
        k_index_A = np.abs(ks - k_val_A).argmin()
        band_index_A = np.abs(energies[k_index_A] - E_val_A).argmin()
        k_val_B = states[1][0]
        E_val_B = states[1][1]
        k_index_B = np.abs(ks - k_val_B).argmin()
        band_index_B = np.abs(energies[k_index_B] - E_val_B).argmin()

        # complete figure setup
        fig = plt.figure(figsize=(12, 8))
        gs_lat = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.1)

        ax_A = fig.add_subplot(gs_lat[0, 0])
        ax_B = fig.add_subplot(gs_lat[0, 1])

        # plot lattice
        # top and bottom layers
        id_range = edge_layers * int(self.lsw.lat.n_sublats / n_unitcells)

        bot_ids = [i for i in range(0, id_range)]
        top_ids = [self.lsw.lat.n_sublats - i for i in range(1, id_range + 1)]

        x_bot = points[bot_ids, 0]
        y_bot = points[bot_ids, 1]

        x_top = points[top_ids, 0]
        y_top = points[top_ids, 1]

        c_bot_A = amps[k_index_A, band_index_A][bot_ids]
        c_top_A = amps[k_index_A, band_index_A][top_ids]
        c_bot_B = amps[k_index_B, band_index_B][bot_ids]
        c_top_B = amps[k_index_B, band_index_B][top_ids]

        lognorm = SymLogNorm(0.001, vmin=0, vmax=1)

        # replicate in x direction, select fitting region
        # reps = 14

        x_shift = self.lsw.lat.num_vects[0,0]
        y_shift = self.lsw.lat.num_vects[1,1]
        top_shift_left = np.min(x_top)
        y_gap = np.min(y_top) - np.max(y_bot)
        buffer = gap_pad * y_shift
        top_shift_down = y_gap - buffer

        left_bound = np.max(x_bot) + left_pad + x_offset
        right_bound = np.min(x_bot) + (reps - 1) * x_shift + right_pad + x_offset

        x_bot_all = np.concatenate([x_bot + i * x_shift for i in range(reps)])
        x_top_all = np.concatenate([x_top + i * x_shift - top_shift_left for i in range(reps)])
        y_bot_all = np.concatenate([y_bot for _ in range(reps)])
        y_top_all = np.concatenate([y_top - top_shift_down for _ in range(reps)])
        c_bot_A_all = np.concatenate([c_bot_A for _ in range(reps)])
        c_top_A_all = np.concatenate([c_top_A for _ in range(reps)])
        c_bot_B_all = np.concatenate([c_bot_B for _ in range(reps)])
        c_top_B_all = np.concatenate([c_top_B for _ in range(reps)])

        # add fade out effect at bounds
        fade_mask_bot = (x_bot_all < left_bound + fade_pad) | (x_bot_all > right_bound - fade_pad)

        alphas_bot = np.ones_like(x_bot_all)
        alphas_bot[fade_mask_bot] = fade_val

        fade_mask_top = (x_top_all < left_bound + fade_pad) | (x_top_all > right_bound - fade_pad)

        alphas_top = np.ones_like(x_top_all)
        alphas_top[fade_mask_top] = fade_val

        latplot_A_bot = ax_A.scatter(x_bot_all, y_bot_all,
                                    marker="o", edgecolors="black", linewidths=0.8, s=30,
                                    c=c_bot_A_all, cmap='Reds', norm=lognorm,
                                    alpha=alphas_bot, rasterized=True)

        latplot_A_top = ax_A.scatter(x_top_all, y_top_all,
                                    marker="o", edgecolors="black", linewidths=0.8, s=30,
                                    c=c_top_A_all, cmap='Reds', norm=lognorm,
                                    alpha=alphas_top, rasterized=True)

        latplot_B_bot = ax_B.scatter(x_bot_all, y_bot_all,
                                    marker="o", edgecolors="black", linewidths=0.8, s=30,
                                    c=c_bot_B_all, cmap='Reds', norm=lognorm,
                                    alpha=alphas_bot, rasterized=True)

        latplot_B_top = ax_B.scatter(x_top_all, y_top_all,
                                    marker="o", edgecolors="black", linewidths=0.8, s=30,
                                    c=c_top_B_all, cmap='Reds', norm=lognorm,
                                    alpha=alphas_top, rasterized=True)
            
        # add dots to mark bulk gap
        center_x = (right_bound + left_bound) / 2
        center_y = (np.min(y_top)- top_shift_down + np.max(y_bot)) / 2

        for i in [-2, -1, 0, 1, 2]:
            for j in [-1, 0, 1]:
                ax_A.scatter(center_x + gap_dot_scale[0] * i / np.abs(center_x), center_y + gap_dot_scale[1] * j / np.abs(center_y), marker = ".", facecolor = 'black', s = 10, rasterized=True)
                ax_B.scatter(center_x + gap_dot_scale[0] * i / np.abs(center_x), center_y + gap_dot_scale[1] * j / np.abs(center_y), marker = ".", facecolor = 'black', s = 10, rasterized=True)

        # customize plots
        ax_A.set_aspect('equal') 
        ax_A.axis('off')
        ax_A.set_xlim(left_bound, right_bound)
        ax_A.set_title('state (A)', fontsize=16)

        ax_B.set_aspect('equal') 
        ax_B.axis('off')
        ax_B.set_xlim(left_bound, right_bound)
        ax_B.set_title('state (B)', fontsize=16)

        # customize colorbar
        cbar_lat = fig.colorbar(latplot_A_bot, ax=[ax_A, ax_B], pad = 0.05, orientation='horizontal', shrink = 0.5, aspect = 40)
        cbar_lat.set_label('sublattice localization', fontsize = 16)
        cbar_lat.set_ticks([1, 10**-1, 10**-2, 10**-3, 0])
        cbar_lat.ax.set_xticklabels([r"$1$", r"$10^{-1}$", r"$10^{-2}$", r"$10^{-3}$", r"$0$"])
        cbar_lat.ax.tick_params(labelsize = 14)
        cbar_lat.ax.xaxis.set_label_position('top')

        return fig

    # def plot_slab_complete(self,
    #                        width: int,
    #                        edge_layers: int,
    #                        state_A: list,
    #                        state_B: list,
    #                        k_space: str = 'positive',
    #                        mask_degen_bands: bool = True,
    #                        reps: int = 10,
    #                        padding: float = 0.25,
    #                        x_offset: float = 0.5) -> Figure:
    #     '''
    #     TODO could be improved a lot, atm just to make it work
    #     '''
    #     assert k_space in ['full', 'half', 'positive'], "Choose k values in (-2pi, 2pi) -> 'full' / (-pi, pi) -> 'half' / (0, 2pi) -> positive."

    #     # latex styling
    #     plt.rcParams.update({
    #         "text.usetex": True,
    #         "font.family": "serif",
    #         "text.latex.preamble": r"\usepackage{lmodern}"})
    #     plt.rcParams['text.usetex'] = True
    #     plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

    #     if k_space == 'full':
    #         tick_positions = [-1, -0.5, 0, 0.5, 1] #[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
    #         tick_labels = [r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$", r"$2\pi$"] #[r"$-2\pi$",r"$-3\pi/2$",r"$-\pi$",r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    #         ks = np.linspace(-1, 1, 1000)
    #     elif k_space == 'half':
    #         tick_positions = [-0.5, -0.25, 0, 0.25, 0.5]
    #         tick_labels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
    #         ks = np.linspace(-0.5, 0.5, 500)
    #     elif k_space == 'positive':
    #         tick_positions = [0, 0.25, 0.5, 0.75, 1]
    #         tick_labels = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    #         ks = np.linspace(0, 1, 500)

    #     dir = self.lsw.lat.num_rcpr_vects[0]
    #     energies = np.empty((len(ks), 2 * self.lsw.lat.n_sublats))
    #     amps = np.empty((len(ks), self.lsw.lat.n_sublats, self.lsw.lat.n_sublats))
    #     eigvects = np.empty((len(ks), 2 * self.lsw.lat.n_sublats, 2 * self.lsw.lat.n_sublats), dtype='complex')

    #     for i, j in enumerate(ks):
    #         k = j*dir # cut dir vector into len(ks)-1 pieces
    #         E, EV = self.lsw.Bogoliubov_trafo(k[0], k[1], k[2])
    #         amp = self.lsw.sublat_localization(EV)
    #         energies[i] = E
    #         amps[i] = amp
    #         eigvects[i] = EV

    #     energies = energies[:, :len(energies[1]) // 2]

    #     layers = 10
    #     loc_ids = layers * int(len(self.lsw.lat.sublats) / width)

    #     loc = amps[:, :, -loc_ids:].sum(axis=-1) - amps[:, :, :loc_ids].sum(axis=-1)
        
    #     if mask_degen_bands:
    #         tol = 1e-5
    #         degen_mask = np.zeros_like(energies, dtype=bool)

    #         nk, nbands = energies.shape

    #         for i in range(nk):
    #             e_on_k = energies[i, :]

    #             for j in range(nbands):
    #                 if j > 0 and np.abs(e_on_k[j] - e_on_k[j-1]) < tol:
    #                     degen_mask[i, j] = True
    #                 if j < (nbands-1) and np.abs(e_on_k[j] - e_on_k[j+1]) < tol:
    #                     degen_mask[i, j] = True

    #         loc[degen_mask] = 0

    #     points = np.array([self.lsw.lat.sublats[i].num_basisvect[0:2] for i in range(0, len(self.lsw.lat.sublats))])

    #     # complete figure setup
    #     fig = plt.figure(figsize=(14, 8))
    #     gs_main = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.1)

    #     # bandstructure: left half
    #     ax_bs = fig.add_subplot(gs_main[0, 0])

    #     # sublattice localization: right half
    #     gs_lat = gs_main[0, 1].subgridspec(1, 2, wspace=0)
    #     ax_A = fig.add_subplot(gs_lat[0, 0])
    #     ax_B = fig.add_subplot(gs_lat[0, 1])

    #     # plot bands
    #     for i in range(nbands):
    #         bandplot = ax_bs.scatter(ks, energies[:,i], marker = '.', c = loc[:, i], edgecolor='none', cmap = 'coolwarm', vmin = -1.0, vmax = 1.0, s = 10)

    #     # mark states
    #     k_lat_A = state_A[0][0]
    #     E_lat_A = state_A[0][1]
    #     k_index_A = np.abs(ks - k_lat_A).argmin()
    #     band_index_A = np.abs(energies[k_index_A] - E_lat_A).argmin()

    #     k_lat_B = state_B[0][0]
    #     E_lat_B = state_B[0][1]
    #     k_index_B = np.abs(ks - k_lat_B).argmin()
    #     band_index_B = np.abs(energies[k_index_B] - E_lat_B).argmin()

    #     ax_bs.scatter(ks[k_index_A], energies[k_index_A, band_index_A], facecolors = 'none', edgecolors= 'black', marker = 'o', linewidths = 1, s = 50, zorder = 3)

    #     ax_bs.text(ks[k_index_A] - 0.01, energies[k_index_A, band_index_A] + 0.05, '(A)', 
    #     fontsize=12, ha=state_A[1][0], va=state_A[1][1])

    #     ax_bs.scatter(ks[k_index_B],energies[k_index_B, band_index_B], facecolors = 'none', edgecolors= 'black', marker = 'o', linewidths = 1, s = 50, zorder = 3)

    #     ax_bs.text(ks[k_index_B] - 0.01, energies[k_index_B, band_index_B] + 0.05, '(B)', 
    #     fontsize=12, ha=state_B[1][0], va=state_B[1][1])

    #     # customize plot
    #     ax_bs.set_xticks(tick_positions, tick_labels, fontsize = 14)
    #     ax_bs.set_xlabel(r'$k \, \, [a^{-1}]$', fontsize=14)
    #     ax_bs.tick_params(axis='y', labelsize=14)
    #     ax_bs.set_ylabel(r'$\varepsilon(k) \, \, [\mathrm{meV}]$', fontsize=14)
    #     ax_bs.margins(x=0, y=0)
    #     ax_bs.set_ylim(0, np.max(energies) + 0.5)

    #     # customize colorbar
    #     cbar_bs = fig.colorbar(bandplot, ax=ax_bs, pad = 0.125, orientation='horizontal')
    #     cbar_bs.set_label('edge localization', fontsize = 14)
    #     cbar_bs.set_ticks([-1, 0, 1])
    #     cbar_bs.set_ticklabels(['bottom', 'bulk', 'top'])
    #     cbar_bs.ax.yaxis.set_label_coords(1.5, 0.5)
    #     cbar_bs.ax.tick_params(labelsize = 12)

    #     # plot lattice
    #     # top and bottom layers
    #     n_per_uc = int(len(self.lsw.lat.sublats) / width)
    #     uc_layers = edge_layers
    #     id_range = n_per_uc * uc_layers

    #     bot_ids = [i for i in range(0, id_range)]
    #     top_ids = [self.lsw.lat.n_sublats - i for i in range(1, id_range + 1)]

    #     x_bot = points[bot_ids, 0]
    #     y_bot = points[bot_ids, 1]

    #     x_top = points[top_ids, 0]
    #     y_top = points[top_ids, 1]

    #     c_bot_A = amps[k_index_A, band_index_A][bot_ids]
    #     c_top_A = amps[k_index_A, band_index_A][top_ids]
    #     c_bot_B = amps[k_index_B, band_index_B][bot_ids]
    #     c_top_B = amps[k_index_B, band_index_B][top_ids]

    #     lognorm = SymLogNorm(0.001, vmin=0, vmax=1)

    #     # replicate in x direction, select fitting region
    #     # reps = 10

    #     x_shift = self.lsw.lat.num_vects[0,0]
    #     y_shift = self.lsw.lat.num_vects[1,1]
    #     top_shift_left = np.min(x_top)
    #     y_gap = np.min(y_top) - np.max(y_bot)
    #     buffer = 1.5 * y_shift
    #     top_shift_down = y_gap - buffer

    #     # padding = 0.25
    #     # x_offset = 0.5
    #     left_bound = np.max(x_bot) - padding + x_offset
    #     right_bound = np.min(x_bot) + (reps - 1) * x_shift + padding + x_offset

    #     for i in range(reps):
    #         latplot_A = ax_A.scatter(x_bot + i * x_shift, y_bot,
    #                                 marker = "o", edgecolors = "black", linewidths = 0.8, s = 30,
    #                                 c = c_bot_A,
    #                                 cmap = 'Reds', norm = lognorm)
            
    #         latplot_A = ax_A.scatter(x_top + i * x_shift - top_shift_left, y_top - top_shift_down,
    #                                 marker = "o", edgecolors = "black", linewidths = 0.8, s = 30,
    #                                 c = c_top_A,
    #                                 cmap = 'Reds', norm = lognorm)
            
    #         latplot_B = ax_B.scatter(x_bot + i * x_shift, y_bot,
    #                                 marker = "o", edgecolors = "black", linewidths = 0.8, s = 30,
    #                                 c = c_bot_B,
    #                                 cmap = 'Reds', norm = lognorm)
            
    #         latplot_B = ax_B.scatter(x_top + i * x_shift - top_shift_left, y_top - top_shift_down,
    #                                 marker = "o", edgecolors = "black", linewidths = 0.8, s = 30,
    #                                 c = c_top_B,
    #                                 cmap = 'Reds', norm = lognorm)
            
    #     # add dots to mark bulk gap
    #     center_x = (right_bound + left_bound) / 2
    #     center_y = (np.min(y_top)- top_shift_down + np.max(y_bot)) / 2

    #     for i in [-2, -1, 0, 1, 2]:
    #         for j in [-1, 0, 1]:
    #             ax_A.scatter(center_x + 30*i / np.abs(center_x), center_y + 5 * j / np.abs(center_y), marker = ".", facecolor = 'black', s = 10)
    #             ax_B.scatter(center_x + 30*i / np.abs(center_x), center_y + 5 * j / np.abs(center_y), marker = ".", facecolor = 'black', s = 10)

    #     # customize plots
    #     ax_A.set_aspect('equal') 
    #     ax_A.axis('off')
    #     ax_A.set_xlim(left_bound, right_bound)
    #     ax_A.set_title('state (A)', fontsize=14, y=-0.1, pad=15)

    #     ax_B.set_aspect('equal') 
    #     ax_B.axis('off')
    #     ax_B.set_xlim(left_bound, right_bound)
    #     ax_B.set_title('state (B)', fontsize=14, y=-0.1, pad=15)

    #     # customize colorbar
    #     cbar_lat = fig.colorbar(latplot_A, ax=[ax_A, ax_B], pad = 0.125, orientation='horizontal')
    #     cbar_lat.set_label('sublattice localization', fontsize = 14)
    #     cbar_lat.set_ticks([1, 10**-1, 10**-2, 10**-3, 0])
    #     cbar_lat.ax.set_xticklabels([r"$1$", r"$10^{-1}$", r"$10^{-2}$", r"$10^{-3}$", r"$0$"])
    #     cbar_lat.ax.tick_params(labelsize = 12)

    #     return fig
    


class HighSymmetryPoints:
    '''
    Auxilliary class for assigning and storing high-symmetry point information of an arbitrary type of lattice.
    '''
    @staticmethod
    def angle_between(v1: npt.NDArray[np.float64], v2: npt.NDArray[np.float64]) -> np.float64:
        '''
        Compute the angle ∈ [0, pi] between two vectors v1 and v2.
        
        Arguments:
        ----------
        v1: npt.NDArray
            First vector.
        v2: npt.NDArray
            Second vector.

        Return:
        -------
        np.arccos(c): np.float64
            Angle between the vectors.
        '''
        c = np.dot(v1, v2) / (norm(v1) * norm(v2))
        return np.arccos(c)
    


class Generic(BandStructure1D):
    '''
    Arbitrary symmetry class.
    '''
    def __init__(self, lsw: LinearSpinWave, system: str) -> None:
        '''
        Arguments:
        ----------
        lsw: LinearSpinWave
            An instance of the LinearSpinWave class.
        system: str
            System name for file managing.
        '''
        hsp = HighSymmetryPoints()

        hsp.g = [r'$\Gamma$', (0, 0, 0)]
        hsp.a = [r'A', (1/2, 0, 0)]
        hsp.ap = [r'A$^\prime$', (-1/2, 0, 0)]
        hsp.b = [r'B', (0, 1/2, 0)]
        hsp.bp = [r'B$^\prime$', (0, -1/2, 0)]
        hsp.c = [r'C', (0, 0, 1/2)]
        hsp.cp = [r'C$^\prime$', (0, 0, -1/2)]
        hsp.l = [r'L', (1/2, 1/2, 0)]
        self.hsp = hsp
        super().__init__(lsw, system)



class Slab(BandStructure1D):
    '''
    Container class for setting up slab lattice plots.
    '''
    def __init__(self, lsw: LinearSpinWave, system: str) -> None:
        '''
        Arguments:
        ----------
        lsw: LinearSpinWave
            An instance of the LinearSpinWave class.
        system: str
            System name for file managing.
        '''
        super().__init__(lsw, system)



class Cubic(BandStructure1D):
    '''
    Contains information about high-symmetry points of cubic (reciprocal) lattices.
    '''
    def __init__(self, lsw: LinearSpinWave, system: str) -> None:
        '''
        Arguments:
        ----------
        lsw: LinearSpinWave
            An instance of the LinearSpinWave class.
        system: str
            System name for file managing.
        '''
        hsp = HighSymmetryPoints()
        
        # set up high-symmetry points: [display name, coords]
        hsp.g = [r'$\Gamma$', (0, 0, 0)]
        hsp.m = [r'M', (1/2, 1/2, 0)]
        hsp.r = [r'R', (1/2, 1/2, 1/2)]
        hsp.x = [r'X', (1/2, 0, 0)]
        self.hsp = hsp
        super().__init__(lsw, system)



class Hexagonal(BandStructure1D):
    '''
    Contains information about high-symmetry points of hexagonal (reciprocal) lattices.
    '''
    def __init__(self, lsw: LinearSpinWave, system: str) -> None:
        '''
        Arguments:
        ----------
        lsw: LinearSpinWave
            An instance of the LinearSpinWave class.
        system: str
            System name for file managing.
        '''
        r1, r2, _ = lsw.lat.num_rcpr_vects
        ang12 = HighSymmetryPoints.angle_between(r1, r2)
        eps = 1e-3

        # set up high-symmetry points: [display name, coords, internal name]
        hsp = HighSymmetryPoints()
        hsp.g = [r'$\Gamma$', (0, 0, 0), 'G']
        hsp.m = [r'M', (1/2, 0, 0), 'M']
        if abs(ang12 - np.pi/3) < eps:
            hsp.k = [r'K', (1/3, 1/3, 0), 'K']
            hsp.kp = [r'K$^\prime$', (-1/3, -1/3, 0), 'Kp']
        elif abs(ang12 - 2*np.pi/3) < eps:
            hsp.k = [r'K', (1/3, -1/3, 0), 'K']
            hsp.kp = [r'K$^\prime$', (-1/3, 1/3, 0), 'Kp']
        else:
            raise ValueError(f'Angle between in-plane reciprocal vectors needs to be 60° or 120°, not {ang12/np.pi*180}°')
        hsp.a = [r'A', (0, 0, 1/2), 'A']
        hsp.l = [r'L', (1/2, 0, 1/2), 'L']
        hsp.h = [r'H', (1/3, 1/3, 1/2), 'H']
        self.hsp = hsp
        super().__init__(lsw, system)



class Orthorhombic(BandStructure1D):
    '''
    Contains information about high-symmetry points of orthorombic (reciprocal) lattices.
    '''
    def __init__(self, lsw: LinearSpinWave, system: str) -> None:
        '''
        Arguments:
        ----------
        lsw: LinearSpinWave
            An instance of the LinearSpinWave class.
        system: str
            System name for file managing.
        '''
        hsp = HighSymmetryPoints()

        # set up high-symmetry points: [display name, coords]
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
        super().__init__(lsw, system)



class Oblique(BandStructure1D):
    '''
    Contains information about high-symmetry points of oblique (reciprocal) lattices.
    '''
    def __init__(self, lsw: LinearSpinWave, system: str) -> None:
        '''
        Arguments:
        ----------
        lsw: LinearSpinWave
            An instance of the LinearSpinWave class.
        system: str
            System name for file managing.
        '''
        hsp = HighSymmetryPoints()

        # set up high-symmetry points: [display name, coords]
        hsp.g = [r'$\Gamma$', (0, 0, 0)]
        hsp.y = [r'Y', (1/2, 0, 0)]
        hsp.h = [r'H', (0.53589838, -0.26794919, 0)]
        hsp.c = [r'C', (1/2, -1/2, 0)]
        hsp.h1 = [r'H1', (0.46410162, -0.73205081, 0)]
        hsp.x = [r'X', (0, -1/2, 0)]
        
        self.hsp = hsp
        super().__init__(lsw, system)



class Oblique2(BandStructure1D):
    '''
    Contains information about high-symmetry points of oblique (reciprocal) lattices.
    '''
    def __init__(self, lsw: LinearSpinWave, system: str) -> None:
        '''
        Arguments:
        ----------
        lsw: LinearSpinWave
            An instance of the LinearSpinWave class.
        system: str
            System name for file managing.
        '''
        hsp = HighSymmetryPoints()

        # set up high-symmetry points: [display name, coords]
        hsp.g = [r'$\Gamma$', (0, 0, 0)]
        hsp.y = [r'Y', (0, 0.5, 0)]
        hsp.a1 = [r'A1', (0.46410162, 0.26794919, 0)]
        hsp.x = [r'X', (0.53589838, -0.26794919, 0)]
        
        self.hsp = hsp
        super().__init__(lsw, system)



class ObliqueAFLOW(BandStructure1D):
    '''
    Contains information about high-symmetry points of oblique (reciprocal) lattices.
    '''
    def __init__(self, lsw: LinearSpinWave, system: str) -> None:
        '''
        Arguments:
        ----------
        lsw: LinearSpinWave
            An instance of the LinearSpinWave class.
        system: str
            System name for file managing.
        '''
        hsp = HighSymmetryPoints()

        t1, t2, _ = lsw.lat.num_vects  # AFLOW uses real space vects
        ang12 = HighSymmetryPoints.angle_between(t1, t2) # 75° or 5pi/12 for elongated triangular
        b = norm(t1)    # 1
        c = norm(t2)    # sqrt(2 + sqrt(3))
        
        eta = (1 - (b / c) * np.cos(ang12)) / (2 * (np.sin(ang12))**2) # 2*sqrt(3) - 3 ~ 0.4641016
        nu = 1/2 - eta * (c / b) * np.cos(ang12)                       # 2 - sqrt(3) ~ 0.2679492

        # set up high-symmetry points: [display name, coords]
        hsp.g = [r'$\Gamma$', (0, 0, 0)]
        hsp.x = [r'X', (1/2, 0, 0)]
        hsp.y = [r'Y', (0, 1/2, 0)]
        hsp.y1 = [r'Y1', (0, -1/2, 0)]
        hsp.z = [r'Z', (0, 0, 1/2)]
        hsp.c = [r'C', (1/2, 1/2, 0)]
        hsp.h = [r'H', (eta, 1 - nu, 0)]        # (0.4641016, 0.7320508)
        hsp.h1 = [r'H1', (1 - eta, nu, 0)]      # (0.5358984, 0.2679492)
        hsp.h2 = [r'H2', (eta, - nu, 0)]        # (0.4641016, - 0.2679492)
        
        self.hsp = hsp
        super().__init__(lsw, system)



class FileManager:
    '''
    Utility class for managing names and directories for generated files.
    '''
    @staticmethod
    def get_unique_fname(directory: str, filename: str, format: str) -> str:
        '''
        TODO
        '''
        filepath = directory + filename + format
    
        # if file does not exist, return original filepath
        if not os.path.exists(filepath):
            return filepath
        
        # find next available number
        counter = 2
        while True:
            next_file = f"{filename}_{counter}{format}"
            next_path = directory + next_file + format
            
            if not os.path.exists(next_path):
                return next_path
            
            counter += 1
    
    @staticmethod
    def get_unique_dname(basename: str, modifier: Callable = None) -> str:
        '''
        Generate an unique directory name and return it.

        Arguments:
        ----------
        basename : str
            Name of the directory that will be used if no other directory exists
            with the same name.
        modifier : function, optional
            See 'get_unique_fname'.
        '''
        if not os.path.isdir(basename):
            return basename
        if modifier is None:
            modifier = lambda i: '_copy' if i == 1 else f'_copy_{i}'
        i = 1
        while True:
            newname = basename + modifier(i)
            i += 1
            if not os.path.isdir(newname):
                return newname

    @staticmethod
    def create_unique_dir(basename: str, modifier: Callable = None) -> str:
        '''
        Create new directory and return its name.

        Arguments:
        ----------
        basename : str
            Name of the directory that will be used if no other directory exists
            with the same name.
        modifier : function, optional
            See 'get_unique_fname'.
        '''
        dname = FileManager.get_unique_dname(basename, modifier)
        os.makedirs(dname)
        return dname
    


class PlotManager():
    '''
    Utility class for managing titles and labels of plots.
    '''
    @staticmethod
    def format_to_latex(key: str) -> str:
        '''
        TODO
        '''
        # chemical potential
        if key == 'chempot':
            return r'\mu'
        
        # single letter J
        if key == 'J':
            return 'J'
        
        # single letter A
        if key == 'A':
            return 'A'
        
        # J with number
        if key.startswith('J') and len(key) > 1:
            num = key[1:]
            if num == '1':
                return r'J_{\mathrm{N}}'
            elif num == '2':
                return r'J_{\mathrm{NN}}'
            # Add more cases if needed
        
        # Dz variations
        if key.startswith('Dz'):
            if key == 'Dz':
                return '|D|'
            else:
                # Dz1, Dz2, etc. -> D_1, D_2, etc.
                num = key[2:]
                return f'|D_{num}|'
            
        return key
    
    @staticmethod
    def find_label_position(k_points: npt.NDArray, band_energies: npt.NDArray, all_bands_energies: npt.NDArray, band_id: int, margin: float = 0.15):
        """
        Find a good position for a label near the band.
        
        Arguments:
        ----------
        k_points:
            1D array of k-points.
        band_energies: 
            1D array of energies for this specific band.
        all_bands_energies: 
            2D array (n_points, n_bands) of all band energies.
        band_id: 
            Index of the current band.
        margin: 
            Minimum distance to be maintained from bands.
        """
        n_bands = all_bands_energies.shape[1]
        
        # Try positions at different k-point indices, with bias towards middle
        k_indices = [len(k_points) // 2,  # Try middle first
                    3 * len(k_points) // 8,
                    5 * len(k_points) // 8,
                    len(k_points) // 4,
                    3 * len(k_points) // 4]
        
        best_pos = None
        best_score = -np.inf
        
        for k_idx in k_indices:
            k_val = k_points[k_idx]
            energy_val = band_energies[k_idx]
            
            # Try positions above and below the band
            for offset_multiplier in [1, -1, 1.5, -1.5, 2, -2]:
                offset = offset_multiplier * margin
                test_energy = energy_val + offset
                
                # Calculate minimum distance to all bands in a neighborhood
                min_dist = np.inf
                # Check not just at one k-point, but in a small window around it
                window_size = max(1, len(k_points) // 20)
                k_start = max(0, k_idx - window_size)
                k_end = min(len(k_points), k_idx + window_size + 1)
                
                for check_k_idx in range(k_start, k_end):
                    for other_band_idx in range(n_bands):
                        other_energy = all_bands_energies[check_k_idx, other_band_idx]
                        # Distance from label position to the band point
                        k_dist = abs(k_points[check_k_idx] - k_val)
                        energy_dist = abs(test_energy - other_energy)
                        # Weight energy distance more heavily for nearby k-points
                        weight = 1.0 / (1.0 + k_dist * 10)
                        effective_dist = energy_dist * weight
                        min_dist = min(min_dist, effective_dist)
                
                # Score this position: prioritize good clearance and middle positions
                if min_dist >= margin * 0.8:  # Only consider if reasonably clear
                    # Bias towards middle: score is higher for positions near k=0.5
                    middle_bias = 1.0 - 2 * abs(k_val - 0.5)  # 1.0 at middle, 0.0 at edges
                    
                    # Bias towards below: give bonus if label is below the band
                    below_bias = 0.8 if offset < 0 else 0.0
                    
                    score = min_dist * (1.0 + middle_bias + below_bias)
                    
                    if score > best_score:
                        best_score = score
                        best_pos = (k_val, test_energy)
        
        # Fallback if no good position found
        if best_pos is None:
            k_idx = len(k_points) // 2
            best_pos = (k_points[k_idx], band_energies[k_idx] + margin)
        
        return best_pos