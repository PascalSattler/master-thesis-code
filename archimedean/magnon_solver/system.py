"""
system.py

Intermediate data container for the magnon solver. Receives numerical 
data from Parser and organises it into a validated, physics-consistent 
structure ready for evaluation by the Hamiltonian class.

Responsibilities:
    - store lattice and reciprocal lattice vectors
    - store site/sublattice data
    - auto-symmetrize and validate interactions
    - build a neighbor list organised by reference sublattice
"""

import numpy as np
from typing import Any



class SpinSystem:
    """
    Intermediate data container representing a magnonic spin system.

    Receives the parsed numerical outputs from Parser and organises them
    into a validated, solver-ready structure.

    Parameters
    ----------
    translation_vectors : list[np.ndarray]
        List of three lattice translation vectors, each of shape (3,).
    spin_data : list[dict]
        List of parsed site dictionaries from Parser. Each dict contains:
            'sublattice'            : int
            'basis_vector'          : np.ndarray of shape (3,)
            'local_frame'           : np.ndarray of shape (3,3), rows are [e1, e2, e3]
            'spin_length'           : float
            'gyromagnetic_matrix'   : np.ndarray of shape (3,3)
    interaction_data : list[dict]
        List of parsed interaction dictionaries from Parser. Each dict contains:
            'reference_sublat'  : int
            'neighbor_sublat'   : int
            'difference_vector' : np.ndarray of shape (3,)  (in lattice coordinates)
            'interaction_matrix': np.ndarray of shape (3,3)
    parameters : dict
        Dictionary of resolved numerical parameters (floats, arrays).
    symmetrize : bool, optional
        If True (default), automatically generate the reverse interaction
        for each entry, guaranteeing J_ji = J_ij^T. If False, then both 
        directions must be explicitly specified. In this case the consistency 
        of both directions is checked.
    atol : float, optional
        Absolute tolerance for symmetry validation checks. Default 1e-8.
    """

    def __init__(
        self,
        translation_vectors: list[np.ndarray],
        spin_data: list[dict],
        interaction_data: list[dict],
        parameters: dict[str, int | float | np.ndarray],
        symmetrize: bool = True,
        atol: float = 1e-8,
    ) -> None:

        self.parameters = parameters
        # self._parameters_changed = False  # flag for Hamiltonian refresh; NOTE for now was moved to parser
        self.symmetrize = symmetrize
        self.atol = atol

        # lattice vectors stored as rows of (3,3) array
        self.lattice_vectors = np.array(translation_vectors, dtype=float)

        # site data stored as provided, indexed by position in list
        self.spin_data = spin_data

        # build sublattice index set (no duplicates) for validation
        self._sublat_indices: set[int] = {
            site['sublattice'] for site in self.spin_data
        }

        # process and validate interactions
        self._raw_inters = interaction_data
        self.interaction_data = self._process_interactions()

        # build derived quantities
        self.reciprocal_vectors = self._compute_reciprocal_vectors()
        self.neighbor_list = self._build_neighbor_list()

    # ------------------------------------------------------------------
    # Properties of the system class
    # ------------------------------------------------------------------

    # advantage over self.xyz attributes: 
    # 1) is computed on demand from the current state of the object, thus if the 
    #    instance is modified after construction, there is no need for it to be 
    #    updated
    # 2) is read only, thus protected from beeing overridden/modified by methods
    #    calculating quantities derived from them

    @property
    def n_sublats(self) -> int:
        """Number of distinct sublattices in the system."""
        return len(self.spin_data)


    @property
    def n_interactions(self) -> int:
        """Total number of interactions (after symmetrization if applied)."""
        return len(self.interaction_data)

    # ------------------------------------------------------------------
    # Interaction processing
    # ------------------------------------------------------------------

    def _process_interactions(self) -> list[dict]:
        """
        Process raw interactions: validate sublattice indices, then either
        symmetrize automatically or check consistency manually depending
        on the symmetrize toggle.

        Returns
        -------
        interactions : list[dict]
            Processed and validated interaction list.
        """
        # validate that all referenced sublattice indices exist
        self._validate_sublattice_indices(self._raw_inters)

        if self.symmetrize:
            return self._symmetrize_interactions(self._raw_inters)
        
        else:
            self._validate_interaction_symmetry(self._raw_inters)
            return list(self._raw_inters)
        

    def _symmetrize_interactions(self, interactions: list[dict]) -> list[dict]:
        """
        Auto-generate reverse interactions to enforce J_ji = J_ij^T.

        For each interaction i->j with difference vector d and matrix J,
        creates the reverse interaction j->i with difference vector -d
        and matrix J^T, unless it already exists in the input.

        Self-interactions (i==j, d=0), e.g. on-site easy-axis anisotropy A, 
        are not symmetrized.

        Returns
        -------
        symmetrized : list of dict
            Complete interaction list with auto-generated reverses.
        """
        symmetrized = []
        original_inters = interactions

        for inter in original_inters:
            # always add the original interaction (never deduplicate the input)
            symmetrized.append(inter)

            i = inter['reference_sublat']
            j = inter['neighbor_sublat']
            d = tuple(inter['difference_vector'])

            # skip reverse symmetrization for self-interactions
            if i == j and d == (0, 0, 0):
                continue

            # check if reverse already exists in original input
            reverse_d = tuple(-np.array(d))
            reverse_exists = any(
                orig['reference_sublat'] == j and
                orig['neighbor_sublat'] == i and
                tuple(orig['difference_vector']) == reverse_d
                for orig in original_inters
            )

            if not reverse_exists:
                # create reverse interaction
                J_original = inter['interaction_matrix']
                J_reverse = J_original.T

                reverse_interaction = {
                    'reference_sublat': j,
                    'neighbor_sublat': i,
                    'difference_vector': np.array(reverse_d),
                    'interaction_matrix': J_reverse,
                }

                symmetrized.append(reverse_interaction)

        return symmetrized


    def _validate_interaction_symmetry(self, interactions: list[dict]) -> None:
        """
        When symmetrize = False, verify that for every interaction from i to j
        with matrix J_ij and difference vector d, there exists a corresponding
        reverse interaction from j to i with matrix J_ij^T and difference
        vector -d.

        Checks symmetric (exchange) and antisymmetric (DMI) parts separately
        for clearer error messages.

        Raises
        ------
        ValueError
            If any required reverse interaction is missing or inconsistent.
        """
        key_to_interaction = {
            self._make_key(
                inter['reference_sublat'],
                inter['neighbor_sublat'],
                inter['difference_vector']
            ): inter
            for inter in interactions
        }

        checked_pairs: set[tuple] = set()

        for inter in interactions:
            i = inter['reference_sublat']
            j = inter['neighbor_sublat']
            d = inter['difference_vector']
            J_ij = inter['interaction_matrix']

            forward_key = self._make_key(i, j, d)
            reverse_key = self._make_key(j, i, -d)

            # skip if pair was already checked
            if (forward_key, reverse_key) in checked_pairs:
                continue
            checked_pairs.add((forward_key, reverse_key))
            checked_pairs.add((reverse_key, forward_key))

            # check if reverse interaction exists
            if reverse_key not in key_to_interaction:
                raise ValueError(
                    f"SpinSystem: interaction from sublattice {i} to {j} with "
                    f"difference vector {d} has no reverse interaction from {j} to {i} "
                    f"with difference vector {-d}. "
                    f"Set symmetrize = True to generate it automatically."
                )

            J_ji = key_to_interaction[reverse_key]['interaction_matrix']

            # decompose into symmetric and anti-symmetric parts
            J_ij_sym  = (J_ij + J_ij.T) / 2
            J_ji_sym  = (J_ji + J_ji.T) / 2
            J_ij_asym = (J_ij - J_ij.T) / 2
            J_ji_asym = (J_ji - J_ji.T) / 2

            # symmetric part: check J_ij_sym == J_ji_sym
            if not np.allclose(J_ij_sym, J_ji_sym, atol=self.atol):
                raise ValueError(
                    f"SpinSystem: symmetric (exchange) part of interaction matrix "
                    f"between sublattices {i} and {j} is inconsistent: "
                    f"J_ij_sym != J_ji_sym."
                )

            # anti-symmetric part: check J_ij_asym == - J_ji_asym
            if not np.allclose(J_ij_asym, -J_ji_asym, atol=self.atol):
                raise ValueError(
                    f"SpinSystem: antisymmetric (DMI) part of interaction matrix "
                    f"between sublattices {i} and {j} is inconsistent: "
                    f"J_ij_asym != - J_ji_asym."
                )

    # ------------------------------------------------------------------
    # Neighbor list
    # ------------------------------------------------------------------

    def _build_neighbor_list(self) -> dict[int, list[dict]]:
        """
        Organise interactions into a dictionary keyed by reference sublattice
        index for fast lookup during Hamiltonian construction.

        Returns
        -------
        neighbor_list : dict[int, list[dict]]
            Maps each reference sublattice index to a list of its interactions.
        """
        neighbor_list: dict[int, list[dict]] = {
            idx: [] for idx in self._sublat_indices
        }

        for inter in self.interaction_data:
            ref = inter['reference_sublat']
            neighbor_list[ref].append(inter)

        return neighbor_list

    # ------------------------------------------------------------------
    # Reciprocal lattice vectors
    # ------------------------------------------------------------------

    def _compute_reciprocal_vectors(self) -> np.ndarray:
        """
        Compute the reciprocal lattice vectors from the real-space
        translation vectors using the standard relation:

            b_i = 2 * pi * (a_j x a_k) / (a_i * (a_j x a_k))

        Returns
        -------
        reciprocal_vectors : np.ndarray of shape (3,3)
            Rows are reciprocal lattice vectors [b1, b2, b3].

        Raises
        ------
        ValueError
            If the lattice vectors are linearly dependent (zero volume).
        """
        a1, a2, a3 = self.lattice_vectors

        vol = np.dot(a1, np.cross(a2, a3))

        if np.isclose(vol, 0.0, atol=self.atol):
            raise ValueError(
                "SpinSystem: lattice vectors are linearly dependent "
                "(unit cell volume is zero)."
            )

        b1 = 2 * np.pi * np.cross(a2, a3) / vol
        b2 = 2 * np.pi * np.cross(a3, a1) / vol
        b3 = 2 * np.pi * np.cross(a1, a2) / vol

        return np.array([b1, b2, b3])

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _validate_sublattice_indices(self, interactions: list[dict]) -> None:
        """
        Check that all sublattice indices referenced in the interaction
        list correspond to sites defined in spin_data.

        Raises
        ------
        ValueError
            If any referenced sublattice index is undefined.
        """
        for i, inter in enumerate(interactions):
            for key in ('reference_sublat', 'neighbor_sublat'):
                idx = inter[key]
                if idx not in self._sublat_indices:
                    raise ValueError(
                        f"SpinSystem: interaction {i} references sublattice "
                        f"index {idx}, which is not defined in spin_data. "
                        f"Defined indices are: {sorted(self._sublat_indices)}."
                    )

    # ------------------------------------------------------------------
    # Key management
    # ------------------------------------------------------------------

    # interactions are uniquely identified by three quantities: ref/nbr sublat ids and
    # their difference vector; diffvec is an array (mutable object) and therefore not
    # hashable -> construct a unique key from this data which can be used in a set

    # hashing quantities has advantages for comparison checks, e.g. looking up if an
    # interaction already exists, because hash lookup from memory is approx O(1) regardless
    # of the amound of interactions, whereas the interaction list must be looped over every
    # time if checked directly  

    @staticmethod
    def _make_key(
        ref: int,
        neighbor: int,
        d: np.ndarray,
        decimals: int = 8
    ) -> tuple:
        """
        Create a hashable key for an interaction from its sublattice
        indices and difference vector.

        Parameters
        ----------
        ref : int
            Reference sublattice index.
        neighbor : int
            Neighbor sublattice index.
        d : np.ndarray of shape (3,)
            Difference vector in lattice coordinates.
        decimals : int
            Rounding precision for the difference vector components.

        Returns
        -------
        key : tuple
            Hashable tuple (ref, neighbor, d0, d1, d2).
        """
        d_rounded = tuple(np.round(d, decimals=decimals)) 
        # TODO: REMOVE; d is in lattice coords, which are always integers, so this is not necessary

        return (ref, neighbor) + d_rounded


    @staticmethod
    def _interaction_key_set(interactions: list[dict]) -> set[tuple]:
        """
        Build a set of hashable keys for all interactions in a list.
        """
        return {
            SpinSystem._make_key(
                inter['reference_sublat'],
                inter['neighbor_sublat'],
                inter['difference_vector'],
            )
            for inter in interactions
        }
    
    # ------------------------------------------------------------------
    # Updating functionality
    # ------------------------------------------------------------------
    
    # NOTE for now was moved to parser, to avoid some bugs that are not resolved yet

    # def update_parameters(self, new_params: dict) -> None:
    #     """
    #     Update system parameters and flag that derived quantities need recomputation.
        
    #     This method updates the parameter dictionary and sets a flag indicating
    #     that any quantities derived from parameters (e.g., Hamiltonian's local
    #     interactions and J(0)) need to be recalculated.
        
    #     Parameters
    #     ----------
    #     new_params : dict
    #         Dictionary of parameters to update, e.g., {'J': -1.5, 'Dz': 0.2}.
    #         Only the specified parameters are changed; others remain unchanged.
            
    #     Example
    #     -------
    #     >>> system.update_parameters({'J': -1.5, 'A': -0.5})
    #     >>> hamiltonian.compute_at_k(k)  # Will auto-refresh if needed
    #     """
    #     self.parameters.update(new_params)
    #     self._parameters_changed = True

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        Returns a printable representation of an object of this class.
        """
        return (
            f"SpinSystem("
            f"n_sublattices={self.n_sublats}, "
            f"n_interactions={self.n_interactions}, "
            f"symmetrize={self.symmetrize})"
        )