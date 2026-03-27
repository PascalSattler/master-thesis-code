"""
hamiltonian.py

Constructs the bilinear bosonic Hamiltonian matrix H(k) for a magnonic system
using Linear Spin Wave Theory (LSWT), following the formalism of
Toth & Lake (2015, doi.org/10.1088/0953-8984/27/16/166002).

The Hamiltonian is constructed in the local coordinate frames at each
sublattice site, accounting for arbitrary non-collinear spin configurations.
"""

import numpy as np
from magnon_solver.system import SpinSystem



class Hamiltonian:
    """
    Bilinear bosonic Hamiltonian for a magnonic system in the LSWT framework.

    Constructs the (2N x 2N) quadratic Hamiltonian matrix H(k) in the basis
    [a_1, a_2, ..., a_N, a_1†, a_2†, ..., a_N†] where N is the number of
    sublattices and a_i, a_i† are bosonic annihilation/creation operators 
    introduced by the Holstein-Primakoff transformation.

    The matrix has block structure following (cf. Toth & Lake eq. (25)):

        H(k) = [[ A(k) - C,  B(k)        ],
                [ B(k)†,     A(k)† - C   ]]

    where A, B, C are (N x N) matrices constructed from the Fourier-transformed
    interaction matrices rotated into the local frames using auxiliary vectors 
    u, u*, v (defined in eq. (9)).

    Parameters
    ----------
    system : SpinSystem
        The spin system containing lattice, site, and interaction data.

    Attributes
    ----------
    system : SpinSystem
        Reference to the input spin system.
    n_sublats : int
        Number of sublattices (extracted for convenience).
    matrix_size : int
        Size of the Hamiltonian matrix (2 * n_sublats).
    has_magfield : bool
        True if any component of the magnetic field is non-zero.
    local_interactions : dict
        Pre-computed interactions transformed to local frames.
        Structure: local_interactions[i][j] = list of (d, J_local) tuples.
    """

    def __init__(self, system: SpinSystem) -> None:
        
        self.system = system
        self.n_sublats = system.n_sublats
        self.matrix_dim = 2 * self.n_sublats

        # check for magnetic field
        params = system.parameters
        self.has_magfield = any(
            abs(params.get(f'B{i}', 0.0)) > 1e-12 for i in ['x', 'y', 'z']
        )

        # pre-compute interactions in local frames (e1, e2, e3) (k-independent)
        self.local_interactions = self._transform_interactions_to_local_frames()

        # pre-compute auxiliary vectors u, u*, v for each sublattice
        self.auxiliary_vectors = self._compute_auxiliary_vectors()

        # pre-compute J(k=0) for C matrix calculation (k-independent)
        k_zero = np.array([0.0, 0.0, 0.0])
        self.J_zero = self._fourier_transform_interactions(k_zero)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_at_k(self, k: np.ndarray) -> np.ndarray:
        """
        Compute the Hamiltonian matrix H(k) at a given k-point.

        Parameters
        ----------
        k : np.ndarray of shape (3,)
            Wave vector in reciprocal space (Cartesian coordinates).

        Returns
        -------
        H_k : np.ndarray of shape (2N, 2N), complex
            The Hamiltonian matrix in the basis
            [a_1, ..., a_N, a_1†, ..., a_N†].
        """
        # auto-refresh if parameters changed
        # self.refresh_if_needed() NOTE for now was moved to parser

        # 1: Fourier transform interactions to get J_ij(k)
        J_k = self._fourier_transform_interactions(k)

        # 2: build the (2N, 2N) Hamiltonian matrix from J(k)
        H_k = self._build_hamiltonian_matrix(J_k)

        # 3: add Zeeman term contribution if magnetic field is present
        if self.has_magfield:
            H_k = self._add_zeeman_term(H_k)

        return H_k

    # ------------------------------------------------------------------
    # Transformation to local frames (k-independent, computed once)
    # ------------------------------------------------------------------

    def _transform_interactions_to_local_frames(self) -> dict:
        """
        Transform all interaction matrices from the global coordinate frame
        to the local frames (e1, e2, e3) at each pair of sites.

        For an interaction from sublattice i to j with matrix J_ij in the
        global frame, the transformed matrix in local coordinates is:

            J_local = R_i^T @ J_ij @ R_j        (cf. eq. (13))

        where R_i is the (3,3) local frame matrix at site i (rows are [e1, e2, e3]).

        Returns
        -------
        local_interactions : dict
            Nested dict structure:
            local_interactions[i][j] = [(d, J_local), ...]
            where i, j are sublattice indices, d is the difference vector in
            lattice coordinates, and J_local is the (3,3) interaction matrix
            in the local (e1, e2, e3) basis.
        """
        # initialize nested dict
        local_int = {
            site['sublattice']: {} for site in self.system.spin_data
        }

        # get local frames indexed by sublattice
        local_frames = {
            site['sublattice']: site['local_frame']
            for site in self.system.spin_data
        }

        # transform each interaction
        for inter in self.system.interaction_data:
            i = inter['reference_sublat']
            j = inter['neighbor_sublat']
            d = inter['difference_vector']
            J_global = inter['interaction_matrix']

            # rotation to local frames: J_local = R_i^T @ J_global @ R_j
            R_i = local_frames[i]
            R_j = local_frames[j]
            J_local = R_i.T @ J_global @ R_j

            # store in nested dict
            if j not in local_int[i]:
                local_int[i][j] = []
            local_int[i][j].append((d, J_local))

        return local_int

    # ------------------------------------------------------------------
    # Auxiliary vector computation
    # ------------------------------------------------------------------

    def _compute_auxiliary_vectors(self) -> dict:
        """
        Compute the auxiliary vectors u, u*, v for each sublattice following
        Toth & Lake eq. (9).

        For a sublattice with local frame R = [e1, e2, e3] (rows):
            u = e1 + i * e2    (complex 3-vector in global frame)
            u* = e1 - i * e2   (complex conjugate of u)
            v = e3             (real 3-vector in global frame)

        However, when computing matrix elements with J_local (which is in the
        local frame basis), we express these in the local basis:
            u_local = (1, i, 0)
            u*_local = (1, -i, 0)
            v_local = (0, 0, 1)

        Returns
        -------
        aux_vectors : dict
            Dictionary mapping sublattice index to dict with keys 'u', 'u_conj', 'v'
            containing the vectors in the local frame basis.
        """
        aux_vectors = {}

        for site in self.system.spin_data:
            sublat = site['sublattice']
            # in the local frame basis, the auxiliary vectors are always
            aux_vectors[sublat] = {
                'u': np.array([1, 1j, 0], dtype=complex),
                'u_conj': np.array([1, -1j, 0], dtype=complex),
                'v': np.array([0, 0, 1], dtype=complex)
            }

        return aux_vectors

    # ------------------------------------------------------------------
    # Fourier transform (k-dependent)
    # ------------------------------------------------------------------

    def _fourier_transform_interactions(self, k: np.ndarray) -> dict:
        """
        Compute the Fourier-transformed interaction matrices J_ij(k) by
        summing over all difference vectors with appropriate phase factors.

        For an interaction from site i to site j, the phase factor is:

            exp(i * k * (r_j - r_i))

        where:
               r_i    = R_n + u_i  (unit cell position R_n + basis vector u_i)
               r_j    = R_m + u_j  (unit cell position R_m + basis vector u_j)
            r_j - r_i = (R_m - R_n) + (u_j - u_i)
                      =      d      + (u_j - u_i)

        The basis vectors u_i, u_j and difference vector d are given in lattice 
        coordinates in the input file, so the entire difference must be converted 
        to Cartesian before taking the dot product with k.

        Parameters
        ----------
        k : np.ndarray of shape (3,)
            Wave vector in Cartesian coordinates.

        Returns
        -------
        J_k : dict
            Dictionary mapping (i, j) sublattice pairs to (3,3) complex
            matrices J_ij(k) in the local (e1, e2, e3) basis.
        """
        J_k = {}

        # get basis vectors in lattice coordinates
        basis_vectors = {
            site['sublattice']: site['basis_vector']
            for site in self.system.spin_data
        }

        # lattice translation vectors
        a1, a2, a3 = self.system.lattice_vectors

        for i in self.local_interactions:
            for j in self.local_interactions[i]:
                # sum over all (d, J_local) pairs for this (i,j) combination
                J_sum = np.zeros((3, 3), dtype=complex)

                for d, J_local in self.local_interactions[i][j]:
                    # compute displacement in lattice coordinates first
                    # d is unit cell difference, basis_vectors are positions within cell
                    lattice_diff = d + basis_vectors[j] - basis_vectors[i]

                    # convert to Cartesian coordinates
                    r_diff = (
                        lattice_diff[0] * a1 +
                        lattice_diff[1] * a2 +
                        lattice_diff[2] * a3
                    )

                    # phase factor: exp(i * k * r_diff)
                    phase = np.exp(1j * np.dot(k, r_diff))

                    J_sum += phase * J_local

                J_k[(i, j)] = J_sum

        return J_k

    # ------------------------------------------------------------------
    # Hamiltonian matrix construction
    # ------------------------------------------------------------------

    def _build_hamiltonian_matrix(self, J_k: dict) -> np.ndarray:
        """
        Construct the (2N x 2N) Hamiltonian matrix from the Fourier-transformed
        interaction matrices J_ij(k) following Toth & Lake eqs. (25), (26), (27).

        Matrix elements are computed using auxiliary vectors defined per site:
            u_i = e1_i + i * e2_i   (expressed as (1, i, 0) in local basis)
            u*_i = e1_i - i * e2_i  (expressed as (1, -i, 0) in local basis)
            v_i = e3_i              (expressed as (0, 0, 1) in local basis)

        Following eqs. (26):
            A_ij(k) = (1/2) * sqrt(S_i * S_j) * u_i^T @ J_ij(k) @ u*_j  (off-diagonal)
            A_ii(k) = (1/2) * S_i * u_i^T @ J_ii(k) @ u*_i              (diagonal)
            B_ij(k) = (1/2) * sqrt(S_i * S_j) * u_i^T @ J_ij(k) @ u_j
            C_ii    = sum_l S_l * v_i^T @ J_ij_zero @ v_i

        The final Hamiltonian diagonal has (A - C) terms.

        Parameters
        ----------
        J_k : dict
            Dictionary of Fourier-transformed interaction matrices in local frame 
            (e1, e2, e3) basis.

        Returns
        -------
        H_k : np.ndarray of shape (2N, 2N), complex
            The full Hamiltonian matrix.
        """
        N = self.n_sublats
        A = np.zeros((N, N), dtype=complex)
        B = np.zeros((N, N), dtype=complex)
        C = np.zeros(N, dtype=float)  # C is always real

        # get spin lengths for each sublattice
        spin_lengths = {
            site['sublattice']: site['spin_length']
            for site in self.system.spin_data
        }

        # map sublattice indices to matrix indices
        sublat_to_idx = {
            sublat: idx for idx, sublat in
            enumerate(sorted(spin_lengths.keys()))
        }

        # build A, B, and C matrices
        # C is computed from J(k=0), it represents the k-independent
        # classical ground state energy contribution
        for (i, j), J_ij in J_k.items():
            idx_i = sublat_to_idx[i]
            idx_j = sublat_to_idx[j]

            S_i = spin_lengths[i]
            S_j = spin_lengths[j]

            # get auxiliary vectors for sites i and j
            u_i = self.auxiliary_vectors[i]['u']
            u_conj_i = self.auxiliary_vectors[i]['u_conj']
            v_i = self.auxiliary_vectors[i]['v']
            
            u_j = self.auxiliary_vectors[j]['u']
            u_conj_j = self.auxiliary_vectors[j]['u_conj']
            v_j = self.auxiliary_vectors[j]['v']

            # compute matrix elements using auxiliary vectors
            # J_ij is in the (e1_i, e2_i, e3_i) x (e1_j, e2_j, e3_j) basis
            
            # A matrix element: u_i^T @ J_ij(k) @ u*_j
            A_element = u_i.T @ J_ij @ u_conj_j

            # A diagonal element: u_i^T @ J_ii(k) @ u*_i
            A_element_diag = u_i.T @ J_ij @ u_conj_i
            
            # B matrix element: u_i^T @ J_ij(k) @ u_j
            B_element = u_i.T @ J_ij @ u_j
            
            # C matrix element: v_i^T @ J_ij(0) @ v_j (k-independent)
            # get J_zero = J(k=0) for this pair
            J_ij_zero = self.J_zero[(i, j)]
            C_element = np.real(v_i.T @ J_ij_zero @ v_j)

            if i == j:
                # self-interaction contributes to diagonal accumulation
                A[idx_i, idx_i] += (S_i / 2) * A_element_diag
                C[idx_i] += S_i * C_element

            else:
                # off-diagonal terms
                prefactor = np.sqrt(S_i * S_j)
                
                # A off-diagonal
                A[idx_i, idx_j] += (prefactor / 2) * A_element
                
                # C diagonal accumulation from this neighbor (index l in paper)
                C[idx_i] += S_j * C_element
                
                # B off-diagonal
                B[idx_i, idx_j] += (prefactor / 2) * B_element

        # construct final A block matrix by subtracting C from block diagonal
        A_final = A - np.diag(C)

        # assemble full (2N, 2N) Hamiltonian matrix
        H_k = np.block([
            [A_final,      B               ],
            [np.conj(B),   np.conj(A_final)]       # TODO check if it is enough to just conjugate lower right block
        ])

        return H_k

    # ------------------------------------------------------------------
    # Zeeman term
    # ------------------------------------------------------------------

    def _add_zeeman_term(self, H_k: np.ndarray) -> np.ndarray:
        """
        Add the Zeeman coupling to the external magnetic field to the
        Hamiltonian matrix. This modifies the diagonal (on-site) terms.

        The Zeeman term couples the spins to the field via the gyromagnetic
        tensor. In the local frame at each site, this contributes to the
        A block diagonal elements (cf. eq. (27)). 

        Parameters
        ----------
        H_k : np.ndarray of shape (2N, 2N)
            Hamiltonian matrix before Zeeman term is added.

        Returns
        -------
        H_k : np.ndarray of shape (2N, 2N)
            Hamiltonian matrix with Zeeman contribution included.
        """
        # extract field components
        Bx = self.system.parameters.get('Bx', 0.0)
        By = self.system.parameters.get('By', 0.0)
        Bz = self.system.parameters.get('Bz', 0.0)
        B_global = np.array([Bx, By, Bz], dtype=float)

        # get spin lengths and gyromagnetic matrices
        spin_data_sorted = sorted(
            self.system.spin_data,
            key = lambda site: site['sublattice']
        )

        for idx, site in enumerate(spin_data_sorted):
            S = site['spin_length']
            g_mat = site['gyromagnetic_matrix']
            R = site['local_frame']

            # transform field to local frame: B_local = R^T @ B_global
            B_local = R.T @ B_global

            # gyromagnetic coupling in local frame
            # Zeeman energy contributes to the diagonal A block
            # H_Z = S * (g @ B)_z in local frame
            g_dot_B = g_mat @ B_local
            zeeman_contribution = S * g_dot_B[2]  # z-component in local frame

            # add to diagonal of A block
            H_k[idx, idx] += zeeman_contribution

            # add to the conjugated diagonal (lower-right block)
            H_k[self.n_sublats + idx, self.n_sublats + idx] += (
                np.conj(zeeman_contribution)
            )

        return H_k
    
    # ------------------------------------------------------------------
    # Updating functionality
    # ------------------------------------------------------------------

    # NOTE for now was moved to parser, to avoid some bugs that are not resolved yet

    # def refresh_if_needed(self) -> None:
    #     """
    #     Recompute k-independent quantities if system parameters have changed.
        
    #     This method checks if SpinSystem.update_parameters() was called and
    #     recomputes:
    #     - local_interactions (interaction matrices in local frames)
    #     - J_zero (Fourier transform at k=0 for C matrix)
    #     - auxiliary_vectors (if they depend on parameters, currently they don't)
        
    #     Called automatically by compute_at_k() to ensure consistency.
    #     """
    #     if getattr(self.system, '_parameters_changed', False):
    #         # Recompute k-independent quantities
    #         self.local_interactions = self._transform_interactions_to_local_frames()
            
    #         # Recompute J(0) for C matrix
    #         k_zero = np.array([0.0, 0.0, 0.0])
    #         self.J_zero = self._fourier_transform_interactions(k_zero)
            
    #         # Reset flag
    #         self.system._parameters_changed = False

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        Returns a printable representation of an object of this class.
        """
        field_status = "with field" if self.has_magfield else "no field"
        return (
            f"Hamiltonian(n_sublattices={self.n_sublats}, "
            f"matrix_size={self.matrix_dim}, {field_status})"
        )