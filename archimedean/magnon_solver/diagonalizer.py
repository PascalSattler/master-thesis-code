"""
diagonalizer.py

Implements Colpa's algorithm for diagonalizing para-Hermitian, quadratic, bosonic
Hamiltonians arising from Linear Spin Wave Theory.
(1978, doi.org/10.1016/0378-4371(78)90160-7)

The algorithm properly handles the bosonic commutation relations encoded
in the metric K = diag(I, -I) and includes ground state stability checks.
"""

import numpy as np
from numpy.linalg import eigh, eigvalsh, cholesky, inv, LinAlgError
from typing import Optional



class Colpa:
    """
    Colpa's diagonalization algorithm for bosonic Hamiltonians in LSWT.

    Diagonalizes the para-Hermitian, quadratic Hamiltonian matrix H(k) using 
    Colpa's algorithm, which accounts for the bosonic metric K = diag(I, -I). 
    The algorithm includes automatic ground state stability validation via 
    Cholesky decomposition.

    Parameters
    ----------
    global_gauge : bool, optional
        If True, apply a global phase to eigenvectors such that the entry
        with largest absolute value is positive and real. Enforces particle-hole
        symmetry except for degeneracies. Default is True.
    gauge_decimals : int, optional
        Number of decimals to round absolute values before comparison when
        applying global gauge. Avoids instabilities when multiple entries
        have similar magnitude. Default is 5.
    force_particle_hole_symmetry : bool, optional
        If True, restrict k-vectors to half of the reciprocal space by ensuring
        the first non-zero component is positive. Uses particle-hole symmetry
        to compute results for negative k. Default is True.
    validate_para_hermiticity : bool, optional
        If True, check that H satisfies K H K = H† before diagonalization.
        Useful for debugging. Default is False.
    stability_potential_start : float, optional
        Initial potential added to diagonal if Cholesky decomposition fails
        at k=0 as an attempt to numerically stabilize the ground state by making
        the Hamiltonian matrix positive definite. Default is 1e-10.
    stability_potential_factor : float, optional
        Factor by which the potential is increased on each retry. Default is 10.
    stability_max_attempts : int, optional
        Maximum number of attempts to stabilize H(k=0) before raising error.
        Default is 6.
    stability_tolerance : float, optional
        Tolerance for considering eigenvalues positive in stability check.
        Default is 1e-8.

    Attributes
    ----------
    n_sublats : int or None
        Number of sublattices, inferred from H(k) shape on first call.
    """

    def __init__(
        self,
        global_gauge: bool = True,
        gauge_decimals: int = 5,
        force_particle_hole_symmetry: bool = True,
        validate_para_hermiticity: bool = False,
        stability_potential_start: float = 1e-10,
        stability_potential_factor: float = 10.0,
        stability_max_attempts: int = 6,
        stability_tolerance: float = 1e-8,
    ) -> None:
        
        self.global_gauge = global_gauge
        self.gauge_decimals = gauge_decimals
        self.force_particle_hole_symmetry = force_particle_hole_symmetry
        self.validate_para_hermiticity = validate_para_hermiticity
        self.stability_potential_start = stability_potential_start
        self.stability_potential_factor = stability_potential_factor
        self.stability_max_attempts = stability_max_attempts
        self.stability_tolerance = stability_tolerance

        self.n_sublats: Optional[int] = None
        self.paulix: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def diagonalize(
        self,
        H_k: np.ndarray,
        k: np.ndarray,
        return_full: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize the Hamiltonian H(k) using Colpa's algorithm.

        Parameters
        ----------
        H_k : np.ndarray of shape (2N, 2N), complex
            The bosonic Hamiltonian matrix at wave vector k.
        k : np.ndarray of shape (3,),
            The wave vector. Required for force_particle_hole_symmetry if True.
            Also used to determine if this is k = 0 for stability checks.
        return_full : bool, optional
            If True (default), return full (2N,) eigenvalues and (2N, 2N) eigenvectors.
            If False, return only positive (N,) eigenvalues and corresponding
            (2N, N) eigenvectors (particle sector).

        Returns
        -------
        eigenvalues : np.ndarray
            Shape (2N,) if return_full = True, else (N,) containing positive energies.
        eigenvectors : np.ndarray
            Shape (2N, 2N) if return_full = True, else (2N, N). Columns are eigenvectors.

        Raises
        ------
        ValueError
            If H(k=0) is not positive definite after stability attempts.
        """
        # infer dimensions on first call
        if self.n_sublats is None:
            self._infer_dimensions(H_k)

        # optional para-Hermiticity check
        if self.validate_para_hermiticity:
            self._validate_para_hermiticity(H_k)

        if self.force_particle_hole_symmetry:
            # ensure particle-hole symmetry by mapping k to -k if necessary
            k, sign, k_is_zero = self._apply_particle_hole_symmetry_to_k(k)
        else:
            # check if k is the zero vector
            k_is_zero = np.allclose(k, 0)      

        # ensure positive definite at Gamma point
        if k_is_zero:
            H_k = self._ensure_positive_definite(H_k)

        # core diagonalization algorithm
        eigenvalues, eigenvectors = self._colpa_diagonalize_core(H_k)

        # apply global gauge
        if self.global_gauge:
            eigenvectors = self._apply_global_gauge(eigenvectors)

        # enforce particle-hole symmetry on results
        if self.force_particle_hole_symmetry:
            eigenvalues, eigenvectors = self._apply_particle_hole_symmetry_to_results(
                eigenvalues, eigenvectors, k_is_zero, sign
            )

        # Return full or reduced results
        if return_full:
            return eigenvalues, eigenvectors
        else:
            return self._extract_positive_spectrum(eigenvalues, eigenvectors)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _infer_dimensions(self, H_k: np.ndarray) -> None:
        """
        Infer the number of sublattices from the Hamiltonian shape
        and initialize helper matrices.
        """
        if H_k.shape[0] != H_k.shape[1]:
            raise ValueError(f"Hamiltonian must be square, got shape {H_k.shape}")
        
        if H_k.shape[0] % 2 != 0:
            raise ValueError(f"Hamiltonian dimension must be even, got {H_k.shape[0]}")

        self.n_sublats = H_k.shape[0] // 2
        N = self.n_sublats

        # Pauli_x matrix in block form: [[0, I], [I, 0]]
        self.paulix = np.block([
            [np.zeros((N, N)), np.eye(N)],
            [np.eye(N), np.zeros((N, N))]
        ])

        # bosonic metric η = diag(I, -I)
        self.eta = np.block([
            [np.eye(N), np.zeros((N, N))],
            [np.zeros((N, N)), -np.eye(N)]
        ])

    # ------------------------------------------------------------------
    # Particle-hole symmetry handling
    # ------------------------------------------------------------------

    def _apply_particle_hole_symmetry_to_k(
        self,
        k: np.ndarray
    ) -> tuple[np.ndarray, int, bool]:
        """
        Apply particle-hole symmetry constraint to k-vector.

        If k is not zero, ensure the first non-zero component is positive.
        This restricts calculations to half of the Brillouin zone.

        Parameters
        ----------
        k : np.ndarray of shape (3,)
            Input wave vector.

        Returns
        -------
        k_modified : np.ndarray of shape (3,)
            Modified wave vector (possibly flipped).
        sign : int
            +1 if k was unchanged, -1 if k was flipped to -k.
        k_is_zero : bool
            True if k is the zero vector.
        """
        components_zero = np.isclose(k, 0)
        k_is_zero = np.all(components_zero)

        if k_is_zero:
            return k.copy(), 1, True

        # ensure first non-zero component is positive
        sign = int(np.sign(k[~components_zero][0]))

        k *= sign

        return k, sign, False

    def _apply_particle_hole_symmetry_to_results(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        k_is_zero: bool,
        sign: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply particle-hole symmetry constraints to diagonalization results.

        At k=0: enforce exact symmetry in transformation matrix.
        At k≠0 with sign<0: computed H(-k), transform back to H(k) convention.

        Parameters
        ----------
        eigenvalues : np.ndarray of shape (2N,)
            Eigenvalues from Colpa diagonalization.
        eigenvectors : np.ndarray of shape (2N, 2N)
            Eigenvectors from Colpa diagonalization.
        k_is_zero : bool
            True if this is the Gamma point.
        sign : int
            +1 if k unchanged, -1 if k was flipped.

        Returns
        -------
        eigenvalues : np.ndarray of shape (2N,)
            Modified eigenvalues.
        eigenvectors : np.ndarray of shape (2N, 2N)
            Modified eigenvectors.
        """
        N = self.n_sublats

        if k_is_zero:
            # enforce exact particle-hole symmetry at Gamma
            eigenvectors[N:, N:] = eigenvectors[:N, :N].conj()
            eigenvectors[:N, N:] = eigenvectors[N:, :N].conj()

        elif sign < 0:
            # computed H(-k), transform back to H(k) convention
            # swap particle and hole sectors and negate eigenvalues
            eigenvalues[:N], eigenvalues[N:] = -eigenvalues[N:].copy(), -eigenvalues[:N].copy()

            # apply Pauli_x matrix
            # from right: swap particle and hole components within a eigenvector
            # from left: swap which eigenvector corresponds to which band
            # both ensures that first N eigvals remain particle bands and second N hole bands
            eigenvectors = self.paulix @ eigenvectors.conj() @ self.paulix

        return eigenvalues, eigenvectors

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_para_hermiticity(self, H_k: np.ndarray) -> None:
        """
        Check that H satisfies the para-Hermiticity condition η H η = H†
        where η = diag(I, -I).

        Raises
        ------
        ValueError
            If para-Hermiticity is violated beyond tolerance.
        """
        etaHeta = self.eta @ H_k @ self.eta
        H_dag = H_k.conj().T

        if not np.allclose(etaHeta, H_dag, atol=1e-10):
            max_diff = np.max(np.abs(etaHeta - H_dag))
            raise ValueError(
                f"Hamiltonian is not para-Hermitian: max|K H K - H†| = {max_diff:.2e}"
            )


    def _ensure_positive_definite(self, H_k: np.ndarray) -> np.ndarray:
        """
        Ensure H(k=0) is positive definite by checking eigenvalues and adding
        stabilization potential if needed.

        For stable ground states, H should have non-negative eigenvalues. If any
        eigenvalue is negative (below tolerance), the ground state is unstable.
        If eigenvalues are merely small (Goldstone modes), adds a small potential
        to make Cholesky decomposition numerically stable.

        Parameters
        ----------
        H_k : np.ndarray of shape (2N, 2N)
            Hamiltonian at Gamma point.

        Returns
        -------
        H_stabilized : np.ndarray of shape (2N, 2N)
            Hamiltonian with added potential for numerical stability.

        Raises
        ------
        ValueError
            If any eigenvalue is significantly negative (unstable ground state).
        """
        # check eigenvalues for stability
        eigenvals = eigvalsh(H_k)
        min_eigenval = np.min(eigenvals)

        # if minimum eigenvalue is significantly negative, ground state is unstable
        if min_eigenval < - self.stability_tolerance:
            raise ValueError(
                f"Colpa: H(k=0) has negative eigenvalue {min_eigenval:.2e}. "
                f"Ground state is unstable. Check that the classical spin "
                f"configuration minimizes the energy."
            )

        # add a small potential to handle Goldstone modes and ensure strict positive definiteness
        # (needed for numerical stability in Cholesky decomposition)
        potential = self.stability_potential_start
        H_stabilized = H_k + potential * np.eye(H_k.shape[0])

        # verify that Cholesky will work
        try:
            cholesky(H_stabilized)
        except LinAlgError:
            # if it still fails, try increasing potential
            for attempt in range(1, self.stability_max_attempts):
                potential *= self.stability_potential_factor
                H_stabilized = H_k + potential * np.eye(H_k.shape[0])
                
                try:
                    cholesky(H_stabilized)
                    break
                except LinAlgError:
                    continue
            else:
                raise ValueError(
                    f"Colpa: Failed to stabilize H(k=0) after {self.stability_max_attempts} attempts. "
                    f"Final potential = {potential:.2e}"
                )

        return H_stabilized

    # ------------------------------------------------------------------
    # Core Colpa algorithm
    # ------------------------------------------------------------------

    def _colpa_diagonalize_core(
        self,
        H_k: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Core Colpa diagonalization algorithm.

        Steps:
        1. Cholesky decomposition: H = K† @ K
        2. Transform to standard eigenvalue problem K η K† and unitarily diagonalize
           to get eigenvalues Es and eigenvectors Vs
        3. Sort eigenvalues: positive energies first, then negative
        4. Back-transform eigenvectors: T = K^-1 @ V @ E^(1/2)
        
        Parameters
        ----------
        H_k : np.ndarray of shape (2N, 2N)
            Hamiltonian matrix.

        Returns
        -------
        eigenvalues : np.ndarray of shape (2N,)
            Eigenvalues sorted with positive energies in first N positions,
            negative in last N positions.
        eigenvectors : np.ndarray of shape (2N, 2N)
            Transformation matrix (columns are eigenvectors).
        """

        # ------------------------------------
        # 1. Cholesky decomposition H = K† @ K
        # ------------------------------------
        
        # lower triangular matrix
        L = cholesky(H_k)

        # upper triangular matrix K = L†
        K = L.conj().T

        # -------------------------------------
        # 2. Unitary diagonalization of K η K†
        # -------------------------------------

        M = K @ self.eta @ L

        # solve M @ v = λ @ v for 1D array of eigvals and 2D array of eigvects (columns)
        Es, Vs = eigh(M)

        # ------------------------------------
        # 3. Reorder eigenvalues- and vectors
        # ------------------------------------

        # separate positive and negative eigvals
        pos_mask = Es > 0
        neg_mask = Es < 0

        pos_idx = np.where(pos_mask)[0]
        neg_idx = np.where(neg_mask)[0]

        # sort each group by absolute value (ascending for pos, descending for neg)
        pos_sorted = pos_idx[np.argsort(Es[pos_idx])]
        neg_sorted = neg_idx[np.argsort(-Es[neg_idx])]

        # combine all: first N diagonal elements positive, second N negative
        sort_idx = np.concatenate([pos_sorted, neg_sorted])
        
        Es = Es[sort_idx]
        Vs = Vs[:, sort_idx]

        # ---------------------------------------------------------
        # 4. Solve for the transformation matrix T = K^-1 U E^(1/2)
        # ---------------------------------------------------------
        
        # create diagonal matrix of sqrt(|eigenvalues|)
        E_sqrt = np.sqrt(self.eta @ np.diag(Es))

        # T = K^(-1) @ U @ E^(1/2)
        Ts = inv(K) @ Vs @ E_sqrt

        return Es, Ts

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _apply_global_gauge(self, eigenvectors: np.ndarray) -> np.ndarray:
        """
        Apply global phase to each eigenvector such that the entry with
        largest absolute value is positive and real.

        Parameters
        ----------
        eigenvectors : np.ndarray of shape (2N, 2N)
            Eigenvector matrix (columns are eigenvectors).

        Returns
        -------
        eigenvectors_gauged : np.ndarray of shape (2N, 2N)
            Eigenvectors with global phase applied.
        """
        N = self.n_sublats
    
        # separate particle and hole blocks
        # particles: first N eigenvectors (upper NxN block)
        # holes: last N eigenvectors (lower NxN block)
        abs_part = np.abs(eigenvectors[:N, :N]).round(decimals=self.gauge_decimals)
        abs_hole = np.abs(eigenvectors[N:, N:]).round(decimals=self.gauge_decimals)
        
        # find row index of maximum absolute value for each column
        # (axis=0 runs along rows and down each column, result has shape (N,))
        max_rows_part = np.argmax(abs_part, axis=0)
        max_rows_hole = np.argmax(abs_hole, axis=0)
        
        # extract elements with largest magnitude using previously found index
        # e.g. particle block: eigenvectors[max_row_i, col_i] for each column i
        col_indices = np.arange(N)
        max_elements_part = eigenvectors[max_rows_part, col_indices]
        max_elements_hole = eigenvectors[max_rows_hole + N, col_indices + N]
        
        # compute phase factors: x / |x| = exp(i*arg(x))
        # conjugate phase to rotate element onto positive real axis after multiplication
        phase_part = np.conj(max_elements_part / np.abs(max_elements_part))
        phase_hole = np.conj(max_elements_hole / np.abs(max_elements_hole))
        
        # store in diagonal phase matrix for multiplication
        phases = np.zeros((2 * N, 2 * N), dtype=complex)
        phases[col_indices, col_indices] = phase_part
        phases[col_indices + N, col_indices + N] = phase_hole
        
        # apply phase correction via matrix multiplication
        # (to each eigenvector column)
        eigenvectors_gauged = eigenvectors @ phases
        
        return eigenvectors_gauged


    def _extract_positive_spectrum(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract only the positive eigenvalues (physical magnon energies)
        and their corresponding eigenvectors.

        Colpa's algorithm produces eigenvalues in ±E pairs. Extract only 
        the positive ones and sort by ascending magnitude.

        Parameters
        ----------
        eigenvalues : np.ndarray of shape (2N,)
            Full eigenvalue spectrum.
        eigenvectors : np.ndarray of shape (2N, 2N)
            Full eigenvector matrix.

        Returns
        -------
        eigenvalues_positive : np.ndarray of shape (N,)
            Positive eigenvalues sorted by ascending magnitude.
        eigenvectors_positive : np.ndarray of shape (2N, N)
            Corresponding eigenvectors (columns).
        """
        N = self.n_sublats

        # first N eigenvalues should be positive (particle sector)
        eigenvalues_positive = eigenvalues[:N]
        # first N columns with 2N row entries
        eigenvectors_positive = eigenvectors[:, :N]

        # sort by ascending magnitude
        sort_idx = np.argsort(np.abs(eigenvalues_positive))
        eigenvalues_positive = eigenvalues_positive[sort_idx]
        eigenvectors_positive = eigenvectors_positive[:, sort_idx]

        return eigenvalues_positive, eigenvectors_positive

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        Returns a printable representation of an object of this class.
        """
        options = []
        if self.global_gauge:
            options.append("global_gauge")
        if self.force_particle_hole_symmetry:
            options.append("force_ph_sym")
        if self.validate_para_hermiticity:
            options.append("validate_para_herm")

        options_str = ", ".join(options) if options else "default"

        if self.n_sublats is not None:
            return f"Colpa(n_sublattices={self.n_sublats}, {options_str})"
        else:
            return f"Colpa({options_str})"