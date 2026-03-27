"""
topology.py

Topological invariant calculations for magnonic systems using the
Fukui-Hatsugai-Suzuki (FHS) method.
(2004, doi.org/10.1143/JPSJ.74.1674)

Computes Berry curvature and Chern numbers from eigenvectors on a
discrete k-space mesh following the lattice gauge theory approach.

References:
    Fukui, T., Hatsugai, Y., & Suzuki, H. (2005).
    Chern numbers in discretized Brillouin zone: Efficient method of computing (spin) Hall conductances.
    Journal of the Physical Society of Japan, 74(6), 1674-1677.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Literal
from pathlib import Path

from magnon_solver.hamiltonian import Hamiltonian
from magnon_solver.diagonalizer import Colpa
from magnon_solver.bz_configs import get_bz_hsp, BZ_LIBRARY, BZ_VERTICES


class TopologySolver:
    """
    Compute topological invariants for magnonic systems.

    Calculates Berry curvature and Chern numbers using the Fukui-Hatsugai-Suzuki
    (FHS) discretization method on a 2D k-space mesh.

    Supports both abelian (single-band) and non-abelian (multi-band) calculations.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Hamiltonian object for computing H(k).
    colpa : Colpa
        Colpa diagonalizer for obtaining eigenvectors.
    bz_type : str
        Brillouin zone type (must be in BZ_LIBRARY).
    system_name : str, optional
        System name for file management. Default is 'system'.

    Attributes
    ----------
    hamiltonian : Hamiltonian
    colpa : Colpa
    bz_type : str
    system_name : str
    k_mesh : np.ndarray or None
        K-point mesh, shape (n_kx, n_ky, 2).
    eigenvalues : np.ndarray or None
        Eigenvalues on mesh, shape (n_kx, n_ky, 2N).
    eigenvectors : np.ndarray or None
        Eigenvectors on mesh, shape (n_kx, n_ky, 2N, 2N).
    berry_curvature : np.ndarray or None
        Berry curvature field, shape (n_kx-1, n_ky-1, 2N) for corner mode.
    chern_numbers : np.ndarray or None
        Chern numbers for each band, shape (2N,).
    degeneracies : np.ndarray or None
        Boolean mask of degenerate points, shape (n_kx, n_ky, 2N).
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        colpa: Colpa,
        bz_type: str,
        system_name: str = 'system',
    ) -> None:

        self.hamiltonian = hamiltonian
        self.colpa = colpa
        self.bz_type = bz_type
        self.system_name = system_name
        self.hsp = self.hsp = get_bz_hsp(bz_type, self.hamiltonian.system.reciprocal_vectors)

        # infer dimension from hamiltonian and create metric
        N = self.hamiltonian.system.n_sublats
        self.eta = np.block([
            [np.eye(N), np.zeros((N, N))],
            [np.zeros((N, N)), -np.eye(N)]
        ])

        # storage for computed results
        self.k_mesh: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.berry_curvature: Optional[np.ndarray] = None
        self.chern_numbers: Optional[np.ndarray] = None
        self.degeneracies: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Main computation methods
    # ------------------------------------------------------------------

    def compute_berry_curvature(
        self,
        n_kx: int = 50,
        n_ky: Optional[int] = None,
        kz: float = 0.0,
        k_range: Optional[tuple] = None,
        bands: Optional[list[int]] = None,
        mode: Literal['abelian', 'nonabelian'] = 'abelian',
        grid_type: Literal['corner'] = 'corner',
        degeneracy_tol: float = 1e-6,
        link_cutoff: float = 1e-6,
        n_cores: int = 1,
        worker_pool = None,
    ) -> None:
        """
        Compute the Berry curvature using the FHS method.

        Parameters
        ----------
        n_kx : int, optional
            Number of k-points along kx direction. Default is 50.
        n_kx : int or None, optional
            Number of k-points along ky direction. If None, it is set
            equal to kx. Default is None.
        kz : float, optional
            Fixed kz value for 2D systems. Default is 0.
        k_range : tuple of tuples, optional
            K-space range: ((kx_min, kx_max), (ky_min, ky_max)) in
            reciprocal lattice units. Default is [0, 1] for both.
        bands : list of int, optional
            Band indices to compute. If None, compute all bands.
        mode : {'abelian', 'nonabelian'}, optional
            Calculation mode. 'abelian' for single-band (default),
            'nonabelian' for multi-band with degeneracies.
        grid_type : {'corner'}, optional
            Grid sampling type. Currently only 'corner' is implemented.
        degeneracy_tol : float, optional
            Energy tolerance for detecting degeneracies. Default is 1e-6.
        link_cutoff : float, optional
            Overlap threshold for gauge links. Overlaps below this are
            set to 1 (zero phase). Default is 1e-6.
        n_cores : int, optional
            Number of CPU cores to run parallel jobs on. If 1, run serially. 
            If -1, use all CPU cores. Default is 1.
        worker_pool : joblib.Parallel, optional
            Pre-initialized worker pool for reuse across multiple calls.
            Improves performance when computing multiple band structures.
            Create with: pool = Parallel(n_jobs = n_cores)
            Default is None (creates new pool each time).
        """
        if grid_type != 'corner':
            raise NotImplementedError("Only 'corner' grid type is currently supported")
        
        # if only one integer number for divisions of a k-space direction is given
        # set the other equal to it for evenly spaced mesh
        if n_ky is None:
            n_ky = n_kx

        # create k-mesh (with periodic wrapping for corner mode)
        self.k_mesh = self._create_kmesh(n_kx, n_ky, k_range, grid_type='corner')

        # compute eigenvectors on mesh
        print(f"Computing eigenvectors on {n_kx}x{n_ky} mesh...")
        self._compute_eigenvectors_on_mesh(kz, n_cores, worker_pool=worker_pool)

        # detect degeneracies (for abelian mode only)
        if mode == 'abelian':
            self.degeneracies = self._detect_degeneracies(degeneracy_tol)
            n_deg = np.sum(self.degeneracies)
            print(f"  Detected {n_deg} degenerate points")

        # compute Berry curvature
        print(f"Computing Berry curvature ({mode} mode)...")
        if mode == 'abelian':
            self.berry_curvature = self._compute_berry_curvature_abelian(
                link_cutoff, bands
            )
        else:
            self.berry_curvature = self._compute_berry_curvature_nonabelian(
                link_cutoff, bands
            )

        print(f"  ✓ Berry curvature computed")


    def compute_chern_numbers(
        self,
        bands: Optional[list[int]] = None,
    ) -> np.ndarray:
        """
        Compute Chern numbers by integrating Berry curvature over BZ.

        The Chern number is:
            C = (1/2π) ∫∫ Ω(k) d²k

        Parameters
        ----------
        bands : list of int, optional
            Band indices to compute. If None, compute all bands.

        Returns
        -------
        chern_numbers : np.ndarray
            Chern numbers for each band. First N are particle bands,
            last N are hole bands (equal but opposite sign).

        Raises
        ------
        RuntimeError
            If Berry curvature has not been computed yet.
        """
        if self.berry_curvature is None:
            raise RuntimeError(
                "Berry curvature not computed. Call compute_berry_curvature() first."
            )

        n_bands = self.berry_curvature.shape[2]

        if bands is None:
            bands = list(range(n_bands))

        chern_numbers = np.zeros(n_bands)

        # Integrate Berry curvature over BZ
        # Each plaquette contributes equally (uniform mesh)
        for band in bands:
            total_curvature = np.sum(self.berry_curvature[:, :, band])
            chern_numbers[band] = total_curvature / (2 * np.pi)

        self.chern_numbers = chern_numbers

        return chern_numbers

    # ------------------------------------------------------------------
    # K-mesh generation
    # ------------------------------------------------------------------

    def _create_kmesh(
        self,
        n_kx: int,
        n_ky: int,
        k_range: Optional[tuple],
        grid_type: str,
    ) -> np.ndarray:
        """
        Create 2D k-point mesh in reciprocal lattice coordinates.

        For 'corner' mode, creates (n_kx+1) x (n_ky+1) grid with periodic
        wrapping to enable closed plaquette calculations.

        Parameters
        ----------
        n_kx, n_ky : int
            Number of divisions along kx, ky.
        k_range : tuple or None
            k-space range.
        grid_type : str
            'corner' or 'center'.

        Returns
        -------
        k_mesh : np.ndarray of shape (n_kx+1, n_ky+1, 2)
            K-point mesh in reciprocal lattice coordinates.
        """
        # example ranges: [0, 1] for first BZ, [−0.5, 0.5] for center at Γ
        if k_range is None:
            k_range = ((-0.5, 0.5), (-0.5, 0.5))

        # for corner mode, create n+1 points to close the mesh
        kx_vals = np.linspace(k_range[0][0], k_range[0][1], n_kx + 1)
        ky_vals = np.linspace(k_range[1][0], k_range[1][1], n_ky + 1)

        kx_mesh, ky_mesh = np.meshgrid(kx_vals, ky_vals, indexing='ij')

        k_mesh = np.stack([kx_mesh, ky_mesh], axis=-1)

        return k_mesh

    # ------------------------------------------------------------------
    # Eigenvector computation on mesh
    # ------------------------------------------------------------------

    def _compute_eigenvectors_on_mesh(
        self,
        kz: float,
        n_cores: int,
        worker_pool = None,
    ) -> None:
        """
        Compute eigenvalues and eigenvectors at all mesh points.

        Stores results in self.eigenvalues and self.eigenvectors.
        """
        n_kx, n_ky = self.k_mesh.shape[:2]

        # convert to Cartesian and flatten
        b1, b2, b3 = self.hamiltonian.system.reciprocal_vectors
        k_cartesian_list = []
        for i in range(n_kx):
            for j in range(n_ky):
                k_lattice = np.array([self.k_mesh[i, j, 0], self.k_mesh[i, j, 1], kz])
                k_cart = k_lattice[0] * b1 + k_lattice[1] * b2 + k_lattice[2] * b3
                k_cartesian_list.append((i, j, k_cart))

        # compute eigenvectors
        if n_cores == 1:
            # serial
            results = []
            for i, j, k_cart in k_cartesian_list:
                H_k = self.hamiltonian.compute_at_k(k_cart)
                eigs, vecs = self.colpa.diagonalize(H_k, k=k_cart)
                results.append((i, j, eigs, vecs))
        else:
            # parallel
            from joblib import Parallel, delayed

            # create a wrapper for the diagonalization
            def compute_at_point(i, j, k_cart):
                H_k = self.hamiltonian.compute_at_k(k_cart)
                eigs, vecs = self.colpa.diagonalize(
                    H_k, k=k_cart
                )
                return i, j, eigs, vecs
            
            # reuse a worker pool if provided, otherwise create new one
            if worker_pool is None:
                parallel = Parallel(n_jobs = n_cores)
            else:
                parallel = worker_pool

            results = parallel(
                delayed(compute_at_point)(i, j, k_cart)
                for i, j, k_cart in k_cartesian_list
            )

        # reconstruct arrays
        n_bands = results[0][2].shape[0]
        self.eigenvalues = np.zeros((n_kx, n_ky, n_bands))
        self.eigenvectors = np.zeros((n_kx, n_ky, n_bands, n_bands), dtype=complex)

        for i, j, eigs, vecs in results:
            self.eigenvalues[i, j, :] = eigs
            self.eigenvectors[i, j, :, :] = vecs

    # ------------------------------------------------------------------
    # Degeneracy detection
    # ------------------------------------------------------------------

    def _detect_degeneracies(self, tolerance: float) -> np.ndarray:
        """
        Detect degenerate bands at each k-point.

        Parameters
        ----------
        tolerance : float
            Energy difference threshold for degeneracy.

        Returns
        -------
        degeneracies : np.ndarray of bool, shape (n_kx, n_ky, n_bands)
            True if band n is degenerate with band n+1 at that k-point.
        """
        n_kx, n_ky, n_bands = self.eigenvalues.shape
        degeneracies = np.zeros((n_kx, n_ky, n_bands), dtype=bool)
        
        # compute energy gaps between adjacent bands
        gaps = np.abs(np.diff(self.eigenvalues, axis=2))  # shape is (n_kx, n_ky, n_bands-1)
        
        # find where gaps are below tolerance
        is_degenerate = gaps < tolerance
        
        # mark both bands in each degenerate pair
        # |= is the OR operator -> if band was already marked True from
        # degeneracy with previous band, it is kept True and not overwritten
        # if the band is not degenerate with the next (which would set it False
        # for regular = assignment)
        degeneracies[:, :, :-1] |= is_degenerate  # band n
        degeneracies[:, :, 1:] |= is_degenerate   # band n+1

        return degeneracies

    # ------------------------------------------------------------------
    # FHS method - Abelian (single-band)
    # ------------------------------------------------------------------

    def _compute_berry_curvature_abelian(
        self,
        link_cutoff: float,
        bands: Optional[list[int]],
    ) -> np.ndarray:
        """
        Compute Berry curvature using abelian (single-band) FHS method.

        For each plaquette and band, computes the U(1) gauge link product
        around the plaquette boundary.

        Parameters
        ----------
        link_cutoff : float
            Overlap threshold for setting links to 1.
        bands : list of int or None
            Bands to compute.

        Returns
        -------
        berry_curvature : np.ndarray of shape (n_kx-1, n_ky-1, n_bands)
            Berry curvature field.
        """
        n_kx, n_ky, n_bands = self.eigenvalues.shape

        # if no specific bands are given, calculate for all
        if bands is None:
            bands = list(range(n_bands))

        # Berry curvature on plaquettes (dim one less than mesh in each direction)
        berry_curv = np.zeros((n_kx - 1, n_ky - 1, n_bands))

        for band in bands:             # TODO remove this band loop and do it vectorized for all at once
            for i in range(n_kx - 1):
                for j in range(n_ky - 1):
                    # check if any corner is degenerate
                    if (self.degeneracies[i, j, band] or
                        self.degeneracies[i + 1, j, band] or
                        self.degeneracies[i + 1, j + 1, band] or
                        self.degeneracies[i, j + 1, band]):
                        berry_curv[i, j, band] = 0.0
                        continue

                    # get eigenvectors at the corners
                    psi1 = self.eigenvectors[i, j, :, band]
                    psi2 = self.eigenvectors[i + 1, j, :, band]
                    psi3 = self.eigenvectors[i + 1, j + 1, :, band]
                    psi4 = self.eigenvectors[i, j + 1, :, band]

                    # Compute gauge links
                    link1 = self._compute_u1_link(psi1, psi2, link_cutoff)
                    link2 = self._compute_u1_link(psi2, psi3, link_cutoff)
                    link3 = self._compute_u1_link(psi3, psi4, link_cutoff)
                    link4 = self._compute_u1_link(psi4, psi1, link_cutoff)

                    # FHS formula (cf. eq. (8)): flux = Im(ln(link1 * link2 * link3 * link4))
                    berry_curv[i, j, band] = self._fhs_plaquette_abelian(
                        link1, link2, link3, link4
                    )

        return berry_curv


    def _compute_u1_link(
        self,
        psi1: np.ndarray,
        psi2: np.ndarray,
        cutoff: float,
    ) -> complex:
        """
        Compute U(1) gauge link between two states w.r.t. to the bosonic metric:

            link = η ev1† η ev2 / norm

        If overlap is below cutoff, returns 1 (zero phase contribution).

        Parameters
        ----------
        psi1, psi2 : np.ndarray
            Eigenvectors (complex).
        cutoff : float
            Minimum overlap magnitude.

        Returns
        -------
        U : complex
            Gauge link (complex phase).
        """
        # calculate the inner product of the eigenvectors w.r.t. the metric space i.e. η ev1† η ev2
        # diagonal elements -> normalization, off-diagonal elements -> similarity of eigenstates
        overlap = self.eta @ psi1.conj().T @ self.eta @ psi2
        overlap_mag = np.abs(overlap)

        if overlap_mag < cutoff:
            return 1.0 + 0.0j

        # normalize to unit circle
        return overlap / overlap_mag # TODO make sure that the overlap calculation is implemented correctly


    def _fhs_plaquette_abelian(
        self,
        U12: complex,
        U23: complex,
        U34: complex,
        U41: complex,
    ) -> float:
        """
        Compute Berry curvature on a plaquette from gauge links:

            Ω = Im(log(U12 * U23 * U34 * U41))

        In this convention the flux is calculated around the plaquette in
        circular order, hence the log is taken of the product of all links.
        In the paper, the links follow the k-directions, thus links going 
        corner to corner against a k direction need to be inversed.

        Parameters
        ----------
        U12, U23, U34, U41 : complex
            Gauge links around plaquette.

        Returns
        -------
        berry_curvature : float
            Berry curvature on this plaquette.
        """
        # product of links around closed loop
        F = U12 * U23 * U34 * U41

        # complex log results in log|x| + i*arg(x), then taking imaginary part
        # leaves just arg(x), but this can lead to large discontinuities between plaquettes
        # if flux wraps around near +-pi -> they can lie on different log branches
        # use arctan2 to directly get the angle -> result always on principal branch
        return np.arctan2(F.imag, F.real)

    # ------------------------------------------------------------------
    # FHS method - Non-abelian (multi-band)
    # ------------------------------------------------------------------

    def _compute_berry_curvature_nonabelian(
        self,
        link_cutoff: float,
        bands: Optional[list[int]],
    ) -> np.ndarray:
        """
        Compute Berry curvature using non-abelian (multi-band) FHS method.

        For degenerate bands, treats them as a manifold and computes
        the U(N) gauge link matrices.

        Parameters
        ----------
        link_cutoff : float
            Overlap threshold.
        bands : list of int or None
            Bands to compute (currently must be all bands for non-abelian).

        Returns
        -------
        berry_curvature : np.ndarray of shape (n_kx-1, n_ky-1, n_bands)
            Berry curvature field.
        """
        n_kx, n_ky, n_bands = self.eigenvalues.shape

        if bands is None:
            bands = list(range(n_bands))

        # For non-abelian, compute for all bands
        berry_curv = np.zeros((n_kx - 1, n_ky - 1, n_bands))

        for i in range(n_kx - 1):
            for j in range(n_ky - 1):
                # Get full eigenvector matrices at 4 corners
                Psi1 = self.eigenvectors[i, j, :, :]
                Psi2 = self.eigenvectors[i+1, j, :, :]
                Psi3 = self.eigenvectors[i+1, j+1, :, :]
                Psi4 = self.eigenvectors[i, j+1, :, :]

                # Compute U(N) gauge link matrices
                U12 = self._compute_uN_link(Psi1, Psi2, link_cutoff)
                U23 = self._compute_uN_link(Psi2, Psi3, link_cutoff)
                U34 = self._compute_uN_link(Psi3, Psi4, link_cutoff)
                U41 = self._compute_uN_link(Psi4, Psi1, link_cutoff)

                # FHS formula for non-abelian case
                curv_matrix = self._fhs_plaquette_nonabelian(U12, U23, U34, U41)

                # Extract diagonal (Berry curvature for each band)
                berry_curv[i, j, :] = np.diagonal(curv_matrix).real

        return berry_curv

    def _compute_uN_link(
        self,
        Psi1: np.ndarray,
        Psi2: np.ndarray,
        cutoff: float,
    ) -> np.ndarray:
        """
        Compute U(N) gauge link matrix between two sets of states.

        U_12 = Psi1† @ Psi2

        This is an (n_bands × n_bands) unitary matrix.

        Parameters
        ----------
        Psi1, Psi2 : np.ndarray of shape (2N, 2N)
            Eigenvector matrices (columns are eigenvectors).
        cutoff : float
            (Not used in non-abelian case, kept for consistency).

        Returns
        -------
        U : np.ndarray of shape (n_bands, n_bands), complex
            Gauge link matrix.
        """
        # U_12 = Psi1† @ Psi2
        return Psi1.conj().T @ Psi2

    def _fhs_plaquette_nonabelian(
        self,
        U12: np.ndarray,
        U23: np.ndarray,
        U34: np.ndarray,
        U41: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Berry curvature matrix on a plaquette from link matrices.

        F = U12 @ U23 @ U34 @ U41
        Ω = Im(log(F))

        Parameters
        ----------
        U12, U23, U34, U41 : np.ndarray
            Link matrices around plaquette.

        Returns
        -------
        berry_curvature_matrix : np.ndarray
            Berry curvature matrix (diagonal gives per-band curvature).
        """
        # Product of link matrices
        F = U12 @ U23 @ U34 @ U41

        # Matrix logarithm
        # Note: np.linalg.logm can have branch cut issues
        # For numerical stability, use eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eig(F)

        # log(F) = V @ diag(log(λ)) @ V†
        log_eigvals = np.log(eigvals)
        log_F = eigvecs @ np.diag(log_eigvals) @ eigvecs.conj().T

        # Berry curvature = imaginary part
        return log_F.imag

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_berry_curvature(
        self,
        band: int,
        show_hsp: bool = True,
        show_bz: bool = True,
        cmap: str = 'RdBu_r',
        figsize: tuple = (8, 7),
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot Berry curvature as heatmap over the Brillouin zone.

        Parameters
        ----------
        band : int
            Band index to plot.
        show_hsp : bool, optional
            Mark high-symmetry points. Default is True.
        show_bz : bool, optional
            Draw BZ boundary. Default is True.
        cmap : str, optional
            Colormap name. Default is 'RdBu_r'.
        figsize : tuple, optional
            Figure size. Default is (8, 7).
        show : bool, optional
            Display plot. Default is True.

        Returns
        -------
        fig : matplotlib.figure.Figure

        Raises
        ------
        RuntimeError
            If Berry curvature not computed.
        """
        if self.berry_curvature is None:
            raise RuntimeError("Berry curvature not computed. Call compute_berry_curvature() first.")

        fig, ax = plt.subplots(figsize=figsize)

        # Get k-mesh coordinates (plaquette centers for visualization)
        kx = (self.k_mesh[:-1, :-1, 0] + self.k_mesh[1:, 1:, 0]) / 2
        ky = (self.k_mesh[:-1, :-1, 1] + self.k_mesh[1:, 1:, 1]) / 2

        # Berry curvature for this band
        omega = self.berry_curvature[:, :, band]

        # Heatmap
        im = ax.pcolormesh(
            kx, ky, omega.T,
            cmap=cmap,
            shading='auto',
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r'Berry Curvature $\Omega(k)$', fontsize=12)

        # BZ boundary
        if show_bz and self.bz_type in BZ_VERTICES:
            vertices = BZ_VERTICES[self.bz_type]
            bz_x = [v[0] for v in vertices] + [vertices[0][0]]
            bz_y = [v[1] for v in vertices] + [vertices[0][1]]
            ax.plot(bz_x, bz_y, 'k-', linewidth=2, label='BZ boundary')

        # High-symmetry points
        if show_hsp:
            for label, (coords, display_label) in self.hsp.items():
                if len(coords) >= 2:
                    ax.plot(coords[0], coords[1], 'ko', markersize=8)
                    ax.text(coords[0], coords[1], f'  {display_label}',
                            fontsize=11, ha='left', va='bottom')

        ax.set_xlabel(r'$k_x$', fontsize=12)
        ax.set_ylabel(r'$k_y$', fontsize=12)

        chern_str = ""
        if self.chern_numbers is not None:
            chern_str = f", C = {self.chern_numbers[band]:.2f}"

        ax.set_title(f'Berry Curvature - Band {band}{chern_str}', fontsize=14)
        ax.set_aspect('equal')
        plt.tight_layout()

        if show:
            plt.show()

        return fig

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "computed" if self.berry_curvature is not None else "not computed"
        return (
            f"TopologySolver(bz_type='{self.bz_type}', "
            f"system='{self.system_name}', status={status})"
        )