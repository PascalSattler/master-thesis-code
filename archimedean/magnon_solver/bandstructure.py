"""
bandstructure.py

Compute and plot magnon band structures along high-symmetry paths
in the Brillouin zone.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Callable
import os

from magnon_solver.hamiltonian import Hamiltonian
from magnon_solver.diagonalizer import Colpa
from magnon_solver.bz_configs import get_bz_hsp, BZ_LIBRARY, BZ_VERTICES


class BandStructure:
    """
    Compute and plot magnon band structures along high-symmetry paths.

    Parameters
    ----------
    hamiltonian : Hamiltonian
        The Hamiltonian object for computing H(k).
    colpa : Colpa
        The Colpa diagonalizer for obtaining eigenvalues.
    bz_type : str
        Type of Brillouin zone. Must be a key in BZ_LIBRARY
        (e.g., 'cubic', 'hexagonal', 'orthorhombic').
    system_name : str, optional
        Name of the system for file management. Default is 'system'.

    Attributes
    ----------
    hamiltonian : Hamiltonian
        Reference to Hamiltonian object.
    colpa : Colpa
        Reference to Colpa diagonalizer.
    bz_type : str
        Brillouin zone type.
    system_name : str
        System name for file saving.
    hsp : dict
        High-symmetry points for the chosen BZ type.
    k_points : np.ndarray or None
        K-points along computed path (in reciprocal lattice coords).
    k_cartesian : np.ndarray or None
        K-points in Cartesian coordinates.
    eigenvalues : np.ndarray or None
        Computed eigenvalues along path, shape (n_kpoints, n_bands).
    x_axis : np.ndarray or None
        1D parametrization of the path for plotting.
    tick_positions : list or None
        Positions of high-symmetry points on x_axis.
    tick_labels : list or None
        Labels for high-symmetry points.
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

        # Load high-symmetry points for this BZ type
        self.hsp = self.hsp = get_bz_hsp(bz_type, self.hamiltonian.system.reciprocal_vectors)

        # Storage for computed results
        self.k_points: Optional[np.ndarray] = None
        self.k_cartesian: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.x_axis: Optional[np.ndarray] = None
        self.tick_positions: Optional[list] = None
        self.tick_labels: Optional[list] = None
        self.fig: Optional[Figure] = None

    # ------------------------------------------------------------------
    # Path computation
    # ------------------------------------------------------------------

    def compute_along_path(
        self,
        path: list[str],
        n_points: int = 100,
        custom_points: Optional[dict[str, list]] = None,
        arc_length: bool = True,
        return_full: bool = False,
        n_cores: int = 1,
        worker_pool = None,
    ) -> None:
        """
        Compute band structure along a high-symmetry path.

        Parameters
        ----------
        path : list of str
            List of high-symmetry point labels defining the path,
            e.g., ['G', 'X', 'M', 'G'].
        n_points : int, optional
            Number of k-points to sample along the path. Default is 100.
        custom_points : dict, optional
            Custom k-point definitions that override BZ_LIBRARY.
            Keys are point labels, values are [kx, ky, kz] coordinates
            in reciprocal lattice units.
        arc_length : bool, optional
            If True, use arc length parametrization (isometric plot).
            If False, space points uniformly. Default is True.
        return_full : bool, optional
            If True, return full (2N,) eigenvalue spectrum at each k.
            If False, return only positive (N,) eigenvalues. Default is False.
        n_cores : int, optional
            Number of CPU cores to run parallel jobs on. If 1, run serially. 
            If -1, use all CPU cores. Default is 1.
        worker_pool : joblib.Parallel, optional
            Pre-initialized worker pool for reuse across multiple calls.
            Improves performance when computing multiple band structures.
            Create with: pool = Parallel(n_jobs = n_cores)
            Default is None (creates new pool each time).
        """
        # Resolve k-point coordinates
        k_coords = self._resolve_path_coords(path, custom_points)

        # Convert to Cartesian coordinates
        k_cartesian = self._to_cartesian(k_coords)

        # Generate interpolated path
        if arc_length:
            interpolate_func, tick_positions = self._arc_length_parametrization(k_cartesian)
        else:
            interpolate_func, tick_positions = self._uniform_parametrization(k_cartesian)

        # Sample points along path
        t = np.linspace(0, 1, n_points)
        k_sampled = np.array([interpolate_func(ti) for ti in t])

        # Compute eigenvalues at each k-point
        if n_cores == 1:
            # Serial execution
            eigenvalues_list = []
            for k in k_sampled:
                H_k = self.hamiltonian.compute_at_k(k)
                eigs, _ = self.colpa.diagonalize(H_k, k=k, return_full=return_full)
                eigenvalues_list.append(eigs)
            eigenvalues = np.array(eigenvalues_list)
        else:
            # Parallel execution
            from joblib import Parallel, delayed

            # Reuse pool if provided, otherwise create new one
            if worker_pool is None:
                parallel = Parallel(n_jobs = n_cores)
            else:
                parallel = worker_pool
            
            def compute_at_single_k(k):
                H_k = self.hamiltonian.compute_at_k(k)
                eigs, _ = self.colpa.diagonalize(H_k, k=k, return_full=return_full)
                return eigs
            
            eigenvalues_list = parallel(
                delayed(compute_at_single_k)(k) for k in k_sampled
            )
            eigenvalues = np.array(eigenvalues_list)

        eigenvalues = np.array(eigenvalues_list)  # shape (n_kpoints, n_bands)

        # Store results
        self.k_points = k_coords
        self.k_cartesian = k_sampled
        self.eigenvalues = eigenvalues
        self.x_axis = t
        self.tick_positions = tick_positions
        self.tick_labels = self._get_tick_labels(path, custom_points)

    def _resolve_path_coords(
        self,
        path: list[str],
        custom_points: Optional[dict[str, list]],
    ) -> np.ndarray:
        """
        Resolve k-point coordinates from path labels.

        Parameters
        ----------
        path : list of str
            High-symmetry point labels.
        custom_points : dict or None
            Custom point definitions.

        Returns
        -------
        coords : np.ndarray of shape (n_points, 3)
            K-point coordinates in reciprocal lattice units.
        """
        coords = []
        for label in path:
            if custom_points is not None and label in custom_points:
                # Use custom point
                coords.append(custom_points[label])
            elif label in self.hsp:
                # Use library point
                coords.append(self.hsp[label][0])
            else:
                raise ValueError(
                    f"Unknown point '{label}'. Not in BZ library for '{self.bz_type}' "
                    f"and not in custom_points."
                )
        return np.array(coords)

    def _get_tick_labels(
        self,
        path: list[str],
        custom_points: Optional[dict[str, list]],
    ) -> list[str]:
        """Get display labels for path points."""
        labels = []
        for label in path:
            if custom_points is not None and label in custom_points:
                # Use simple label for custom points
                labels.append(label)
            elif label in self.hsp:
                # Use fancy label from library
                labels.append(self.hsp[label][1])
            else:
                labels.append(label)
        return labels

    def _to_cartesian(self, k_lattice: np.ndarray) -> np.ndarray:
        """
        Convert k-points from reciprocal lattice coordinates to Cartesian.

        Parameters
        ----------
        k_lattice : np.ndarray of shape (n_points, 3)
            K-points in reciprocal lattice units.

        Returns
        -------
        k_cartesian : np.ndarray of shape (n_points, 3)
            K-points in Cartesian coordinates.
        """
        b1, b2, b3 = self.hamiltonian.system.reciprocal_vectors
        k_cartesian = np.array([
            k[0] * b1 + k[1] * b2 + k[2] * b3
            for k in k_lattice
        ])
        return k_cartesian

    # ------------------------------------------------------------------
    # Path parametrization
    # ------------------------------------------------------------------

    @staticmethod
    def _arc_length_parametrization(
        points: np.ndarray
    ) -> tuple[Callable, np.ndarray]:
        """
        Create arc length parametrization of a path through k-space.

        The path is parametrized by t ∈ [0, 1] where t represents the
        fraction of total path length traversed.

        Parameters
        ----------
        points : np.ndarray of shape (n_points, 3)
            K-points defining the path.

        Returns
        -------
        interpolate_func : callable
            Function that takes t ∈ [0, 1] and returns k-point.
        normalized_distances : np.ndarray of shape (n_points,)
            Normalized cumulative distances to each point (for tick positions).
        """
        n = points.shape[0] - 1

        # Compute segment lengths
        segment_lengths = np.array([
            np.linalg.norm(points[i+1] - points[i]) for i in range(n)
        ])

        # Cumulative distances
        cumulative_dist = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_dist = cumulative_dist[-1]

        # Normalize to [0, 1]
        if total_dist > 0:
            normalized_dist = cumulative_dist / total_dist
        else:
            normalized_dist = cumulative_dist

        def interpolate_func(t: float) -> np.ndarray:
            """Interpolate k-point at parameter value t ∈ [0, 1]."""
            # Find segment containing t
            idx = np.searchsorted(normalized_dist, t) - 1
            idx = np.clip(idx, 0, n - 1)

            # Segment boundaries
            start = normalized_dist[idx]
            end = normalized_dist[idx + 1]

            # Position within segment
            if end == start:
                dt = 0
            else:
                dt = (t - start) / (end - start)

            # Linear interpolation
            return points[idx] + dt * (points[idx + 1] - points[idx])

        return interpolate_func, normalized_dist

    @staticmethod
    def _uniform_parametrization(
        points: np.ndarray
    ) -> tuple[Callable, np.ndarray]:
        """
        Create uniform parametrization of a path (equal spacing between points).

        Parameters
        ----------
        points : np.ndarray of shape (n_points, 3)
            K-points defining the path.

        Returns
        -------
        interpolate_func : callable
            Function that takes t ∈ [0, 1] and returns k-point.
        point_positions : np.ndarray of shape (n_points,)
            Positions of input points on [0, 1] interval.
        """
        n = points.shape[0] - 1
        point_positions = np.arange(n + 1) / n

        def interpolate_func(t: float) -> np.ndarray:
            """Interpolate k-point at parameter value t ∈ [0, 1]."""
            idx = int(t * n)
            idx = np.clip(idx, 0, n - 1)

            # Position within segment
            dt = n * t - idx

            # Linear interpolation
            return points[idx] + dt * (points[idx + 1] - points[idx])

        return interpolate_func, point_positions

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        bands: Optional[list[int]] = None,
        figsize: tuple = (8, 6),
        color: str = 'blue',
        linewidth: float = 1.5,
        title: Optional[str] = None,
        ylabel: str = 'Energy',
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot the computed band structure.

        Parameters
        ----------
        bands : list of int, optional
            Indices of bands to plot. If None, plot all bands.
        figsize : tuple, optional
            Figure size (width, height) in inches. Default is (8, 6).
        color : str, optional
            Line color. Default is 'blue'.
        linewidth : float, optional
            Line width. Default is 1.5.
        title : str, optional
            Plot title. If None, uses system name.
        ylabel : str, optional
            Y-axis label. Default is 'Energy'.
        show : bool, optional
            If True, call plt.show(). Default is True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.

        Raises
        ------
        RuntimeError
            If compute_along_path has not been called yet.
        """
        if self.eigenvalues is None:
            raise RuntimeError(
                "No band structure data. Call compute_along_path() first."
            )

        self.fig, ax = plt.subplots(figsize=figsize)

        # Determine which bands to plot
        n_bands = self.eigenvalues.shape[1]
        if bands is None:
            bands = list(range(n_bands))

        # Plot selected bands
        for band_idx in bands:
            if band_idx >= n_bands:
                raise ValueError(f"Band index {band_idx} out of range (max {n_bands-1})")
            ax.plot(
                self.x_axis,
                self.eigenvalues[:, band_idx],
                color=color,
                linewidth=linewidth,
            )

        # Add vertical lines at high-symmetry points
        for tick_pos in self.tick_positions:
            ax.axvline(tick_pos, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

        # Set x-axis ticks
        ax.set_xticks(self.tick_positions)
        ax.set_xticklabels(self.tick_labels, fontsize=12)

        # Labels and title
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xlim(0, 1)

        if title is None:
            title = f"Band Structure - {self.system_name}"
        ax.set_title(title, fontsize=16)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if show:
            plt.show()

        return self.fig

    # ------------------------------------------------------------------
    # File saving
    # ------------------------------------------------------------------

    def save(
        self,
        filepath: Optional[str] = None,
        directory: Optional[str] = None,
        filename: Optional[str] = None,
        format: str = 'pdf',
        dpi: int = 600,
        overwrite: bool = False,
    ) -> str:
        """
        Save the band structure plot to file.

        Parameters
        ----------
        filepath : str, optional
            Full path including filename and extension. If provided,
            directory, filename, and format are ignored.
        directory : str, optional
            Directory to save in. Default is './results/{system_name}/'.
        filename : str, optional
            Filename without extension. Default is 'bandstructure_{system_name}'.
        format : str, optional
            File format ('pdf' or 'png'). Default is 'pdf'.
        dpi : int, optional
            Resolution for raster formats. Default is 600.
        overwrite : bool, optional
            If False, append '_2', '_3', etc. to avoid overwriting.
            Default is False.

        Returns
        -------
        saved_path : str
            The actual path where the file was saved.

        Raises
        ------
        RuntimeError
            If no figure has been created yet (call plot() first).
        """
        # if not hasattr(plt, 'gcf') or plt.gcf().get_axes() == []:
        if self.fig is None:
            raise RuntimeError("No figure to save. Call plot() first.")

        # Use explicit filepath if provided
        if filepath is not None:
            final_path = filepath
        else:
            # Build path from components
            if directory is None:
                directory = f'./results/{self.system_name}/'

            if filename is None:
                filename = f'bandstructure_{self.system_name}'

            # Ensure directory exists
            os.makedirs(directory, exist_ok=True)

            # Build full path
            ext = format if format.startswith('.') else f'.{format}'
            final_path = os.path.join(directory, filename + ext)

            # Handle overwrite
            if not overwrite:
                final_path = self._get_unique_filepath(final_path)

        # Save figure
        # fig = plt.gcf()
        self.fig.savefig(
            final_path,
            facecolor='w',
            transparent=False,
            dpi=dpi,
            bbox_inches='tight'
        )

        print(f"Band structure saved to: {final_path}")
        return final_path

    @staticmethod
    def _get_unique_filepath(filepath: str) -> str:
        """
        Generate unique filepath by appending _2, _3, etc. if file exists.

        Parameters
        ----------
        filepath : str
            Desired filepath.

        Returns
        -------
        unique_path : str
            Filepath that doesn't exist yet.
        """
        if not os.path.exists(filepath):
            return filepath

        # Split into base and extension
        base, ext = os.path.splitext(filepath)

        # Find next available number
        counter = 2
        while True:
            new_path = f"{base}_{counter}{ext}"
            if not os.path.exists(new_path):
                return new_path
            counter += 1

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "computed" if self.eigenvalues is not None else "not computed"
        return (
            f"BandStructure(bz_type='{self.bz_type}', "
            f"system='{self.system_name}', status={status})"
        )


# ======================================================================
# 3D Band Surface Visualization
# ======================================================================

class BandStructure3D:
    """
    Compute and visualize 3D magnon band surfaces over the Brillouin zone.

    Uses Plotly for interactive 3D visualization with support for:
    - Multiple band surfaces with transparency
    - Degeneracy point detection and marking
    - BZ boundary and high-symmetry point annotations
    - Energy gradient coloring or solid band colors
    - Interactive HTML or static image export

    Parameters
    ----------
    hamiltonian : Hamiltonian
        The Hamiltonian object for computing H(k).
    colpa : Colpa
        The Colpa diagonalizer for obtaining eigenvalues.
    bz_type : str
        Type of Brillouin zone (must be a 2D system: 'cubic', 'hexagonal', etc.).
    system_name : str, optional
        Name of the system for file management. Default is 'system'.

    Attributes
    ----------
    hamiltonian : Hamiltonian
    colpa : Colpa
    bz_type : str
    system_name : str
    hsp : dict
        High-symmetry points for the chosen BZ type.
    k_mesh : np.ndarray or None
        2D mesh of k-points (in reciprocal lattice coords), shape (n_kx, n_ky, 2).
    eigenvalues : np.ndarray or None
        Computed eigenvalues on mesh, shape (n_kx, n_ky, n_bands).
    degeneracies : list or None
        List of (kx_idx, ky_idx, band_idx) tuples marking degenerate points.
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

        # Storage for computed results
        self.k_mesh: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.degeneracies: Optional[list] = None

    # ------------------------------------------------------------------
    # Mesh generation and computation
    # ------------------------------------------------------------------

    def compute_on_mesh(
        self,
        n_kx: int = 50,
        n_ky: int = 50,
        kz: float = 0.0,
        k_range: Optional[tuple] = None,
        return_full: bool = False,
        detect_degeneracies: bool = True,
        degeneracy_tol: float = 1e-3,
        n_cores: int = 1,
        worker_pool = None,
    ) -> None:
        """
        Compute band structure on a 2D mesh in the kx-ky plane.

        Parameters
        ----------
        n_kx : int, optional
            Number of k-points along kx direction. Default is 50.
        n_ky : int, optional
            Number of k-points along ky direction. Default is 50.
        kz : float, optional
            Fixed kz value (for 2D systems). Default is 0.
        k_range : tuple of tuples, optional
            Range for k-mesh: ((kx_min, kx_max), (ky_min, ky_max))
            in reciprocal lattice units. If None, uses [-0.5, 0.5] for both.
        return_full : bool, optional
            If True, compute full (2N,) eigenvalues. If False, only positive.
            Default is False.
        detect_degeneracies : bool, optional
            If True, detect and store degenerate points. Default is True.
        degeneracy_tol : float, optional
            Energy difference threshold for considering bands degenerate.
            Default is 1e-3.
        n_cores : int, optional
            Number of CPU cores to run parallel jobs on. If 1, run serially. 
            If -1, use all CPU cores. Default is 1.
        worker_pool : joblib.Parallel, optional
            Pre-initialized worker pool for reuse across multiple calls.
            Improves performance when computing multiple band structures.
            Create with: pool = Parallel(n_jobs = n_cores)
            Default is None (creates new pool each time).
        """
        # Default k-range covers first BZ in lattice coordinates
        if k_range is None:
            k_range = ((-0.5, 0.5), (-0.5, 0.5))

        # Create 2D mesh in reciprocal lattice coordinates
        kx_vals = np.linspace(k_range[0][0], k_range[0][1], n_kx)
        ky_vals = np.linspace(k_range[1][0], k_range[1][1], n_ky)
        kx_mesh, ky_mesh = np.meshgrid(kx_vals, ky_vals, indexing='ij')

        # Store mesh
        self.k_mesh = np.stack([kx_mesh, ky_mesh], axis=-1)

        # Compute eigenvalues at each k-point
        if n_cores == 1:
            # Serial execution
            eigenvalues = []
            for i in range(n_kx):
                row = []
                for j in range(n_ky):
                    k_lattice = np.array([kx_mesh[i, j], ky_mesh[i, j], kz])
                    k_cartesian = self._to_cartesian(k_lattice)
                    H_k = self.hamiltonian.compute_at_k(k_cartesian)
                    eigs, _ = self.colpa.diagonalize(H_k, k=k_cartesian, return_full=return_full)
                    row.append(eigs)
                eigenvalues.append(row)
            self.eigenvalues = np.array(eigenvalues) # shape (n_kx, n_ky, n_bands)
        else:
            # Parallel execution - flatten the 2D loop
            from joblib import Parallel, delayed

            # Reuse pool if provided, otherwise create new one
            if worker_pool is None:
                parallel = Parallel(n_jobs=n_cores)
            else:
                parallel = worker_pool
            
            # Create list of all (i, j) pairs
            ij_pairs = [(i, j) for i in range(n_kx) for j in range(n_ky)]
            
            def compute_at_mesh_point(i, j):
                k_lattice = np.array([kx_mesh[i, j], ky_mesh[i, j], kz])
                k_cartesian = self._to_cartesian(k_lattice)
                H_k = self.hamiltonian.compute_at_k(k_cartesian)
                eigs, _ = self.colpa.diagonalize(H_k, k=k_cartesian, return_full=return_full)
                return i, j, eigs
            
            results = parallel(
                delayed(compute_at_mesh_point)(i, j) for i, j in ij_pairs
            )
            
            # Reconstruct 2D array from results
            eigenvalues = [[None for _ in range(n_ky)] for _ in range(n_kx)]
            for i, j, eigs in results:
                eigenvalues[i][j] = eigs
            
            self.eigenvalues = np.array(eigenvalues)
        
        # Detect degeneracies
        if detect_degeneracies:
            self.degeneracies = self._detect_degeneracies(degeneracy_tol)
        else:
            self.degeneracies = []

    def _to_cartesian(self, k_lattice: np.ndarray) -> np.ndarray:
        """Convert k-point from lattice to Cartesian coordinates."""
        b1, b2, b3 = self.hamiltonian.system.reciprocal_vectors
        return k_lattice[0] * b1 + k_lattice[1] * b2 + k_lattice[2] * b3

    def _detect_degeneracies(self, tol: float) -> list:
        """
        Detect points where adjacent bands are degenerate.

        Returns
        -------
        degeneracies : list of tuples
            List of (kx_idx, ky_idx, band_idx) marking where band_idx and
            band_idx+1 are degenerate within tolerance.
        """
        degeneracies = []
        n_kx, n_ky, n_bands = self.eigenvalues.shape

        for i in range(n_kx):
            for j in range(n_ky):
                energies = self.eigenvalues[i, j, :]
                for band in range(n_bands - 1):
                    if np.abs(energies[band+1] - energies[band]) < tol:
                        degeneracies.append((i, j, band))

        return degeneracies

    # ------------------------------------------------------------------
    # Plotting with Plotly
    # ------------------------------------------------------------------

    def plot(
        self,
        bands: Optional[list[int]] = None,
        mode: str = 'overlay',
        color_by: str = 'band',
        opacity: float = 0.7,
        mark_degeneracies: bool = True,
        show_bz: bool = True,
        show_hsp: bool = True,
        title: Optional[str] = None,
        show: bool = True,
    ):
        """
        Create interactive 3D plot of band surfaces using Plotly.

        Parameters
        ----------
        bands : list of int, optional
            Indices of bands to plot. If None, plot all bands.
        mode : str, optional
            'overlay': all bands on same plot (default)
            'subplots': separate subplot for each band
        color_by : str, optional
            'band': solid color per band (default)
            'energy': color gradient by energy value
        opacity : float, optional
            Surface opacity (0-1). Default is 0.7 for overlay, 1.0 for subplots.
        mark_degeneracies : bool, optional
            If True, mark degenerate points. Default is True.
        show_bz : bool, optional
            If True, draw BZ boundary. Default is True.
        show_hsp : bool, optional
            If True, mark high-symmetry points. Default is True.
        title : str, optional
            Plot title. If None, auto-generated.
        show : bool, optional
            If True, display plot in browser. Default is True.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The Plotly figure object.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError(
                "Plotly is required for 3D visualization. "
                "Install with: pip install plotly"
            )

        if self.eigenvalues is None:
            raise RuntimeError("No data computed. Call compute_on_mesh() first.")

        # Determine which bands to plot
        n_bands = self.eigenvalues.shape[2]
        if bands is None:
            bands = list(range(n_bands))

        # Create figure
        if mode == 'subplots':
            n_plots = len(bands)
            cols = int(np.ceil(np.sqrt(n_plots)))
            rows = int(np.ceil(n_plots / cols))
            fig = make_subplots(
                rows=rows, cols=cols,
                specs=[[{'type': 'surface'}] * cols for _ in range(rows)],
                subplot_titles=[f'Band {b}' for b in bands],
            )
            opacity = 1.0  # Solid for individual plots
        else:
            fig = go.Figure()

        # Color palette for bands
        colors = [
            'blue', 'red', 'green', 'purple', 'orange', 'cyan',
            'magenta', 'yellow', 'brown', 'pink', 'gray', 'lime'
        ]

        # Get mesh coordinates
        kx = self.k_mesh[:, :, 0]
        ky = self.k_mesh[:, :, 1]

        # Plot each band
        for idx, band in enumerate(bands):
            energies = self.eigenvalues[:, :, band]

            # Determine coloring
            if color_by == 'energy':
                colorscale = 'Viridis'
                surfacecolor = energies
                showscale = True
            else:  # color_by == 'band'
                colorscale = [[0, colors[band % len(colors)]], [1, colors[band % len(colors)]]]
                surfacecolor = None
                showscale = False

            surface = go.Surface(
                x=kx,
                y=ky,
                z=energies,
                colorscale=colorscale,
                surfacecolor=surfacecolor,
                opacity=opacity,
                name=f'Band {band}',
                showscale=showscale,
            )

            if mode == 'subplots':
                row = idx // cols + 1
                col = idx % cols + 1
                fig.add_trace(surface, row=row, col=col)
            else:
                fig.add_trace(surface)

        # Mark degeneracies
        if mark_degeneracies and self.degeneracies and mode == 'overlay':
            deg_kx = [kx[i, j] for i, j, _ in self.degeneracies]
            deg_ky = [ky[i, j] for i, j, _ in self.degeneracies]
            deg_e = [self.eigenvalues[i, j, b] for i, j, b in self.degeneracies]

            fig.add_trace(go.Scatter3d(
                x=deg_kx,
                y=deg_ky,
                z=deg_e,
                mode='markers',
                marker=dict(size=5, color='black', symbol='diamond'),
                name='Degeneracies',
            ))

        # BZ boundary (simplified - draw rectangle for now)
        if show_bz and mode == 'overlay':
            self._add_bz_boundary(fig, kx, ky)

        # High-symmetry points
        if show_hsp and mode == 'overlay':
            self._add_hsp_markers(fig)

        # Layout
        if title is None:
            title = f"3D Band Structure - {self.system_name}"

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='kx',
                yaxis_title='ky',
                zaxis_title='Energy',
            ),
            showlegend=True,
        )

        if show:
            fig.show()

        return fig

    def _add_bz_boundary(self, fig, kx, ky) -> None:
        """Add BZ boundary lines to figure using proper BZ shape."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return

        # Get vertices for this BZ type, fallback to rectangle if not defined
        if self.bz_type in BZ_VERTICES:
            vertices = BZ_VERTICES[self.bz_type]
        else:
            # Fallback to rectangular boundary
            kx_min, kx_max = kx.min(), kx.max()
            ky_min, ky_max = ky.min(), ky.max()
            vertices = [
                [kx_min, ky_min], [kx_max, ky_min],
                [kx_max, ky_max], [kx_min, ky_max]
            ]

        # Close the polygon by appending first vertex at end
        boundary_x = [v[0] for v in vertices] + [vertices[0][0]]
        boundary_y = [v[1] for v in vertices] + [vertices[0][1]]
        boundary_z = [0] * len(boundary_x)

        fig.add_trace(go.Scatter3d(
            x=boundary_x,
            y=boundary_y,
            z=boundary_z,
            mode='lines',
            line=dict(color='black', width=4),
            name='BZ Boundary',
            showlegend=False,
        ))

    def _add_hsp_markers(self, fig) -> None:
        """Add high-symmetry point markers to figure."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return

        hsp_kx = []
        hsp_ky = []
        hsp_labels = []

        for label, (coords, display_label) in self.hsp.items():
            if len(coords) >= 2:  # Only plot 2D points
                hsp_kx.append(coords[0])
                hsp_ky.append(coords[1])
                hsp_labels.append(display_label)

        if hsp_kx:
            fig.add_trace(go.Scatter3d(
                x=hsp_kx,
                y=hsp_ky,
                z=[0] * len(hsp_kx),
                mode='markers+text',
                marker=dict(size=8, color='red', symbol='x'),
                text=hsp_labels,
                textposition='top center',
                name='High-symmetry points',
                showlegend=False,
            ))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save(
        self,
        fig,
        filepath: Optional[str] = None,
        format: str = 'html',
        camera: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Save the 3D plot to file.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            Figure returned from plot().
        filepath : str, optional
            Full path for output file. If None, auto-generated.
        format : str, optional
            'html' for interactive (default)
            'png', 'pdf' for static images
        camera : str, optional
            Camera angle preset for static exports. Options:
            'top' - top-down view
            'isometric' - 3D perspective view (default for static)
            'side' - side view
            If None and format is not 'html', uses 'isometric'.
        **kwargs
            Additional arguments passed to fig.write_html() or fig.write_image()

        Returns
        -------
        saved_path : str
            Path where file was saved.
        """
        # Camera angle presets
        CAMERA_PRESETS = {
            'top': dict(
                eye=dict(x=0, y=0, z=2.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=1, z=0)
            ),
            'isometric': dict(
                eye=dict(x=1.5, y=1.5, z=1.3),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            'side': dict(
                eye=dict(x=2.0, y=0, z=0.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
        }

        # Apply camera angle for static exports
        if format != 'html':
            if camera is None:
                camera = 'isometric'  # Default for static images
            if camera in CAMERA_PRESETS:
                fig.update_layout(scene_camera=CAMERA_PRESETS[camera])

        if filepath is None:
            directory = f'./results/{self.system_name}/'
            os.makedirs(directory, exist_ok=True)
            filename = f'bandstructure_3d_{self.system_name}'
            ext = '.html' if format == 'html' else f'.{format}'
            filepath = os.path.join(directory, filename + ext)

        if format == 'html':
            fig.write_html(filepath, **kwargs)
        else:
            # Requires kaleido: pip install kaleido
            fig.write_image(filepath, **kwargs)

        print(f"3D band structure saved to: {filepath}")
        return filepath

    def __repr__(self) -> str:
        status = "computed" if self.eigenvalues is not None else "not computed"
        n_deg = len(self.degeneracies) if self.degeneracies else 0
        return (
            f"BandStructure3D(bz_type='{self.bz_type}', "
            f"system='{self.system_name}', status={status}, "
            f"degeneracies={n_deg})"
        )