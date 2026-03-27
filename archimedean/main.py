"""
main.py

Top-level entry point for the magnon_solver pipeline.
Orchestrates the full calculation sequence:
    Initialize -> Parser -> SpinSystem -> Hamiltonian -> Colpa -> Topology

Usage:
    python main.py <input_file.csv>

Or import and use programmatically:
    from main import run_pipeline
    run_pipeline('input.csv', compute_bands=True)
"""

import sys
import numpy as np
from pathlib import Path

from magnon_solver import (
    InputReader,
    Parser,
    SpinSystem,
    Hamiltonian,
    Colpa,
    BandStructure,
    BandStructure3D,
    TopologySolver,
)


def run_pipeline(
    input_file: str,
    compute_bands: bool = True,
    band_path: list[str] = None,
    n_points: int = 200,
    bz_type: str = 'cubic',
    show_plot: bool = True,
    save_plot: bool = False,
) -> dict:
    """
    Run the complete magnon solver pipeline.

    Parameters
    ----------
    input_file : str
        Path to .csv input file.
    compute_bands : bool, optional
        If True, compute and plot band structure. Default is True.
    band_path : list of str, optional
        High-symmetry path for band structure. If None, uses ['G','X','M','G'].
    n_points : int, optional
        Number of k-points along band structure path. Default is 200.
    bz_type : str, optional
        Brillouin zone type. Default is 'cubic'.
    show_plot : bool, optional
        If True, display band structure plot. Default is True.
    save_plot : bool, optional
        If True, save band structure to file. Default is False.

    Returns
    -------
    results : dict
        Dictionary containing:
            'reader' : InputReader
            'parser' : Parser
            'system' : SpinSystem
            'hamiltonian' : Hamiltonian
            'colpa' : Colpa
            'bands' : BandStructure (if compute_bands=True)

    Example
    -------
    >>> results = run_pipeline('input.csv', band_path=['G','X','M','G'])
    >>> hamiltonian = results['hamiltonian']
    >>> bands = results['bands']
    """
    print("=" * 60)
    print("MAGNON SOLVER PIPELINE")
    print("=" * 60)

    # Step 1: Read input file
    print(f"\n[1/6] Reading input file: {input_file}")
    reader = InputReader(input_file)
    print(f"      {reader}")

    # Step 2: Parse symbolic expressions
    print("\n[2/6] Parsing symbolic expressions...")
    parser = Parser(
        reader.translation_vectors,
        reader.spin_data,
        reader.interaction_data,
        reader.parameters,
    )
    parser.parse()
    print(f"      {parser}")

    # Step 3: Build spin system
    print("\n[3/6] Building spin system...")
    system = SpinSystem(
        parser.translation_vectors,
        parser.spin_data,
        parser.interaction_data,
        parser.parameters,
        symmetrize=True,
    )
    print(f"      {system}")

    # Step 4: Create Hamiltonian
    print("\n[4/6] Creating Hamiltonian...")
    hamiltonian = Hamiltonian(system)
    print(f"      {hamiltonian}")

    # Step 5: Initialize Colpa diagonalizer
    print("\n[5/6] Initializing Colpa diagonalizer...")
    colpa = Colpa(
        global_gauge=True,
        force_particle_hole_symmetry=True,
    )
    print(f"      {colpa}")

    results = {
        'reader': reader,
        'parser': parser,
        'system': system,
        'hamiltonian': hamiltonian,
        'colpa': colpa,
    }

    # Step 6: Compute band structure (optional)
    if compute_bands:
        print("\n[6/6] Computing band structure...")

        if band_path is None:
            band_path = ['G', 'X', 'M', 'G']

        system_name = Path(input_file).stem

        bands = BandStructure(
            hamiltonian=hamiltonian,
            colpa=colpa,
            bz_type=bz_type,
            system_name=system_name,
        )

        bands.compute_along_path(
            path=band_path,
            n_points=n_points,
            arc_length=True,
        )

        print(f"      {bands}")
        print(f"      Path: {' → '.join(band_path)}")
        print(f"      Energy range: [{np.min(bands.eigenvalues):.3f}, "
              f"{np.max(bands.eigenvalues):.3f}]")

        # Plot
        fig = bands.plot(show=show_plot)

        # Save if requested
        if save_plot:
            saved_path = bands.save()
            print(f"      Saved to: {saved_path}")

        results['bands'] = bands

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    return results


def main():
    """Command-line entry point."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_file.csv>")
        print("\nExample:")
        print("  python main.py examples/square.csv")
        sys.exit(1)

    input_file = sys.argv[1]

    try:
        results = run_pipeline(
            input_file,
            compute_bands=True,
            show_plot=True,
            save_plot=True,
        )

        print("\nResults stored in 'results' dictionary with keys:")
        for key in results.keys():
            print(f"  - {key}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()