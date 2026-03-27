"""
magnon_solver

A modular solver for magnonic systems based on Linear Spin Wave Theory (LSWT),
Colpa's Bogoliubov diagonalization algorithm, and topological invariant calculation
via the Fukui-Hatsugai-Suzuki (FHS) method.

Follows the theoretical work laid out in:
    - LSWT scheme: S. Toth & B. Lake, 2015, doi.org/10.1088/0953-8984/27/16/166002
    - Bogoliubov: J.H.P. Colpa, 1978, doi.org/10.1016/0378-4371(78)90160-7
    - FHS method: Fukui, Hatsugai, Suzuki, 2004, doi.org/10.1143/JPSJ.74.1674

Modules
-------
reader : InputReader
    Read and validate .csv input files.
parser : Parser
    Parse symbolic expressions into numerical arrays.
system : SpinSystem
    Data container with validation and symmetrization.
hamiltonian : Hamiltonian
    LSWT Hamiltonian construction following Toth & Lake.
diagonalizer : Colpa
    Bogoliubov diagonalization with ground state checks.
bandstructure : BandStructure, BandStructure3D
    Band structure computation and visualization.

Quick Start
-----------
>>> from magnon_solver import InputReader, Parser, SpinSystem, Hamiltonian, Colpa
>>> reader = InputReader('input.csv')
>>> parser = Parser(reader.translation_vectors, reader.spin_data,
...                 reader.interaction_data, reader.parameters)
>>> parser.parse()
>>> system = SpinSystem(parser.translation_vectors, parser.spin_data,
...                     parser.interaction_data, parser.parameters)
>>> hamiltonian = Hamiltonian(system)
>>> colpa = Colpa()
"""

from magnon_solver.reader import InputReader
from magnon_solver.parser import Parser
from magnon_solver.system import SpinSystem
from magnon_solver.hamiltonian import Hamiltonian
from magnon_solver.diagonalizer import Colpa
from magnon_solver.bandstructure import BandStructure, BandStructure3D
from magnon_solver.topology import TopologySolver
# from magnon_solver.bz_configs import get_bz_hsp, BZ_LIBRARY, BZ_VERTICES

__version__ = "1.0.0"

__all__ = [
    "InputReader",
    "Parser",
    "SpinSystem",
    "Hamiltonian",
    "Colpa",
    "BandStructure",
    "BandStructure3D",
    "TopologySolver",
]