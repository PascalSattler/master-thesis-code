"""
reader.py

Input file reader for the magnon solver.

Reads and validates .csv input files containing lattice structure,
spin data, interaction data, and parameters for LSWT calculations.
"""

import pandas as pd
from pathlib import Path
from typing import Optional



class InputReader:
    """
    Read and validate .csv input files for magnonic systems.

    The CSV file should have columns organized as:
    - #translation vectors: 3 rows for lattice vectors
    - #spins section: sublattice, basis vector, ground state direction,
      spin length, gyromagnetic matrix
    - #interactions section: reference sublat, neighbor sublat,
      difference vector, interaction matrix
    - #parameters section: parameter names and values

    Parameters
    ----------
    csv_file : str or Path
        Path to the input .csv file.

    Attributes
    ----------
    csv_file : Path
        Path to the input.csv file.
    translation_vectors : list of str
        Translation vector expressions (e.g., '[a, 0, 0]').
    spin_data : list of dict
        Spin/sublattice data with keys: 'sublattice', 'basis vector',
        'ground state direction', 'spin length', 'gyromagnetic matrix'.
    interaction_data : list of dict
        Interaction data with keys: 'reference sublat', 'neighbor sublat',
        'difference vector', 'interaction matrix'.
    parameters : dict
        Parameter name -> value mapping (all strings).

    Examples
    --------
    >>> input = InputReader('input.csv')
    >>> parser = Parser(
    ...     input.translation_vectors,
    ...     input.spin_data,
    ...     input.interaction_data,
    ...     input.parameters
    ... )
    """

    def __init__(self, csv_file: str | Path) -> None:
        
        self.csv_file = Path(csv_file)

        # check that file exists
        if not self.csv_file.exists():
            raise FileNotFoundError(
                f"Input file not found: {self.csv_file}\n"
                f"Please provide a valid .csv file path."
            )

        # read and validate
        self._data = self._read_csv()
        self._validate_structure()

        # extract data
        self.translation_vectors = self._get_translation_vectors()
        self.spin_data = self._get_spin_data()
        self.interaction_data = self._get_interaction_data()
        self.parameters = self._get_parameters()

    # ------------------------------------------------------------------
    # Reading and validation
    # ------------------------------------------------------------------

    def _read_csv(self) -> pd.DataFrame:
        """
        Read the .csv file with all cells interpreted as strings.

        Returns
        -------
        data : pd.DataFrame
            Raw data from .csv file.

        Raises
        ------
        ValueError
            If .csv file is empty or faulty otherwise.
        """
        try:
            data = pd.read_csv(self.csv_file, dtype=str)
        except pd.errors.EmptyDataError:
            raise ValueError(f".csv file is empty: {self.csv_file}")
        except Exception as exc:
            raise ValueError(
                f"Failed to read .csv file {self.csv_file} because: {exc}"
            ) from exc

        if data.empty:
            raise ValueError(f".csv file contains no data: {self.csv_file}")

        return data


    def _validate_structure(self) -> None:
        """
        Validate that the required columns are present.

        Raises
        ------
        ValueError
            If required columns are missing or falsely named.
        """
        required_columns = {
            '#translation vectors',
            'sublattice',
            'basis vector',
            'ground state direction',
            'spin length',
            'gyromagnetic matrix',
            'reference sublat',
            'neighbor sublat',
            'difference vector',
            'interaction matrix',
            '#parameters',
            'value',
        }

        missing = required_columns - set(self._data.columns)
        if missing:
            raise ValueError(
                f"Input file is missing required columns: {missing}\n"
                f"Required columns are: {list(self._data.columns)}"
            )

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def _get_translation_vectors(self) -> list[str]:
        """
        Extract translation vectors from '#translation vectors' column.

        Returns
        -------
        vectors : list of str
            3 translation vector expressions.

        Raises
        ------
        ValueError
            If not exactly 3 translation vectors found.
        """
        vectors = self._data['#translation vectors'].dropna().tolist()

        if len(vectors) != 3:
            raise ValueError(
                f"Expected 3 translation vectors, found {len(vectors)}. "
                f"Ensure the '#translation vectors' column has exactly 3 entries."
            )

        return vectors


    def _get_spin_data(self) -> list[dict[str, str]]:
        """
        Extract spin/sublattice data.

        Returns
        -------
        spin_data : list of dict
            Each dict contains: sublattice, basis vector, ground state direction,
            spin length, gyromagnetic matrix.

        Raises
        ------
        ValueError
            If no spin data is found.
        """
        columns = [
            'sublattice',
            'basis vector',
            'ground state direction',
            'spin length',
            'gyromagnetic matrix'
        ]

        spin_data = self._data[columns].dropna(how='all').dropna(subset=['sublattice']).to_dict('records')

        if not spin_data:
            raise ValueError(
                f"No spin data found. Ensure that the {columns} columns have entries."  # TODO test if this error gets shown properly
            )

        return spin_data


    def _get_interaction_data(self) -> list[dict[str, str]]:
        """
        Extract interaction data.

        Returns
        -------
        interaction_data : list of dict
            Each dict contains: reference sublat, neighbor sublat,
            difference vector, interaction matrix.

        Raises
        ------
        ValueError
            If no interaction data found.
        """
        columns = [
            'reference sublat',
            'neighbor sublat',
            'difference vector',
            'interaction matrix'
        ]

        interaction_data = (
            self._data[columns]
            .dropna(how='all')
            .dropna(subset=['reference sublat'])
            .to_dict('records')
        )

        if not interaction_data:
            raise ValueError(
                f"No interaction data found. Ensure the {columns} "     # TODO test if this error gets shown properly
                "columns have entries."
            )

        return interaction_data


    def _get_parameters(self) -> dict[str, str]:
        """
        Extract parameters as key-value pairs.

        Returns
        -------
        parameters : dict
            Parameter name -> numerical value string mapping.

        Raises
        ------
        ValueError
            If no parameters found or duplicate parameter names exist.
        """
        params = (
            self._data[['#parameters', 'value']]
            .dropna()
            .set_index('#parameters')['value']
            .to_dict()
        )

        if not params:
            raise ValueError(
                "No parameters found. Ensure the '#parameters' and 'value' "
                "columns have entries."
            )

        return params

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    # a __repr__ dunder method is defined to be able to pass the object to print()
    def __repr__(self) -> str:
        """
        Returns a printable representation of an object of this class.
        """
        n_spins = len(self.spin_data)
        n_interactions = len(self.interaction_data)
        n_params = len(self.parameters)

        return (
            f"InputReader(file = '{self.csv_file.name}', "
            f"spins = {n_spins}, interactions = {n_interactions}, "
            f"parameters = {n_params})"
        )


    def summary(self) -> str:
        """
        Return a detailed summary of the loaded data.

        Returns
        -------
        summary : str
            Formatted summary of file contents.
        """
        lines = [
            f"Input File: {self.csv_file}",
            f"\nTranslation Vectors: {len(self.translation_vectors)}",
        ]
        for i, vec in enumerate(self.translation_vectors, 1):
            lines.append(f"  a{i}: {vec}")

        lines.append(f"\nSpins/Sublattices: {len(self.spin_data)}")
        for spin in self.spin_data:
            sublat = spin.get('sublattice', '?')
            lines.append(f"  Sublattice {sublat}")

        lines.append(f"\nInteractions: {len(self.interaction_data)}")
        # group by interaction type if #interactions column has cells with comments
        if '#interactions' in self._data.columns:
            types = self._data['#interactions'].dropna().unique()
            for itype in types:
                if pd.notna(itype):
                    lines.append(f"  {itype}")

        lines.append(f"\nParameters: {len(self.parameters)}")
        for key, val in self.parameters.items():
            lines.append(f"  {key} = {val}")

        return "\n".join(lines)