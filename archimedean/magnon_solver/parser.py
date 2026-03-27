"""
parser.py

Intermediate parsing layer between the Initialize input reader and the
numerical solver. Evaluates symbolic string expressions using a controlled
eval namespace built from the user-supplied parameter dictionary.

All data leaving this module is numerical (numpy arrays, floats, ints).
No symbolic expressions exist downstream of this module.
"""

import numpy as np
from numpy import sqrt, pi, sin, cos, tan, arcsin, arccos, arctan, exp, log
from numpy.linalg import norm, det
from typing import Self, Callable



class Parser:
    """
    Parses and evaluates the raw string-based data structures produced by
    the InputReader module into numerical numpy arrays.

    Parameters
    ----------
    translation_vectors : list of str
        List of translation vector strings, e.g. ['[a,0,0]', '[0,a,0]', '[0,0,1]'].
    spin_data : list of dict
        List of site/sublattice dictionaries with string-valued fields.
    interaction_data : list of dict
        List of interaction dictionaries with string-valued fields.
    parameters : dict
        Dictionary mapping parameter names to their string values,
        e.g. {'a': '1', 'J': '-1', 'S1': '1', ...}.
    """

    def __init__(self, 
                 translation_vectors: list[str], 
                 spin_data: list[dict[str, str]], 
                 interaction_data: list[dict[str, str]], 
                 parameters: dict[str, str]
                ) -> None:
        
        self.raw_transl_vects = translation_vectors
        self.raw_spin_data = spin_data
        self.raw_inter_data = interaction_data
        self.raw_params = parameters

        # build the evaluation namespace of standard math functions once at init time
        self._namespace = self._build_namespace()

        # parsed outputs, assigned by parse()
        self.translation_vectors = None
        self.spin_data = None
        self.interaction_data = None
        self.parameters = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    # method that can be called by the user, calls all other internal methods 

    def parse(self) -> Self:
        """
        Run the full parsing module. After calling this method, the
        attributes translation_vectors, spin_data, interaction_data, and
        parameters are assigned with numerical data.

        Returns
        -------
        self : Parser
            Returns self to allow method chaining if desired.
        """
        self.parameters = self._parse_parameters()
        # rebuild namespace after parameters are resolved numerically
        self._namespace = self._build_namespace(numerical_params=self.parameters)

        self.translation_vectors = self._parse_translation_vectors()
        self.spin_data = self._parse_spin_data()
        self.interaction_data = self._parse_interaction_data()

        return self

    # ------------------------------------------------------------------
    # Namespace construction
    # ------------------------------------------------------------------

    # a namespace maps objects to a name (like a tensor is an object that transforms like a tensor)

    def _build_namespace(self, numerical_params: dict | None = None) -> dict[Callable, np.float64]:
        """
        Build a restricted evaluation namespace containing:
        - numpy math functions
        - mathematical constants
        - user parameter values (if provided)

        Parameters
        ----------
        numerical_params : dict or None
            Numerically resolved parameter dictionary. If None, only
            math functions and constants are included.

        Returns
        -------
        namespace : dict
            The namespace dict to pass to eval().
        """
        namespace = {
            "__builtins__": {},  # block access to Python builtins
            # math functions from numpy
            "sqrt": sqrt,
            "pi": pi,
            "sin": sin,
            "cos": cos,
            "tan": tan,
            "arcsin": arcsin,
            "arccos": arccos,
            "arctan": arctan,
            "exp": exp,
            "log": log,
            "abs": np.abs,
            # constants
            "e": np.e,
        }

        if numerical_params is not None:
            namespace.update(numerical_params)

        return namespace

    # ------------------------------------------------------------------
    # Parameter parsing
    # ------------------------------------------------------------------

    def _parse_parameters(self) -> dict[str, int | float | np.ndarray]:
        """
        Evaluate the raw parameter dictionary. Each value is a string
        expression that may reference only math functions and constants
        (parameters cannot reference each other).

        Vector/tuple values like '(0,0,0)' are converted to numpy arrays.

        Returns
        -------
        parsed : dict
            Dictionary mapping parameter names to numerical values
            (float, int, or numpy array).
        """
        parsed = {}
        # use only math namespace here, parameters should not reference each other
        math_namespace = self._build_namespace()

        for key, value_str in self.raw_params.items():
            try:
                evaluated = eval(str(value_str), math_namespace)
                # convert parameters declared as tuple or list to numpy arrays TODO decide if this is stil needed or not
                if isinstance(evaluated, (tuple, list)):
                    evaluated = np.array(evaluated, dtype=float)
                parsed[key] = evaluated

            except Exception as exc:
                raise ValueError(
                    f"Parser: failed to evaluate parameter '{key}' = '{value_str}'. "
                    f"Reason: {exc}"
                ) from exc

        return parsed

    # ------------------------------------------------------------------
    # Translation vector parsing
    # ------------------------------------------------------------------

    def _parse_translation_vectors(self) -> list[np.ndarray]:
        """
        Evaluate the list of translation vector strings into a list of
        1D numpy arrays of shape (3,) -> vectors are assigned to rows.

        Returns
        -------
        vectors : list of np.ndarray
            Each element is a numpy array of shape (3,).
        """
        vectors = []
        for i, vec_str in enumerate(self.raw_transl_vects):
            try:
                evaluated = eval(str(vec_str), self._namespace)
                vectors.append(np.array(evaluated, dtype=float))

            except Exception as exc:
                raise ValueError(
                    f"Parser: failed to evaluate translation vector {i} = '{vec_str}'. "
                    f"Reason: {exc}"
                ) from exc

        # validate shape -> 3D vectors
        for i, vec in enumerate(vectors):
            self._check_shape(vec, (3,), f"translation vector {i}")

        return vectors

    # ------------------------------------------------------------------
    # Spin data parsing
    # ------------------------------------------------------------------

    def _parse_spin_data(self) -> list[dict]:
        """
        Parse the list of spin/site dictionaries. String cells are
        evaluated; sublattice indices are converted to integers.

        Each output dict contains:
            'sublattice'           : int
            'basis_vector'         : np.ndarray of shape (3,)
            'ground_state_direction': np.ndarray of shape (3,), normalized
            'spin_length'          : float
            'gyromagnetic_matrix'  : np.ndarray of shape (3,3)

        Returns
        -------
        parsed_sites : list of dict
        """
        parsed_sites = []

        for i, site in enumerate(self.raw_spin_data):
            try:
                parsed = {}

                # sublat index is forced as integer
                parsed['sublattice'] = int(site['sublattice'])

                # basis vector
                bv = eval(str(site['basis vector']), self._namespace)
                parsed['basis_vector'] = np.array(bv, dtype=float)
                self._check_shape(parsed['basis_vector'], (3,), f"basis_vector of site {i}")

                # construct local frame from ground state direction if single 
                # vector for classical spin direction is given, otherwise take input 
                # as is if already a local frame (tripod)
                gsd_raw = eval(str(site['ground state direction']), self._namespace)
                gsd_array = np.array(gsd_raw, dtype=float)

                if gsd_array.shape == (3, 3):
                    # local frame provided directly, only validate and store
                    frame = gsd_array
                    self._check_orthonormal(frame, f"local frame of site {i}")
                    parsed['local_frame'] = frame

                elif gsd_array.shape == (3,):
                    # single direction vector provided, full frame must be constructed
                    parsed['local_frame'] = self._build_local_frame(gsd_array)

                else:
                    raise ValueError(
                        f"Parser: 'ground state direction' of site {i} has shape "
                        f"{gsd_array.shape}, expected (3,) or (3,3).")

                # spin length
                parsed['spin_length'] = float(
                    eval(str(site['spin length']), self._namespace)
                )
                if parsed['spin_length'] <= 0:
                    raise ValueError("Spin length must be positive.")

                # gyromagnetic matrix
                gm = eval(str(site['gyromagnetic matrix']), self._namespace)
                parsed['gyromagnetic_matrix'] = np.array(gm, dtype=float)
                self._check_shape(
                    parsed['gyromagnetic_matrix'], (3, 3),
                    f"gyromagnetic_matrix of site {i}"
                )

                parsed_sites.append(parsed)

            except Exception as exc:
                raise ValueError(
                    f"Parser: failed to parse spin data for site index {i}. "
                    f"Reason: {exc}"
                ) from exc

        return parsed_sites

    # ------------------------------------------------------------------
    # Interaction data parsing
    # ------------------------------------------------------------------

    def _parse_interaction_data(self) -> list[dict]:
        """
        Parse the list of interaction dictionaries. String cells are
        evaluated; sublattice indices are converted to integers.

        Each output dict contains:
            'reference_sublat'  : int
            'neighbor_sublat'   : int
            'difference_vector' : np.ndarray of shape (3,)  [lattice coordinates]
            'interaction_matrix': np.ndarray of shape (3,3)

        Returns
        -------
        parsed_interactions : list of dict
        """
        parsed_interactions = []

        for i, inter in enumerate(self.raw_inter_data):
            try:
                parsed = {}

                parsed['reference_sublat'] = int(inter['reference sublat'])
                parsed['neighbor_sublat'] = int(inter['neighbor sublat'])

                # difference vector (in lattice coords)
                dv = eval(str(inter['difference vector']), self._namespace)
                parsed['difference_vector'] = np.array(dv, dtype=float)
                self._check_shape(
                    parsed['difference_vector'], (3,),
                    f"difference_vector of interaction {i}"
                )

                # interaction matrix
                im = eval(str(inter['interaction matrix']), self._namespace)
                parsed['interaction_matrix'] = np.array(im, dtype=float)
                self._check_shape(
                    parsed['interaction_matrix'], (3, 3),
                    f"interaction_matrix of interaction {i}"
                )

                parsed_interactions.append(parsed)

            except Exception as exc:
                raise ValueError(
                    f"Parser: failed to parse interaction data for interaction index {i}. "
                    f"Reason: {exc}"
                ) from exc

        return parsed_interactions

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _check_shape(array: np.ndarray, expected_shape: tuple[int], name: str) -> None:
        """
        Raise a descriptive ValueError if array does not have the expected shape.
        """
        if array.shape != expected_shape:
            raise ValueError(
                f"Parser: '{name}' has shape {array.shape}, expected {expected_shape}."
            )


    def _build_local_frame(self, e3: np.ndarray) -> np.ndarray:
        """
        Construct a right-handed orthonormal local coordinate frame
        given e3 (the classical spin direction).

        Finds the smallest component of e3 and constructs e1 in the
        corresponding coordinate plane, orthogonal to e3, by applying 
        Gram-Schmidt orthogonalization. e2 = e3 x e1.

        Parameters
        ----------
        e3 : np.ndarray of shape (3,)
            Classical spin direction, not necessarily normalized on input.

        Returns
        -------
        frame : np.ndarray of shape (3,3)
            Rows are [e1, e2, e3], all unit vectors, right-handed.
        """
        # make sure e3 is normalized
        e3 = e3 / norm(e3)

        # pick the cardinal axis 'most orthogonal' to e3 to avoid near-zero cross products
        # -> numerical stability when normalizing
        ref = np.zeros(3)
        ref[np.argmin(np.abs(e3))] = 1.0

        # subtract the projection of ref onto e3 from itself -> removes components parallel 
        # to e3, thus the result is perpendicular to e3
        e1 = ref - np.dot(ref, e3) * e3
        e1 = e1 / norm(e1)

        e2 = np.cross(e3, e1)
        e2 = e2 / norm(e2)  # renormalize to absorb floating point error

        return np.array([e1, e2, e3])


    @staticmethod
    def _check_orthonormal(frame: np.ndarray, name: str) -> None:
        """
        Validate that a (3,3) matrix represents a right-handed
        orthonormal frame. Raises ValueError if not.
        """
        if not np.allclose(frame @ frame.T, np.eye(3), atol=1e-6):
            raise ValueError(
                f"Parser: '{name}' is not orthonormal "
                f"(frame @ frame.T deviates from identity)."
            )
        if not np.isclose(det(frame), 1.0, atol=1e-6):
            raise ValueError(
                f"Parser: '{name}' is not right-handed (det = {det(frame):.6f})."
            )
        
    # ------------------------------------------------------------------
    # Updating parameters
    # ------------------------------------------------------------------

    def update_parameters(self, new_params: dict[str, str]) -> None:
        """
        Update parameters and re-parse all data structures.

        This method updates the parameter dictionary with new values and
        re-evaluates all symbolic expressions (translation vectors, spin data,
        interaction data) to reflect the new parameters.

        Parameters
        ----------
        new_params : dict[str, str]
            Dictionary of parameters to update. Keys are parameter symbols,
            values are string expressions (e.g., {'J': '-2.0', 'Dz': '0.5'}).
            Only specified parameters are changed; others remain unchanged.

        Notes
        -----
        After calling this method, SpinSystem and Hamiltonian objects 
        need to be rebuilt to use the updated parameters:

        >>> parser.update_parameters({'J': '-2.0'})
        >>> system = SpinSystem(parser.translation_vectors, ...)
        >>> hamiltonian = Hamiltonian(system)

        Example
        -------
        >>> parser = Parser(translation_vectors, spin_data, interaction_data, parameters)
        >>> parser.parse()
        >>> # ... use parser outputs ...
        >>> parser.update_parameters({'J': '-1.5', 'A': '-0.8'})
        >>> # rebuild downstream objects with new parameters
        """
        # update raw parameters
        self.raw_params.update(new_params)

        # re-parse everything
        self.parse()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        Returns a printable representation of an object of this class.
        """
        status = "parsed" if self.translation_vectors is not None else "unparsed"
        n_sites = len(self.raw_spin_data)
        n_interactions = len(self.raw_inter_data)
        return (
            f"Parser(status = {status}, sites = {n_sites}, "
            f"interactions = {n_interactions})"
        )