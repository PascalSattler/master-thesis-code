import pandas as pd

class Initialize:
    '''
    Utility class for initializing the calculation. Converts data from an input.csv file into
    a format that can be interpreted by a parser before being passed to a solver.
    '''
    @classmethod
    def from_csv(cls, csv_file: str) -> tuple[list[str], dict[str, str], list[dict[str, str]]]:
        '''
        Initialize by reading an input file in .csv format.
        
        Arguments:
        ----------
        csv_file: str
            Path to input file.

        Returns:
        --------
        vects: list[str]
            Translation vector data.
        spins: list[dict[str, str]]
            Spin data.
        inters: list[dict[str, str]]
            Interaction data.
        params: dict[str, str]
            Prameter data.
        '''
        data = pd.read_csv(csv_file, dtype=str)

        vects = cls.get_translation_vectors(data)
        spins = cls.get_spin_data(data)
        inters = cls.get_interaction_data(data)
        params = cls.get_parameters(data)

        # print(f"transl_vects = {vects}", 
        #       f"spins = {spins}", 
        #       f"interactions = {inters}", 
        #       f"params = {params}", 
        #       sep='\n')

        return vects, spins, inters, params

    @classmethod
    def show_setup(cls):
        # print out the input file as a pandas dataframe
        # maybe showsetup=False as default argument of from_xyz and then if statement that prints data if True
        raise NotImplementedError('Will be added later.')

    @staticmethod
    def get_translation_vectors(data: pd.DataFrame) -> list[str]:
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
        return data['#translation vectors'].dropna().to_list()

    @staticmethod
    def get_spin_data(data: pd.DataFrame) -> list[dict[str, str]]:
        '''
        Collects data of all spins inside the unit cell of a lattice and stores it in a list. Each spin has information 
        about its sublattice index, basis vector, ground state direction and spin length, which is stored in a dictionary.
        
        Arguments:
        ----------
        data: pd.DataFrame
            pandas DataFrame object containing all data given by the setup file.

        Return:
        -------
        list[dict[str, str]]
            List of dictionaries corresponding to each spin, containing key:value pairs of their data.
        '''
        return data[['sublattice', 
                      'basis vector', 
                      'ground state direction', 
                      'spin length', 
                      'gyromagnetic matrix']].dropna().to_dict('records')

    @staticmethod
    def get_interaction_data(data: pd.DataFrame) -> list[dict[str, str]]:
        '''
        Collects data of all interactions between spins on sublattice sites and stores it in a list. Each interaction has 
        information about the a sublattice inside the reference unit cell, a neighboring sublattice inside the same or an 
        adjacent unit cell, the difference vector between their unit cells (zero vector if both inside the reference cell) 
        and the interaction matrix. The information is stored in a dictionary.
        
        Arguments:
        ----------
        data: pd.DataFrame
            pandas DataFrame object containing all data given by the setup file.

        Return:
        -------
        list[dict[str, str]]
            List of dictionaries corresponding to each interaction, containing key:value pairs of their data.
        '''
        # TODO filter for interaction types by finding their starting index (i.e. grouping them)
        return data[['reference sublat', 
                      'neighbor sublat', 
                      'difference vector', 
                      'interaction matrix']].dropna().to_dict('records')

    @staticmethod
    def get_parameters(data: pd.DataFrame) -> dict[str, str]:
        '''
        Collects data of all paramteters.
        
        Arguments:
        ----------
        data: pd.DataFrame
            pandas DataFrame object containing all data given by the setup file.

        Return:
        -------
        dict[str, str]
            Dictionary with the parameter key strings and their values as floats.
        '''
        return data[['#parameters', 'value']].dropna().set_index('#parameters')['value'].to_dict()
    

# Initialize.from_csv('archimedean/module_testing/setup.csv')