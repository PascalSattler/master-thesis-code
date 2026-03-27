import pandas as pd
import os

def build_zigzag_zigzag(ncells: int) -> pd.DataFrame:
    '''
    TODO
    '''
    dir = os.path.dirname(os.path.abspath(__file__))
    read = os.path.join(dir, 'template.csv')
    template = pd.read_csv(read, dtype=str)

    df = template.copy()

    # lattice vectors
    df.loc[0, '#translation vectors'] = '[a, 0, 0]'
    df.loc[1, '#translation vectors'] = '[0.5*a, sqrt(3)/2*a, 0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 2*i, sublat_cols] = f'{1 + 2*i}', f'[0, {i}, 0]', *others
        df.loc[1 + 2*i, sublat_cols] = f'{2 + 2*i}', f'[1/3, {1/3 + i}, 0]', *others

        # interactions nearest neighbor
        # intra cell
        df.loc[3*i + 0, inter_cols] = f'{2*i + 1}', f'{2*i + 2}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

        # inter cell
        df.loc[3*i + 1, inter_cols] = f'{2*i + 1}', f'{2*i + 2}', '[-1, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

        # edge cases
        if i < ncells-1:
            # intra cell
            df.loc[3*i + 2, inter_cols] = f'{2*i + 2}', f'{2*i + 3}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

    nn_ints = df['reference sublat'].last_valid_index() + 1

    for i in range(0, ncells):
        # DM interactions next-nearest neighbor
        df.loc[6*i + 0 + nn_ints, inter_cols] = f'{2*i + 1}', f'{2*i + 1}', '[-1, 0, 0]', '[[0,Dz,0],[-Dz,0,0],[0,0,0]]'
        df.loc[6*i + 1 + nn_ints, inter_cols] = f'{2*i + 2}', f'{2*i + 2}', '[1, 0, 0]', '[[0,Dz,0],[-Dz,0,0],[0,0,0]]'

        # edge cases
        if i < ncells-1:
            # intra cell
            df.loc[6*i + 2 + nn_ints, inter_cols] = f'{2*i + 1}', f'{2*i + 3}', '[0, 0, 0]', '[[0,Dz,0],[-Dz,0,0],[0,0,0]]'
            df.loc[6*i + 3 + nn_ints, inter_cols] = f'{2*i + 4}', f'{2*i + 2}', '[0, 0, 0]', '[[0,Dz,0],[-Dz,0,0],[0,0,0]]'

            # inter cell
            df.loc[6*i + 4 + nn_ints, inter_cols] = f'{2*i + 3}', f'{2*i + 1}', '[1, 0, 0]', '[[0,Dz,0],[-Dz,0,0],[0,0,0]]'
            df.loc[6*i + 5 + nn_ints, inter_cols] = f'{2*i + 2}', f'{2*i + 4}', '[-1, 0, 0]', '[[0,Dz,0],[-Dz,0,0],[0,0,0]]'
            
    write = os.path.join(dir, 'hexagonal_zigzag_zigzag.csv')
    with open(write, 'w') as f:
        df.to_csv(write, index=False)

    print(f"Setup file saved to: {write}")

    return df



def build_armchair_armchair(ncells: int) -> pd.DataFrame:
    '''
    TODO
    '''
    dir = os.path.dirname(os.path.abspath(__file__))
    read = os.path.join(dir, 'template.csv')
    template = pd.read_csv(read, dtype=str)

    df = template.copy()

    # lattice vectors
    df.loc[0, '#translation vectors'] = '[3*a, 0, 0]'
    df.loc[1, '#translation vectors'] = '[1.5*a, sqrt(3)/2*a, 0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 2*i, sublat_cols] = f'{1 + 2*i}', f'[0, {i}, 0]', *others
        df.loc[1 + 2*i, sublat_cols] = f'{2 + 2*i}', f'[1/3, {i}, 0]', *others

        # interactions nearest neighbor
        # intra cell
        df.loc[3*i + 0, inter_cols] = f'{2*i + 1}', f'{2*i + 2}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

        # edge cases
        if i < ncells-1:
            # intra cell
            df.loc[3*i + 1, inter_cols] = f'{2*i + 2}', f'{2*i + 3}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
            df.loc[3*i + 2, inter_cols] = f'{2*i + 1}', f'{2*i + 4}', '[-1, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

    nn_ints = df['reference sublat'].last_valid_index() + 1

    for i in range(0, ncells):
        # DM interactions next-nearest neighbor -> only if at least two layers of unit cells
        if i < ncells-1:
            # inter cell
            df.loc[6*i + 0 + nn_ints, inter_cols] = f'{2*i + 4}', f'{2*i + 2}', '[1, 0, 0]', '[[0,Dz,0],[-Dz,0,0],[0,0,0]]'
            df.loc[6*i + 1 + nn_ints, inter_cols] = f'{2*i + 1}', f'{2*i + 3}', '[0, 0, 0]', '[[0,Dz,0],[-Dz,0,0],[0,0,0]]'
            df.loc[6*i + 2 + nn_ints, inter_cols] = f'{2*i + 1}', f'{2*i + 3}', '[-1, 0, 0]', '[[0,Dz,0],[-Dz,0,0],[0,0,0]]'
            df.loc[6*i + 3 + nn_ints, inter_cols] = f'{2*i + 4}', f'{2*i + 2}', '[0, 0, 0]', '[[0,Dz,0],[-Dz,0,0],[0,0,0]]'

        # only if at least 3 layers
        if i < ncells-2:
            # inter cell
            df.loc[6*i + 4 + nn_ints, inter_cols] = f'{2*i + 2}', f'{2*i + 6}', '[-1, 0, 0]', '[[0,Dz,0],[-Dz,0,0],[0,0,0]]'
            df.loc[6*i + 5 + nn_ints, inter_cols] = f'{2*i + 5}', f'{2*i + 1}', '[1, 0, 0]', '[[0,Dz,0],[-Dz,0,0],[0,0,0]]'
            
    write = os.path.join(dir, 'hexagonal_armchair_armchair.csv')
    with open(write, 'w') as f:
        df.to_csv(write, index=False)

    print(f"Setup file saved to: {write}")

    return df



# df = build_zigzag_zigzag(10)