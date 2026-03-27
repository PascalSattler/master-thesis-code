import pandas as pd
import os

def build_flat_flat(ncells: int) -> pd.DataFrame:
    '''
    TODO
    '''
    dir = os.path.dirname(os.path.abspath(__file__))
    read = os.path.join(dir, 'template.csv')
    template = pd.read_csv(read, dtype=str)

    df = template.copy()

    # lattice vectors
    df.loc[0, '#translation vectors'] = '[a, 0, 0]'
    df.loc[1, '#translation vectors'] = '[0, a, 0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[i, sublat_cols] = f'{i+1}', f'[0, {i}, 0]', *others
        
        # interactions
        # inter cell
        df.loc[2*i + 0, inter_cols] = f'{i + 1}', f'{i + 1}', '[1, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'


        # edge cases
        if i < ncells-1:
            # inter cell
            df.loc[2*i + 1, inter_cols] = f'{i + 1}', f'{i + 2}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

    write = os.path.join(dir, 'square_flat_flat.csv')
    with open(write, 'w') as f:
        df.to_csv(write, index=False)

    print(f"Setup file saved to: {write}")

    return df

# df = build_flat_flat(10)