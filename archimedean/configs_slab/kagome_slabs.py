import pandas as pd
import os

def build_flat_hook(ncells: int) -> pd.DataFrame:
    '''
    TODO
    '''
    dir = os.path.dirname(os.path.abspath(__file__))
    read = os.path.join(dir, 'template.csv')
    template = pd.read_csv(read, dtype=str)

    df = template.copy()

    # lattice vectors
    df.loc[0, '#translation vectors'] = '[2*a, 0, 0]'
    df.loc[1, '#translation vectors'] = '[a, sqrt(3)*a, 0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 3*i, sublat_cols] = f'{1 + 3*i}', f'[0, {i}, 0]', *others
        df.loc[1 + 3*i, sublat_cols] = f'{2 + 3*i}', f'[0.5, {i}, 0]', *others
        df.loc[2 + 3*i, sublat_cols] = f'{3 + 3*i}', f'[0, {0.5 + i}, 0]', *others

        # interactions
        # intra cell
        df.loc[6*i + 0, inter_cols] = f'{3*i + 1}', f'{3*i + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[6*i + 1, inter_cols] = f'{3*i + 2}', f'{3*i + 3}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[6*i + 2, inter_cols] = f'{3*i + 3}', f'{3*i + 1}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

        # inter cell
        df.loc[6*i + 3, inter_cols] = f'{3*i + 1}', f'{3*i + 2}', '[-1, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

        # edge cases
        if i < ncells-1:
            # intra cell
            df.loc[6*i + 4, inter_cols] = f'{3*i+3}', f'{3*i+4}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
            # inter cell
            df.loc[6*i + 5, inter_cols] = f'{3*i+3}', f'{3*i+5}', '[-1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'

    write = os.path.join(dir, 'kagome_flat_hook.csv')
    with open(write, 'w') as f:
        df.to_csv(write, index=False)

    print(f"Setup file saved to: {write}")

    return df



def build_flat_flat(ncells: int) -> pd.DataFrame:
    '''
    TODO
    '''
    dir = os.path.dirname(os.path.abspath(__file__))
    read = os.path.join(dir, 'template.csv')
    template = pd.read_csv(read, dtype=str)

    df = template.copy()

    # lattice vectors
    df.loc[0, '#translation vectors'] = '[2*a, 0, 0]'
    df.loc[1, '#translation vectors'] = '[a, sqrt(3)*a, 0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors, ground state, spin length, gyromat
        df.loc[0 + 3*i, sublat_cols] = f'{1 + 3*i}', f'[0, {i}, 0]', *others
        df.loc[1 + 3*i, sublat_cols] = f'{2 + 3*i}', f'[0.5, {i}, 0]', *others
        df.loc[2 + 3*i, sublat_cols] = f'{3 + 3*i}', f'[0, {0.5 + i}, 0]', *others

        # interactions
        # intra cell
        df.loc[6*i + 0, inter_cols] = f'{3*i + 1}', f'{3*i + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[6*i + 1, inter_cols] = f'{3*i + 2}', f'{3*i + 3}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[6*i + 2, inter_cols] = f'{3*i + 3}', f'{3*i + 1}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

        # inter cell
        df.loc[6*i + 3, inter_cols] = f'{3*i + 1}', f'{3*i + 2}', '[-1, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

        # extra atom 1
        df.loc[6*i + 4, inter_cols] = f'{3*i+3}', f'{3*i+4}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]' 

        # extra atom 2
        df.loc[6*i + 5, inter_cols] = f'{3*i+3}', f'{3*i+5}', '[-1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'

    # additional atoms to flatten top edge
    df.loc[3*ncells + 0, sublat_cols] = f'{1 + 3*ncells}', f'[0, {ncells}, 0]', *others
    df.loc[3*ncells + 1, sublat_cols] = f'{2 + 3*ncells}', f'[0.5, {ncells}, 0]', *others
            
    # additional interactions
    df.loc[6*ncells + 1, inter_cols] = f'{3*ncells + 1}', f'{3*ncells + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
    df.loc[6*ncells + 2, inter_cols] = f'{3*ncells + 1}', f'{3*ncells + 2}', '[-1, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

    write = os.path.join(dir, 'kagome_flat_flat.csv')
    with open(write, 'w') as f:
        df.to_csv(write, index=False)

    print(f"Setup file saved to: {write}")

    return df



def build_hook_hook(ncells: int) -> pd.DataFrame:
    '''
    TODO
    '''
    dir = os.path.dirname(os.path.abspath(__file__))
    read = os.path.join(dir, 'template.csv')
    template = pd.read_csv(read, dtype=str)

    df = template.copy()

    # lattice vectors
    df.loc[0, '#translation vectors'] = '[2*a, 0, 0]'
    df.loc[1, '#translation vectors'] = '[a, sqrt(3)*a, 0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 3*i + 1, sublat_cols] = f'{1 + 3*i + 1}', f'[0, {i}, 0]', *others
        df.loc[1 + 3*i + 1, sublat_cols] = f'{2 + 3*i + 1}', f'[0.5, {i}, 0]', *others
        df.loc[2 + 3*i + 1, sublat_cols] = f'{3 + 3*i + 1}', f'[0, {0.5 + i}, 0]', *others

        # interactions
        # intra cell
        df.loc[6*i + 0 + 2, inter_cols] = f'{3*i + 1 + 1}', f'{3*i + 2 + 1}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[6*i + 1 + 2, inter_cols] = f'{3*i + 2 + 1}', f'{3*i + 3 + 1}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[6*i + 2 + 2, inter_cols] = f'{3*i + 3 + 1}', f'{3*i + 1 + 1}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

        # inter cell
        df.loc[6*i + 3 + 2, inter_cols] = f'{3*i + 1 + 1}', f'{3*i + 2 + 1}', '[-1, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

        # edge cases
        if i < ncells-1:
            # intra cell
            df.loc[6*i + 4 + 2, inter_cols] = f'{3*i + 3 + 1}', f'{3*i + 4 + 1}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
            # inter cell
            df.loc[6*i + 5 + 2, inter_cols] = f'{3*i + 3 + 1}', f'{3*i + 5 + 1}', '[-1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'

    # additional atom to create hook on bottom edge -> shifted all other sublat ids by 1
    df.loc[0, sublat_cols] = f'{1}', f'[0, -0.5, 0]', *others
            
    # additional interactions
    df.loc[0, inter_cols] = f'{1}', f'{2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
    df.loc[1, inter_cols] = f'{1}', f'{3}', '[-1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'

    write = os.path.join(dir, 'kagome_hook_hook.csv')
    with open(write, 'w') as f:
        df.to_csv(write, index=False)

    print(f"Setup file saved to: {write}")

    return df



def build_armchair_zigzag(ncells: int) -> pd.DataFrame:
    '''
    TODO
    '''
    dir = os.path.dirname(os.path.abspath(__file__))
    read = os.path.join(dir, 'template.csv')
    template = pd.read_csv(read, dtype=str)

    df = template.copy()

    # lattice vectors
    df.loc[0, '#translation vectors'] = '[2*sqrt(3)*a,0,0]'
    df.loc[1, '#translation vectors'] = '[0,2*a,0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 6*i, sublat_cols] = f'{1 + 6*i}', f'[0, {i}, 0]', *others
        df.loc[1 + 6*i, sublat_cols] = f'{2 + 6*i}', f'[0, {0.5 + i}, 0]', *others
        df.loc[2 + 6*i, sublat_cols] = f'{3 + 6*i}', f'[0.25, {0.25 + i}, 0]', *others
        df.loc[3 + 6*i, sublat_cols] = f'{4 + 6*i}', f'[0.5, {i}, 0]', *others
        df.loc[4 + 6*i, sublat_cols] = f'{5 + 6*i}', f'[0.5, {0.5 + i}, 0]', *others
        df.loc[5 + 6*i, sublat_cols] = f'{6 + 6*i}', f'[0.75, {0.75 + i}, 0]', *others

        # interactions
        # intra cell
        df.loc[12*i + 0, inter_cols] = f'{6*i + 2}', f'{6*i + 1}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 1, inter_cols] = f'{6*i + 1}', f'{6*i + 3}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 2, inter_cols] = f'{6*i + 3}', f'{6*i + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 3, inter_cols] = f'{6*i + 4}', f'{6*i + 5}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 4, inter_cols] = f'{6*i + 5}', f'{6*i + 3}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 5, inter_cols] = f'{6*i + 3}', f'{6*i + 4}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 6, inter_cols] = f'{6*i + 5}', f'{6*i + 6}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

        # inter cell
        df.loc[12*i + 7, inter_cols] = f'{6*i + 2}', f'{6*i + 6}', '[-1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
        

        # edge cases
        if i < ncells-1:
            # intra cell
            df.loc[12*i + 8, inter_cols] = f'{6*i + 2}', f'{6*i + 7}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
            df.loc[12*i + 9, inter_cols] = f'{6*i + 5}', f'{6*i + 10}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
            df.loc[12*i + 10, inter_cols] = f'{6*i + 6}', f'{6*i + 10}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

            # inter cell
            df.loc[12*i + 11, inter_cols] = f'{6*i + 6}', f'{6*i + 7}', '[1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'

    write = os.path.join(dir, 'kagome_armchair_zigzag.csv')
    with open(write, 'w') as f:
        df.to_csv(write, index=False)

    print(f"Setup file saved to: {write}")

    return df



def build_zigzag_zigzag(ncells: int) -> pd.DataFrame:
    '''
    TODO
    '''
    dir = os.path.dirname(os.path.abspath(__file__))
    read = os.path.join(dir, 'template.csv')
    template = pd.read_csv(read, dtype=str)

    df = template.copy()

    # lattice vectors
    df.loc[0, '#translation vectors'] = '[2*sqrt(3)*a,0,0]'
    df.loc[1, '#translation vectors'] = '[0,2*a,0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 6*i + 1, sublat_cols] = f'{1 + 6*i + 1}', f'[0, {i}, 0]', *others
        df.loc[1 + 6*i + 1, sublat_cols] = f'{2 + 6*i + 1}', f'[0, {0.5 + i}, 0]', *others
        df.loc[2 + 6*i + 1, sublat_cols] = f'{3 + 6*i + 1}', f'[0.25, {0.25 + i}, 0]', *others
        df.loc[3 + 6*i + 1, sublat_cols] = f'{4 + 6*i + 1}', f'[0.5, {i}, 0]', *others
        df.loc[4 + 6*i + 1, sublat_cols] = f'{5 + 6*i + 1}', f'[0.5, {0.5 + i}, 0]', *others
        df.loc[5 + 6*i + 1, sublat_cols] = f'{6 + 6*i + 1}', f'[0.75, {0.75 + i}, 0]', *others

        # interactions
        # intra cell
        df.loc[12*i + 0 + 2, inter_cols] = f'{6*i + 2 + 1}', f'{6*i + 1 + 1}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 1 + 2, inter_cols] = f'{6*i + 1 + 1}', f'{6*i + 3 + 1}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 2 + 2, inter_cols] = f'{6*i + 3 + 1}', f'{6*i + 2 + 1}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 3 + 2, inter_cols] = f'{6*i + 4 + 1}', f'{6*i + 5 + 1}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 4 + 2, inter_cols] = f'{6*i + 5 + 1}', f'{6*i + 3 + 1}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 5 + 2, inter_cols] = f'{6*i + 3 + 1}', f'{6*i + 4 + 1}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 6 + 2, inter_cols] = f'{6*i + 5 + 1}', f'{6*i + 6 + 1}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

        # inter cell
        df.loc[12*i + 7 + 2, inter_cols] = f'{6*i + 2 + 1}', f'{6*i + 6 + 1}', '[-1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
        
        # edge cases
        if i < ncells-1:
            # intra cell
            df.loc[12*i + 8 + 2, inter_cols] = f'{6*i + 2 + 1}', f'{6*i + 7 + 1}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
            df.loc[12*i + 9 + 2, inter_cols] = f'{6*i + 5 + 1}', f'{6*i + 10 + 1}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
            df.loc[12*i + 10 + 2, inter_cols] = f'{6*i + 6 + 1}', f'{6*i + 10 + 1}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

            # inter cell
            df.loc[12*i + 11 + 2, inter_cols] = f'{6*i + 6 + 1}', f'{6*i + 7 + 1}', '[1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'

    # additional atom to create continuous zigzag pattern on bottom edge
    df.loc[0, sublat_cols] = f'{1}', f'[-0.25, -0.25, 0]', *others
            
    # additional interactions
    df.loc[0, inter_cols] = f'{1}', f'{2}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
    df.loc[1, inter_cols] = f'{1}', f'{5}', '[-1, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

    write = os.path.join(dir, 'kagome_zigzag_zigzag.csv')
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
    df.loc[0, '#translation vectors'] = '[2*sqrt(3)*a,0,0]'
    df.loc[1, '#translation vectors'] = '[0,2*a,0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 6*i, sublat_cols] = f'{1 + 6*i}', f'[0, {i}, 0]', *others
        df.loc[1 + 6*i, sublat_cols] = f'{2 + 6*i}', f'[0, {0.5 + i}, 0]', *others
        df.loc[2 + 6*i, sublat_cols] = f'{3 + 6*i}', f'[0.25, {0.25 + i}, 0]', *others
        df.loc[3 + 6*i, sublat_cols] = f'{4 + 6*i}', f'[0.5, {i}, 0]', *others
        df.loc[4 + 6*i, sublat_cols] = f'{5 + 6*i}', f'[0.5, {0.5 + i}, 0]', *others
        df.loc[5 + 6*i, sublat_cols] = f'{6 + 6*i}', f'[0.75, {0.75 + i}, 0]', *others

        # interactions
        # intra cell
        df.loc[12*i + 0, inter_cols] = f'{6*i + 2}', f'{6*i + 1}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 1, inter_cols] = f'{6*i + 1}', f'{6*i + 3}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 2, inter_cols] = f'{6*i + 3}', f'{6*i + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 3, inter_cols] = f'{6*i + 4}', f'{6*i + 5}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 4, inter_cols] = f'{6*i + 5}', f'{6*i + 3}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 5, inter_cols] = f'{6*i + 3}', f'{6*i + 4}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 6, inter_cols] = f'{6*i + 5}', f'{6*i + 6}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

        # inter cell
        df.loc[12*i + 7, inter_cols] = f'{6*i + 2}', f'{6*i + 6}', '[-1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
        

        # edge cases
        if i < ncells-1:
            # intra cell
            df.loc[12*i + 8, inter_cols] = f'{6*i + 2}', f'{6*i + 7}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
            df.loc[12*i + 9, inter_cols] = f'{6*i + 5}', f'{6*i + 10}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
            df.loc[12*i + 10, inter_cols] = f'{6*i + 6}', f'{6*i + 10}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

            # inter cell
            df.loc[12*i + 11, inter_cols] = f'{6*i + 6}', f'{6*i + 7}', '[1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'

    # additional atoms to create continuous armchair pattern on bottom edge
    df.loc[6*ncells + 0, sublat_cols] = f'{6*ncells + 1}', f'[0.5, {ncells}, 0]', *others
    df.loc[6*ncells + 1, sublat_cols] = f'{6*ncells + 2}', f'[0, {ncells}, 0]', *others
            
    # additional interactions
    df.loc[12*ncells + 0, inter_cols] = f'{6*ncells - 1}', f'{6*ncells + 1}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
    df.loc[12*ncells + 1, inter_cols] = f'{6*ncells - 4}', f'{6*ncells + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
    df.loc[12*ncells + 2, inter_cols] = f'{6*ncells + 0}', f'{6*ncells + 1}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
    df.loc[12*ncells + 3, inter_cols] = f'{6*ncells + 2}', f'{6*ncells + 0}', '[-1, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

    write = os.path.join(dir, 'kagome_armchair_armchair.csv')
    with open(write, 'w') as f:
        df.to_csv(write, index=False)

    print(f"Setup file saved to: {write}")

    return df



# df = build_flat_hook(15)

# super cell construction: ncells * latvects and then basis vectors in terms of unit cell vects?