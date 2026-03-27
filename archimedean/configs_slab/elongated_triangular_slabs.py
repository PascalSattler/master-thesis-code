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
    df.loc[1, '#translation vectors'] = '[a/2, (1+sqrt(3)/2)*a, 0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 2*i, sublat_cols] = f'{1 + 2*i}', f'[0, {i}, 0]', *others
        df.loc[1 + 2*i, sublat_cols] = f'{2 + 2*i}', f'[0.26794919, {0.46410162 + i:.8f}, 0]', *others

        # interactions nearest neighbor
        # intra cell
        df.loc[5*i + 0, inter_cols] = f'{2*i + 1}', f'{2*i + 2}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

        # inter cell
        df.loc[5*i + 1, inter_cols] = f'{2*i + 1}', f'{2*i + 2}', '[-1, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
        df.loc[5*i + 2, inter_cols] = f'{2*i + 1}', f'{2*i + 1}', '[1, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[5*i + 3, inter_cols] = f'{2*i + 2}', f'{2*i + 2}', '[1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'

        # edge cases
        if i < ncells-1:
            # inter cell
            df.loc[5*i + 4, inter_cols] = f'{2*i + 3}', f'{2*i + 2}', '[-1, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
    
    write = os.path.join(dir, 'elongated_triangular_flat_flat.csv')
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
    df.loc[0, '#translation vectors'] = '[(1+sqrt(3)/2)*a, 0.5*a, 0]'
    df.loc[1, '#translation vectors'] = '[0, a, 0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 2*i, sublat_cols] = f'{1 + 2*i}', f'[0, {i}, 0]', *others
        df.loc[1 + 2*i, sublat_cols] = f'{2 + 2*i}', f'[0.46410162, {0.26794919 + i:.8f}, 0]', *others

        # interactions nearest neighbor
        # intra cell
        df.loc[5*i + 0, inter_cols] = f'{2*i + 1}', f'{2*i + 2}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

        # inter cell
        df.loc[5*i + 1, inter_cols] = f'{2*i + 2}', f'{2*i + 1}', '[1, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
        
        # edge cases
        if i < ncells-1:
            # intra cell
            df.loc[5*i + 2, inter_cols] = f'{2*i + 1}', f'{2*i + 3}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
            df.loc[5*i + 3, inter_cols] = f'{2*i + 2}', f'{2*i + 4}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
            df.loc[5*i + 4, inter_cols] = f'{2*i + 2}', f'{2*i + 3}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
            
    write = os.path.join(dir, 'elongated_triangular_zigzag_zigzag.csv')
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
    df.loc[0, '#translation vectors'] = '[(1+sqrt(3)/2)*a, -0.5*a, 0]'
    df.loc[1, '#translation vectors'] = '[0, a, 0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 2*i, sublat_cols] = f'{1 + 2*i}', f'[0, {i}, 0]', *others
        df.loc[1 + 2*i, sublat_cols] = f'{2 + 2*i}', f'[0.46410162, {0.73205081 + i:.8f}, 0]', *others

        # interactions nearest neighbor
        # intra cell
        df.loc[5*i + 0, inter_cols] = f'{2*i + 1}', f'{2*i + 2}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
        
        # edge cases
        if i < ncells-1:
            # inter cell
            df.loc[5*i + 1, inter_cols] = f'{2*i + 2}', f'{2*i + 3}', '[1, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

            # intra cell
            df.loc[5*i + 2, inter_cols] = f'{2*i + 1}', f'{2*i + 3}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
            df.loc[5*i + 3, inter_cols] = f'{2*i + 2}', f'{2*i + 4}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
            df.loc[5*i + 4, inter_cols] = f'{2*i + 2}', f'{2*i + 3}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
            
    write = os.path.join(dir, 'elongated_triangular_hook_hook.csv')
    with open(write, 'w') as f:
        df.to_csv(write, index=False)

    print(f"Setup file saved to: {write}")

    return df



# df = build_hook_hook(20)