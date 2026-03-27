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
    df.loc[0, '#translation vectors'] = '[sqrt(2 + sqrt(3))*a, 0, 0]'
    df.loc[1, '#translation vectors'] = '[0, sqrt(2 + sqrt(3))*a, 0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 4*i, sublat_cols] = f'{1 + 4*i}', f'[0, {i}, 0]', *others
        df.loc[1 + 4*i, sublat_cols] = f'{2 + 4*i}', f'[0.5, {0.1339746 + i:.8f}, 0]', *others
        df.loc[2 + 4*i, sublat_cols] = f'{3 + 4*i}', f'[0.8660254, {0.5 + i:.8f}, 0]', *others
        df.loc[3 + 4*i, sublat_cols] = f'{4 + 4*i}', f'[0.3660254, {0.6339746 + i:.8f}, 0]', *others

        # interactions
        # intra cell
        df.loc[10*i + 0, inter_cols] = f'{4*i + 1}', f'{4*i + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[10*i + 1, inter_cols] = f'{4*i + 2}', f'{4*i + 3}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
        df.loc[10*i + 2, inter_cols] = f'{4*i + 3}', f'{4*i + 4}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
        df.loc[10*i + 3, inter_cols] = f'{4*i + 4}', f'{4*i + 2}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
        

        # inter cell
        df.loc[10*i + 4, inter_cols] = f'{4*i + 1}', f'{4*i + 3}', '[-1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
        df.loc[10*i + 5, inter_cols] = f'{4*i + 4}', f'{4*i + 3}', '[-1, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[10*i + 6, inter_cols] = f'{4*i + 1}', f'{4*i + 2}', '[-1, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

        # edge cases
        if i < ncells-1:
            # intra cell
            df.loc[10*i + 7, inter_cols] = f'{4*i + 4}', f'{4*i + 5}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
            df.loc[10*i + 8, inter_cols] = f'{4*i + 4}', f'{4*i + 6}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'

            # inter cell
            df.loc[10*i + 9, inter_cols] = f'{4*i + 3}', f'{4*i + 5}', '[1, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

    write = os.path.join(dir, 'snub_square_zigzag_zigzag.csv')
    with open(write, 'w') as f:
        df.to_csv(write, index=False)

    print(f"Setup file saved to: {write}")

    return df



def build_armchair_arc(ncells: int) -> pd.DataFrame:
    '''
    TODO
    '''
    dir = os.path.dirname(os.path.abspath(__file__))
    read = os.path.join(dir, 'template.csv')
    template = pd.read_csv(read, dtype=str)

    df = template.copy()

    # lattice vectors
    df.loc[0, '#translation vectors'] = '[(1 + sqrt(3))*a, 0, 0]'
    df.loc[1, '#translation vectors'] = '[0, (1 + sqrt(3))*a, 0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 8*i, sublat_cols] = f'{1 + 8*i}', f'[0, {i}, 0]', *others
        df.loc[1 + 8*i, sublat_cols] = f'{2 + 8*i}', f'[0.3169873, {0.1830127 + i:.8f}, 0]', *others
        df.loc[2 + 8*i, sublat_cols] = f'{3 + 8*i}', f'[0.6339746, {i}, 0]', *others
        df.loc[3 + 8*i, sublat_cols] = f'{4 + 8*i}', f'[0.8169873, {0.3169873 + i:.8f}, 0]', *others
        df.loc[4 + 8*i, sublat_cols] = f'{5 + 8*i}', f'[0.1339746, {0.5 + i}, 0]', *others
        df.loc[5 + 8*i, sublat_cols] = f'{6 + 8*i}', f'[0.5, {0.5 + i}, 0]', *others
        df.loc[6 + 8*i, sublat_cols] = f'{7 + 8*i}', f'[0.8169873, {0.6830127 + i:.8f}, 0]', *others
        df.loc[7 + 8*i, sublat_cols] = f'{8 + 8*i}', f'[0.3169873, {0.8169873 + i:.8f}, 0]', *others

        # interactions
        # intra cell
        df.loc[20*i + 0, inter_cols] = f'{8*i + 1}', f'{8*i + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 1, inter_cols] = f'{8*i + 2}', f'{8*i + 3}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 2, inter_cols] = f'{8*i + 3}', f'{8*i + 4}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 3, inter_cols] = f'{8*i + 2}', f'{8*i + 5}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 4, inter_cols] = f'{8*i + 2}', f'{8*i + 6}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
        df.loc[20*i + 5, inter_cols] = f'{8*i + 4}', f'{8*i + 6}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 6, inter_cols] = f'{8*i + 4}', f'{8*i + 7}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
        df.loc[20*i + 7, inter_cols] = f'{8*i + 5}', f'{8*i + 8}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 8, inter_cols] = f'{8*i + 6}', f'{8*i + 8}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
        df.loc[20*i + 9, inter_cols] = f'{8*i + 5}', f'{8*i + 6}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
        df.loc[20*i + 10, inter_cols] = f'{8*i + 6}', f'{8*i + 7}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        
        # inter cell
        df.loc[20*i + 11, inter_cols] = f'{8*i + 1}', f'{8*i + 3}', '[-1, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
        df.loc[20*i + 12, inter_cols] = f'{8*i + 1}', f'{8*i + 4}', '[-1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
        df.loc[20*i + 13, inter_cols] = f'{8*i + 5}', f'{8*i + 4}', '[-1, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 14, inter_cols] = f'{8*i + 5}', f'{8*i + 7}', '[-1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'

        # edge cases
        if i < ncells-1:
            # intra cell
            df.loc[20*i + 15, inter_cols] = f'{8*i + 8}', f'{8*i + 9}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
            df.loc[20*i + 16, inter_cols] = f'{8*i + 8}', f'{8*i + 10}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
            df.loc[20*i + 17, inter_cols] = f'{8*i + 8}', f'{8*i + 11}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
            df.loc[20*i + 18, inter_cols] = f'{8*i + 7}', f'{8*i + 11}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

            # inter cell
            df.loc[20*i + 19, inter_cols] = f'{8*i + 7}', f'{8*i + 9}', '[1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'

    write = os.path.join(dir, 'snub_square_armchair_arc.csv')
    with open(write, 'w') as f:
        df.to_csv(write, index=False)

    print(f"Setup file saved to: {write}")

    return df



def build_arc_arc(ncells: int) -> pd.DataFrame:
    '''
    TODO
    '''
    dir = os.path.dirname(os.path.abspath(__file__))
    read = os.path.join(dir, 'template.csv')
    template = pd.read_csv(read, dtype=str)

    df = template.copy()

    # lattice vectors
    df.loc[0, '#translation vectors'] = '[(1 + sqrt(3))*a, 0, 0]'
    df.loc[1, '#translation vectors'] = '[0, (1 + sqrt(3))*a, 0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 8*i + 2, sublat_cols] = f'{1 + 8*i + 2}', f'[0, {i}, 0]', *others
        df.loc[1 + 8*i + 2, sublat_cols] = f'{2 + 8*i + 2}', f'[0.3169873, {0.1830127 + i:.8f}, 0]', *others
        df.loc[2 + 8*i + 2, sublat_cols] = f'{3 + 8*i + 2}', f'[0.6339746, {i}, 0]', *others
        df.loc[3 + 8*i + 2, sublat_cols] = f'{4 + 8*i + 2}', f'[0.8169873, {0.3169873 + i:.8f}, 0]', *others
        df.loc[4 + 8*i + 2, sublat_cols] = f'{5 + 8*i + 2}', f'[0.1339746, {0.5 + i}, 0]', *others
        df.loc[5 + 8*i + 2, sublat_cols] = f'{6 + 8*i + 2}', f'[0.5, {0.5 + i}, 0]', *others
        df.loc[6 + 8*i + 2, sublat_cols] = f'{7 + 8*i + 2}', f'[0.8169873, {0.6830127 + i:.8f}, 0]', *others
        df.loc[7 + 8*i + 2, sublat_cols] = f'{8 + 8*i + 2}', f'[0.3169873, {0.8169873 + i:.8f}, 0]', *others

        # interactions
        # intra cell
        df.loc[20*i + 0 + 5, inter_cols] = f'{8*i + 1 + 2}', f'{8*i + 2 + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 1 + 5, inter_cols] = f'{8*i + 2 + 2}', f'{8*i + 3 + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 2 + 5, inter_cols] = f'{8*i + 3 + 2}', f'{8*i + 4 + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 3 + 5, inter_cols] = f'{8*i + 2 + 2}', f'{8*i + 5 + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 4 + 5, inter_cols] = f'{8*i + 2 + 2}', f'{8*i + 6 + 2}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
        df.loc[20*i + 5 + 5, inter_cols] = f'{8*i + 4 + 2}', f'{8*i + 6 + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 6 + 5, inter_cols] = f'{8*i + 4 + 2}', f'{8*i + 7 + 2}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
        df.loc[20*i + 7 + 5, inter_cols] = f'{8*i + 5 + 2}', f'{8*i + 8 + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 8 + 5, inter_cols] = f'{8*i + 6 + 2}', f'{8*i + 8 + 2}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
        df.loc[20*i + 9 + 5, inter_cols] = f'{8*i + 5 + 2}', f'{8*i + 6 + 2}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
        df.loc[20*i + 10 + 5, inter_cols] = f'{8*i + 6 + 2}', f'{8*i + 7 + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        
        # inter cell
        df.loc[20*i + 11 + 5, inter_cols] = f'{8*i + 1 + 2}', f'{8*i + 3 + 2}', '[-1, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
        df.loc[20*i + 12 + 5, inter_cols] = f'{8*i + 1 + 2}', f'{8*i + 4 + 2}', '[-1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
        df.loc[20*i + 13 + 5, inter_cols] = f'{8*i + 5 + 2}', f'{8*i + 4 + 2}', '[-1, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 14 + 5, inter_cols] = f'{8*i + 5 + 2}', f'{8*i + 7 + 2}', '[-1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'

        # edge cases
        if i < ncells-1:
            # intra cell
            df.loc[20*i + 15 + 5, inter_cols] = f'{8*i + 8 + 2}', f'{8*i + 9 + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
            df.loc[20*i + 16 + 5, inter_cols] = f'{8*i + 8 + 2}', f'{8*i + 10 + 2}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
            df.loc[20*i + 17 + 5, inter_cols] = f'{8*i + 8 + 2}', f'{8*i + 11 + 2}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
            df.loc[20*i + 18 + 5, inter_cols] = f'{8*i + 7 + 2}', f'{8*i + 11 + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

            # inter cell
            df.loc[20*i + 19 + 5, inter_cols] = f'{8*i + 7 + 2}', f'{8*i + 9 + 2}', '[1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'

    # additional atoms to create arc pattern on bottom edge -> shifted all other sublat ids by 2
    df.loc[0, sublat_cols] = f'{1}', f'[-0.1830127, -0.3169873, 0]', *others
    df.loc[1, sublat_cols] = f'{2}', f'[0.3169873, -0.1830127, 0]', *others
            
    # additional interactions
    # intra
    df.loc[0, inter_cols] = f'{1}', f'{3}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
    df.loc[1, inter_cols] = f'{2}', f'{3}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
    df.loc[2, inter_cols] = f'{2}', f'{4}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
    df.loc[3, inter_cols] = f'{2}', f'{5}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'

    # inter
    df.loc[4, inter_cols] = f'{1}', f'{5}', '[-1, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

    write = os.path.join(dir, 'snub_square_arc_arc.csv')
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
    df.loc[0, '#translation vectors'] = '[(1 + sqrt(3))*a, 0, 0]'
    df.loc[1, '#translation vectors'] = '[0, (1 + sqrt(3))*a, 0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 8*i, sublat_cols] = f'{1 + 8*i}', f'[0, {i}, 0]', *others
        df.loc[1 + 8*i, sublat_cols] = f'{2 + 8*i}', f'[0.3169873, {0.1830127 + i:.8f}, 0]', *others
        df.loc[2 + 8*i, sublat_cols] = f'{3 + 8*i}', f'[0.6339746, {i}, 0]', *others
        df.loc[3 + 8*i, sublat_cols] = f'{4 + 8*i}', f'[0.8169873, {0.3169873 + i:.8f}, 0]', *others
        df.loc[4 + 8*i, sublat_cols] = f'{5 + 8*i}', f'[0.1339746, {0.5 + i}, 0]', *others
        df.loc[5 + 8*i, sublat_cols] = f'{6 + 8*i}', f'[0.5, {0.5 + i}, 0]', *others
        df.loc[6 + 8*i, sublat_cols] = f'{7 + 8*i}', f'[0.8169873, {0.6830127 + i:.8f}, 0]', *others
        df.loc[7 + 8*i, sublat_cols] = f'{8 + 8*i}', f'[0.3169873, {0.8169873 + i:.8f}, 0]', *others

        # interactions
        # intra cell
        df.loc[20*i + 0, inter_cols] = f'{8*i + 1}', f'{8*i + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 1, inter_cols] = f'{8*i + 2}', f'{8*i + 3}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 2, inter_cols] = f'{8*i + 3}', f'{8*i + 4}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 3, inter_cols] = f'{8*i + 2}', f'{8*i + 5}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 4, inter_cols] = f'{8*i + 2}', f'{8*i + 6}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
        df.loc[20*i + 5, inter_cols] = f'{8*i + 4}', f'{8*i + 6}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 6, inter_cols] = f'{8*i + 4}', f'{8*i + 7}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
        df.loc[20*i + 7, inter_cols] = f'{8*i + 5}', f'{8*i + 8}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 8, inter_cols] = f'{8*i + 6}', f'{8*i + 8}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
        df.loc[20*i + 9, inter_cols] = f'{8*i + 5}', f'{8*i + 6}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
        df.loc[20*i + 10, inter_cols] = f'{8*i + 6}', f'{8*i + 7}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        
        # inter cell
        df.loc[20*i + 11, inter_cols] = f'{8*i + 1}', f'{8*i + 3}', '[-1, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
        df.loc[20*i + 12, inter_cols] = f'{8*i + 1}', f'{8*i + 4}', '[-1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
        df.loc[20*i + 13, inter_cols] = f'{8*i + 5}', f'{8*i + 4}', '[-1, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 14, inter_cols] = f'{8*i + 5}', f'{8*i + 7}', '[-1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'

        # extra atom 1
        df.loc[20*i + 15, inter_cols] = f'{8*i + 8}', f'{8*i + 9}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 16, inter_cols] = f'{8*i + 7}', f'{8*i + 9}', '[1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'

        # extra atom 2
        df.loc[20*i + 17, inter_cols] = f'{8*i + 8}', f'{8*i + 10}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

        # extra atom 3
        df.loc[20*i + 18, inter_cols] = f'{8*i + 8}', f'{8*i + 11}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
        df.loc[20*i + 19, inter_cols] = f'{8*i + 7}', f'{8*i + 11}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'     

    # additional atoms to create armchair pattern on top edge
    df.loc[8*ncells + 0, sublat_cols] = f'{1 + 8*ncells}', f'[0, {ncells}, 0]', *others
    df.loc[8*ncells + 1, sublat_cols] = f'{2 + 8*ncells}', f'[0.3169873, {0.1830127 + ncells:.8f}, 0]', *others
    df.loc[8*ncells + 2, sublat_cols] = f'{3 + 8*ncells}', f'[0.6339746, {ncells}, 0]', *others
            
    # additional interactions
    df.loc[20*ncells + 1, inter_cols] = f'{8*ncells + 1}', f'{8*ncells + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
    df.loc[20*ncells + 2, inter_cols] = f'{8*ncells + 2}', f'{8*ncells + 3}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
    df.loc[20*ncells + 3, inter_cols] = f'{8*ncells + 3}', f'{8*ncells + 1}', '[1, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

    write = os.path.join(dir, 'snub_square_armchair_zigzag.csv')
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
    df.loc[0, '#translation vectors'] = '[(1 + sqrt(3))*a, 0, 0]'
    df.loc[1, '#translation vectors'] = '[0, (1 + sqrt(3))*a, 0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 8*i, sublat_cols] = f'{1 + 8*i}', f'[0, {i}, 0]', *others
        df.loc[1 + 8*i, sublat_cols] = f'{2 + 8*i}', f'[0.3169873, {0.1830127 + i:.8f}, 0]', *others
        df.loc[2 + 8*i, sublat_cols] = f'{3 + 8*i}', f'[0.6339746, {i}, 0]', *others
        df.loc[3 + 8*i, sublat_cols] = f'{4 + 8*i}', f'[0.8169873, {0.3169873 + i:.8f}, 0]', *others
        df.loc[4 + 8*i, sublat_cols] = f'{5 + 8*i}', f'[0.1339746, {0.5 + i}, 0]', *others
        df.loc[5 + 8*i, sublat_cols] = f'{6 + 8*i}', f'[0.5, {0.5 + i}, 0]', *others
        df.loc[6 + 8*i, sublat_cols] = f'{7 + 8*i}', f'[0.8169873, {0.6830127 + i:.8f}, 0]', *others
        df.loc[7 + 8*i, sublat_cols] = f'{8 + 8*i}', f'[0.3169873, {0.8169873 + i:.8f}, 0]', *others

        # interactions
        # intra cell
        df.loc[20*i + 0, inter_cols] = f'{8*i + 1}', f'{8*i + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 1, inter_cols] = f'{8*i + 2}', f'{8*i + 3}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 2, inter_cols] = f'{8*i + 3}', f'{8*i + 4}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 3, inter_cols] = f'{8*i + 2}', f'{8*i + 5}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 4, inter_cols] = f'{8*i + 2}', f'{8*i + 6}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
        df.loc[20*i + 5, inter_cols] = f'{8*i + 4}', f'{8*i + 6}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 6, inter_cols] = f'{8*i + 4}', f'{8*i + 7}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
        df.loc[20*i + 7, inter_cols] = f'{8*i + 5}', f'{8*i + 8}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 8, inter_cols] = f'{8*i + 6}', f'{8*i + 8}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
        df.loc[20*i + 9, inter_cols] = f'{8*i + 5}', f'{8*i + 6}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
        df.loc[20*i + 10, inter_cols] = f'{8*i + 6}', f'{8*i + 7}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        
        # inter cell
        df.loc[20*i + 11, inter_cols] = f'{8*i + 1}', f'{8*i + 3}', '[-1, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
        df.loc[20*i + 12, inter_cols] = f'{8*i + 1}', f'{8*i + 4}', '[-1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
        df.loc[20*i + 13, inter_cols] = f'{8*i + 5}', f'{8*i + 4}', '[-1, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[20*i + 14, inter_cols] = f'{8*i + 5}', f'{8*i + 7}', '[-1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'

        # edge cases
        if i < ncells-1:
            # intra cell
            df.loc[20*i + 15, inter_cols] = f'{8*i + 8}', f'{8*i + 9}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
            df.loc[20*i + 16, inter_cols] = f'{8*i + 8}', f'{8*i + 10}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
            df.loc[20*i + 17, inter_cols] = f'{8*i + 8}', f'{8*i + 11}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
            df.loc[20*i + 18, inter_cols] = f'{8*i + 7}', f'{8*i + 11}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

            # inter cell
            df.loc[20*i + 19, inter_cols] = f'{8*i + 7}', f'{8*i + 9}', '[1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'         

   # additional atoms to create armchair pattern on top edge
    df.loc[8*ncells + 0, sublat_cols] = f'{1 + 8*ncells}', f'[0, {ncells}, 0]', *others
    df.loc[8*ncells + 1, sublat_cols] = f'{2 + 8*ncells}', f'[0.6339746, {ncells}, 0]', *others
            
    # additional interactions
    df.loc[20*ncells + 1, inter_cols] = f'{8*ncells + 2}', f'{8*ncells + 1}', '[1, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
    df.loc[20*ncells + 2, inter_cols] = f'{8*ncells + 0}', f'{8*ncells + 1}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
    df.loc[20*ncells + 3, inter_cols] = f'{8*ncells + 0}', f'{8*ncells + 2}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
    df.loc[20*ncells + 4, inter_cols] = f'{8*ncells - 1}', f'{8*ncells + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
    df.loc[20*ncells + 5, inter_cols] = f'{8*ncells - 1}', f'{8*ncells + 1}', '[1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'

    write = os.path.join(dir, 'snub_square_armchair_armchair.csv')
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
    df.loc[0, '#translation vectors'] = '[sqrt(2 + sqrt(3))*a, 0, 0]'
    df.loc[1, '#translation vectors'] = '[0, sqrt(2 + sqrt(3))*a, 0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 4*i, sublat_cols] = f'{1 + 4*i}', f'[0.5, {-0.8660254 + i:.8f}, 0]', *others
        df.loc[1 + 4*i, sublat_cols] = f'{2 + 4*i}', f'[0.1339746, {-0.5 + i:.1f}, 0]', *others
        df.loc[2 + 4*i, sublat_cols] = f'{3 + 4*i}', f'[0.6339746, {-0.3660254 + i:.8f}, 0]', *others
        df.loc[3 + 4*i, sublat_cols] = f'{4 + 4*i}', f'[0, {i}, 0]', *others

        # interactions
        # intra cell
        df.loc[10*i + 0, inter_cols] = f'{4*i + 1}', f'{4*i + 2}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
        df.loc[10*i + 1, inter_cols] = f'{4*i + 2}', f'{4*i + 3}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[10*i + 2, inter_cols] = f'{4*i + 1}', f'{4*i + 3}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
        df.loc[10*i + 3, inter_cols] = f'{4*i + 2}', f'{4*i + 4}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
        

        # inter cell
        df.loc[10*i + 4, inter_cols] = f'{4*i + 2}', f'{4*i + 3}', '[-1, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[10*i + 5, inter_cols] = f'{4*i + 4}', f'{4*i + 3}', '[-1, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

        # edge cases
        if i < ncells-1:
            # intra cell
            df.loc[10*i + 6, inter_cols] = f'{4*i + 4}', f'{4*i + 6}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
            df.loc[10*i + 7, inter_cols] = f'{4*i + 4}', f'{4*i + 5}', '[0, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'
            df.loc[10*i + 8, inter_cols] = f'{4*i + 3}', f'{4*i + 5}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

            # inter cell
            df.loc[10*i + 9, inter_cols] = f'{4*i + 4}', f'{4*i + 5}', '[-1, 0, 0]', '[[J,-Dz,0],[Dz,J,0],[0,0,J]]'

    write = os.path.join(dir, 'snub_square_hook_hook.csv')
    with open(write, 'w') as f:
        df.to_csv(write, index=False)

    print(f"Setup file saved to: {write}")

    return df



# df = build_hook_hook(10)