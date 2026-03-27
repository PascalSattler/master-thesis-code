import pandas as pd
import os

def build_zigzag_zigzag(ncells: int) -> pd.DataFrame:
    '''
    TODO
    '''
    dir = os.path.dirname(os.path.abspath(__file__))
    read = os.path.join(dir, 'template2.csv')
    template = pd.read_csv(read, dtype=str)

    df = template.copy()

    # lattice vectors
    df.loc[0, '#translation vectors'] = '[sqrt(7)*a, 0, 0]'
    df.loc[1, '#translation vectors'] = '[sqrt(7)/2*a, sqrt(21)/2*a, 0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 6*i, sublat_cols] = f'{1 + 6*i}', f'[0.28571429, {0.14285714 + i:.8f}, 0]', *others
        df.loc[1 + 6*i, sublat_cols] = f'{2 + 6*i}', f'[0.57142857, {0.28571429 + i:.8f}, 0]', *others
        df.loc[2 + 6*i, sublat_cols] = f'{3 + 6*i}', f'[0.85714286, {0.42857143 + i:.8f}, 0]', *others
        df.loc[3 + 6*i, sublat_cols] = f'{4 + 6*i}', f'[0.71428571, {0.85714286 + i:.8f}, 0]', *others
        df.loc[4 + 6*i, sublat_cols] = f'{5 + 6*i}', f'[0.42857143, {0.71428571 + i:.8f}, 0]', *others
        df.loc[5 + 6*i, sublat_cols] = f'{6 + 6*i}', f'[0.14285714, {0.57142857 + i:.8f}, 0]', *others

        # interactions
        # intra cell
        df.loc[15*i + 0, inter_cols] = f'{6*i + 1}', f'{6*i + 2}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[15*i + 1, inter_cols] = f'{6*i + 2}', f'{6*i + 3}', '[0, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
        df.loc[15*i + 2, inter_cols] = f'{6*i + 3}', f'{6*i + 4}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[15*i + 3, inter_cols] = f'{6*i + 4}', f'{6*i + 5}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[15*i + 4, inter_cols] = f'{6*i + 5}', f'{6*i + 6}', '[0, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
        df.loc[15*i + 5, inter_cols] = f'{6*i + 6}', f'{6*i + 1}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[15*i + 6, inter_cols] = f'{6*i + 2}', f'{6*i + 6}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[15*i + 7, inter_cols] = f'{6*i + 5}', f'{6*i + 3}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[15*i + 8, inter_cols] = f'{6*i + 2}', f'{6*i + 5}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

        # inter cell
        df.loc[15*i + 9, inter_cols] = f'{6*i + 6}', f'{6*i + 4}', '[-1, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
        df.loc[15*i + 10, inter_cols] = f'{6*i + 3}', f'{6*i + 1}', '[1, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
        df.loc[15*i + 11, inter_cols] = f'{6*i + 3}', f'{6*i + 6}', '[1, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

        # edge cases
        if i < ncells-1:
            # intra cell
            df.loc[15*i + 12, inter_cols] = f'{6*i + 4}', f'{6*i + 8}', '[0, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
            df.loc[15*i + 13, inter_cols] = f'{6*i + 7}', f'{6*i + 5}', '[0, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
            df.loc[15*i + 14, inter_cols] = f'{6*i + 4}', f'{6*i + 7}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

    write = os.path.join(dir, 'snub_trihexagonal_zigzag_zigzag.csv')
    with open(write, 'w') as f:
        df.to_csv(write, index=False)

    print(f"Setup file saved to: {write}")

    return df



def build_armchair_armchair(ncells: int) -> pd.DataFrame:
    '''
    TODO
    '''
    dir = os.path.dirname(os.path.abspath(__file__))
    read = os.path.join(dir, 'template2.csv')
    template = pd.read_csv(read, dtype=str)

    df = template.copy()

    # lattice vectors
    df.loc[0, '#translation vectors'] = '[4.5*a, sqrt(3)/2*a, 0]'
    df.loc[1, '#translation vectors'] = '[2*a, sqrt(3)*a, 0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 6*i, sublat_cols] = f'{1 + 6*i}', f'[0.28571429, {-0.14285714 + i:.8f}, 0]', *others
        df.loc[1 + 6*i, sublat_cols] = f'{2 + 6*i}', f'[0.57142857, {-0.28571429 + i:.8f}, 0]', *others
        df.loc[2 + 6*i, sublat_cols] = f'{3 + 6*i}', f'[0.85714286, {-0.42857143 + i:.8f}, 0]', *others
        df.loc[3 + 6*i, sublat_cols] = f'{4 + 6*i}', f'[0.71428571, {0.14285714 + i:.8f}, 0]', *others
        df.loc[4 + 6*i, sublat_cols] = f'{5 + 6*i}', f'[0.42857143, {0.28571429 + i:.8f}, 0]', *others
        df.loc[5 + 6*i, sublat_cols] = f'{6 + 6*i}', f'[0.14285714, {0.42857143 + i:.8f}, 0]', *others

        # interactions
        # intra cell
        df.loc[15*i + 0, inter_cols] = f'{6*i + 1}', f'{6*i + 2}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[15*i + 1, inter_cols] = f'{6*i + 2}', f'{6*i + 3}', '[0, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
        df.loc[15*i + 2, inter_cols] = f'{6*i + 3}', f'{6*i + 4}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[15*i + 3, inter_cols] = f'{6*i + 4}', f'{6*i + 5}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[15*i + 4, inter_cols] = f'{6*i + 5}', f'{6*i + 6}', '[0, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
        df.loc[15*i + 5, inter_cols] = f'{6*i + 6}', f'{6*i + 1}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[15*i + 6, inter_cols] = f'{6*i + 2}', f'{6*i + 6}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[15*i + 7, inter_cols] = f'{6*i + 5}', f'{6*i + 3}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[15*i + 8, inter_cols] = f'{6*i + 2}', f'{6*i + 5}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

        # edge cases
        if i < ncells-1:
            # inter cell
            df.loc[15*i + 9, inter_cols] = f'{6*i + 6}', f'{6*i + 10}', '[-1, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
            df.loc[15*i + 10, inter_cols] = f'{6*i + 9}', f'{6*i + 1}', '[1, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
            df.loc[15*i + 11, inter_cols] = f'{6*i + 9}', f'{6*i + 6}', '[1, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

            # intra cell
            df.loc[15*i + 12, inter_cols] = f'{6*i + 4}', f'{6*i + 8}', '[0, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
            df.loc[15*i + 13, inter_cols] = f'{6*i + 7}', f'{6*i + 5}', '[0, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
            df.loc[15*i + 14, inter_cols] = f'{6*i + 4}', f'{6*i + 7}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

    write = os.path.join(dir, 'snub_trihexagonal_armchair_armchair.csv')
    with open(write, 'w') as f:
        df.to_csv(write, index=False)

    print(f"Setup file saved to: {write}")

    return df



# df = build_armchair_armchair(10)