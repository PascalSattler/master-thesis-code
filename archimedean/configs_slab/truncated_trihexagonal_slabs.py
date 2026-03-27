import pandas as pd
import os

def build_zigzag_zigzag(ncells: int) -> pd.DataFrame:
    '''
    TODO
    '''
    dir = os.path.dirname(os.path.abspath(__file__))
    read = os.path.join(dir, 'template3.csv')
    template = pd.read_csv(read, dtype=str)

    df = template.copy()

    # lattice vectors
    df.loc[0, '#translation vectors'] = '[(sqrt(3)+3)*a, 0, 0]'
    df.loc[1, '#translation vectors'] = '[(sqrt(3)+3)/2*a, 1.5*(sqrt(3)+1)*a, 0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 12*i, sublat_cols] = f'{1 + 12*i}', f'[0.33333333, {0.12200847 + i:.8f}, 0]', *others
        df.loc[1 + 12*i, sublat_cols] = f'{2 + 12*i}', f'[0.54465820, {0.12200847 + i:.8f}, 0]', *others
        df.loc[2 + 12*i, sublat_cols] = f'{3 + 12*i}', f'[0.54465820, {0.33333333 + i:.8f}, 0]', *others
        df.loc[3 + 12*i, sublat_cols] = f'{4 + 12*i}', f'[0.33333333, {0.54465820 + i:.8f}, 0]', *others
        df.loc[4 + 12*i, sublat_cols] = f'{5 + 12*i}', f'[0.12200847, {0.54465820 + i:.8f}, 0]', *others
        df.loc[5 + 12*i, sublat_cols] = f'{6 + 12*i}', f'[0.12200847, {0.33333333 + i:.8f}, 0]', *others
        df.loc[6 + 12*i, sublat_cols] = f'{7 + 12*i}', f'[0.66666667, {0.45534180 + i:.8f}, 0]', *others
        df.loc[7 + 12*i, sublat_cols] = f'{8 + 12*i}', f'[0.87799153, {0.45534180 + i:.8f}, 0]', *others
        df.loc[8 + 12*i, sublat_cols] = f'{9 + 12*i}', f'[0.87799153, {0.66666667 + i:.8f}, 0]', *others
        df.loc[9 + 12*i, sublat_cols] = f'{10 + 12*i}', f'[0.66666667, {0.87799153 + i:.8f}, 0]', *others
        df.loc[10 + 12*i, sublat_cols] = f'{11 + 12*i}', f'[0.45534180, {0.87799153 + i:.8f}, 0]', *others
        df.loc[11 + 12*i, sublat_cols] = f'{12 + 12*i}', f'[0.45534180, {0.66666667 + i:.8f}, 0]', *others

        # interactions
        # intra cell
        df.loc[18*i + 0, inter_cols] = f'{12*i + 1}', f'{12*i + 2}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[18*i + 1, inter_cols] = f'{12*i + 2}', f'{12*i + 3}', '[0, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
        df.loc[18*i + 2, inter_cols] = f'{12*i + 3}', f'{12*i + 4}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[18*i + 3, inter_cols] = f'{12*i + 4}', f'{12*i + 5}', '[0, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
        df.loc[18*i + 4, inter_cols] = f'{12*i + 5}', f'{12*i + 6}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[18*i + 5, inter_cols] = f'{12*i + 6}', f'{12*i + 1}', '[0, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'

        df.loc[18*i + 6, inter_cols] = f'{12*i + 7}', f'{12*i + 8}', '[0, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
        df.loc[18*i + 7, inter_cols] = f'{12*i + 8}', f'{12*i + 9}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[18*i + 8, inter_cols] = f'{12*i + 9}', f'{12*i + 10}', '[0, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
        df.loc[18*i + 9, inter_cols] = f'{12*i + 10}', f'{12*i + 11}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[18*i + 10, inter_cols] = f'{12*i + 11}', f'{12*i + 12}', '[0, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
        df.loc[18*i + 11, inter_cols] = f'{12*i + 12}', f'{12*i + 7}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'

        df.loc[18*i + 12, inter_cols] = f'{12*i + 3}', f'{12*i + 7}', '[0, 0, 0]', '[[J,-Dz3,0],[Dz3,J,0],[0,0,J]]'
        df.loc[18*i + 13, inter_cols] = f'{12*i + 12}', f'{12*i + 4}', '[0, 0, 0]', '[[J,-Dz3,0],[Dz3,J,0],[0,0,J]]'

        # inter cell
        df.loc[18*i + 14, inter_cols] = f'{12*i + 8}', f'{12*i + 6}', '[1, 0, 0]', '[[J,-Dz3,0],[Dz3,J,0],[0,0,J]]'
        df.loc[18*i + 15, inter_cols] = f'{12*i + 5}', f'{12*i + 9}', '[-1, 0, 0]', '[[J,-Dz3,0],[Dz3,J,0],[0,0,J]]'

        # edge cases
        if i < ncells-1:
            # intra cell
            df.loc[18*i + 16, inter_cols] = f'{12*i + 10}', f'{12*i + 14}', '[0, 0, 0]', '[[J,-Dz3,0],[Dz3,J,0],[0,0,J]]'
            df.loc[18*i + 17, inter_cols] = f'{12*i + 13}', f'{12*i + 11}', '[0, 0, 0]', '[[J,-Dz3,0],[Dz3,J,0],[0,0,J]]'

    write = os.path.join(dir, 'truncated_trihexagonal_zigzag_zigzag.csv')
    with open(write, 'w') as f:
        df.to_csv(write, index=False)

    print(f"Setup file saved to: {write}")

    return df



def build_armchair_armchair(ncells: int) -> pd.DataFrame:
    '''
    TODO
    '''
    dir = os.path.dirname(os.path.abspath(__file__))
    read = os.path.join(dir, 'template3.csv')
    template = pd.read_csv(read, dtype=str)

    df = template.copy()

    # lattice vectors
    df.loc[0, '#translation vectors'] = '[3*(sqrt(3)+1)*a, 0, 0]'
    df.loc[1, '#translation vectors'] = '[3*(sqrt(3)+1)/2*a, (sqrt(3)+3)/2*a, 0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 12*i, sublat_cols] = f'{1 + 12*i}', f'[0.24401694, {i:.8f}, 0]', *others
        df.loc[1 + 12*i, sublat_cols] = f'{2 + 12*i}', f'[0.4553418, {-0.21132487 + i:.8f}, 0]', *others
        df.loc[2 + 12*i, sublat_cols] = f'{3 + 12*i}', f'[0.4553418, {i:.8f}, 0]', *others
        df.loc[3 + 12*i, sublat_cols] = f'{4 + 12*i}', f'[0.24401694, {0.42264973 + i:.8f}, 0]', *others
        df.loc[4 + 12*i, sublat_cols] = f'{5 + 12*i}', f'[0.03269207, {0.63397460 + i:.8f}, 0]', *others
        df.loc[5 + 12*i, sublat_cols] = f'{6 + 12*i}', f'[0.03269207, {0.42264973 + i:.8f}, 0]', *others
        df.loc[6 + 12*i, sublat_cols] = f'{7 + 12*i}', f'[0.57735027, {i:.8f}, 0]', *others
        df.loc[7 + 12*i, sublat_cols] = f'{8 + 12*i}', f'[0.78867513, {-0.21132487 + i:.8f}, 0]', *others
        df.loc[8 + 12*i, sublat_cols] = f'{9 + 12*i}', f'[0.78867513, {i:.8f}, 0]', *others
        df.loc[9 + 12*i, sublat_cols] = f'{10 + 12*i}', f'[0.57735027, {0.42264973 + i:.8f}, 0]', *others
        df.loc[10 + 12*i, sublat_cols] = f'{11 + 12*i}', f'[0.36602540, {0.63397460 + i:.8f}, 0]', *others
        df.loc[11 + 12*i, sublat_cols] = f'{12 + 12*i}', f'[0.36602540, {0.42264973 + i:.8f}, 0]', *others

        # interactions
        # intra cell
        df.loc[18*i + 0, inter_cols] = f'{12*i + 1}', f'{12*i + 2}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[18*i + 1, inter_cols] = f'{12*i + 2}', f'{12*i + 3}', '[0, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
        df.loc[18*i + 2, inter_cols] = f'{12*i + 3}', f'{12*i + 4}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[18*i + 3, inter_cols] = f'{12*i + 4}', f'{12*i + 5}', '[0, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
        df.loc[18*i + 4, inter_cols] = f'{12*i + 5}', f'{12*i + 6}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[18*i + 5, inter_cols] = f'{12*i + 6}', f'{12*i + 1}', '[0, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'

        df.loc[18*i + 6, inter_cols] = f'{12*i + 7}', f'{12*i + 8}', '[0, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
        df.loc[18*i + 7, inter_cols] = f'{12*i + 8}', f'{12*i + 9}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[18*i + 8, inter_cols] = f'{12*i + 9}', f'{12*i + 10}', '[0, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
        df.loc[18*i + 9, inter_cols] = f'{12*i + 10}', f'{12*i + 11}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'
        df.loc[18*i + 10, inter_cols] = f'{12*i + 11}', f'{12*i + 12}', '[0, 0, 0]', '[[J,Dz2,0],[-Dz2,J,0],[0,0,J]]'
        df.loc[18*i + 11, inter_cols] = f'{12*i + 12}', f'{12*i + 7}', '[0, 0, 0]', '[[J,Dz1,0],[-Dz1,J,0],[0,0,J]]'

        df.loc[18*i + 12, inter_cols] = f'{12*i + 3}', f'{12*i + 7}', '[0, 0, 0]', '[[J,-Dz3,0],[Dz3,J,0],[0,0,J]]'
        df.loc[18*i + 13, inter_cols] = f'{12*i + 12}', f'{12*i + 4}', '[0, 0, 0]', '[[J,-Dz3,0],[Dz3,J,0],[0,0,J]]'

        # edge cases
        if i < ncells-1:
            # inter cell
            df.loc[18*i + 14, inter_cols] = f'{12*i + 20}', f'{12*i + 6}', '[1, 0, 0]', '[[J,-Dz3,0],[Dz3,J,0],[0,0,J]]'
            df.loc[18*i + 15, inter_cols] = f'{12*i + 5}', f'{12*i + 21}', '[-1, 0, 0]', '[[J,-Dz3,0],[Dz3,J,0],[0,0,J]]'

            # intra cell
            df.loc[18*i + 16, inter_cols] = f'{12*i + 10}', f'{12*i + 14}', '[0, 0, 0]', '[[J,-Dz3,0],[Dz3,J,0],[0,0,J]]'
            df.loc[18*i + 17, inter_cols] = f'{12*i + 13}', f'{12*i + 11}', '[0, 0, 0]', '[[J,-Dz3,0],[Dz3,J,0],[0,0,J]]'

    write = os.path.join(dir, 'truncated_trihexagonal_armchair_armchair.csv')
    with open(write, 'w') as f:
        df.to_csv(write, index=False)

    print(f"Setup file saved to: {write}")

    return df



# df = build_armchair_armchair(10)