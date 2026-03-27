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
    df.loc[0, '#translation vectors'] = '[(sqrt(2)+1)*a, 0, 0]'
    df.loc[1, '#translation vectors'] = '[0, (sqrt(2)+1)*a, 0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 4*i, sublat_cols] = f'{1 + 4*i}', f'[0, {i}, 0]', *others
        df.loc[1 + 4*i, sublat_cols] = f'{2 + 4*i}', f'[0.29289322, {0.29289322 + i:.8f}, 0]', *others
        df.loc[2 + 4*i, sublat_cols] = f'{3 + 4*i}', f'[0, {0.58578644 + i:.8f}, 0]', *others
        df.loc[3 + 4*i, sublat_cols] = f'{4 + 4*i}', f'[0.70710678, {0.29289322 + i:.8f}, 0]', *others

        # interactions
        # intra cell
        df.loc[6*i + 0, inter_cols] = f'{4*i + 1}', f'{4*i + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[6*i + 1, inter_cols] = f'{4*i + 2}', f'{4*i + 3}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[6*i + 2, inter_cols] = f'{4*i + 2}', f'{4*i + 4}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'
        

        # inter cell
        df.loc[6*i + 3, inter_cols] = f'{4*i + 3}', f'{4*i + 4}', '[-1, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[6*i + 4, inter_cols] = f'{4*i + 4}', f'{4*i + 1}', '[1, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'

        # edge cases
        if i < ncells-1:
            # intra cell
            df.loc[6*i + 5, inter_cols] = f'{4*i+3}', f'{4*i+5}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

    write = os.path.join(dir, 'truncated_square_zigzag_zigzag.csv')
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
    df.loc[0, '#translation vectors'] = '[(sqrt(2)+1)*a, (sqrt(2)+1)*a, 0]'
    df.loc[1, '#translation vectors'] = '[-(sqrt(2)+1)*a, (sqrt(2)+1)*a, 0]'
    df.loc[2, '#translation vectors'] = '[0, 0, 1]'

    sublat_cols = ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']
    others = '[0,0,1]', 'S1', '[[g, 0, 0], [0, g, 0], [0, 0, g]]'
    inter_cols = ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']

    # spins, interactions
    for i in range(0, ncells):
        # sublat id, basis vectors
        df.loc[0 + 8*i, sublat_cols] = f'{1 + 8*i}', f'[0, {i}, 0]', *others
        df.loc[1 + 8*i, sublat_cols] = f'{2 + 8*i}', f'[0.29289322, {i}, 0]', *others
        df.loc[2 + 8*i, sublat_cols] = f'{3 + 8*i}', f'[0.29289322, {0.29289322 + i:.8f}, 0]', *others
        df.loc[3 + 8*i, sublat_cols] = f'{4 + 8*i}', f'[0, {0.29289322 + i:.8f}, 0]', *others
        df.loc[4 + 8*i, sublat_cols] = f'{5 + 8*i}', f'[0.5, {0.5 + i:.8f}, 0]', *others
        df.loc[5 + 8*i, sublat_cols] = f'{6 + 8*i}', f'[0.79289322, {0.5 + i:.8f}, 0]', *others
        df.loc[6 + 8*i, sublat_cols] = f'{7 + 8*i}', f'[0.79289322, {0.79289322 + i:.8f}, 0]', *others
        df.loc[7 + 8*i, sublat_cols] = f'{8 + 8*i}', f'[0.5, {0.79289322 + i:.8f}, 0]', *others
        
        # interactions
        # intra cell
        df.loc[12*i + 0, inter_cols] = f'{8*i + 1}', f'{8*i + 2}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 1, inter_cols] = f'{8*i + 2}', f'{8*i + 3}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 2, inter_cols] = f'{8*i + 3}', f'{8*i + 4}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 3, inter_cols] = f'{8*i + 4}', f'{8*i + 1}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 4, inter_cols] = f'{8*i + 5}', f'{8*i + 6}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 5, inter_cols] = f'{8*i + 6}', f'{8*i + 7}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 6, inter_cols] = f'{8*i + 7}', f'{8*i + 8}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 7, inter_cols] = f'{8*i + 8}', f'{8*i + 5}', '[0, 0, 0]', '[[J,Dz,0],[-Dz,J,0],[0,0,J]]'
        df.loc[12*i + 8, inter_cols] = f'{8*i + 3}', f'{8*i + 5}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

        # inter cell
        df.loc[12*i + 9, inter_cols] = f'{8*i + 6}', f'{8*i + 4}', '[1, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

        # edge cases
        if i < ncells-1:
            # intra cell
            df.loc[12*i + 10, inter_cols] = f'{8*i + 8}', f'{8*i + 10}', '[0, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

            # inter cell
            df.loc[12*i + 11, inter_cols] = f'{8*i + 7}', f'{8*i + 9}', '[1, 0, 0]', '[[J,0,0],[0,J,0],[0,0,J]]'

    write = os.path.join(dir, 'truncated_square_armchair_armchair.csv')
    with open(write, 'w') as f:
        df.to_csv(write, index=False)

    print(f"Setup file saved to: {write}")

    return df



# df = build_armchair_armchair(10)

# import numpy as np

# # Original square-octagon lattice
# a = 1.0
# factor = np.sqrt(2) + 1

# a1_orig = np.array([factor * a, 0])
# a2_orig = np.array([0, factor * a])

# # Original positions (fractional)
# positions_orig = np.array([
#     [0.0, 0.0],
#     [0.29289322, 0.29289322],
#     [0.0, 0.58578644],
#     [0.70710678, 0.29289322]
# ])

# # For 45° armchair edge:
# # New lattice vectors - rotate by 45°
# # a1_new should be along [1,1] direction (parallel to edge)
# # a2_new should be along [1,-1] direction (perpendicular, will be terminated)

# # Method 1: Rotation matrix approach
# theta = np.pi / 4  # 45 degrees
# rotation = np.array([[np.cos(theta), -np.sin(theta)],
#                      [np.sin(theta), np.cos(theta)]])

# # This doesn't preserve the lattice - need integer linear combinations instead

# # Method 2: Find integer combinations (like we discussed before)
# # For square lattice symmetry, natural choice:
# # a1_new = a1_orig + a2_orig  (diagonal direction [1,1])
# # a2_new = -a1_orig + a2_orig (perpendicular diagonal [1,-1])

# a1_new = a1_orig + a2_orig  # Along [1,1]
# a2_new = -a1_orig + a2_orig  # Along [-1,1]

# print(f"New lattice vectors:")
# print(f"a1_new (parallel to edge): {a1_new}")
# print(f"a2_new (perpendicular): {a2_new}")

# # Transform sublattice positions to new basis
# M_orig = np.column_stack([a1_orig, a2_orig])
# M_new = np.column_stack([a1_new, a2_new])
# M_new_inv = np.linalg.inv(M_new)

# positions_new_frac = []
# search_range = 3

# for nx in range(-search_range, search_range + 1):
#     for ny in range(-search_range, search_range + 1):
#         for pos_orig_frac in positions_orig:
#             # Position in original fractional coords
#             pos_frac = pos_orig_frac + np.array([nx, ny])
            
#             # Convert to Cartesian
#             pos_cart = M_orig @ pos_frac
            
#             # Convert to new fractional coords
#             pos_new_frac = M_new_inv @ pos_cart
            
#             # Keep if inside new unit cell [0,1) x [0,1)
#             if (0 <= pos_new_frac[0] < 1.0 and 
#                 0 <= pos_new_frac[1] < 1.0):
                
#                 # Check for duplicates
#                 is_duplicate = False
#                 for existing in positions_new_frac:
#                     if np.allclose(pos_new_frac, existing, atol=1e-6):
#                         is_duplicate = True
#                         break
                
#                 if not is_duplicate:
#                     positions_new_frac.append(pos_new_frac)

# positions_new_frac = np.array(positions_new_frac)
# print(f"\nOriginal unit cell: {len(positions_orig)} atoms")
# print(f"New unit cell: {len(positions_new_frac)} atoms")
# print(f"New positions (fractional):\n{positions_new_frac}")

# import matplotlib.pyplot as plt

# def visualize_both_orientations():
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
#     # Plot 1: Original orientation (your current setup)
#     for nx in range(-2, 3):
#         for ny in range(-2, 3):
#             for i, pos_frac in enumerate(positions_orig):
#                 pos = (pos_frac[0] + nx) * a1_orig + (pos_frac[1] + ny) * a2_orig
#                 ax1.scatter(pos[0], pos[1], s=50, c='blue', alpha=0.6)
                
#                 # Add index labels only for origin cell
#                 if nx == 0 and ny == 0:
#                     ax1.text(pos[0], pos[1], f'{i}', 
#                             fontsize=12, fontweight='bold',
#                             ha='center', va='bottom', color='red')
    
#     # Draw original unit cell
#     cell_orig = np.array([[0, 0], a1_orig, a1_orig + a2_orig, a2_orig, [0, 0]])
#     ax1.plot(cell_orig[:, 0], cell_orig[:, 1], 'r-', linewidth=2, label='Original UC')
#     ax1.set_title('Original orientation (zigzag edge along y)')
#     ax1.set_aspect('equal')
#     ax1.grid(True, alpha=0.3)
#     ax1.legend()
    
#     # Plot 2: 45° rotated (armchair edge)
#     for nx in range(-2, 3):
#         for ny in range(-2, 3):
#             for i, pos_frac in enumerate(positions_new_frac):
#                 pos = (pos_frac[0] + nx) * a1_new + (pos_frac[1] + ny) * a2_new
#                 ax2.scatter(pos[0], pos[1], s=50, c='green', alpha=0.6)
                
#                 # Add index labels only for origin cell
#                 if nx == 0 and ny == 0:
#                     ax2.text(pos[0], pos[1], f'{i}', 
#                             fontsize=12, fontweight='bold',
#                             ha='center', va='bottom', color='red')
    
#     # Draw new unit cell
#     cell_new = np.array([[0, 0], a1_new, a1_new + a2_new, a2_new, [0, 0]])
#     ax2.plot(cell_new[:, 0], cell_new[:, 1], 'r-', linewidth=2, label='Rotated UC')
    
#     # Show edge direction
#     ax2.arrow(0, 0, a1_new[0]*0.5, a1_new[1]*0.5, 
#               head_width=0.3, head_length=0.2, fc='orange', ec='orange',
#               linewidth=3, label='Edge direction (45°)')
    
#     ax2.set_title('Rotated orientation (armchair edge)')
#     ax2.set_aspect('equal')
#     ax2.grid(True, alpha=0.3)
#     ax2.legend()
    
#     plt.tight_layout()
#     plt.show()

# visualize_both_orientations()

# print("\nOriginal unit cell atoms:")
# for i, pos_frac in enumerate(positions_orig):
#     pos_cart = M_orig @ pos_frac
#     print(f"Atom {i}: fractional {pos_frac}, Cartesian {pos_cart}")

# print(f"\nNew unit cell atoms ({len(positions_new_frac)} total):")
# for i, pos_frac in enumerate(positions_new_frac):
#     pos_cart = M_new @ pos_frac
#     print(f"Atom {i}: fractional {pos_frac}, Cartesian {pos_cart}")