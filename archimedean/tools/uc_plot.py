import numpy as np
import matplotlib.pyplot as plt

def plot_unit_cell(lattice_vectors, positions_frac, title="Lattice Unit Cell"):
    """
    Plot unit cell with numbered atoms.
    
    Parameters:
    -----------
    lattice_vectors : list or array of shape (2, 2)
        [a1, a2] where a1 and a2 are 2D vectors
    positions_frac : array of shape (n_atoms, 2)
        Atomic positions in fractional coordinates (in lattice vector basis)
    title : str
        Plot title
    """
    
    # Convert to numpy arrays
    a1 = np.array(lattice_vectors[0])
    a2 = np.array(lattice_vectors[1])
    positions_frac = np.array(positions_frac)
    
    # Create transformation matrix
    M = np.column_stack([a1, a2])
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot neighboring cells for context
    n_cells = 2  # number of neighboring cells to show
    for nx in range(-n_cells, n_cells + 1):
        for ny in range(-n_cells, n_cells + 1):
            for i, pos_frac in enumerate(positions_frac):
                # Position in fractional coords
                pos_frac_shifted = pos_frac + np.array([nx, ny])
                
                # Convert to Cartesian
                pos_cart = M @ pos_frac_shifted
                
                # Plot atom
                if nx == 0 and ny == 0:
                    # Origin cell - larger and colored
                    ax.scatter(pos_cart[0], pos_cart[1], s=200, c='blue', 
                              edgecolors='black', linewidth=2, alpha=0.8, zorder=3)
                    # Add index label
                    ax.text(pos_cart[0], pos_cart[1], f'{i+1}', 
                           fontsize=14, fontweight='bold',
                           ha='center', va='center', color='white', zorder=4)
                else:
                    # Neighboring cells - smaller and faded
                    ax.scatter(pos_cart[0], pos_cart[1], s=80, c='lightblue', 
                              edgecolors='gray', linewidth=0.5, alpha=0.4, zorder=1)
    
    # Draw unit cell boundary
    cell_corners = np.array([[0, 0], a1, a1 + a2, a2, [0, 0]])
    ax.plot(cell_corners[:, 0], cell_corners[:, 1], 'r-', 
           linewidth=3, label='Unit cell', zorder=2)
    
    # Draw lattice vectors from origin
    ax.arrow(0, 0, a1[0], a1[1], head_width=0.15, head_length=0.15, 
            fc='green', ec='green', linewidth=2, alpha=0.7, zorder=2)
    ax.text(a1[0]/2, a1[1]/2, 'a₁', fontsize=14, fontweight='bold', 
           color='green', ha='center', va='bottom')
    
    ax.arrow(0, 0, a2[0], a2[1], head_width=0.15, head_length=0.15, 
            fc='orange', ec='orange', linewidth=2, alpha=0.7, zorder=2)
    ax.text(a2[0]/2, a2[1]/2, 'a₂', fontsize=14, fontweight='bold', 
           color='orange', ha='right', va='center')
    
    # Formatting
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print positions
    print(f"\n{title}")
    print("="*60)
    print(f"Lattice vector a1: {a1}")
    print(f"Lattice vector a2: {a2}")
    print(f"\nNumber of atoms in unit cell: {len(positions_frac)}")
    print("\nAtomic positions:")
    print(f"{'Index':<6} {'Fractional (a1, a2)':<25} {'Cartesian (x, y)':<20}")
    print("-"*60)
    for i, pos_frac in enumerate(positions_frac):
        pos_cart = M @ pos_frac
        print(f"{i:<6} ({pos_frac[0]:.6f}, {pos_frac[1]:.6f})     ({pos_cart[0]:.6f}, {pos_cart[1]:.6f})")


# # Example 1: Kagome lattice
# print("\n" + "="*60)
# print("EXAMPLE 1: KAGOME LATTICE")
# print("="*60)

# a = 1.0
# kagome_vectors = [
#     [2*a, 0],
#     [a, np.sqrt(3)*a]
# ]

# kagome_positions = [
#     [0.0, 0.0],
#     [0.5, 0.0],
#     [0.0, 0.5]
# ]

# plot_unit_cell(kagome_vectors, kagome_positions, "Kagome Lattice")


# # Example 2: Square-Octagon (Truncated Square) lattice
# print("\n" + "="*60)
# print("EXAMPLE 2: SQUARE-OCTAGON LATTICE")
# print("="*60)

# a = 1.0
# factor = np.sqrt(2) + 1

# sq_oct_vectors = [
#     [factor * a, 0],
#     [0, factor * a]
# ]

# sq_oct_positions = [
#     [0.0, 0.0],
#     [0.29289322, 0.29289322],
#     [0.0, 0.58578644],
#     [0.70710678, 0.29289322]
# ]

# plot_unit_cell(sq_oct_vectors, sq_oct_positions, "Square-Octagon Lattice")


# # Example 3: Simple square lattice (for comparison)
# print("\n" + "="*60)
# print("EXAMPLE 3: SIMPLE SQUARE LATTICE")
# print("="*60)

# square_vectors = [
#     [1, 0],
#     [0, 1]
# ]

# square_positions = [
#     [0.0, 0.0]
# ]

# plot_unit_cell(square_vectors, square_positions, "Simple Square Lattice")


vects = [
    [4.5, 0.8660254], 
    [2, 1.73205081]
    ]
basis = [
    [0.28571429, -0.14285714], 
    [0.57142857, -0.28571429], 
    [0.85714286, -0.42857143],
    [0.71428571, 0.14285714], 
    [0.42857143, 0.28571429], 
    [0.14285714, 0.42857143],
    ]

plot_unit_cell(vects, basis, ".")