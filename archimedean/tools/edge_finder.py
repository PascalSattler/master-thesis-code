import numpy as np
import matplotlib.pyplot as plt
from math import gcd

# Kagome lattice vectors
a = 1.0  # lattice constant
a1 = np.array([2*a, 0])
a2 = np.array([a, np.sqrt(3)*a])

# Sublattice positions (in units of lattice vectors, or direct Cartesian)
# For kagome, typical sublattice positions within unit cell:
sublattice_positions = np.array([
    [0.0, 0.0],
    [0.5*a, 0.0],
    [0.25*a, 0.25*np.sqrt(3)*a]  # or adjust to your convention
])

# Note: Your positions (0,0,0), (0.5,0,0), (0,0.5) seem to be in fractional coordinates
# Let me also show that version:

# If using fractional coordinates:
sublattice_frac = np.array([
    [0.0, 0.0],
    [0.5, 0.0],
    [0.0, 0.5]  # assuming this is what you meant
])

# Convert to Cartesian
def frac_to_cart(frac_coords, a1, a2):
    """Convert fractional coordinates to Cartesian"""
    cart = np.zeros((len(frac_coords), 2))
    for i, frac in enumerate(frac_coords):
        cart[i] = frac[0] * a1 + frac[1] * a2
    return cart

sublattice_positions = frac_to_cart(sublattice_frac, a1, a2)

# Generate lattice points for visualization
n_cells_x = 5  # number of unit cells in x direction
n_cells_y = 5  # number of unit cells in y direction

lattice_points = []
for nx in range(-n_cells_x, n_cells_x+1):
    for ny in range(-n_cells_y, n_cells_y+1):
        cell_origin = nx * a1 + ny * a2
        for sub_pos in sublattice_positions:
            lattice_points.append(cell_origin + sub_pos)

lattice_points = np.array(lattice_points)

# Plot the lattice
plt.figure(figsize=(10, 10))
plt.scatter(lattice_points[:, 0], lattice_points[:, 1], s=30, alpha=0.6)

# Draw unit cell
unit_cell = np.array([
    [0, 0],
    a1,
    a1 + a2,
    a2,
    [0, 0]
])
plt.plot(unit_cell[:, 0], unit_cell[:, 1], 'r-', linewidth=2, label='Unit cell')

# Optionally draw bonds (nearest neighbors for kagome)
# This helps visualize the structure
bond_length = 1 * a  # adjust based on your lattice
for i, p1 in enumerate(lattice_points):
    for j, p2 in enumerate(lattice_points[i+1:], i+1):
        dist = np.linalg.norm(p1 - p2)
        if np.abs(dist - bond_length) < 0.1:  # tolerance for nearest neighbors
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.3, linewidth=0.5)

plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Kagome Lattice')
plt.show()

# Add a cutting line at different angles
def plot_cutting_line(angle_deg, offset=0.25):
    angle = np.radians(angle_deg)
    # Line perpendicular to edge direction
    normal = np.array([np.cos(angle), np.sin(angle)])
    
    # Generate line for plotting
    t = np.linspace(-10, 10, 100)
    tangent = np.array([-normal[1], normal[0]])
    line = offset * normal[:, None] + tangent[:, None] * t
    
    plt.plot(line[0], line[1], 'r--', linewidth=2, label=f'Cut at {angle_deg}°')

# Try different angles
for angle in [0, 30, 60, 90]:
    plt.figure(figsize=(10, 10))
    plt.scatter(lattice_points[:, 0], lattice_points[:, 1], s=50, alpha=0.6)
    plot_cutting_line(angle)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.title(f'Termination at {angle}°')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()

# 0 : shift left/right between subalts -> armchair/deep zigzag, stay on line and terminate sublats -> double deep zigzag, shift line to left/right sublats and terminate -> double armchair
# 30 : shift left/right between subalts -> hook/flat, stay on line and terminate sublats -> double flat, shift line to left/right sublats and terminate -> double hook
# 60 : same as 0 (60 degree rotational symmetry of lattice ?)
# 90 : same as 30

import numpy as np

def find_slab_vectors(a1_bulk, a2_bulk, edge_angle_deg, max_search=5, max_det=5):
    """
    Find slab lattice vectors for a given edge orientation.
    
    Parameters:
    -----------
    a1_bulk, a2_bulk : array
        Bulk lattice vectors
    edge_angle_deg : float
        Desired angle of edge in degrees (measured from x-axis)
    max_search : int
        Maximum index to search for lattice vector combinations
    max_det : int
        Maximum determinant value to allow (unit cell size multiplier)
    
    Returns:
    --------
    results : list of dict
        List of possible solutions, each containing:
        - 'a1_slab': vector parallel to edge
        - 'a2_slab': vector perpendicular to slab (finite direction)
        - 'indices': ((n1, n2), (m1, m2))
        - 'det': determinant value
        - 'angle_error': angular deviation from target (degrees)
        - 'n_atoms': number of atoms in new unit cell
    """
    
    edge_angle = np.radians(edge_angle_deg)
    edge_dir = np.array([np.cos(edge_angle), np.sin(edge_angle)])
    
    results = []
    
    # Search for integer combinations
    for n1 in range(-max_search, max_search+1):
        for n2 in range(-max_search, max_search+1):
            if n1 == 0 and n2 == 0:
                continue
            
            # Candidate for a1_slab (parallel to edge)
            a1_candidate = n1 * a1_bulk + n2 * a2_bulk
            
            # Check angle match with desired edge direction
            a1_norm = a1_candidate / np.linalg.norm(a1_candidate)
            
            # Angle between vectors
            cos_angle = np.dot(a1_norm, edge_dir)
            angle_error_rad = np.arccos(np.clip(cos_angle, -1, 1))
            # Consider both directions (angle and angle+180°)
            angle_error_rad = min(angle_error_rad, np.pi - angle_error_rad)
            angle_error_deg = np.degrees(angle_error_rad)
            
            # Only consider if angle is close enough (within 5 degrees)
            if angle_error_deg > 5.0:
                continue
            
            # Now search for a2_slab
            for m1 in range(-max_search, max_search+1):
                for m2 in range(-max_search, max_search+1):
                    if m1 == 0 and m2 == 0:
                        continue
                    
                    # Check determinant
                    det = n1 * m2 - n2 * m1
                    
                    if det == 0 or abs(det) > max_det:
                        continue
                    
                    a2_candidate = m1 * a1_bulk + m2 * a2_bulk
                    
                    # Check that vectors are linearly independent (not parallel)
                    cross = np.cross(a1_candidate, a2_candidate)
                    if np.abs(cross) < 1e-10:
                        continue
                    
                    # Calculate angle between a1 and a2
                    dot_prod = np.dot(a1_candidate, a2_candidate)
                    mag_product = np.linalg.norm(a1_candidate) * np.linalg.norm(a2_candidate)
                    angle_between = np.degrees(np.arccos(np.clip(dot_prod / mag_product, -1, 1)))
                    
                    results.append({
                        'a1_slab': a1_candidate.copy(),
                        'a2_slab': a2_candidate.copy(),
                        'indices': ((n1, n2), (m1, m2)),
                        'det': det,
                        'angle_error': angle_error_deg,
                        'angle_between_vectors': angle_between,
                        'n_atoms': abs(det) * 3  # 3 atoms per kagome unit cell
                    })
    
    # Sort by: smallest |det|, then smallest angle error
    results.sort(key=lambda x: (abs(x['det']), x['angle_error']))
    
    return results


def print_slab_solutions(results, n_show=5):
    """Print the top solutions in a readable format."""
    print(f"\nFound {len(results)} possible slab configurations")
    print(f"Showing top {min(n_show, len(results))} solutions:\n")
    
    for i, sol in enumerate(results[:n_show]):
        (n1, n2), (m1, m2) = sol['indices']
        print(f"Solution {i+1}:")
        print(f"  a1_slab = {n1:2d}*a1 + {n2:2d}*a2  →  {sol['a1_slab']}")
        print(f"  a2_slab = {m1:2d}*a1 + {m2:2d}*a2  →  {sol['a2_slab']}")
        print(f"  Determinant: {sol['det']:2d}  (unit cell has {sol['n_atoms']} atoms)")
        print(f"  Angle error: {sol['angle_error']:.2f}°")
        print(f"  Angle between vectors: {sol['angle_between_vectors']:.1f}°")
        print()


# Example usage
a = 1.0
a1_bulk = np.array([2*a, 0])
a2_bulk = np.array([a, np.sqrt(3)*a])

# Try different edge orientations
for edge_angle in [0, 30, 60, 90]:
    print(f"\n{'='*60}")
    print(f"Searching for edge at {edge_angle}°")
    print('='*60)
    
    results = find_slab_vectors(a1_bulk, a2_bulk, edge_angle, 
                                max_search=5, max_det=5)
    print_slab_solutions(results, n_show=3)

def visualize_slab_vectors(a1_bulk, a2_bulk, a1_slab, a2_slab, 
                          sublattice_frac_bulk):
    """
    Visualize the old and new lattice vectors overlaid.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot bulk lattice points
    for nx in range(-4, 5):
        for ny in range(-4, 5):
            for sub in sublattice_frac_bulk:
                pos = (sub[0] + nx) * a1_bulk + (sub[1] + ny) * a2_bulk
                ax.scatter(pos[0], pos[1], c='lightblue', s=30, alpha=0.5)
    
    # Draw bulk unit cell
    bulk_cell = np.array([[0, 0], a1_bulk, a1_bulk + a2_bulk, a2_bulk, [0, 0]])
    ax.plot(bulk_cell[:, 0], bulk_cell[:, 1], 'b--', linewidth=1, 
            label='Bulk unit cell', alpha=0.5)
    
    # Draw slab unit cell
    slab_cell = np.array([[0, 0], a1_slab, a1_slab + a2_slab, a2_slab, [0, 0]])
    ax.plot(slab_cell[:, 0], slab_cell[:, 1], 'r-', linewidth=2, 
            label='Slab unit cell')
    
    # Draw a few repetitions of slab cell
    for i in range(-2, 3):
        offset = i * a1_slab
        shifted_cell = slab_cell + offset
        ax.plot(shifted_cell[:, 0], shifted_cell[:, 1], 'r-', 
                linewidth=1, alpha=0.3)
    
    # Draw vectors from origin
    ax.arrow(0, 0, a1_bulk[0], a1_bulk[1], head_width=0.1, 
             head_length=0.1, fc='blue', ec='blue', alpha=0.5, label='a1_bulk')
    ax.arrow(0, 0, a2_bulk[0], a2_bulk[1], head_width=0.1, 
             head_length=0.1, fc='cyan', ec='cyan', alpha=0.5, label='a2_bulk')
    
    ax.arrow(0, 0, a1_slab[0], a1_slab[1], head_width=0.15, 
             head_length=0.15, fc='red', ec='red', linewidth=2, label='a1_slab')
    ax.arrow(0, 0, a2_slab[0], a2_slab[1], head_width=0.15, 
             head_length=0.15, fc='orange', ec='orange', linewidth=2, label='a2_slab')
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Bulk lattice with slab unit cell overlay')
    
    plt.show()

# Example: visualize a specific solution
results = find_slab_vectors(a1_bulk, a2_bulk, 30, max_search=5, max_det=5)
if results:
    best = results[0]
    sublattice_frac_bulk = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]])
    visualize_slab_vectors(a1_bulk, a2_bulk, best['a1_slab'], best['a2_slab'],
                          sublattice_frac_bulk)