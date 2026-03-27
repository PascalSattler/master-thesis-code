import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def find_degeneracy_points(hamiltonian_solver, kx_range, ky_range, num_points=100,
                          band_pairs=None, threshold=0.01, refine=True):
    """
    Find points in k-space where bands are degenerate or nearly degenerate.
    
    Parameters:
    -----------
    hamiltonian_solver : callable
        Function that takes (kx, ky) and returns array of energies
    kx_range : tuple
        (kx_min, kx_max) for x-axis of k-space
    ky_range : tuple
        (ky_min, ky_max) for y-axis of k-space
    num_points : int
        Number of points to sample in each k-direction
    band_pairs : list of tuples or None
        Specific band pairs to check, e.g., [(0,1), (1,2)]
        None checks all pairs
    threshold : float
        Energy difference below which bands are considered degenerate
    refine : bool
        If True, use local refinement around detected degeneracies
    
    Returns:
    --------
    degeneracies : dict
        Dictionary with degeneracy information for each band pair
    """
    kx = np.linspace(kx_range[0], kx_range[1], num_points)
    ky = np.linspace(ky_range[0], ky_range[1], num_points)
    
    # Get number of bands
    test_E = hamiltonian_solver(kx[0], ky[0],0)[0]
    num_bands = len(test_E)
    
    if band_pairs is None:
        band_pairs = [(i, j) for i in range(num_bands) for j in range(i+1, num_bands)]
    
    degeneracies = {}
    
    for b1, b2 in band_pairs:
        print(f"Searching for degeneracies between bands {b1} and {b2}...")
        degen_kx = []
        degen_ky = []
        degen_E = []
        degen_gap = []
        
        for i, kx_val in enumerate(kx):
            for j, ky_val in enumerate(ky):
                E = hamiltonian_solver(kx_val, ky_val, 0)[0]
                gap = abs(E[b1] - E[b2])
                
                if gap < threshold:
                    degen_kx.append(kx_val)
                    degen_ky.append(ky_val)
                    degen_E.append((E[b1] + E[b2]) / 2)
                    degen_gap.append(gap)
        
        degeneracies[f"bands_{b1}_{b2}"] = {
            'kx': np.array(degen_kx),
            'ky': np.array(degen_ky),
            'energy': np.array(degen_E),
            'gap': np.array(degen_gap),
            'num_points': len(degen_kx)
        }
        
        print(f"  Found {len(degen_kx)} points with gap < {threshold}")
    
    return degeneracies


def plot_bandstructure_3d(hamiltonian_solver, kx_range, ky_range, num_points=50, 
                          band_indices=None, figsize=(12, 8), cmap='viridis',
                          show_colorbar=True, elevation=30, azimuth=45,
                          highlight_degeneracies=True, degeneracy_threshold=0.01,
                          degeneracy_color='red', degeneracy_size=80):
    """
    Create a 3D plot of electronic bandstructure.
    
    Parameters:
    -----------
    hamiltonian_solver : callable
        Function that takes (kx, ky) and returns energies. 
        Should return either a single energy or array of energies for multiple bands.
    kx_range : tuple
        (kx_min, kx_max) for x-axis of k-space
    ky_range : tuple
        (ky_min, ky_max) for y-axis of k-space
    num_points : int
        Number of points to sample in each k-direction
    band_indices : list or None
        If your solver returns multiple bands, specify which to plot (e.g., [0, 1])
        None will plot all bands
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Colormap name (e.g., 'viridis', 'plasma', 'coolwarm')
    show_colorbar : bool
        Whether to show colorbar
    elevation : float
        Viewing elevation angle in degrees
    azimuth : float
        Viewing azimuth angle in degrees
    highlight_degeneracies : bool
        Whether to highlight points where bands are degenerate
    degeneracy_threshold : float
        Energy difference below which bands are considered degenerate
    degeneracy_color : str
        Color for degeneracy points (e.g., 'red', 'yellow', 'cyan')
    degeneracy_size : float
        Size of degeneracy marker points
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    degeneracy_points : dict (only if highlight_degeneracies=True)
        Dictionary with 'kx', 'ky', 'energy' arrays of degeneracy locations
    """
    
    # Create k-space grid
    # Ensure (0,0) is included if the range spans zero
    if kx_range[0] < 0 < kx_range[1]:
        # Include 0 explicitly by using odd number of points
        if num_points % 2 == 0:
            num_points_x = num_points + 1
        else:
            num_points_x = num_points
        kx = np.linspace(kx_range[0], kx_range[1], num_points_x)
    else:
        kx = np.linspace(kx_range[0], kx_range[1], num_points)
    
    if ky_range[0] < 0 < ky_range[1]:
        if num_points % 2 == 0:
            num_points_y = num_points + 1
        else:
            num_points_y = num_points
        ky = np.linspace(ky_range[0], ky_range[1], num_points_y)
    else:
        ky = np.linspace(ky_range[0], ky_range[1], num_points)
    
    KX, KY = np.meshgrid(kx, ky)
    actual_num_points_x = len(kx)
    actual_num_points_y = len(ky)
    
    # Compute energies on the grid
    print(f"Computing energies on {actual_num_points_x}x{actual_num_points_y} grid...")
    
    # Initialize degeneracy tracking
    degeneracy_points = {'kx': [], 'ky': [], 'energy': []}
    
    # Check if solver returns single or multiple bands
    test_result = hamiltonian_solver(kx[0], ky[0], 0)[0]
    if np.isscalar(test_result):
        # Single band case
        energies = np.zeros((actual_num_points_y, actual_num_points_x))
        for i in range(actual_num_points_y):
            for j in range(actual_num_points_x):
                energies[i, j] = hamiltonian_solver(KX[i, j], KY[i, j], 0)[0]
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create surface plot
        surf = ax.plot_surface(KX, KY, energies, cmap=cmap, 
                              linewidth=0, antialiased=True, alpha=0.9)
        
        if show_colorbar:
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Energy')
        
    else:
        # Multiple bands case
        num_bands = len(test_result)
        if band_indices is None:
            band_indices = list(range(num_bands))
        
        energies = np.zeros((len(band_indices), actual_num_points_y, actual_num_points_x))
        
        for i in range(actual_num_points_y):
            for j in range(actual_num_points_x):
                E = hamiltonian_solver(KX[i, j], KY[i, j], 0)[0]
                for b_idx, band in enumerate(band_indices):
                    energies[b_idx, i, j] = E[band]
                
                # Check for degeneracies if requested
                if highlight_degeneracies and len(band_indices) > 1:
                    # Check all pairs of bands
                    for b1 in range(len(band_indices)):
                        for b2 in range(b1 + 1, len(band_indices)):
                            energy_diff = abs(energies[b1, i, j] - energies[b2, i, j])
                            if energy_diff < degeneracy_threshold:
                                degeneracy_points['kx'].append(KX[i, j])
                                degeneracy_points['ky'].append(KY[i, j])
                                # Use average energy of degenerate bands
                                degeneracy_points['energy'].append(
                                    (energies[b1, i, j] + energies[b2, i, j]) / 2
                                )
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each band with different color
        colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(band_indices)))
        
        for b_idx, band in enumerate(band_indices):
            surf = ax.plot_surface(KX, KY, energies[b_idx], 
                                  color=colors[b_idx], 
                                  linewidth=0, antialiased=True, 
                                  alpha=0.7, label=f'Band {band}')
    
    # Set labels and title
    ax.set_xlabel('$k_x$', fontsize=12)
    ax.set_ylabel('$k_y$', fontsize=12)
    ax.set_zlabel('Energy', fontsize=12)
    ax.set_title('3D Band Structure', fontsize=14, fontweight='bold')
    
    # Highlight degeneracy points if requested
    if highlight_degeneracies and len(degeneracy_points['kx']) > 0:
        ax.scatter(degeneracy_points['kx'], 
                  degeneracy_points['ky'], 
                  degeneracy_points['energy'],
                  c=degeneracy_color, 
                  s=degeneracy_size, 
                  marker='o',
                  edgecolors='black',
                  linewidths=1,
                  label='Degeneracy',
                  zorder=10,
                  alpha=0.9)
        print(f"Found {len(degeneracy_points['kx'])} degeneracy points")
        ax.legend(loc='upper right')
    
    # Set viewing angle
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if highlight_degeneracies:
        return fig, ax, degeneracy_points
    else:
        return fig, ax


# Example usage with a simple 2D tight-binding model
def example_hamiltonian(kx, ky, t=1.0):
    """
    Example: Simple 2D tight-binding dispersion
    E(k) = -2t(cos(kx) + cos(ky))
    
    Returns single band energy.
    """
    return -2 * t * (np.cos(kx) + np.cos(ky))


def example_two_band_hamiltonian(kx, ky, t=1.0, delta=0.5):
    """
    Example: Two-band model (e.g., simplified graphene-like)
    
    Returns array of energies for two bands.
    """
    # Simple model with avoided crossing
    h0 = -2 * t * (np.cos(kx) + np.cos(ky))
    h1 = t * (np.sin(kx) + np.sin(ky))
    
    # Eigenvalues of 2x2 Hamiltonian
    E_plus = h0 + np.sqrt(delta**2 + h1**2)
    E_minus = h0 - np.sqrt(delta**2 + h1**2)
    
    return np.array([E_minus, E_plus])


# if __name__ == "__main__":
#     print("Example 1: Single band structure")
#     fig1, ax1 = plot_bandstructure_3d(
#         hamiltonian_solver=example_hamiltonian,
#         kx_range=(-np.pi, np.pi),
#         ky_range=(-np.pi, np.pi),
#         num_points=50,
#         cmap='viridis',
#         elevation=25,
#         azimuth=45
#     )
#     plt.savefig('/mnt/user-data/outputs/bandstructure_single_band.png', dpi=150, bbox_inches='tight')
#     print("Saved: bandstructure_single_band.png")
    
#     print("\nExample 2: Two-band structure")
#     fig2, ax2 = plot_bandstructure_3d(
#         hamiltonian_solver=example_two_band_hamiltonian,
#         kx_range=(-np.pi, np.pi),
#         ky_range=(-np.pi, np.pi),
#         num_points=50,
#         band_indices=[0, 1],  # Plot both bands
#         cmap='plasma',
#         elevation=25,
#         azimuth=45
#     )
#     plt.savefig('/mnt/user-data/outputs/bandstructure_two_bands.png', dpi=150, bbox_inches='tight')
#     print("Saved: bandstructure_two_bands.png")
    
#     plt.show()

from magnonics import *

hamil, lat, basis = Initialize.from_csv('archimedean/configs/elongated_triangular.csv')

hamil.HP_trafo(symbolic=True)

hamil.parameterize_hamil(1e-8)

# hamil.update_parameters({'Dz1': 0.1, 'Dz2': 0.1}, 1e-3)

fig, ax, degen_info = plot_bandstructure_3d(
    hamiltonian_solver=hamil.Bogoliubov_trafo,
    kx_range=(-np.pi, np.pi),
    ky_range=(-np.pi, np.pi),
    num_points=100,
    band_indices = [0,1],
    show_colorbar=True,
    elevation=0,             
    azimuth=45  
)

plt.show()

print(f"Found {len(degen_info['kx'])} degenerate points")
print(f"Locations: k = ({degen_info['kx']}, {degen_info['ky']})")
print(f"Energies: E = {degen_info['energy']}")


# plot the reciprocal lattice  and its HSP underneath it