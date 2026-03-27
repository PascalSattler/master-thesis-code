from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial import KDTree

# goal: create lattices as instances of a class for a given vertex config, with methods like .coords, .plot, .ribbon, etc.

# get lmodern font from LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{lmodern}"
})

def internal_angle(n_sides:int):
    # internal angle of a regular polygon, all interior angels add up to (n-2)*pi
    return (n_sides - 2) * np.pi / n_sides

def circumradius(n_sides:int, side_length:float):
    # vertex to center distance of a polygon -> circumradius R = a/(2sin(pi/n)) i.e. the radius of a circle inscribing the polygon
    return side_length / (2 * np.sin(np.pi / n_sides))

def get_polygon_coords(n_sides:int, side_length:float, center:tuple = (0,0), rotation_deg:float = 0):

    if n_sides < 3:
        raise ValueError("Polygon must have at least 3 sides.")

    radius = circumradius(n_sides, side_length)

    # angles around the circumcircle
    angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False) + rotation_deg * np.pi/180

    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)

    return np.column_stack((x,y))

def generate_vertices(config:tuple, side_length:float = 1, central_vertex:tuple = (0,0), global_rotation:float = 0):

    # check if config is possible
    # exterior angles around the vertex must sum to 2*pi
    # adjacent polygons add up their internal angles, which are the exterior angles around the vertex
    exterior_sum = np.sum([internal_angle(n) for n in config])
    if not np.isclose(exterior_sum, 2*np.pi):
        raise ValueError(f"WARNING: Angles sum to {(exterior_sum*180/np.pi):.1f}°, not 360°. Not a valid vertex configuration.")

    # total number of vertices of the config
    # n_vertices = np.sum([n for n in config])

    # initialize list of tiling coordinates
    tiling_coords = []

    # angle by which the polygons positions get rotated around the vertex
    current_angle = 0 + global_rotation * np.pi/180

    # subsequently arranging the polygons around central vertex
    for n in config:

        # 1. calculate internal angle for current polygon n in config
        internal = internal_angle(n)

        # 2. calculate the bisector angle of the interval [current_angle, internal_angle] -> current + internal/2
        bisector = current_angle + internal/2

        # 3. place polygon center at distance of circumradius from central vertex on bisector
        distance = circumradius(n, side_length)
        polygon_center = (distance * np.cos(bisector) + central_vertex[0], distance * np.sin(bisector) + central_vertex[1])

        # 4. rotate polygon such that one vertex aligns with central vertex
        # first vertex of polygon will be placed at polygon_center + (side_length,0)
        # to align this vertex onto the center_vertex, rotate onto the bisector and then by 180° to the opposite point on the bisector
        polygon_rotation = bisector * 180/np.pi + 180

        polygon_coords = get_polygon_coords(n, side_length, center=polygon_center, rotation_deg=polygon_rotation)
        tiling_coords.append(polygon_coords)
        

        # 5. update current angle -> add internal of previous polygon
        current_angle += internal
    
    # make sure the shape of the stack is uniform 
    stacked_list = np.vstack(tiling_coords)

    # round down the coordinates to get rid of orders ~ e-10 and below
    rounded = np.round(stacked_list, decimals=8)

    # remove duplicate (overlapping) vertices for more efficient plotting
    unique_coords = np.sort(np.unique(rounded, axis=0, return_index=True)[1])
    no_duplicates = rounded[unique_coords]
    
    return no_duplicates

def plot_vertex_figure(config:tuple, side_length:float = 1, central_vertex:tuple = (0,0), global_rotation:float = 0, mark_central_vertex:bool = True, show_edges:bool = True, label=None, filename: str = None):

    vertex_coords = generate_vertices(config, side_length=side_length, central_vertex=central_vertex, global_rotation = global_rotation)

    plt.figure(figsize=(6, 6))

    cmap = plt.get_cmap('coolwarm')
    cnorm = Normalize(vmin=0, vmax=1)
    color= cmap(cnorm(0.95))
    
    # mark central vertex
    if mark_central_vertex:
        plt.scatter(vertex_coords[0,0], vertex_coords[0,1], c=color, s=500, zorder=10)

    # plot vertices
    plt.scatter(vertex_coords[:,0], vertex_coords[:,1], c='black', s=100)

    # plot edges by finding nearest neighbors using KDTree ball search method 
    if show_edges:
        tree = KDTree(vertex_coords)
        search_radius = side_length * (1 + 1e-5) # add small epsilon for tolerance 

        edges = set() # use undirected set to avoid drawing edges twice (automatic duplicate removal)

        for i, point in enumerate(vertex_coords):
            neighbors = tree.query_ball_point(point, r=search_radius) # search for indices of NN inside distance of side length 

            for j in neighbors: # exclude the vertex itself
                if j == i:
                    continue
                k, l = sorted((i, j)) # sort to avoid duplicate storing
                edges.add((k, l))

        for i, j in edges:
            plt.plot([vertex_coords[i,0], vertex_coords[j,0]], [vertex_coords[i,1], vertex_coords[j,1]], color='black', linewidth=3)
    else:
        pass
    
    if label is not None:
        # label = f"{config}"
        plt.title("Vertex Configuration: " + label, fontsize = 14)
    plt.axis('equal')
    plt.axis('off')
    plt.grid(False)
    if filename is not None:
        plt.savefig(f'archimedean/tools/tiling_generation/minimal_figures/{filename}.pdf')
    plt.show()



archimedean = {
    'triangular' : [(3,3,3,3,3,3), r'$(3^6)$'],
    'square' : [(4,4,4,4), r'$(4^4)$'],
    'hexagonal' : [(6,6,6), r'$(6^3)$'],
    'kagome' : [(3,6,3,6), r'$(3.6.3.6)$'],
    'snub_square' : [(3,3,4,3,4), r'$(3^2.4.3.4)$'],
    'truncated_square' : [(4,8,8), r'$(4.8^2)$'],
    'truncated_hexagonal' : [(3,12,12), r'$(3.12^2)$'],
    'snub_trihexagonal' : [(3,3,3,3,6), r'$(3^4.6)$'],
    'truncated_trihexagonal' : [(4,6,12), r'$(4.6.12)$'],
    'rhombitrihexagonal' : [(3,4,6,4), r'$(3.4.6.4)$'],
    'elongated_triangular' : [(3,3,3,4,4), r'$(3^3.4^2)$'],
}

# fig, ax = plt.subplots(figsize=(6,6))

# triangle = get_polygon_coords(3, 1, rotation_deg=0)
# square = get_polygon_coords(4, 1, rotation_deg=0)
# hexagon = get_polygon_coords(6, 1, rotation_deg=0)

# ax.scatter(triangle[:,0], triangle[:,1])
# ax.scatter(square[:,0], square[:,1])
# ax.scatter(hexagon[:,0], hexagon[:,1])

# vertex_figure = generate_vertices(archimedean['kagome'])

# ax.scatter(vertex_figure[:,0], vertex_figure[:,1])

# plt.show()

# filename = 'hexagonal'
# tiling = archimedean.get(filename)

for key, value in archimedean.items():
    plot_vertex_figure(value[0], filename=key, global_rotation=0, central_vertex=(0,0)) # label=tiling[1]