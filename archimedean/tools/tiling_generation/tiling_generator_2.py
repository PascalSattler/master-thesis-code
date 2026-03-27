from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from collections import deque

class Tiling:
    def __init__(self, config:tuple, bond_length:float = 1, global_rotation:float = 0):
        self.config = config
        self.bond_length = bond_length
        self.global_rotation = global_rotation

        # angles between the bond vectors originating from the central vertex (fixed at (0,0)), starting at 0 (or global_rotation)
        self.vertex_spokes = self._get_vertex_spokes()

        # storage to keep track of calulated vertices and avoid duplicates, key is a unique position and value is an index
        self.vertex_registry = {}

        # list of vertex coordinates for plotting
        self.vertex_coords = []

    @staticmethod
    def _get_internal_angle(n_sides:int):
        # internal angle of a regular polygon, all interior angels add up to (n-2)*pi
        return (n_sides - 2) * np.pi / n_sides
    
    def _get_circumradius(self, n_sides:int):
        # vertex to center distance of a polygon -> circumradius R = a/(2sin(pi/n)) i.e. the radius of a circle inscribing the polygon
        return self.bond_length / (2 * np.sin(np.pi / n_sides))
    
    def _get_vertex_spokes(self):
        # calculate inner angles of the bonds (spokes) originating from the origin and determine the polygons adjacent to them
        
        spokes = []
        current = 0 + self.global_rotation * np.pi/180

        for i in range(len(self.config)):
            #determine adjacent polygons
            left_polygon = self.config[i] # counter clock-wise
            right_polygon = self.config[i-1] # clock-wise
            
            # store information of each spoke
            spokes.append({
                'angle': current,
                'left': left_polygon,
                'right': right_polygon,
                'index': i
            })

            # advance by internal angle of polygon on the left (next i n ccw order)
            internal = self._get_internal_angle(left_polygon)
            current += internal

        return spokes
    
    def _get_registry_key(self, vertex: np.ndarray):
        # round vertex coordinates to get unique key for avoiding duplicate storage 
        return (np.round(vertex[0], decimals=4), np.round(vertex[1], decimals=4))
    
    def generate(self, steps:int = 30):
        # generate all vertices from vertex config

        # check if config is possible
        exterior_sum = np.sum([self._get_internal_angle(n) for n in self.config])
        if not np.isclose(exterior_sum, 2*np.pi):
            raise ValueError(f"WARNING: Angles sum to {(exterior_sum*180/np.pi):.1f}°, not 360°. Not a valid vertex configuration.")
        
        # calculate total number of vertices in the vertex figure given by the config and then multiply by the amount of steps to repeat the vertex figure
        #n_vertices = steps*np.sum([n for n in self.config])

        # generate the tiling starting from a central vertex figure (origin) and initialize in storage
        origin = np.array([0.0, 0.0])
        self.vertex_registry[self._get_registry_key(origin)] = 0
        self.vertex_coords.append(origin)

        # initialize a "slider" where: the current vertex (first entry) is placed on the last spoke (third entry) rotated around a reference vertex (second entry)
        slider = deque([(0, None, None)]) 

        # iterate in order of the slider entries until the maximum number of vertices has been calculated
        while slider and len(self.vertex_coords) < steps:
            # get current vertex index and index of the reference vertex the spokes are rotated around, then remove them from slider 
            current_id, reference_id, last_spoke = slider.popleft()

            # get current vertex coordinates from corresponding index 
            current_coord = self.vertex_coords[current_id]
            if current_id < 11:
                print(current_id, reference_id, last_spoke)
                print(current_coord)

            # initialize starting spoke rotation if no reference vertex exists
            rotation = 0.0

            if reference_id is not None:
                # get reference vertex coordinates from corresponding index (if it exists)
                reference_coord = self.vertex_coords[reference_id]

                # bond vector pointing back to previous vertex and its angle starting from 0
                bond_vec = reference_coord - current_coord
                bond_angle = np.arctan2(bond_vec[1], bond_vec[0])

                # next spoke must have matching adjacent polygons
                match_left = last_spoke['right']
                match_right = last_spoke['left']
 
                find_match = None

                for spoke in self.vertex_spokes:
                    if spoke['left'] == match_left and spoke['right'] == match_right:
                        find_match = spoke
                        break 

                if find_match is None:
                    continue 

                # the bond vector has to align with the matching last spoke originating from the reference vertex
                # thus we have to find the rotation such that vertex_angle + rotation = bond_angle
                # to minimize the rotation angle it suffices to rotate up to the first vertex angle
                rotation = bond_angle - find_match['angle']

            # find the neighboring vertices to the current vertex
            for spoke in self.vertex_spokes:
                # get the current angle after rotation
                current_angle = spoke['angle'] + rotation

                # calulate the neighbor coordinates
                x = current_coord[0] + self.bond_length*np.cos(current_angle)
                y = current_coord[1] + self.bond_length*np.sin(current_angle)
                new_coord = np.array([x,y])

                # generate a new registry key and check if neighbor already exists
                check_key = self._get_registry_key(new_coord)

                if check_key in self.vertex_registry:
                    # already exists, do nothing
                    continue # new_id = self.vertex_registry[check_key]
                else:
                    # does not yet exist, create new point
                    new_id = len(self.vertex_coords)
                    self.vertex_registry[check_key] = new_id
                    self.vertex_coords.append(new_coord)

                    # add new pair to slider
                    slider.append((new_id, current_id, spoke))

    def plot(self, mark_central_vertex:bool = True, show_edges:bool = True):
        # plot the tiling
        
        coords = np.array(self.vertex_coords)

        plt.figure(figsize=(6, 6))
    
        # mark central vertex
        if mark_central_vertex:
            plt.scatter(coords[0,0], coords[0,1], c='red', s=50, zorder=10)

        # plot vertices
        plt.scatter(coords[:,0], coords[:,1], c='black', s=25)

        # plot edges by finding nearest neighbors using KDTree ball search method 
        if show_edges:
            tree = KDTree(coords)
            search_radius = self.bond_length * (1 + 1e-5) # add small epsilon for tolerance 

            edges = set() # use undirected set to avoid drawing edges twice (automatic duplicate removal)

            for i, point in enumerate(coords):
                neighbors = tree.query_ball_point(point, r=search_radius) # search for indices of NN inside distance of side length 

                for j in neighbors: # exclude the vertex itself
                    if j == i:
                        continue
                    k, l = sorted((i, j)) # sort to avoid duplicate storing
                    edges.add((k, l))

            for i, j in edges:
                plt.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]], color='black', linewidth=0.6)
        else:
            pass
        
        # if label==None:
        #     label = f"{config}"
        # plt.title("Vertex Configuration: " + label, fontsize = 14)
        plt.axis('equal')
        plt.axis('off')
        plt.grid(False, alpha=0.3)
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
# snub_square, snub_trihexagonal, truncated_trihexagonal, elongated_triangular

tiling = archimedean.get('kagome')[0]

lattice = Tiling(tiling, global_rotation=0)
lattice.generate()
lattice.plot(show_edges=False)