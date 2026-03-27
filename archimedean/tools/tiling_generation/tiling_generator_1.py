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
        self.vertex_angles = self._get_vertex_angles()

        # storage to keep track of calulated vertices and avoid duplicates, key is a unique position and value is an index
        self.vertex_registry = {}

        # list of vertex coordinates for plotting
        self.vertex_coords = []

    def _get_internal_angle(self, n_sides:int):
        # internal angle of a regular polygon, all interior angels add up to (n-2)*pi
        return (n_sides - 2) * np.pi / n_sides
    
    def _get_circumradius(self, n_sides:int):
        # vertex to center distance of a polygon -> circumradius R = a/(2sin(pi/n)) i.e. the radius of a circle inscribing the polygon
        return self.bond_length / (2 * np.sin(np.pi / n_sides))
    
    def _get_vertex_angles(self):
        # calculate inner angles of the polygons around a vertex
        
        angles = []
        current = 0 + self.global_rotation * np.pi/180

        for n in self.config:
            angles.append(current)
            internal = self._get_internal_angle(n)
            current += internal

        return np.array(angles)
    
    def _get_registry_key(self, vertex: np.ndarray):
        # round vertex coordinates to get unique key for avoiding duplicate storage 
        return (np.round(vertex[0], decimals=4), np.round(vertex[1], decimals=4))
    
    def generate(self, steps:int = 20):
        # generate all vertices from vertex config

        # check if config is possible
        exterior_sum = np.sum([self._get_internal_angle(n) for n in self.config])
        if not np.isclose(exterior_sum, 2*np.pi):
            raise ValueError(f"WARNING: Angles sum to {(exterior_sum*180/np.pi):.1f}°, not 360°. Not a valid vertex configuration.")
        
        # calculate total number of vertices in the vertex figure given by the config and then multiply by the amount of steps to repeat the vertex figure
        n_vertices = steps*np.sum([n for n in self.config])

        # generate the tiling starting from a central vertex figure (origin) and initialize in storage
        origin = np.array([0.0, 0.0])
        self.vertex_registry[self._get_registry_key(origin)] = 0
        self.vertex_coords.append(origin)

        # initialize a "slider" to go from previous vertex (None at start) to current vertex
        slider = deque([(0, None)]) 

        # iterate in order of the slider entries until the maximum number of vertices has been calculated
        while slider and len(self.vertex_coords) < n_vertices:
            # get current and previous index and remove them from slider 
            current_id, previous_id = slider.popleft()

            # get current coordinates from corresponding index
            current_coord = self.vertex_coords[current_id]

            # initialize starting rotation if no previous vertex exists
            rotation = 0.0

            if previous_id is not None:
                # get previous coordinates from corresponding index if it exists
                previous_coord = self.vertex_coords[previous_id]

                # bond vector pointing back to previous vertex and its angle starting from 0
                bond_vec = previous_coord - current_coord
                bond_angle = np.arctan2(bond_vec[1], bond_vec[0])

                # this bond vector has to align with one originating from the vertex figure given by the config
                # thus we have to find the rotation such that vertex_angle + rotation = bond_angle
                # to minimize the rotation angle it suffices to rotate up to the first vertex angle
                rotation = bond_angle - self.vertex_angles[0]

            # find the neighboring vertices to the current vertex
            for angle in self.vertex_angles:
                # get the current angle after rotation
                current_angle = angle + rotation

                # calulate the neighbor coordinates
                x = current_coord[0] + self.bond_length*np.cos(current_angle)
                y = current_coord[1] + self.bond_length*np.sin(current_angle)
                new_coord = np.array([x,y])

                # generate a new registry key and check if neighbor already exists
                check_key = self._get_registry_key(new_coord)

                if check_key in self.vertex_registry:
                    # already exists, do nothing
                    pass # new_id = self.vertex_registry[check_key]
                else:
                    # does not yet exist, create new point
                    new_id = len(self.vertex_coords)
                    self.vertex_registry[check_key] = new_id
                    self.vertex_coords.append(new_coord)

                    # add new pair to slider
                    slider.append((new_id, current_id))

    def plot(self, mark_central_vertex:bool = True, show_edges:bool = True):
        # plot the tiling
        
        coords = np.array(self.vertex_coords)

        plt.figure(figsize=(6, 6))
    
        # mark central vertex
        if mark_central_vertex:
            plt.scatter(coords[0,0], coords[0,1], c='red', s=100, zorder=10)

        # plot vertices
        plt.scatter(coords[:,0], coords[:,1], c='black', s=50)

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


kagome = (3,6,3,6)

lattice = Tiling(kagome)
lattice.generate()
lattice.plot(show_edges=False)