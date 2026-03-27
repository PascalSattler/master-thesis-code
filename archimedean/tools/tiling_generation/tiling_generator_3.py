import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from collections import deque

class Tiling:
    def __init__(self, vertex_configuration:tuple, bond_length:float = 1, global_rotation_deg:float = 0):
        self.config = vertex_configuration
        self.bond_length = bond_length
        self.global_rotation = global_rotation_deg

        # spokes (bond vectors) originating from the central vertex (fixed at (0,0)), starting at angle 0 (or global_rotation), two copies respectively in standard (as given) and reversed config order to account for chirality of lattice (e.g. truncated trihexagonal)
        self.spoke_orders = {
            'standard': self._get_vertex_spokes('standard'), 
            'reversed':self._get_vertex_spokes('reversed')
        }

        # storage to keep track of calulated vertices and avoid duplicates, key is a unique position and value is an index
        self.vertex_registry = {}

        # list of vertex coordinates for plotting
        self.vertex_coords = []

        # track which order is used at each vertex
        self.track_order = {}

    @staticmethod
    def _internal_angle(n_sides:int):
        # internal angle of a regular polygon, all interior angels add up to (n-2)*pi
        return (n_sides - 2) * np.pi / n_sides
    
    @staticmethod
    def _map_angle(angle:float):
        # remap angle to the interval (-pi, pi]
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    @staticmethod
    def _chord_length(n_sides:int, side_length:float, vertices_inbetween:int = 2):
        # for a polygon with n sides the chord length is the secant intersecting the circumcircle at two vertices
        # for two adjacent vertices it is equal to the side length 2*circumradius*sin(pi/n)
        # we will need the chord between two vertices separated by two vertices inbetween -> 2*circumradius*sin((k+1)*pi/n)
        # https://de.wikipedia.org/wiki/Regelm%C3%A4%C3%9Figes_Polygon#Diagonalen
        if n_sides < 3: return 0

        def _circumradius(n_sides:int, side_length:float):
            # vertex to center distance of a polygon -> circumradius R = a/(2sin(pi/n)) i.e. the radius of a circle inscribing the polygon
            return side_length / (2 * np.sin(np.pi / n_sides))
        
        return 2 * _circumradius(n_sides, side_length) * np.sin((vertices_inbetween+1) * np.pi / n_sides)
    
    @staticmethod
    def _unit_vec(angle:float):
        # unit vector in polac coordinates
        return np.array([np.cos(angle), np.sin(angle)])
    
    def _get_vertex_spokes(self, order:str = 'standard'):
        # calculate angles of the spokes (bonds) originating from the central vertex and determine the polygons adjacent to them

        spokes = []
        current_angle = 0.0 + self.global_rotation * np.pi/180

        if order == 'standard':
            config_ordered = self.config
        elif order == 'reversed':
            config_ordered = self.config[::-1]
        else:
            raise ValueError("WARNING: Config order can only be standard (as given) or reversed.")
        
        n_polygons = len(config_ordered)

        config_state = deque(config_ordered)
        config_state.rotate(1)

        # determine all spokes of the vertex figure of the given config
        for id in range(n_polygons):
            # determine a spokes adjacent polygons in counter-clockwise order of appearence
            next_polygon = config_ordered[id]
            previous_polygon = config_ordered[id-1]

            config_state.rotate(-1)
            
            # store attributes of each spoke in a dictionary
            spokes.append({
                'index': id,                            # spoke id <-> next polygon
                'angle': current_angle,                 # angle of spoke w.r.t. horizontal
                'adjacent_next': next_polygon,          # next polygon in config order (current spoke id)
                'adjacent_previous': previous_polygon,  # previous polygon in config order (previous spoke id)
                'order': order,                         # standard or reversed
                'angle_to_next': None,                  # angle to next spoke
                'angle_to_previous': None,              # angle to previous spoke
                'config_state': [*config_state]
            })

            # advance to the next spoke by internal angle of next adjacent polygon
            current_angle += self._internal_angle(next_polygon)

        # determine the angle differences to neighboring spokes -> difference of internal angles
        for id, spoke in enumerate(spokes):
            # modulo divison for looping config order
            next_id = (id+1) % n_polygons
            previous_id = (id-1) % n_polygons

            # map the angles onto (-pi, pi] because no inner polygon angle can be greater than 180°
            spoke['angle_to_next'] = self._map_angle(spokes[next_id]['angle'] - spoke['angle'])
            spoke['angle_to_previous'] = self._map_angle(spokes[previous_id]['angle'] - spoke['angle'])

        return spokes
    
    def _get_registry_key(self, vertex: np.ndarray):
        # round vertex coordinates to get unique key for avoiding duplicate storage 
        return (np.round(vertex[0], decimals=4), np.round(vertex[1], decimals=4))
    
    def generate(self, number_of_vertices:int = 400):
        # generate all vertices from vertex config

        # check if config is possible
        exterior_sum = np.sum([self._internal_angle(n) for n in self.config])
        if not np.isclose(exterior_sum, 2*np.pi):
            raise ValueError(f"WARNING: Angles sum to {(exterior_sum*180/np.pi):.1f}°, not 360°. Not a valid vertex configuration.")

        # generate the tiling starting from a central vertex figure (origin) and initialize in storage
        origin = np.array([0.0, 0.0])
        self.vertex_registry[self._get_registry_key(origin)] = 0
        self.vertex_coords.append(origin)
        self.track_order[0] = 'standard'

        # initialize a queue to track where the current vertex (first entry) is placed on the current spoke (third entry) rotated around a reference vertex (second entry)
        # origin has id = 0 but no reference vertex or spoke
        queue = deque([(0, None, None)])

        # iterate in order of the queue entries until the maximum number of vertices has been calculated
        while queue and len(self.vertex_coords) < number_of_vertices:
            # get current vertex index and index of the reference vertex the spokes are rotated around, then remove them from the queue because all new vertices to this reference have been found
            current_id, reference_id, current_spoke = queue.popleft()

            # get current vertex coordinates from corresponding index 
            current_coord = self.vertex_coords[current_id]

            if reference_id is None:
                # begin with standard config order and no rotation
                selected_order = self.spoke_orders['standard']
                rotation = 0.0
            else:
                # get reference vertex coordinates from corresponding index (if it exists)
                reference_coord = self.vertex_coords[reference_id]
                reference_order = self.track_order[reference_id]

                # bond vector pointing back from reference to current vertex and its angle w.r.t to the horizontal -> add pi because it points backwards
                bond_vec = reference_coord - current_coord
                bond_angle = np.arctan2(bond_vec[1], bond_vec[0]) #arctan2 for correct quadrants
                angle_to_horizontal = bond_angle + np.pi

                # determine the neighboring polygons of the current spoke
                # angles w.r.t horizontal
                angle_next_from_ref = angle_to_horizontal + current_spoke['angle_to_next']
                coord_next_from_ref = reference_coord + self.bond_length * self._unit_vec(angle_next_from_ref)

                angle_prev_from_ref = angle_to_horizontal + current_spoke['angle_to_previous']
                coord_prev_from_ref = reference_coord + self.bond_length * self._unit_vec(angle_prev_from_ref)
 
                # match is searched for in both config orders
                # the first entry in the priority list is the first order tested in the loop, which is always the order of the reference vertex
                # if a match is found in this order, exit the loop, otherwise we check in the other order

                order_priority = [reference_order]
                if reference_order == 'standard': order_priority.append('reversed')
                else: order_priority.append('standard')

                match_found = False

                for order in order_priority:
                    if match_found: break # exit loop if match is found

                    # once order is selected, find spoke with minimal rotation
                    spoke_order = self.spoke_orders[order]
                    candidates = [] # store the possible rotation and corresponding spoke

                    for spoke in spoke_order:
                        # first condition: the next neighboring vertex to the reference vertex on side of the bond following ccw cyclic order (i.e. "left" of it) must be the same next neighbor to the current vertex but on its right (cw order) of the bond -> inversed matches
                        # this is to make sure that next polygon in the config order matches from both perspectives
                        if spoke['adjacent_next'] == current_spoke['adjacent_previous'] and spoke['adjacent_previous'] == current_spoke['adjacent_next']:
                            
                            test_rotation = bond_angle - spoke['angle']

                            # second condition: even if both perspectives match, the minimal rotation could be ambiguous when multiple identical polygons follow each other in the config order
                            # we test for the correct rotation of the current vertex's figure by aligning a canditate spoke with the bond and checking if its neighbors also align with the reference vertex's neighbors
                            # the neighbors to be tested, let's call them N and M, will have a certain distance depending of the type of polygon: for a triangle they are the same, hence their distance is 0, for a square they are separated by exactly one bond length, for a hexagon by two bond lengths, and so on
                            # this distance must be equal to the length of a chord (or diagonal) cutting through the polygon, where the chord spans the vertices N -> ref -> cur -> M that lie on the perimeter of the polygon: this is again 0 for triangles, 1 for squares, 2 for hexagons, etc.
                            # we check this for the neighbors on both sides of the bond
                            angle_prev_from_cur = spoke['angle'] + spoke['angle_to_previous'] + test_rotation
                            coord_prev_from_cur = current_coord + self.bond_length * self._unit_vec(angle_prev_from_cur)
                            
                            distance_match1 = np.linalg.norm(coord_next_from_ref - coord_prev_from_cur)
                            distance_required1 = self._chord_length(current_spoke['adjacent_next'], self.bond_length)

                            angle_next_from_cur = spoke['angle'] + spoke['angle_to_next'] + test_rotation
                            coord_next_from_cur = current_coord + self.bond_length * self._unit_vec(angle_next_from_cur)
                            
                            distance_match2 = np.linalg.norm(coord_prev_from_ref - coord_next_from_cur)
                            distance_required2 = self._chord_length(current_spoke['adjacent_previous'], self.bond_length)

                            # only if both conditions are met the candidate is considered
                            if abs(distance_match1 - distance_required1) < 1e-3 and abs(distance_match2 - distance_required2) < 1e-3:
                                candidates.append((abs(self._map_angle(test_rotation)), test_rotation, spoke))

                    # a final check to resolve potential ambiguity of the candidates
                    if candidates:
                        candidates.sort(key=lambda x: x[0]) # sort the list using the first element (here the rotation) of each entry as the sort key

                        best_match = candidates[0] # first entry after sorting has smallest rotation
                        rotation = best_match[1]
                        selected_order = spoke_order # order of config was correct, thus continue with it
                        self.track_order[current_id] = order
                        match_found = True

                if not match_found: continue

            # find the next new vertices from the current vertex
            for spoke in selected_order:
                # get the total angle after rotation of a spoke
                total_angle = spoke['angle'] + rotation

                # calculate the next vertex coordinate
                new_coord = current_coord + self.bond_length * self._unit_vec(total_angle)

                # generate a new registry key and check if vertex already exists
                check_key = self._get_registry_key(new_coord)

                if check_key not in self.vertex_registry:
                    # does not yet exist, create new vertex
                    new_id = len(self.vertex_coords)
                    self.vertex_registry[check_key] = new_id
                    self.vertex_coords.append(new_coord)

                    # add new pair to slider
                    queue.append((new_id, current_id, spoke))
                else:
                    # already exists, do nothing
                    pass

    def plot(self, mark_central_vertex:bool = True, show_edges:bool = True):
        # plot the tiling
        
        coords = np.array(self.vertex_coords)

        plt.figure(figsize=(6, 6))
    
        # mark central vertex
        if mark_central_vertex:
            plt.scatter(coords[0,0], coords[0,1], c='red', s=50, zorder=10)

        # plot vertices
        plt.scatter(coords[:,0], coords[:,1], c='black', s=25)

        # plot edges by finding nearest neighbors using KDTree ball search method which has time complexity O(logN) compared to O(n^2) of a brute force approach
        if show_edges:
            tree = KDTree(coords)
            search_radius = self.bond_length * (1 + 1e-5) # add small epsilon for tolerance 

            edges = set() # use undirected set to avoid drawing edges twice (automatic duplicate removal)

            for i, point in enumerate(coords):
                neighbors = tree.query_ball_point(point, r=search_radius, workers=4) # search for indices of NN inside distance of side length 

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
        plt.show()



archimedean = {
    'triangular' : [(3,3,3,3,3,3), r'$(3^6)$'],
    'square' : [(4,4,4,4), r'$(4^4)$'],
    'hexagonal' : [(6,6,6), r'$(6^3)$'],
    'kagome' : [(3,6,3,6), r'$(3.6.3.6)$'],
    'rhombitrihexagonal' : [(3,4,6,4), r'$(3.4.6.4)$'],
    'truncated_square' : [(4,8,8), r'$(4.8^2)$'],
    'truncated_hexagonal' : [(3,12,12), r'$(3.12^2)$'],
    'truncated_trihexagonal' : [(4,6,12), r'$(4.6.12)$'],
    'snub_square' : [(3,3,4,3,4), r'$(3^2.4.3.4)$'],
    'snub_trihexagonal' : [(3,3,3,3,6), r'$(3^4.6)$'],
    'elongated_triangular' : [(3,3,3,4,4), r'$(3^3.4^2)$'],
}
# snub_square, snub_trihexagonal, elongated_triangular

tiling = archimedean.get('elongated_triangular')[0]

lattice = Tiling(tiling, global_rotation_deg=0)
lattice.generate()
lattice.plot(show_edges=True)