import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from collections import deque

class Tiling:
    def __init__(self, vertex_configuration:tuple, bond_length:float = 1, global_rotation_deg:float = 0):
        self.config = vertex_configuration
        self.n_polygons = len(self.config)
        self.bond_length = bond_length
        self.global_rotation = global_rotation_deg

        # check if config is possible
        exterior_sum = np.sum([self._internal_angle(n) for n in self.config])
        if not np.isclose(exterior_sum, 2*np.pi):
            raise ValueError(f"WARNING: Angles sum to {(exterior_sum*180/np.pi):.1f}°, not 360°.")

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
    def _norm_angle(angle:float):
        # remap angle to the interval (-pi, pi]
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    @staticmethod
    def _chord_length(n_sides:int, side_length:float, vertices_inbetween:int = 2):
        # for a polygon with n sides the chord length is the connecting line between two of its vertices
        # for two adjacent vertices it is equal to the side length 2*circumradius*sin(pi/n)
        # we will need the chord between two vertices separated by another two vertices along the perimeter -> 2*circumradius*sin((k+1)*pi/n) for k = 2
        # https://de.wikipedia.org/wiki/Regelm%C3%A4%C3%9Figes_Polygon#Diagonalen
        
        def _circumradius(n_sides:int, side_length:float):
            # vertex to center distance of a polygon -> circumradius R = a/(2sin(pi/n)) i.e. the radius of a circle inscribing the polygon
            return side_length / (2 * np.sin(np.pi / n_sides))
        
        if n_sides <= 3: return 0
        else: return 2 * _circumradius(n_sides, side_length) * np.sin((vertices_inbetween+1) * np.pi / n_sides)
    
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

        config_state = deque(config_ordered)

        # determine all spokes of the vertex figure of the given config
        for id, polygon in enumerate(config_ordered):
            # determine a spokes adjacent polygons in counter-clockwise order of appearence

            state = tuple([*config_state])
            next = self._internal_angle(state[0])
            prev = -self._internal_angle(state[-1])

            # store attributes of each spoke in a dictionary
            spokes.append({
                'index': id,
                'order': order,
                'config_state': state,
                'angle': current_angle,
                'angle_to_next': next,
                'angle_to_prev': prev
            })

            config_state.rotate(-1)

            # advance to the next spoke by internal angle of next adjacent polygon
            current_angle += self._internal_angle(polygon)

        return spokes
    
    def _get_registry_key(self, vertex: np.ndarray):
        # round vertex coordinates to get unique key for avoiding duplicate storage 
        return (np.round(vertex[0], decimals=4), np.round(vertex[1], decimals=4))
    
    def generate(self, number_of_vertices:int = 300):
        # start setup
        origin = np.array([0.0, 0.0])
        self.vertex_registry[self._get_registry_key(origin)] = 0
        self.vertex_coords.append(origin)
        self.track_order[0] = 'standard'

        queue = deque([(0, None, None)])

        # candidate finding
        while queue and len(self.vertex_coords) < number_of_vertices:
            current_id, reference_id, active_spoke = queue.popleft()
            current_coord = self.vertex_coords[current_id]

            if reference_id is None:
                selected_order = self.spoke_orders['standard']
                rotation = 0.0
            else:
                reference_coord = self.vertex_coords[reference_id]
                reference_order = self.track_order[reference_id]

                bond_vec = current_coord - reference_coord
                bond_angle = np.arctan2(bond_vec[1], bond_vec[0])
                # angle_to_horizontal = bond_angle + np.pi
                # if (angle_to_horizontal - active_spoke['angle']) < 1e-3: print('yes')

                angle_next_from_ref = bond_angle + active_spoke['angle_to_next']
                coord_next_from_ref = reference_coord + self.bond_length * self._unit_vec(angle_next_from_ref)

                angle_prev_from_ref = bond_angle + active_spoke['angle_to_prev']
                coord_prev_from_ref = reference_coord + self.bond_length * self._unit_vec(angle_prev_from_ref)
 
                order_priority = [reference_order]
                if reference_order == 'standard': order_priority.append('reversed')
                else: order_priority.append('standard')

                match_found = False

                for order in order_priority:
                    if match_found: break

                    spoke_order = self.spoke_orders[order]
                    candidates = []

                    # Retrieve the cyclic state of the Reference spoke
                    # This is the sequence of polygons CCW around Ref starting from the bond
                    ref_state = active_spoke['config_state']

                    for cand_spoke in spoke_order:
                        cand_state = cand_spoke['config_state']
                        
                        # --- NEW: State Compatibility Check ---
                        # We compare the Reference Sequence (CCW) with the Candidate Sequence (CW).
                        # Since 'cand_state' is stored CCW, we must traverse it backwards to simulate CW.
                        # Ref[0] (Left of bond) must match Cand[-1] (Right of bond)
                        # Ref[1] must match Cand[-2], and so on.
                        
                        states_compatible = True
                        for k in range(self.n_polygons):
                            # Compare k-th element forward with k-th element backward
                            if ref_state[k] != cand_state[-(k+1)]:
                                states_compatible = False
                                break
                        
                        if not states_compatible:
                            continue

                        # --- Geometric Check (Chord Lengths) ---
                        # If states match, we verify geometry to find the correct rotation
                        test_rotation = bond_angle + np.pi - cand_spoke['angle']

                        angle_prev_from_cur = cand_spoke['angle'] + cand_spoke['angle_to_prev'] + test_rotation
                        coord_prev_from_cur = current_coord + self.bond_length * self._unit_vec(angle_prev_from_cur)
                        dist_1 = np.linalg.norm(coord_next_from_ref - coord_prev_from_cur)
                        req_1 = self._chord_length(active_spoke['config_state'][0], self.bond_length)

                        angle_next_from_cur = cand_spoke['angle'] + cand_spoke['angle_to_next'] + test_rotation
                        coord_next_from_cur = current_coord + self.bond_length * self._unit_vec(angle_next_from_cur)
                        dist_2 = np.linalg.norm(coord_prev_from_ref - coord_next_from_cur)
                        req_2 = self._chord_length(active_spoke['config_state'][-1], self.bond_length)

                        if abs(dist_1 - req_1) < 1e-3 and abs(dist_2 - req_2) < 1e-3:
                            candidates.append((abs(self._norm_angle(test_rotation)), test_rotation, cand_spoke))

                    if candidates:
                        candidates.sort(key=lambda x: x[0])
                        best_match = candidates[0]
                        rotation = best_match[1]
                        selected_order = spoke_order
                        self.track_order[current_id] = order
                        match_found = True

                if not match_found: continue

            # create new vertices
            for spoke in selected_order:
                total_angle = spoke['angle'] + rotation
                new_coord = current_coord + self.bond_length * self._unit_vec(total_angle)
                check_key = self._get_registry_key(new_coord)

                if check_key not in self.vertex_registry:
                    new_id = len(self.vertex_coords)
                    self.vertex_registry[check_key] = new_id
                    self.vertex_coords.append(new_coord)
                    # Pass the spoke (with its config_state) to the queue for the next generation
                    queue.append((new_id, current_id, spoke))

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

            edges = set() # use a set to avoid drawing edges twice (built-in duplicate removal)

            for i, point in enumerate(coords):
                neighbors = tree.query_ball_point(point, r=search_radius) # search for indices of NN inside distance of side length 

                for j in neighbors: # exclude the vertex itself
                    if j == i:
                        continue
                    k, l = sorted((i, j)) # sort to avoid duplicate storing
                    edges.add((k, l))

            for i, j in edges:
                plt.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]], color='black', linewidth=0.6)
        
        # if label==None:
        #     label = f"{config}"
        # plt.title("Vertex Configuration: " + label, fontsize = 14)
        plt.axis('equal')
        plt.axis('off')
        plt.grid(alpha=0.3)
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

tiling = archimedean.get('snub_square')[0]

lattice = Tiling(tiling, global_rotation_deg=0)
# print(lattice.spoke_orders['standard'])
lattice.generate()
lattice.plot(show_edges=False)