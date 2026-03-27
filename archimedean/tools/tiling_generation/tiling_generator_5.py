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

        # NEW: Generate all rotational phases and chiralities
        self.spoke_orders = self._generate_all_spoke_orders()
        
        self.vertex_registry = {}
        self.vertex_coords = []
        self.track_order = {}  # Now stores (phase, chirality) tuples

    @staticmethod
    def _internal_angle(n_sides:int):
        return (n_sides - 2) * np.pi / n_sides
    
    @staticmethod
    def _norm_angle(angle:float):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    @staticmethod
    def _chord_length(n_sides:int, side_length:float, vertices_inbetween:int = 2):
        def _circumradius(n_sides:int, side_length:float):
            return side_length / (2 * np.sin(np.pi / n_sides))
        
        if n_sides <= 3: return 0
        else: return 2 * _circumradius(n_sides, side_length) * np.sin((vertices_inbetween+1) * np.pi / n_sides)
    
    @staticmethod
    def _unit_vec(angle:float):
        return np.array([np.cos(angle), np.sin(angle)])
    
    def _generate_all_spoke_orders(self):
        """Generate all possible rotational phases and both chiralities"""
        spoke_orders = {}
        
        # For each rotational phase (which spoke starts at angle 0)
        for phase in range(self.n_polygons):
            # For each chirality
            for chirality in ['standard', 'reversed']:
                key = (phase, chirality)
                spoke_orders[key] = self._get_vertex_spokes(phase, chirality)
        
        return spoke_orders
    
    def _get_vertex_spokes(self, rotation_phase:int = 0, chirality:str = 'standard'):
        """Calculate spokes for a specific rotational phase and chirality"""
        spokes = []
        current_angle = 0.0 + self.global_rotation * np.pi/180

        # Apply chirality
        if chirality == 'standard':
            config_ordered = self.config
        elif chirality == 'reversed':
            config_ordered = self.config[::-1]
        else:
            raise ValueError("Chirality must be 'standard' or 'reversed'")

        # Apply rotational phase (which polygon comes first)
        config_rotated = config_ordered[rotation_phase:] + config_ordered[:rotation_phase]
        
        config_state = deque(config_rotated)

        for id, polygon in enumerate(config_rotated):
            state = tuple([*config_state])
            next = self._internal_angle(state[0])
            prev = -self._internal_angle(state[-1])

            spokes.append({
                'index': id,
                'phase': rotation_phase,
                'chirality': chirality,
                'config_state': state,
                'angle': current_angle,
                'angle_to_next': next,
                'angle_to_prev': prev,
                'polygon_left': state[0],
                'polygon_right': state[-1]
            })

            config_state.rotate(-1)
            current_angle += self._internal_angle(polygon)

        return spokes
    
    def _get_registry_key(self, vertex: np.ndarray):
        return (np.round(vertex[0], decimals=4), np.round(vertex[1], decimals=4))
    
    def generate(self, number_of_vertices:int = 300):
        # start setup
        origin = np.array([0.0, 0.0])
        self.vertex_registry[self._get_registry_key(origin)] = 0
        self.vertex_coords.append(origin)
        self.track_order[0] = (0, 'standard')  # Default: phase 0, standard chirality

        queue = deque([(0, None, None)])

        # candidate finding
        while queue and len(self.vertex_coords) < number_of_vertices:
            current_id, reference_id, active_spoke = queue.popleft()
            current_coord = self.vertex_coords[current_id]

            if reference_id is None:
                # First vertex - use default configuration
                selected_order_key = (0, 'standard')
                selected_order = self.spoke_orders[selected_order_key]
                rotation = 0.0
            else:
                reference_coord = self.vertex_coords[reference_id]
                ref_phase, ref_chirality = self.track_order[reference_id]

                bond_vec = current_coord - reference_coord
                bond_angle = np.arctan2(bond_vec[1], bond_vec[0])

                # Calculate verification points from reference vertex perspective
                angle_next_from_ref = bond_angle + active_spoke['angle_to_next']
                coord_next_from_ref = reference_coord + self.bond_length * self._unit_vec(angle_next_from_ref)

                angle_prev_from_ref = bond_angle + active_spoke['angle_to_prev']
                coord_prev_from_ref = reference_coord + self.bond_length * self._unit_vec(angle_prev_from_ref)
 
                # NEW: Try all possible configurations (all phases and chiralities)
                match_found = False
                best_match = None
                best_score = float('inf')

                for order_key, spoke_order in self.spoke_orders.items():
                    for cand_spoke in spoke_order:
                        # Check that the shared polygons match
                        if (active_spoke['polygon_left'] != cand_spoke['polygon_right'] or 
                            active_spoke['polygon_right'] != cand_spoke['polygon_left']):
                            continue
                        
                        # Calculate the rotation needed for this candidate
                        test_rotation = bond_angle + np.pi - cand_spoke['angle']

                        # Geometric validation: check chord lengths
                        angle_prev_from_cur = cand_spoke['angle'] + cand_spoke['angle_to_prev'] + test_rotation
                        coord_prev_from_cur = current_coord + self.bond_length * self._unit_vec(angle_prev_from_cur)
                        dist_1 = np.linalg.norm(coord_next_from_ref - coord_prev_from_cur)
                        req_1 = self._chord_length(active_spoke['polygon_left'], self.bond_length)

                        angle_next_from_cur = cand_spoke['angle'] + cand_spoke['angle_to_next'] + test_rotation
                        coord_next_from_cur = current_coord + self.bond_length * self._unit_vec(angle_next_from_cur)
                        dist_2 = np.linalg.norm(coord_prev_from_ref - coord_next_from_cur)
                        req_2 = self._chord_length(active_spoke['polygon_right'], self.bond_length)

                        # Check if both geometric constraints are satisfied
                        if abs(dist_1 - req_1) < 1e-3 and abs(dist_2 - req_2) < 1e-3:
                            score = abs(self._norm_angle(test_rotation))
                            if score < best_score:
                                best_score = score
                                best_match = (test_rotation, cand_spoke, order_key, spoke_order)
                                match_found = True

                if match_found:
                    rotation, selected_spoke, selected_order_key, selected_order = best_match
                    self.track_order[current_id] = selected_order_key
                else:
                    continue

            # create new vertices
            for spoke in selected_order:
                total_angle = spoke['angle'] + rotation
                new_coord = current_coord + self.bond_length * self._unit_vec(total_angle)
                check_key = self._get_registry_key(new_coord)

                if check_key not in self.vertex_registry:
                    new_id = len(self.vertex_coords)
                    self.vertex_registry[check_key] = new_id
                    self.vertex_coords.append(new_coord)
                    queue.append((new_id, current_id, spoke))

    def plot(self, mark_central_vertex:bool = True, show_edges:bool = True, title:str = None):
        coords = np.array(self.vertex_coords)

        plt.figure(figsize=(8, 8))
    
        if mark_central_vertex:
            plt.scatter(coords[0,0], coords[0,1], c='red', s=50, zorder=10)

        plt.scatter(coords[:,0], coords[:,1], c='black', s=25)

        if show_edges:
            tree = KDTree(coords)
            search_radius = self.bond_length * (1 + 1e-5)
            edges = set()

            for i, point in enumerate(coords):
                neighbors = tree.query_ball_point(point, r=search_radius)

                for j in neighbors:
                    if j == i:
                        continue
                    k, l = sorted((i, j))
                    edges.add((k, l))

            for i, j in edges:
                plt.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]], 
                        color='black', linewidth=0.6)
        
        if title:
            plt.title(title, fontsize=14)
        plt.axis('equal')
        plt.axis('off')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


# Test all Archimedean tilings
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

# Test all tilings
print("Testing all Archimedean tilings:\n")
for name, (config, notation) in archimedean.items():
    print(f"Generating {name} {notation}...")
    try:
        lattice = Tiling(config, global_rotation_deg=0)
        print(f"  Total configurations generated: {len(lattice.spoke_orders)}")
        lattice.generate(number_of_vertices=200)
        print(f"  ✓ Generated {len(lattice.vertex_coords)} vertices")
        
        # Show which configurations were actually used
        used_configs = set(lattice.track_order.values())
        print(f"  Used {len(used_configs)} different configurations: {used_configs}\n")
        
        lattice.plot(show_edges=True, title=f"{name} {notation}")
    except Exception as e:
        print(f"  ✗ Failed: {e}\n")