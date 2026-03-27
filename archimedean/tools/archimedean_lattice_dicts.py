from dataclasses import dataclass
import numpy as np

triangular = {
    'label'         : 'triangular',
    'config'        : (3,3,3,3,3,3),
    'config_label'  : r'$(3^6)$',
    'lat_vects'     : [f'[a, 0, 0]', 
                       f'[a/2, sqrt(3)*a/2, 0]', 
                       f'[0, 0, 1]'],
    'n_sublats'     : 1,
    'positions'     : ['[0, 0, 0]']
}

square = {
    'label'         : 'square',
    'config'        : (4,4,4,4), 
    'config_label'  : r'$(4^4)$',
    'lat_vects'     : [f'[a, 0, 0]', 
                       f'[0, a, 0]', 
                       f'[0, 0, 1]'],
    'n_sublats'     : 1,
    'positions'     : ['[0, 0, 0]']
}

hexagonal = {
    'label'         : 'hexagonal',
    'config'        : (6,6,6), 
    'config_label'  : r'$(6^3)$',
    'lat_vects'     : [f'[sqrt(3)*a, 0, 0]', 
                       f'[sqrt(3)/2*a, 1.5*a, 0]', 
                       f'[0, 0, 1]'],
    'zigzag'        : [f'[sqrt(3)*a, 0, 0]', 
                       f'[sqrt(3)/2*a, 1.5*a, 0]', 
                       f'[0, 0, 1]'],
    'armchair'      : [f'[3*a, 0, 0]', 
                       f'[1.5*a, sqrt(3)/2*a, 0]', 
                       f'[0, 0, 1]'],
    'n_sublats'     : 2,
    'positions'     : ['[0, 0, 0]', 
                       '[1/3, 1/3, 0]']
}

kagome = {
    'label'         : 'kagome',
    'config'        : (3,6,3,6),
    'config_label'  : r'$(3.6.3.6)$',
    'lat_vects'     : [f'[2*a, 0, 0]', 
                       f'[a, sqrt(3)*a, 0]', 
                       f'[0, 0, 1]'],
    'n_sublats'     : 3,
    'positions'     : ['[0, 0, 0]', 
                       '[0, 0, 0.5]', 
                       '[0, 0.5, 0]'] # SWAPPED 2 AND 3
}

rhombitrihexagonal = {
    'label'         : 'rhombitrihexagonal',
    'config'        : (3,4,6,4), 
    'config_label'  : r'$(3.4.6.4)$',
    'lat_vects'     : [f'[(sqrt(3)+1)*a, 0, 0]', 
                       f'[(sqrt(3)+1)/2*a, (sqrt(3)+3)/2*a, 0]', 
                       f'[0, 0, 1]'],
    'zigzag'        : [f'[(sqrt(3)+1)*a, 0, 0]', 
                       f'[(sqrt(3)+1)/2*a, (sqrt(3)+3)/2*a, 0]', 
                       f'[0, 0, 1]'],
    'armchair'      : [f'[(sqrt(3)+3)*a, 0, 0]', 
                       f'[(((sqrt(3)+1)/2)+1)*a, (sqrt(3)+1)/2*a, 0]', 
                       f'[0, 0, 1]'],
    'n_sublats'     : 6,
    'positions'     : ['[0, 0, 0]', 
                       '[0.3660254, 0, 0]', 
                       '[0.57735027, 0.21132487, 0]', 
                       '[0.57735027, 0.57735027, 0]', 
                       '[0.21132487, 0.57735027, 0]' , 
                       '[0, 0.3660254, 0]']
}

truncated_square = {
    'label'         : 'truncated_square',
    'config'        : (4,8,8), 
    'config_label'  : r'$(4.8^2)$',
    'lat_vects'     : [f'[(sqrt(2)+1)*a, 0, 0]', 
                       f'[0, (sqrt(2)+1)*a, 0]', 
                       f'[0, 0, 1]'],
    'n_sublats'     : 4,
    'positions'     : ['[0, 0, 0]', 
                       '[0.29289322, 0.29289322, 0]', 
                       '[0, 0.58578644, 0]', 
                       '[0.70710678, 0.29289322, 0]']
}

truncated_hexagonal = {
    'label'         : 'truncated_hexagonal',
    'config'        : (3,12,12), 
    'config_label'  : r'$(3.12^2)$',
    'lat_vects'     : [f'[(sqrt(3)+2)*a, 0, 0]', 
                       f'[(sqrt(3)/2+1)*a, (sqrt(3)+3/2)*a, 0]', 
                       f'[0, 0, 1]'],
    'zigzag'        : [f'[(sqrt(3)+2)*a, 0, 0]', 
                       f'[(sqrt(3)/2+1)*a, (sqrt(3)+3/2)*a, 0]', 
                       f'[0, 0, 1]'],
    'armchair'      : [f'[(2*sqrt(3)+3)*a, 0, 0]', 
                       f'[(sqrt(3)+3/2)*a, (sqrt(3)/2+1)*a, 0]', 
                       f'[0, 0, 1]'],
    'n_sublats'     : 6,
    'positions'     : ['[0, 0, 0]', 
                       '[0.3660254, 0, 0]', 
                       '[0.57735027, 0.21132487, 0]', 
                       '[0.57735027, 0.57735027, 0]', 
                       '[0.21132487, 0.57735027, 0]' , 
                       '[0, 0.3660254, 0]']
}

truncated_trihexagonal = {
    'label'         : 'truncated_trihexagonal',
    'config'        : (4,6,12), 
    'config_label'  : r'$(4.6.12)$',
    'lat_vects'     : [f'[(sqrt(3)+3)*a, 0, 0]', 
                       f'[(sqrt(3)+3)/2*a, 3*(sqrt(3)+1)/2*a, 0]', 
                       f'[0, 0, 1]'],
    'zigzag'        : [f'[(sqrt(3)+3)*a, 0, 0]', 
                       f'[(sqrt(3)+3)/2*a, 3*(sqrt(3)+1)/2*a, 0]', 
                       f'[0, 0, 1]'],
    'armchair'      : [f'[3*(sqrt(3)+1)*a, 0, 0]', 
                       f'[3*(sqrt(3)+1)/2*a, (sqrt(3)+3)/2*a, 0]', 
                       f'[0, 0, 1]'],
    'n_sublats'     : 12,
    'positions'     : ['[0.33333333, 0.12200847, 0]',
                       '[0.5446582,  0.12200847, 0]', 
                       '[0.5446582,  0.33333333, 0]', 
                       '[0.33333333, 0.5446582, 0]', 
                       '[0.12200847, 0.5446582, 0]', 
                       '[0.12200847, 0.33333333, 0]', 
                       '[0.66666667, 0.4553418, 0]', 
                       '[0.87799153, 0.4553418, 0]', 
                       '[0.87799153, 0.66666667, 0]', 
                       '[0.66666667, 0.87799153, 0]', 
                       '[0.4553418,  0.87799153, 0]', 
                       '[0.4553418,  0.66666667, 0]']
}

snub_square = {
    'label'         : 'snub_square',
    'config'        : (3,3,4,3,4), 
    'config_label'  : r'$(3^2.4.3.4)$',
    'lat_vects'     : [f'[sqrt(2 + sqrt(3))*a, 0, 0]', 
                       f'[0, sqrt(2 + sqrt(3))*a, 0]', 
                       f'[0, 0, 1]'],
    'n_sublats'     : 4,
    'positions'     : ['[0, 0, 0]',
                       '[0.5, 0.1339746, 0]',
                       '[sqrt(3)/2, 0.5, 0]',
                       '[0.3660254, 0.6339746, 0]']
}

snub_trihexagonal = {
    'label'         : 'snub_trihexagonal',
    'config'        : (3,3,3,3,6), 
    'config_label'  : r'$(3^4.6)$',
    'lat_vects'     : [f'[sqrt(7)*a, 0, 0]', 
                       f'[sqrt(7)/2*a, sqrt(21)/2*a, 0]', 
                       f'[0, 0, 1]'],
    'zigzag'        : [f'[sqrt(7)*a, 0, 0]', 
                       f'[sqrt(7)/2*a, sqrt(21)/2*a, 0]', 
                       f'[0, 0, 1]'],
    'armchair'      : [f'[4.5*a, sqrt(3)/2*a, 0]', 
                       f'[2*a, sqrt(3)*a, 0]', 
                       f'[0, 0, 1]'],
    'n_sublats'     : 6,
    'positions'     : ['[0.28571429, 0.14285714, 0]', 
                       '[0.57142857, 0.28571429, 0]', 
                       '[0.85714286, 0.42857143, 0]', 
                       '[0.71428571, 0.85714286, 0]', 
                       '[0.42857143, 0.71428571, 0]' , 
                       '[0.14285714, 0.57142857, 0]']
}

elongated_triangular = {
    'label'         : 'elongated_triangular',
    'config'        : (3,3,3,4,4), 
    'config_label'  : r'$(3^3.4^2)$',
    'lat_vects'     : [f'[a, 0, 0]', 
                       f'[-a/2, (1+sqrt(3)/2)*a, 0]', 
                       f'[0, 0, 1]'],
    'flat'          : [f'[a, 0, 0]', 
                       f'[-a/2, (1+sqrt(3)/2)*a, 0]', 
                       f'[0, 0, 1]'],
    'bolt'          : [f'[(1+sqrt(3)/2)*a, a/2, 0]', 
                       f'[0, a, 0]', 
                       f'[0, 0, 1]'],
    'hook'          : [f'[(1+sqrt(3)/2)*a, -a/2, 0]', 
                       f'[0, a, 0]', 
                       f'[0, 0, 1]'],
    'n_sublats'     : 2,
    'positions'     : ['[0, 0, 0]', 
                       '[sqrt(3)-1, 0.46410162, 0]']
}

archimedean = {
    'triangular'            : triangular,
    'square'                : square,
    'hexagonal'             : hexagonal,
    'kagome'                : kagome,
    'rhombitrihexagonal'    : rhombitrihexagonal,
    'truncated_square'      : truncated_square,
    'truncated_hexagonal'   : truncated_hexagonal,
    'truncated_trihexagonal': truncated_trihexagonal,
    'snub_square'           : snub_square,
    'snub_trihexagonal'     : snub_trihexagonal,
    'elongated_triangular'  : elongated_triangular
}

def archimedean_lattice(lattice:str, key:str = None) -> dict:
    if key != None:
        return archimedean[lattice][key]
    else:
        return archimedean[lattice]

print(archimedean_lattice('kagome'))


def get_interaction_str():
    pass

def get_parameter_str():
    pass

def find_NN():
    pass

def find_NNN():
    pass

def find_DMI():
    pass

def generate_input_file():
    pass

@dataclass
class ArchimedeanLattice:
    pass