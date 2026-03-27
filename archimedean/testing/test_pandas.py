from magnonics import *
import pandas as pd
import numpy as np
import sympy as sp
from copy import copy, deepcopy

dataframe = pd.read_csv("archimedean/setup.csv")

# print(dataframe.iloc[0:, 11:13])

transl_vects = dataframe.loc[0:3, ['#translation vectors']]

spins_ = dataframe.loc[0:, ['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']]

interact = dataframe.loc[0:, ['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']]

parameters = dataframe.loc[0:, ['#parameters', 'value']]

# print(transl_vects, spins, interactions, parameters, sep='\n')

# groups = dataframe[dataframe['#interactions'].notna()].index
groups = dataframe['#interactions'].dropna().index.tolist()

group_names = dataframe.loc[groups, ['#interactions']]

# print(groups, group_names, sep='\n')

vects = dataframe['#translation vectors'].dropna().to_list()

params = (dataframe[['#parameters', 'value']]
                                            .dropna()
                                            .set_index('#parameters')['value']
                                            .to_dict()
          ) # .astype({'value': float}) \ insert after dropna()

spins = (dataframe[['sublattice', 'basis vector', 'ground state direction', 'spin length', 'gyromagnetic matrix']]
                                            .dropna()
                                            .to_dict('records')
         )

ints = (dataframe[['reference sublat', 'neighbor sublat', 'difference vector', 'interaction matrix']]
                                            .dropna()
                                            .to_dict('records')
        )

print(vects, params, spins, ints, sep='\n')

intmat = sp.Matrix(sp.sympify(ints[0]['interaction matrix']))

# Initialize.from_csv("archimedean/setup.csv")

vects_sym = [[sp.sympify(x) for x in v.strip('[]').split(',')] 
                     for v in vects]
mat = sp.Matrix(vects_sym)

basisvect = sp.Matrix(sp.sympify(spins[0]['basis vector']))
basisvect = mat.T * basisvect
num_basisvect = np.array(basisvect.T.subs(params), dtype=np.float64)[0]

grdstatedir = sp.Matrix(sp.sympify(spins[0]['ground state direction']))

spinlen = sp.sympify(spins[0]['spin length'])

# print(f'basisvect={basisvect}', f'numerical={num_basisvect}', sep='\n')

tripod = Sublattice.construct_local_tripod(grdstatedir)
num_tripod = np.array(tripod.T.subs(params), dtype=np.float64)
# print(f'tripod={tripod}', f'numerical={num_tripod}', sep='\n')

# sub = Sublattice.create(basisvect, tripod, spinlen, params)

lat, basis, lsw = Initialize.from_csv("archimedean/setup.csv") #Lattice.create(vects, spins, params)

# print(lat.num_rcpr_vects)

Bx, By, Bz = sp.symbols('B_x B_y B_z', real=True)
B = sp.Matrix([Bx, By, Bz])

bx, by, bz = pd.eval(params['B'])
symbs = {B[0]: bx, B[1]: by, B[2]: bz}
num_B = np.array(B.T.subs(symbs), dtype=np.float64)[0]

print(num_B)