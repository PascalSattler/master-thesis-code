import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from spinwave_visualizer import spin, view, check_lines, angle_color

# plot setup
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(1,1,1, projection='3d')


# view parameters
pos_def = (0,0,0)   # defaults
dir_def = (0,0,1)
scale = 1 # size of the spin object
resolution = 20
params = {'s_r':0.1, 'c_r':0.2, 'c_l':0.6, 'cn_r':1.2, 'cn_l':0.4} # change form of the spin object

zoom = 1.5
azim = 5
elev = 20

show_grid = False


# spin parameters
lattice_const = 0.75
n_spins = 23

rows = 6
spacing = 2

phi0 = 0
phi1 = np.pi/16
phi2 = np.pi/8
phi3 = np.pi/4
phi4 = np.pi/2
phi5 = np.pi
phis = [phi0, phi1, phi2, phi3, phi4, phi5] # angle incerements for each row

theta = np.pi/12    # fixed polar angle
offset = -np.pi/2     # starting angle offset


# set spin positions along negative x direction and new rows along y
positions = [(-i*lattice_const, j*spacing, 0) for j in range(rows) for i in range(n_spins)]

 
# set angles along the chains
for j in range(rows):
    pos = positions[j*n_spins:(j+1)*n_spins]
    angle_increment = phis[j]
    # print(pos)
    # print(angle_increment)
    for i in range(n_spins):
        dir = (np.cos(angle_increment*i+offset)*np.sin(theta),
               np.sin(angle_increment*i+offset)*np.sin(theta), 
               np.cos(theta))
        # print(dir)
        spin(ax, position=pos[i], direction=dir, scale=scale, parameters=params, resolution=resolution, color=angle_color(angle_increment*i))


# for i, pos in enumerate(positions):
#     # azimuthal angle increments, phi = k(r_q-r_p) = ka between neighbors
#     phi = np.pi/4 * i
#     print(pos)
#     theta = np.pi/12    # fixed polar angle
#     offset = np.pi      # starting angle offset

#     dir = (np.cos(phi+offset)*np.sin(theta),    # direction in spherical coordinates
#             np.sin(phi+offset)*np.sin(theta), 
#             np.cos(theta))                       
#     print(dir)
#     spin(ax, position=pos, direction=dir, scale=scale, parameters=params, resolution=resolution, color=angle_color(phi))


# set up markers
# mark lattice constant
ax.plot([0, -1*lattice_const], [0, 0], [-0.6, -0.6], color='black', linewidth=1)
ax.plot([0, 0], [0, 0], [-0.55, -0.65], color='black', linewidth=1)
ax.plot([-1*lattice_const, -1*lattice_const], [0, 0], [-0.55, -0.65], color='black', linewidth=1)
ax.plot([0, 0], [0, 0], [0, -0.55], color='black', linewidth=1, linestyle=':')
ax.plot([-1*lattice_const, -1*lattice_const], [0, 0], [0, -0.55], color='black', linewidth=1, linestyle=':')
ax.text(-0.5*lattice_const, 0.05, -0.75, r"$a$", color='black', fontsize=15)

# wavelength 0
ax.plot([0, -20*lattice_const], [0.4, 0.4], [0, 0], color='black', linewidth=1)
ax.plot([-20*lattice_const, -22*lattice_const], [0.4, 0.4], [0, 0], color='black', linewidth=1, linestyle=':')
ax.plot([0, 0], [0.35, 0.45], [0, 0], color='black', linewidth=1)
ax.plot([0, 0], [0, 0.35], [0, 0], color='black', linewidth=1, linestyle=':')
ax.text(-22/2*lattice_const, 0.5, 0, r"$\lambda \rightarrow 0$", color='black', fontsize=16, zdir='x', va="center")

# wavelength 1
ax.plot([0, -20*lattice_const], [1*spacing + 0.4, 1*spacing + 0.4], [0, 0], color='black', linewidth=1)
ax.plot([-20*lattice_const, -22*lattice_const], [1*spacing + 0.4, 1*spacing + 0.4], [0, 0], color='black', linewidth=1, linestyle=':')
ax.plot([0, 0], [1*spacing + 0.35, 1*spacing + 0.45], [0, 0], color='black', linewidth=1)
ax.plot([0, 0], [1*spacing + 0, 1*spacing + 0.35], [0, 0], color='black', linewidth=1, linestyle=':')
ax.text(-22/2*lattice_const, 1*spacing + 0.5, 0, r"$\lambda = 32a$", color='black', fontsize=16, zdir='x', va="center")

# wavelength 2
ax.plot([0, -16*lattice_const], [2*spacing + 0.4, 2*spacing + 0.4], [0, 0], color='black', linewidth=1)
ax.plot([0, 0], [2*spacing + 0.35, 2*spacing + 0.45], [0, 0], color='black', linewidth=1)
ax.plot([0, 0], [2*spacing + 0, 2*spacing + 0.35], [0, 0], color='black', linewidth=1, linestyle=':')
ax.plot([-16*lattice_const, -16*lattice_const], [2*spacing + 0.35, 2*spacing + 0.45], [0, 0], color='black', linewidth=1)
ax.plot([-16*lattice_const, -16*lattice_const], [2*spacing + 0, 2*spacing + 0.35], [0, 0], color='black', linewidth=1, linestyle=':')
ax.text(-16/2*lattice_const, 2*spacing + 0.5, 0, r"$\lambda = 16a$", color='black', fontsize=16, zdir='x', va="center")

# wavelength 3
ax.plot([0, -8*lattice_const], [3*spacing + 0.4, 3*spacing + 0.4], [0, 0], color='black', linewidth=1)
ax.plot([0, 0], [3*spacing + 0.35, 3*spacing + 0.45], [0, 0], color='black', linewidth=1)
ax.plot([0, 0], [3*spacing + 0, 3*spacing + 0.35], [0, 0], color='black', linewidth=1, linestyle=':')
ax.plot([-8*lattice_const, -8*lattice_const], [3*spacing + 0.35, 3*spacing + 0.45], [0, 0], color='black', linewidth=1)
ax.plot([-8*lattice_const, -8*lattice_const], [3*spacing + 0, 3*spacing + 0.35], [0, 0], color='black', linewidth=1, linestyle=':')
ax.text(-8/2*lattice_const, 3*spacing + 0.5, 0, r"$\lambda = 8a$", color='black', fontsize=16, zdir='x', va="center")

# wavelength 4
ax.plot([0, -4*lattice_const], [4*spacing + 0.4, 4*spacing + 0.4], [0, 0], color='black', linewidth=1)
ax.plot([0, 0], [4*spacing + 0.35, 4*spacing + 0.45], [0, 0], color='black', linewidth=1)
ax.plot([0, 0], [4*spacing + 0, 4*spacing + 0.35], [0, 0], color='black', linewidth=1, linestyle=':')
ax.plot([-4*lattice_const, -4*lattice_const], [4*spacing + 0.35, 4*spacing + 0.45], [0, 0], color='black', linewidth=1)
ax.plot([-4*lattice_const, -4*lattice_const], [4*spacing + 0, 4*spacing + 0.35], [0, 0], color='black', linewidth=1, linestyle=':')
ax.text(-4/2*lattice_const, 4*spacing + 0.5, 0, r"$\lambda = 4a$", color='black', fontsize=16, zdir='x', va="center")

# wavelength 5
ax.plot([0, -2*lattice_const], [5*spacing + 0.4, 5*spacing + 0.4], [0, 0], color='black', linewidth=1)
ax.plot([0, 0], [5*spacing + 0.35, 5*spacing + 0.45], [0, 0], color='black', linewidth=1)
ax.plot([0, 0], [5*spacing + 0, 5*spacing + 0.35], [0, 0], color='black', linewidth=1, linestyle=':')
ax.plot([-2*lattice_const, -2*lattice_const], [5*spacing + 0.35, 5*spacing + 0.45], [0, 0], color='black', linewidth=1)
ax.plot([-2*lattice_const, -2*lattice_const], [5*spacing + 0, 5*spacing + 0.35], [0, 0], color='black', linewidth=1, linestyle=':')
ax.text(-2/2*lattice_const, 5*spacing + 0.5, 0, r"$\lambda = 2a$", color='black', fontsize=16, zdir='x', va="center")

# row labels
ax.text2D(0.07, 0.225, r"$(a)$", transform=ax.transAxes, fontsize=17)
ax.text2D(0.22, 0.225, r"$(b)$", transform=ax.transAxes, fontsize=17)
ax.text2D(0.375, 0.225, r"$(c)$", transform=ax.transAxes, fontsize=17)
ax.text2D(0.53, 0.225, r"$(d)$", transform=ax.transAxes, fontsize=17)
ax.text2D(0.685, 0.225, r"$(e)$", transform=ax.transAxes, fontsize=17)
ax.text2D(0.84, 0.225, r"$(f)$", transform=ax.transAxes, fontsize=17)


# set view mode
view(ax, positions=positions, zoom=zoom, azimuth=azim, elevation=elev, grid=show_grid, margin=1)

check_lines(ax, pos_def, dir_def, scale=scale, state=False) # lines for checking correct spin position and rotation

plt.tight_layout()
plt.savefig('plotting_tools/spin_wave_visualization/ferromagnetic_chain_spinwaves.pdf', facecolor='w', transparent=False, dpi = 1200)
plt.show()