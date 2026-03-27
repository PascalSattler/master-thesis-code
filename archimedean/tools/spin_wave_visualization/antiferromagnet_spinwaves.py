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

zoom = 1.2
azim = 30
elev = 25

show_grid = False


# spin parameters
lattice_const = 0.75
n_spins = 10

rows = 4
spacing = 3

phi0 = 0
phi1 = np.pi/2
phi2 = np.pi/4
phis = [phi1, phi2, phi2, phi1] # angle incerements for each row

# deflection angle follows theta_q ~ sqrt(2 a_q^dagger a_q / S_q) = sqrt(2 N_q(k) / S_q N_cells), here N_1,2(k) = 0.5*(1 +- |sin(ka)|), thus theta ~ sqrt((1 +- |sin(ka)|) / N_cells) = sqrt(2*(1 +- |sin(ka)|) / n_spins) for S_q = 1
theta1 = np.sqrt(4/n_spins)
theta2 = np.sqrt(2*(1+1/np.sqrt(2))/n_spins)
theta3 = np.sqrt(2*(1-1/np.sqrt(2))/n_spins)
theta4 = 0
theta = [theta1, theta2, theta3, theta4]
# theta_down = theta_up[::-1]
offset = np.pi    # starting angle offset

positions_up = [(-2*i*lattice_const, j*spacing, 0) for j in range(rows) for i in range(n_spins//2)]
positions_down = [(-2*(i+1/2)*lattice_const, j*spacing, 0) for j in range(rows) for i in range(n_spins//2)]
# print("positions_up=", positions_up)
positions = [None]*(len(positions_up)+len(positions_down))
positions[::2] = positions_up
positions[1::2] = positions_down

for j in range(rows):
    # if j == 1:
    #     pass
    # else:
    #     continue

    pos = positions[j*n_spins:(j+1)*n_spins]
    # print('pos =',pos)
    angle_increment = phis[j]
    # print(angle_increment)
    # print('z_up=', np.cos(theta[j]))
    # print('z_down=', np.cos(theta[j]))

    if j == 3:
            offset -= np.pi/2

    for i in range(n_spins):
        if i % 2 == 0:
            if theta[j] == 0:
                color = 'grey'
            else:
                color = angle_color(angle_increment*i)

            dir = (np.cos(angle_increment*i+offset)*np.sin(theta[j]),
                   np.sin(angle_increment*i+offset)*np.sin(theta[j]), 
                   np.cos(theta[j]))
            spin(ax, position=pos[i], direction=dir, scale=scale, parameters=params, resolution=resolution, color=color)
        else:
            if theta[::-1][j] == 0:
                color = 'grey'
            else:
                color = angle_color(angle_increment*i+offset)

            dir = (-np.cos(angle_increment*i+offset)*np.sin(theta[::-1][j]),
                   -np.sin(angle_increment*i+offset)*np.sin(theta[::-1][j]), 
                   -np.cos(theta[::-1][j]))
            spin(ax, position=pos[i], direction=dir, scale=scale, parameters=params, resolution=resolution, color=color)

# set up markers
# mark lattice constant
ax.plot([0, -2*lattice_const], [0, 0], [-1.1, -1.1], color='black', linewidth=1)
ax.plot([0, 0], [0, 0], [-1.05, -1.15], color='black', linewidth=1)
ax.plot([-2*lattice_const, -2*lattice_const], [0, 0], [-1.05, -1.15], color='black', linewidth=1)
ax.plot([0, 0], [0, 0], [0, -1.05], color='black', linewidth=1, linestyle=':')
ax.plot([-2*lattice_const, -2*lattice_const], [0, 0], [0, -1.05], color='black', linewidth=1, linestyle=':')
ax.text(-1*lattice_const, 0.0, -1.35, r"$2a$", color='black', fontsize=15)

# wavelength 1
ax.plot([0, -4*lattice_const], [0*spacing + 1.3, 0*spacing + 1.3], [0, 0], color='black', linewidth=1)
ax.plot([0, 0], [0*spacing + 1.25, 0*spacing + 1.35], [0, 0], color='black', linewidth=1)
ax.plot([0, 0], [0*spacing + 0, 0*spacing + 1.25], [0, 0], color='black', linewidth=1, linestyle=':')
ax.plot([-4*lattice_const, -4*lattice_const], [0*spacing + 1.25, 0*spacing + 1.35], [0, 0], color='black', linewidth=1)
ax.plot([-4*lattice_const, -4*lattice_const], [0*spacing + 0, 0*spacing + 1.25], [0, 0], color='black', linewidth=1, linestyle=':')
ax.text(-3/2*lattice_const, 0*spacing + 1.2, 0, r"$\lambda = 4a$", color='black', fontsize=16, zdir='x', va="center")

# wavelength 2
ax.plot([0, -8*lattice_const], [1*spacing + 1.3, 1*spacing + 1.3], [0, 0], color='black', linewidth=1)
ax.plot([0, 0], [1*spacing + 1.25, 1*spacing + 1.35], [0, 0], color='black', linewidth=1)
ax.plot([0, 0], [1*spacing + 0, 1*spacing + 1.25], [0, 0], color='black', linewidth=1, linestyle=':')
ax.plot([-8*lattice_const, -8*lattice_const], [1*spacing + 1.25, 1*spacing + 1.35], [0, 0], color='black', linewidth=1)
ax.plot([-8*lattice_const, -8*lattice_const], [1*spacing + 0, 1*spacing + 1.25], [0, 0], color='black', linewidth=1, linestyle=':')
ax.text(-7/2*lattice_const, 1*spacing + 1.2, 0, r"$\lambda = 8a$", color='black', fontsize=16, zdir='x', va="center")

# wavelength 2
ax.plot([-1*lattice_const, -9*lattice_const], [2*spacing + 1.3, 2*spacing + 1.3], [0, 0], color='black', linewidth=1)
ax.plot([-1*lattice_const, -1*lattice_const], [2*spacing + 1.25, 2*spacing + 1.35], [0, 0], color='black', linewidth=1)
ax.plot([-1*lattice_const, -1*lattice_const], [2*spacing + 0, 2*spacing + 1.25], [0, 0], color='black', linewidth=1, linestyle=':')
ax.plot([-9*lattice_const, -9*lattice_const], [2*spacing + 1.25, 2*spacing + 1.35], [0, 0], color='black', linewidth=1)
ax.plot([-9*lattice_const, -9*lattice_const], [2*spacing + 0, 2*spacing + 1.25], [0, 0], color='black', linewidth=1, linestyle=':')
ax.text(-8/2*lattice_const, 2*spacing + 1.2, 0, r"$\lambda = 8a$", color='black', fontsize=16, zdir='x', va="center")

# wavelength 4
ax.plot([-1*lattice_const, -5*lattice_const], [3*spacing + 1.3, 3*spacing + 1.3], [0, 0], color='black', linewidth=1)
ax.plot([-1*lattice_const, -1*lattice_const], [3*spacing + 1.25, 3*spacing + 1.35], [0, 0], color='black', linewidth=1)
ax.plot([-1*lattice_const, -1*lattice_const], [3*spacing + 0, 3*spacing + 1.25], [0, 0], color='black', linewidth=1, linestyle=':')
ax.plot([-5*lattice_const, -5*lattice_const], [3*spacing + 1.25, 3*spacing + 1.35], [0, 0], color='black', linewidth=1)
ax.plot([-5*lattice_const, -5*lattice_const], [3*spacing + 0, 3*spacing + 1.25], [0, 0], color='black', linewidth=1, linestyle=':')
ax.text(-5/2*lattice_const, 3*spacing + 1.2, 0, r"$\lambda = 4a$", color='black', fontsize=16, zdir='x', va="center")

# row labels
ax.text(0.5, -0.5, 0, r"$(c)$", fontsize=17)
ax.text(0.5, -0.5+1*spacing, 0, r"$(d)$", fontsize=17)
ax.text(0.5, -0.5+2*spacing, 0, r"$(e)$", fontsize=17)
ax.text(0.5, -0.5+3*spacing, 0, r"$(f)$", fontsize=17)

view(ax, positions=np.stack(positions_up+positions_down), zoom=zoom, azimuth=azim, elevation=elev, grid=show_grid, margin=1)

plt.tight_layout()
plt.savefig('plotting_tools/spin_wave_visualization/antiferromagnetic_chain_spinwaves.pdf', facecolor='w', transparent=False, dpi = 1200)
plt.show()