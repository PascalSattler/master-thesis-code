import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# get lmodern font from LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{lmodern}"
})

def sphere(center:tuple = (0,0,0), radius:float = 0.2, resolution:int = 20):
    # spherical coordinate angles: phi -> azimuthal, thetha -> polar
    phi = np.linspace(0, 2*np.pi, resolution)
    theta = np.linspace(0, np.pi, resolution //2)

    # convert angles to array of points on a grid
    phi, theta = np.meshgrid(phi,theta)

    # convert spherical to cartesian coordinates
    center = np.array(center)
    x = radius * np.cos(phi) * np.sin(theta) + center[0]
    y = radius * np.sin(phi) * np.sin(theta) + center[1]
    z = radius * np.cos(theta) + center[2]
    return x, y, z
    
def cylinder(center:tuple, start_z:float = 0, size:float = 0.5, radius:float = 0.1, resolution:int = 20):
    # parametrize cylinder along z-axis and convert z points, angles to array of points on a grid
    z = np.linspace(start_z, start_z + size, 10)
    phi = np.linspace(0, 2*np.pi, resolution)
    phi, z = np.meshgrid(phi, z) # shape(z) = shape(phi) = (len(z) rows, len(phi) columns) -> each angle has one column of z points
    x = radius * np.cos(phi) + center[0]
    y = -radius * np.sin(phi) + + center[1] # sign to match shading, reason unclear!?
    z = z + center[2]
    return x, y, z

def cone(center:tuple, start_z:float = 0.5, size:float = 0.3, radius:float = 0.2, resolution:int = 20):
    # parametrize cylinder along z-axis and convert z points, angles to array of points on a grid
    z = np.linspace(start_z, start_z + size, 10)
    phi = np.linspace(0, 2*np.pi, resolution)
    phi, z = np.meshgrid(phi, z) # shape(z) = shape(phi) = (len(z) rows, len(phi) columns) -> each angle has one column of z points

    # cone radius decreases along the orientation vector
    radius = radius * (start_z + size - z) / size
    x = radius * np.cos(phi) + center[0]
    y = -radius * np.sin(phi) + center[1] # sign to match shading, reason unclear!?
    z = z + center[2]
    return x, y, z

def spin(plot, position:tuple = (0,0,0), direction:tuple = (0,0,1), scale:float = 1.0, parameters:dict = {'s_r':0.2, 'c_r':0.5, 'c_l':0.6, 'cn_r':1.2, 'cn_l':0.4}, color:str = 'red', resolution:int = 20, transparency:float = 1.0, edgecolor = None):
    # parameters
    sphere_radius = parameters['s_r'] * scale
    cylinder_radius = parameters['c_r'] * sphere_radius
    cylinder_start = sphere_radius - np.sqrt(sphere_radius**2 - cylinder_radius**2)
    cylinder_length = parameters['c_l'] * scale
    cone_radius = parameters['cn_r'] * sphere_radius
    cone_start = cylinder_start + cylinder_length
    cone_length = parameters['cn_l'] * scale

    # coordinates
    x_sph, y_sph, z_sph = sphere(center=position, radius=sphere_radius, resolution=resolution)
    x_cyl, y_cyl, z_cyl = cylinder(center=position, start_z=cylinder_start, size=cylinder_length, radius=cylinder_radius, resolution=resolution)
    x_cone, y_cone, z_cone = cone(center=position, start_z=cone_start, size=cone_length, radius=cone_radius, resolution=resolution)

    x_tot = np.concatenate((x_sph, x_cyl, x_cone), axis=0)
    y_tot = np.concatenate((y_sph, y_cyl, y_cone), axis=0)
    z_tot = np.concatenate((z_sph, z_cyl, z_cone), axis=0)
    mesh_shape = z_tot.shape

    X, Y, Z = x_tot.ravel(), y_tot.ravel(), z_tot.ravel()   # shape(Z) = (len(z_tot)*len(phi),) -> rows are subsequently listet after each other into one single row
    XYZ = np.stack([X,Y,Z]) - np.array(position)[:, None]                                # shape(XYZ) = (3, len(z_tot)*len(phi)) -> each column of XYZ is a point in 3D space

    direction_vec = np.array(direction) # direction in which we want to align the spin
    magnitude = np.linalg.norm(direction_vec)
    if magnitude == 0:
        raise ValueError("position and direction must differ")
    direction_normal = direction_vec/magnitude
    
    # rotate onto target direction
    # 1) build rotation matrix
    initial_dir = np.array((0,0,1))    # initially aligned with z axis by default
    dot_prod = np.dot(initial_dir, direction_normal)        # cosine of angle between unit vectors
    cross_prod = np.cross(initial_dir, direction_normal)    # axis of rotation
    cross_prod_norm = np.linalg.norm(cross_prod)            # magnitude of crossproduct is sine of angle between both vectors, checks how "similar" they are
    if cross_prod_norm < 1e-8:                              # if value is close to zero then both vectors are parallel
        if dot_prod > 0:
            rot_mat = np.eye(3)                             # then no rotation is necessary -> unit matrix
        else:
            K = np.array([[0, 0, 0],[0,0,-1],[0,1,0]])      # matrix build from x axis which it is rotated about in the parallel case
            rot_mat = np.eye(3) + 2 * K @ K                 # Rodrigues rotation formula for theta = 180°
    else:
        cross_normed = cross_prod/cross_prod_norm                           # normalized axis of rotation, afterwards written in matrix form
        K = np.array([[0, -cross_normed[2], cross_normed[1]],[cross_normed[2],0,-cross_normed[0]],[-cross_normed[1],cross_normed[0],0]])
        rot_mat = np.eye(3) + cross_prod_norm * K + (1-dot_prod) * K @ K    # Rodrigues rotation formula

    # 2)perform rotation
    XYZ_rotated = rot_mat @ XYZ                                 # applies rotation onto each column vector (i.e. point in space)
    x_rot = XYZ_rotated[0,:].reshape(mesh_shape) + position[0]
    y_rot = XYZ_rotated[1,:].reshape(mesh_shape) + position[1]
    z_rot = XYZ_rotated[2,:].reshape(mesh_shape) + position[2]  # reshape back to original meshgrid and translate by start point

    # create surface in the plot
    plot.plot_surface(x_rot, y_rot, z_rot,
                    rstride=1, cstride=1,
                    facecolors=None, color=color,
                    linewidth=0 if edgecolor is None else 0.3,
                    edgecolor=edgecolor,
                    alpha=transparency,
                    antialiased=True, shade=True)
    
def check_lines(plot, pos:tuple, dir:tuple, scale:float = 1.0, state:bool = True):
    if state:
        # control lines to walls for checking correct position
        xlim = plot.get_xlim()
        ylim = plot.get_ylim()
        zlim = plot.get_zlim()

        # points to check
        p = np.array(pos)
        d = np.array(dir)/np.linalg.norm(dir) * scale + p

        # formatting helper
        def fmt(coord, pos = p, dir = d):
            if np.all(coord == pos):
                return "pos="f"({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})"
            if np.all(coord == dir):
                return "dir="f"({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})"
            
        # X-lines to negative wall
        if p[0] > xlim[0]:
            end = [xlim[0], p[1], p[2]]
            plot.plot([p[0], end[0]], [p[1], end[1]], [p[2], end[2]], color='gray', lw=0.8)
            plot.text(*end, fmt(p), color='black', fontsize=8)

        if d[0] > xlim[0]:
            end = [xlim[0], d[1], d[2]]
            plot.plot([d[0], end[0]], [d[1], end[1]], [d[2], end[2]], color='green', lw=0.8)
            plot.text(*end, fmt(d), color='green', fontsize=8)

        # Y-lines to negative wall
        if p[1] > ylim[0]:
            end = [p[0], ylim[0], p[2]]
            plot.plot([p[0], end[0]], [p[1], end[1]], [p[2], end[2]], color='gray', lw=0.8)
            plot.text(*end, fmt(p), color='black', fontsize=8)

        if d[1] > ylim[0]:
            end = [d[0], ylim[0], d[2]]
            plot.plot([d[0], end[0]], [d[1], end[1]], [d[2], end[2]], color='green', lw=0.8)
            plot.text(*end, fmt(d), color='green', fontsize=8)

        # Z-lines to negative wall
        if p[2] > zlim[0]:
            end = [p[0], p[1], zlim[0]]
            plot.plot([p[0], end[0]], [p[1], end[1]], [p[2], end[2]], color='gray', lw=0.8)
            plot.text(*end, fmt(p), color='black', fontsize=8)

        if d[2] < zlim[1]:
            end = [d[0], d[1], zlim[1]]
            plot.plot([d[0], end[0]], [d[1], end[1]], [d[2], end[2]], color='green', lw=0.8)
            plot.text(*end, fmt(d), color='green', fontsize=8)
        else:
            pass

def angle_color(angle:float, cmap:str = 'hsv', darken:float = 0.3):
    phi = angle % (2*np.pi) # map to [0,2pi)
    t = phi/(2*np.pi)       # normalize to [0,1)
    cmap = plt.get_cmap(cmap)
    r, g, b, _ = cmap(t)
    return tuple(np.array([r, g, b])*(1-darken))

def view(plot, positions: list, zoom:float = 1.0, azimuth:int = 20, elevation:int = 20, margin:float = 0.5, z_limits:tuple = (-1.5, 1.5), grid:bool = False):
    plot.view_init(elev=elevation, azim=azimuth) # set initial view point

    # build array of all positions and list all x and y values seperately
    pos = np.stack(positions)
    x_vals = pos[:,0]
    y_vals = pos[:,1]
    
    # find max and min values of x and y
    xmin, xmax = x_vals.min(), x_vals.max()
    ymin, ymax = y_vals.min(), y_vals.max()

    # for single position point (spin) set a default such that axis sizes are not zero
    if np.isclose(xmin, xmax):
        xmin -= 0.5
        xmax += 0.5
    if np.isclose(ymin, ymax):
        ymin -= 0.5
        ymax += 0.5
    
    # add margins
    xmin -= margin
    xmax += margin
    ymin -= margin
    ymax += margin

    # compute midpoints and ranges
    xmid = 0.5*(xmin + xmax)
    ymid = 0.5*(ymin + ymax)
    dx = xmax - xmin
    dy = ymax - ymin
    dz = z_limits[1] - z_limits[0]

    # set limits centered on midpoints
    plot.set_xlim((xmid - dx/2), (xmid + dx/2))
    plot.set_ylim((ymid - dy/2), (ymid + dy/2))
    plot.set_zlim(z_limits[0], z_limits[1])

    # set box aspect for equal proportions and adjust zoom
    plot.set_box_aspect([dx, dy, dz], zoom=zoom)

    # toggle grid and labels
    if grid:
        plot.set_xlabel('X')
        plot.set_ylabel('Y')
        plot.set_zlabel('Z')
    else:
        plot.set_axis_off()


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