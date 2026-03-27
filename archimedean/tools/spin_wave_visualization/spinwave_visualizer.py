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