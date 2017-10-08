from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import SphericalVoronoi
import math
import matplotlib.pyplot as plt
import numpy as np
import random


#########################################################################
########################### PLOT FUNCTIONS ##############################
#########################################################################

def subdivide(verts, faces):
    """
    Subdivide each triangle into four triangles, pushing verts to the unit sphere
    """
    triangles = len(faces)
    for faceIndex in range(triangles):

        # Create three new verts at the midpoints of each edge:
        face = faces[faceIndex]
        [a, b, c] = [verts[vertIndex] for vertIndex in face]
        verts.append((a + b)/np.linalg.norm(a + b))
        verts.append((b + c)/np.linalg.norm(b + c))
        verts.append((a + c)/np.linalg.norm(a + c))

        # Split the current triangle into four smaller triangles:
        i = len(verts) - 3
        j, k = i+1, i+2
        faces.append((i, j, k))
        faces.append((face[0], i, k))
        faces.append((i, face[1], j))
        faces[faceIndex] = (k, j, face[2])

    return verts, faces


def plot_voronoi(sv):
    """
    displays the voronoi cells associated to points
    """
    sv.sort_vertices_of_regions()
    points = sv.points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='y', alpha=0.1)

    # plot generator points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')
    # plot Voronoi vertices
    # ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2], c='g')

    # indicate Voronoi regions (as Euclidean polygons)
    for region in sv.regions:
        random_color = colors.rgb2hex(np.random.rand(3))
        points = list(sv.vertices[region])
        triangles = []
        for i in range(len(points)-1):
            triangles.append([0, i, i+1])
        verts, faces = subdivide(points, triangles)
        # verts, faces = subdivide(verts, faces)
        # verts, faces = subdivide(verts, faces)
        for triangle in faces:
            i, j, k = triangle
            a, b, c = verts[i], verts[j], verts[k]
            polygon = Poly3DCollection([[a, b, c]], alpha=1.0)
            polygon.set_color(random_color)
            ax.add_collection3d(polygon)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


#########################################################################
########################### LLOYD RELAXATION ############################
#########################################################################

def lloyd_relaxation(points):
    """
    performs a spherical Lloyd's relaxation over the given set of points
    """
    center = np.array([0, 0, 0])
    radius = 1
    n_it = 100

    sv = None
    for i in range(n_it):
        sv = SphericalVoronoi(np.array(points), radius, center)
        points = []
        for region in sv.regions:
            polygon = sv.vertices[region]
            # pol_tup = [(vert[0], vert[1], vert[2]) for vert in polygon]
            centroid = get_centroid(polygon)
            points.append(centroid)
    return sv

#############################################################################
############################ CENTROID COMPUTATION ###########################
#############################################################################


def spher2cartesian(theta, phi):
    """
    returns the cartesian coordinates of (1, \theta, \phi)
    """
    return [np.cos(theta) * np.cos(phi), np.sin(phi), np.sin(theta)*np.cos(phi)]


def get_centroid(polygon):
    """
    computes the spherical centroid of the given polygon
    -----------------------------------------------------------------------------
    code taken from goo.gl/vWPNF6 (thanks to KobeJohn !)
    """
    # get base polygon data based on unit sphere
    r = 1.0
    point_count = len(polygon)
    reference = ok_reference_for_polygon(polygon)
    # decompose the polygon into triangles and record each area and 3d centroid
    areas, subcentroids = list(), list()
    for ia, a in enumerate(polygon):
        # build an a-b-c point set
        ib = (ia + 1) % point_count
        b, c = polygon[ib], reference
        if points_are_equivalent(a, b, 0.001):
            continue  # skip nearly identical points
        # store the area and 3d centroid
        areas.append(area_of_spherical_triangle(r, a, b, c))
        tx, ty, tz = zip(a, b, c)
        subcentroids.append((sum(tx)/3.0,
                             sum(ty)/3.0,
                             sum(tz)/3.0))
    # combine all the centroids, weighted by their areas
    total_area = sum(areas)
    subxs, subys, subzs = zip(*subcentroids)
    _3d_centroid = (sum(a*subx for a, subx in zip(areas, subxs))/total_area,
                    sum(a*suby for a, suby in zip(areas, subys))/total_area,
                    sum(a*subz for a, subz in zip(areas, subzs))/total_area)
    # shift the final centroid to the surface
    surface_centroid = scale_v(1.0 / mag(_3d_centroid), _3d_centroid)
    return surface_centroid


def ok_reference_for_polygon(polygon):
    point_count = len(polygon)
    # fix the average of all vectors to minimize float skew
    polyx, polyy, polyz = zip(*polygon)
    # /10 is for visualization. Remove it to maximize accuracy
    return (sum(polyx)/(point_count*10.0),
            sum(polyy)/(point_count*10.0),
            sum(polyz)/(point_count*10.0))


def points_are_equivalent(a, b, vague_tolerance):
    # vague tolerance is something like a percentage tolerance (1% = 0.01)
    (ax, ay, az), (bx, by, bz) = a, b
    return all(((ax-bx)/ax < vague_tolerance,
                (ay-by)/ay < vague_tolerance,
                (az-bz)/az < vague_tolerance))


def degree_spherical_to_cartesian(point):
    rad_lon, rad_lat, r = math.radians(point[0]), math.radians(point[1]), point[2]
    x = r * math.cos(rad_lat) * math.cos(rad_lon)
    y = r * math.cos(rad_lat) * math.sin(rad_lon)
    z = r * math.sin(rad_lat)
    return x, y, z


def area_of_spherical_triangle(r, a, b, c):
    # points abc
    # build an angle set: A(CAB), B(ABC), C(BCA)
    # http://math.stackexchange.com/a/66731/25581
    A, B, C = surface_points_to_surface_radians(a, b, c)
    E = A + B + C - math.pi  # E is called the spherical excess
    area = r**2 * E
    # add or subtract area based on clockwise-ness of a-b-c
    # http://stackoverflow.com/a/10032657/377366
    if clockwise_or_counter(a, b, c) == 'counter':
        area *= -1.0
    return area


def surface_points_to_surface_radians(a, b, c):
    """build an angle set: A(cab), B(abc), C(bca)"""
    points = a, b, c
    angles = list()
    for i, mid in enumerate(points):
        start, end = points[(i - 1) % 3], points[(i + 1) % 3]
        x_startmid, x_endmid = xprod(start, mid), xprod(end, mid)
        ratio = (dprod(x_startmid, x_endmid)
                 / (mag(x_startmid) * mag(x_endmid)))
        angles.append(math.acos(ratio))
    return angles


def clockwise_or_counter(a, b, c):
    ab = diff_cartesians(b, a)
    bc = diff_cartesians(c, b)
    x = dprod(ab, bc)
    if x < 0:
        return 'clockwise'
    elif x > 0:
        return 'counter'
    else:
        raise RuntimeError('The reference point is in the polygon.')


def diff_cartesians(positive, negative):
    return tuple(p - n for p, n in zip(positive, negative))


def xprod(v1, v2):
    x = v1[1] * v2[2] - v1[2] * v2[1]
    y = v1[2] * v2[0] - v1[0] * v2[2]
    z = v1[0] * v2[1] - v1[1] * v2[0]
    return [x, y, z]


def dprod(v1, v2):
    dot = 0
    for i in range(3):
        dot += v1[i] * v2[i]
    return dot


def mag(v1):
    return math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)


def scale_v(scalar, v):
    return tuple(scalar * vi for vi in v)

#############################################################################


if __name__ == '__main__':
    n = 40
    points = []
    for i in range(n):
        points.append(spher2cartesian(random.uniform(0., 2*np.pi), random.uniform(-np.pi/2, np.pi/2)))
    # sort vertices (optional, helpful for plotting)
    sv_init = SphericalVoronoi(np.array(points), 1.0, [0, 0, 0])
    sv = lloyd_relaxation(points)
    plot_voronoi(sv_init)
    plot_voronoi(sv)

