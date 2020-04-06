"""
Module to create a non-uniform lattice with quenched disorder to model heterogeneous catalysts on amorphous supports.

Additional functions can:
  * decorate the lattice surface with functional groups
  * locate empty sites that can graft a metal atom
  * compute local coordinates a site
"""


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.spatial import distance


def make_quenched_disorder_lattice(lattice_length, cov, use_seed=False, plot_dist_r=False):
    """
    Make square n x n x 2 lattice with quenched disorder.
    A uniform n x n lattice is made where each entry is a 2 x 1 matrix with Cartesian coordinates
    To create the quench disordered lattice, a displacement vector sampled from a disc is applied to each lattice point.

    Inputs:
        lattice_length - int for number of lattice points along edge
        displacement_amt - magnitude of displacement. Samples displacement vector from circle with radius displacement_amt
    """
    if use_seed:
        np.random.seed(484)

    # theta = 2 * np.pi * np.random.uniform(size=(lattice_length**2, 1))
    # if dist_type == "uniform":
    #     r = displacement_amt*np.sqrt(np.random.uniform(size=(lattice_length**2, 1)))
    #     # r = displacement_amt * np.random.uniform(size=(lattice_length**2, 1))

    # 2d normal distribution
    # TODO: Check and remove any sites that have r > 0.5
    # Low priority since cov matrix is so small
    a = np.random.multivariate_normal([0,0], cov * np.asarray([[1, 0], [0, 1]]), lattice_length**2)
    np.random.seed()
    x = a[:, 0]
    y = a[:, 1]
    if plot_dist_r:
        plt.hist(np.linalg.norm(a, axis=1), density = True, bins=50)
        plt.xlabel(r'$\hat{r}$')
        plt.ylabel(r'$\rho(\hat{r})$')
        plt.tight_layout()
        plt.show()

    # Convert to cartesian
    # x = r * np.cos(theta)
    # y = r * np.sin(theta)

    uniform_lattice = np.asarray([[(x, y) for x in range(lattice_length)] for y in range(lattice_length)])
    return uniform_lattice + np.dstack((x.reshape(lattice_length,lattice_length), y.reshape(lattice_length,lattice_length)))

def decorate_lattice(lattice, grafting_fraction, OH_fraction, O_fraction):
    """
    Assign groups to nodes on the lattice. Types are added by stacking a 2d array over the Cartesian coordinates
    0: Empty site (potentially a graftable site)
    1: -OH (Silanol group)
    2: -O- (Siloxane, makes labile bond)
    """
    np.random.seed(542)
    decorated_lattice = np.dstack((lattice, np.random.choice([0, 1, 2], size=lattice.shape[0:2], p=[grafting_fraction, OH_fraction, O_fraction])))
    np.random.seed()
    return decorated_lattice

def locate_grafting_sites(decorated_lattice):
    """
    Locate sites which can graft: empty sites sites that have two pairs of -OH and Si-O-Si groups, where each pair is on opposite sides
    Also finds "trouble sites": pairs of graftable sites that share and OH or Si-O-Si group. Only one pair of sites can graft, which will be important in the population balance model.

    input: n x n x 3 lattice with functionalized nodes
    outputs: 
       - (m x 2) np array of indices of sites that can graft
       - 1d array of indices of graftable sites that share OH or Si-O-Si pairs
    """
    site_type = decorated_lattice[:, :, 2]

    empty_sites = np.vstack(np.where(decorated_lattice[1:-1, 1:-1, 2] == 0)).T + 1
    n_empty_sites = empty_sites.shape[0] 
    
    graftable_sites = []

    for i in empty_sites:
        # Start if statement thunderdome. TODO: This is in no way general if we look at other grafting environments
        # See if top and bottom are the same
        if site_type[i[0] + 1, i[1]] == site_type[i[0] - 1, i[1]]:
            # See if left and right are the same
            if site_type[i[0], i[1]  + 1] == site_type[i[0], i[1] - 1]:
                # See if cross sites are different
                if site_type[i[0], i[1]  + 1] != site_type[i[0] - 1, i[1]]:
                    # Finally, make sure site pairs are not bare sites
                    if site_type[i[0], i[1]  + 1] != 0 and site_type[i[0] - 1, i[1]] != 0:
                        graftable_sites.append(i)
    
    # Find sites that compete to graft, i.e. two empty sites share a neighboring -OH group
    # These cases are important in the population balance model, since only one of the pairs can graft.
    # This procedure also only ID's pairs of competing sites, which are found when the Manhattan distance < 2 in index space.  
    # Its possible for "3 body interactions" where 1 site grafting shuts down 2 neighboring sites, or for even higher order terms
    # These seem like they'll be a lot less common though so they're being ignored for now

    # Make distance matrix (in 1D array format)
    distance_matrix = distance.pdist(graftable_sites, metric='cityblock')
    n_graftable_sites = len(graftable_sites)

    # Convert to 2D distance matrix using upper triangular index notation
    tri = np.zeros((n_graftable_sites, n_graftable_sites))
    tri[np.triu_indices(n_graftable_sites,1)] = distance_matrix
    # Sites competing to graft have a Manhattan distance of 2 in index space
    competing_site_indices = np.where(tri==2)

    # lattice positions of sites that compete (not currently used)
    competing_sites = np.asarray([[graftable_sites[competing_site_indices[0][i]], graftable_sites[competing_site_indices[1][i]]] for i in range(len(competing_site_indices[0]))])

    # Second return is unique array of graftable site indices that will compete to graft
    return np.asarray(graftable_sites), np.asarray(list(set([*competing_site_indices[0], *competing_site_indices[1]])))

def nearest_neighbor_distances(lattice, graftable_sites):
    """
    Returns the positions of the four nearest neighbors as a n x 4 matrix, where n is # of graftable sites.
    The 4 x 1 distances along the boundaries are zero since boundaries are ignored.

    The order of the distances are:
    0 - 1: ordered -OH distances 
    2 - 3: order Si-O-Si distances
    """

    ### Old method of calculating NN distances
    # lattice_len = len(lattice)
    # near_neighbor_d = np.zeros((lattice_len, lattice_len, 4))
    # Maybe theres a way for this to be more efficient, but this calculation is only done once
    # for x in range(1, lattice_len - 1):
    #     for y in range(1, lattice_len - 1):
    #         near_neighbor_d[x, y, 3] = np.linalg.norm(lattice[x, y, :] - lattice[x - 1, y, :]) # bottom
    #         near_neighbor_d[x, y, 2] = np.linalg.norm(lattice[x, y, :] - lattice[x + 1, y, :]) # top
    #         near_neighbor_d[x, y, 1] = np.linalg.norm(lattice[x, y, :] - lattice[x, y + 1, :]) # right
    #         near_neighbor_d[x, y, 0] = np.linalg.norm(lattice[x, y, :] - lattice[x, y - 1, :]) # left
    
    # TODO: Not general if we look at other grafting environments. Maybe pass in a list with pattern(s) to match that will be looped over?
    near_neighbor_d = np.zeros((len(graftable_sites), 4))    
    y = 0
    for x in graftable_sites:
        # Place -OH sites in first 2 columns
        if lattice[x[0] - 1, x[1], 2] == 1:
            near_neighbor_d[y, 0] = np.linalg.norm(lattice[x[0], x[1], 0:2] - lattice[x[0] - 1, x[1], 0:2])  
            near_neighbor_d[y, 1] = np.linalg.norm(lattice[x[0], x[1], 0:2] - lattice[x[0] + 1, x[1], 0:2]) 
            near_neighbor_d[y, 2] = np.linalg.norm(lattice[x[0], x[1], 0:2] - lattice[x[0], x[1] + 1, 0:2]) 
            near_neighbor_d[y, 3] = np.linalg.norm(lattice[x[0], x[1], 0:2] - lattice[x[0], x[1] - 1, 0:2]) 
        # Place Si-O-Si sites in last 2 columns
        else:
            near_neighbor_d[y, 3] = np.linalg.norm(lattice[x[0], x[1], 0:2] - lattice[x[0] - 1, x[1], 0:2])  
            near_neighbor_d[y, 2] = np.linalg.norm(lattice[x[0], x[1], 0:2] - lattice[x[0] + 1, x[1], 0:2]) 
            near_neighbor_d[y, 1] = np.linalg.norm(lattice[x[0], x[1], 0:2] - lattice[x[0], x[1] + 1, 0:2]) 
            near_neighbor_d[y, 0] = np.linalg.norm(lattice[x[0], x[1], 0:2] - lattice[x[0], x[1] - 1, 0:2]) 
        y+=1

    # Sort -OH and Si-O-Si distances separately
    # TODO: also not general for different grafting environments. 
    # Would need some additional logic if empty sites may or may not be present and number of bond types is variable 
    near_neighbor_d[:,0:2].sort()
    near_neighbor_d[:,2:].sort()
    return near_neighbor_d

def compute_local_coordinates(lattice, graftable_sites):
    """
    Compute local coordinates of a site for ML model
    inputs:
      - lattice
      - graftable_sites
    outputs:
      local_coords: n x 4 array of local coordinates.
        - 0: OH-OH distance
        - 1: Siloxane distance
        - 2: angle between OH and Si-O-Si
        - 3: distance between midpoints of OH-OH and Siloxane
    """
    local_coords = np.zeros((len(graftable_sites), 4))    
    y = 0
    for x in graftable_sites:
        # Above and below = -OH group
        if lattice[x[0] - 1, x[1], 2] == 1:
            OH_vector = lattice[x[0] - 1, x[1], 0:2] - lattice[x[0] + 1, x[1], 0:2]
            OH_midpt = 0.5 * (lattice[x[0] - 1, x[1], 0:2] + lattice[x[0] + 1, x[1], 0:2])

            siloxane_vector = lattice[x[0], x[1] - 1, 0:2] - lattice[x[0], x[1] + 1, 0:2]
            siloxane_midpt = 0.5 * (lattice[x[0], x[1] - 1, 0:2] + lattice[x[0], x[1] + 1, 0:2])
        # Above and below = Siloxanes
        else:
            OH_vector = lattice[x[0], x[1] - 1, 0:2] - lattice[x[0], x[1] + 1, 0:2]
            OH_midpt = 0.5 * (lattice[x[0], x[1] - 1, 0:2] + lattice[x[0], x[1] + 1, 0:2])

            siloxane_vector = lattice[x[0] - 1, x[1], 0:2] - lattice[x[0] + 1, x[1], 0:2] 
            siloxane_midpt = 0.5 * (lattice[x[0] - 1, x[1], 0:2] + lattice[x[0] + 1, x[1], 0:2])

        # OH-OH distance
        local_coords[y, 0] = np.linalg.norm(OH_vector)
        # Si-O-Si - Si-O-Si distance
        local_coords[y, 1] = np.linalg.norm(siloxane_vector)
        # Angle between OH and siloxane vectors
        local_coords[y, 2] = vec_angle(OH_vector, siloxane_vector) * 180/np.pi
        # Parallelagramity
        local_coords[y, 3] = np.linalg.norm(OH_midpt - siloxane_midpt)
        # q1, not in use
        # local_coords[y, 3] = q_l(lattice[x[0], x[1] + 1, 0:2], lattice[x[0] + 1, x[1], 0:2], lattice[x[0], x[1] - 1, 0:2], lattice[x[0] - 1, x[1], 0:2], 2)
        y+=1
    return local_coords

def vec_angle(a, b):
    return np.arccos(np.dot(a,b)/((np.linalg.norm(a))*(np.linalg.norm(b)))) 

def q_l(a, b, c, d, l=1):
    """
    Compute q_l coordinate of 4 points, a, b, c, and d, taking the center of mass as the origin
    """
    com = (a + b + c + d)/4
    distances = np.asarray([np.linalg.norm(a - com), np.linalg.norm(b - com), np.linalg.norm(c - com), np.linalg.norm(d - com)])

    # Compute directed angles from [-pi, pi]
    angles = np.asarray([0, np.arctan2(b[1] - com[1], b[0] - com[0]) - np.arctan2(a[1] - com[1], a[0] - com[0]), 
                            np.arctan2(c[1] - com[1], c[0] - com[0]) - np.arctan2(a[1] - com[1], a[0] - com[0]),
                            np.arctan2(d[1] - com[1], d[0] - com[0]) - np.arctan2(a[1] - com[1], a[0] - com[0])])
    
    # Convert range to [0, 2pi]
    angles = np.where(angles < 0, angles + 2 * np.pi, angles)
    real = np.dot(distances, np.cos(l * angles))
    imag = np.dot(distances, np.sin(l * angles))

    return np.sqrt(real**2 + imag**2)