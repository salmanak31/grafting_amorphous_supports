# Standard libraries
import json

# pip imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.integrate as integrate

# non-pip
from potentials import *
from visualization import *
from lattice import *
from il import *

def adsorption_energies(lattice, dissociation_type=1, potential_type=1, **kwargs):
    """
    Get activation energy of a site based on local environment.
    Based off Morse potential, V(r) = D(1 - exp(-a(r-r_eq)))
    inputs:
    r -  4 x 1 np array of distances to nearest neighbors on the lattice
    dissociation_type - int
        1: takes weakest bond as bond broken
        2: takes ensemble avg over all bonds

    potential_type - in
        1: Morse potential, w(r) = D(1 - exp(-a(r-r_eq)))
            kwargs:
            r_eq - float of non-dimensional "equilibrium" bond length between lattice sites 
            D - well depth
            a - force constant like value

        2: Gaussian potential, w(r) = -D * exp(r**2/(2 * sigma**2))
            kwargs:
            D - potential depth
            std - standard deviation of Gaussian
    """
    # FIXME: Be able to take in NN distances with edges pruned
    # FIXME: Theres some reworking needing to be done in the arguments here
    NN_distances = nearest_neighbor_distances(lattice, None)
    
    if potential_type == 1:
        V = morse_potential(NN_distances, kwargs['r_eq'], kwargs['D'], kwargs['a'])

    elif potential_type == 2:
        V = normal(NN_distances, kwargs['D'], 0, kwargs['std'])

    if dissociation_type == 1:
        adsorption_energies = -np.asarray(np.max(V,axis=2)) -0.2*kwargs['D']
    
    elif dissociation_type == 2:
        B = 1000./(8.314 * kwargs['T'])
        exp_V = np.exp(-1 * B * V)
        adsorption_energies = 1 / B * np.asarray(np.log(np.sum(exp_V,axis=2))) -0.2*kwargs['D']

    # Return non-boundary sites
    return adsorption_energies[1:-1, 1:-1]


def histogram_graft_mod(h,b,t,T):
    kb = 1.38064852*10**(-23)
    h_1 = 6.62607004*10**(-34)
    mid_hist = np.asarray([(b[i]+b[i+1])/2 for i in range(np.shape(b)[0]-1)])
    return h*np.exp(-(kb*T/h_1)*np.exp(-(mid_hist*1000)/(8.314*T))*(20/760)*t)

def histogram_graft_mod_non_dim( h, b, Tau, T):
    # Tau is non-dim time
    kb = 1.38064852*10**(-23)
    h_1 = 6.62607004*10**(-34)
    mid_hist = np.asarray([(b[i]+b[i+1])/2 for i in range(np.shape(b)[0]-1)])
    k0 = (kb*T/h_1)*np.exp(-((131.3)*1000)/(8.314*T))
    return h*np.exp(-(((kb*T/h_1)*np.exp(-(mid_hist*1000)/(8.314*T)))*(20/760)/k0)*Tau)

def histogram_graft_mod_non_dim_2( h, b, Tau, T):
    # Tau is non-dim time
    mid_hist = np.asarray([(b[i]+b[i+1])/2 for i in range(np.shape(b)[0]-1)])
    return h*np.exp(-(20/760)*(np.exp(-(mid_hist*1000)/(8.314*T)))*Tau)

def grafting_population(graft_E, ads_E, n_bins, time):
    
    graft_E = np.reshape(graft_E,(np.shape(graft_E)[0],1))
    ads_E = np.reshape(ads_E,(np.shape(ads_E)[0],1))

    histogram_graft, bins_graft = np.histogram(graft_E,n_bins)

    # bin index of graft_E elements
    bin_index_r = np.digitize(graft_E, bins_graft, right=False)
    bin_index_l = np.digitize(graft_E, bins_graft, right=True)
    diff = bin_index_r - bin_index_l
    sum = bin_index_r + bin_index_l
    diff_sum = diff*sum
    for item in np.where(diff_sum > 1)[0]:
        bin_index_r[item] = np.shape(bins_graft)[0] - 1 
    for item in np.where(diff_sum == 1)[0]:
        bin_index_r[item] = 1
    bin_index = bin_index_r
    bin_index = np.reshape(bin_index,(np.shape(bin_index)[0],1))
    ##END##

    # array combining grafting bin index, ads_E, and grafting energy (sorted using bin index)
    combined = np.hstack((ads_E,graft_E,bin_index))
    combined_sorted = combined[combined[:,2].argsort()]
    
    # np.savetxt("test_101.txt",combined_sorted)
    
    # set up populations = # of bins. This is the starting population i.e. all the sites are ungrafted at the moment
    populations = []
    for i in range(0,int(combined_sorted[np.argmax(combined_sorted[:,2])][2])):
        populations.append(combined_sorted[np.where(combined_sorted[:,2]==float(i+1))])

    # for i in range(0,len(populations)):
    #     np.savetxt(str(i)+".txt",np.asarray(populations[i]))
    
    # Calculate the decay in ungrafted population as a function of time
    # This is a list of length = # of bins and each element is the number of sites that have to be delted
    delta_pop = [int(np.round(histogram_graft - histogram_graft_mod(histogram_graft, bins_graft, time))[i]) for i in range(np.shape(histogram_graft)[0])]

    # generate random numbers to delete sites based on the population decrease
    rand_delete = []
    for i in range(0,len(histogram_graft)):
        # generate random numbers for bins which have a non zero population
        if len(populations[i])>0:
            rand_delete.append(np.random.randint( 0, len(populations[i]), size = delta_pop[i]))
        # append an empty list if the bin population = 0 (since there is nothing left to graft)
        else:
            rand_delete.append([])

    # grafted sites
    populations = np.asarray(populations)

    # list of adsorption energies of grafted sites
    grafted_pop_ads = np.array([])

    for i in range(0,len(populations)):
        # concatenate the list of all grafted sites
        grafted_pop_ads = np.concatenate([grafted_pop_ads, populations[i][rand_delete[i]][:,0]])

    # population change after grafting
    for i in range(len(rand_delete)):
        # Skip if the number of sites in a histogram are zero (nothing left to graft) or if there is no reduction in the number of sites (i.e. the random list for this bin is empty)
        if len(rand_delete[i]) == 0 or np.shape(populations[i])[0] == 0 :
            pass
        else:
            populations[i] = np.delete(populations[i],rand_delete[i],axis = 0)
    # populations = new population

    # population of grafted sites (only grafting energies)
    # this can be used to construct the decay of graftable site populations (as a function of deltaG_graft)
    graft_E_modified = np.array([])
    for i in range(0,len(populations)):
        graft_E_modified = np.concatenate([graft_E_modified, populations[i][:,1]])

    return grafted_pop_ads, graft_E_modified



if __name__ == "__main__":

    initial_pool_size = 50    
    output = 'C:\\Users\\Craig\\Desktop\\repos\il-pedagogical\\logs\\morse_potential_{}.json'.format(initial_pool_size)

    lattice_size = 1200

    T = 300
    cov = 0.00022

    displacement_type = 'normal'
    disp_kwargs = {'covariance' : [[cov, 0], [0, cov]]}

    D_MO = 500.
    r_eq_MO = 1.
    a_MO = 1.9

    D_M_O = 120.
    r_eq_M_O = 1.16
    a_M_O = 2.3

    E_MA = 160.

    empty_fraction = 0.3
    OH_fraction = 0.3
    siloxane_fraction = 0.4

    MO_Morse = {'D' : D_MO, 'a' : a_MO, 'r_eq' : r_eq_MO}
    siloxane_Morse = {'D' : D_M_O, 'a' : a_M_O, 'r_eq' : r_eq_M_O}
    lattice_fractions = {'empty' : empty_fraction, 'OH' : OH_fraction, 'Siloxane' : siloxane_fraction}

    lattice = make_quenched_disorder_lattice(lattice_size, cov, True, False)
    decorated_lattice = decorate_lattice(lattice, empty_fraction, OH_fraction, siloxane_fraction)

    graftable_sites, competing_sites = locate_grafting_sites(decorated_lattice)
    # plot_lattice(decorated_lattice, True, graftable_sites)

    local_coordinates = compute_local_coordinates(decorated_lattice, graftable_sites)
    local_coordinates_dict = {'OH-OH-distance' : local_coordinates[:, 0],
                              'siloxane-distances' : local_coordinates[:, 1],
                              'OH-siloxane-angle' : local_coordinates[:, 2], 
                              'OH-siloxane-midpoint' : local_coordinates[:, 3]}

    # local_coordinates = np.delete(local_coordinates, (2,3), axis=1)
    graft_E, ads_E = grafting_energies(MO_Morse, siloxane_Morse, E_MA, T, graftable_sites, decorated_lattice)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.set_xlabel(r'$E_a$, kJ/mol')
    ax2.set_xlabel(r'$E_a$, kJ/mol')
    histogram(ax1, ax2, ads_E, T, n_bins=20)
    plt.show()

    site_avg_E = k_weighted_avg_activation_E(ads_E, T) + 50

    barrier_distribution = ads_E + 50

    # local_coordinates = np.delete(local_coordinates, (0, 1), axis=1)

    IL = importance_learning(barrier_distribution, local_coordinates, T, initial_pool_size, 100, plot_every=25)
    
    parameters = {
        'lattice-length' : lattice_size,
        'displacements' : {'displacement-type' : 'Normal',
                           'displacement-kwargs' : disp_kwargs},
        # 'dissociation-type' : dissociation_type,
        'site-fractions' : lattice_fractions,
        'potentials' : {'MO-potential' : MO_Morse,
                        'siloxane-potential': siloxane_Morse,
                        'metal-adsorbate-bond' : E_MA},
        'number-graftable-sites' : len(graftable_sites),
        'number-competing-sites' : len(competing_sites),
        'T' : T,
        'initial-pool-size' : initial_pool_size
    }

    # Export stuff
    results = {
        'parameters' : parameters,
        'local-coordinates' : local_coordinates.tolist(),
        'competing-sites' : competing_sites.tolist(),
        'grafting-barrier-heights' : graft_E.tolist(),
        'barrier-heights' : barrier_distribution.tolist(),
        'True <Ea>k' : site_avg_E,
        'importance-learning' : IL
    }

    a = json.dumps(results, ensure_ascii=False, indent=2)
    with open(output, 'w') as outfile:
        outfile.write(a)    

