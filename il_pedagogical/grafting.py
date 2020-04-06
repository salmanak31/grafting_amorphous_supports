# Standard libraries
import json
import matplotlib
# pip imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scipy.integrate as integrate
import seaborn as sns
from matplotlib import animation

# non-pip
from potentials import *
from visualization import *
from lattice import *
from il import *
from il_pedagogical import *


def histogram_graft_mod(h,b,t,T):
    kb = 1.38064852*10**(-23)
    h_1 = 6.62607004*10**(-34)
    mid_hist = np.asarray([(b[i]+b[i+1])/2 for i in range(np.shape(b)[0]-1)])
    return h*np.exp(-(kb*T/h_1)*np.exp(-(mid_hist*1000)/(8.314*T))*t*(20/760))

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
            rand_delete.append(np.random.choice(len(populations[i]), size = delta_pop[i],replace= False))
        # append an empty list if the bin population = 0 (since there is nothing left to graft)
        else:
            rand_delete.append([])

    # grafted sites
    populations = np.asarray(populations)

    # list of adsorption energies of grafted sites
    grafted_pop_ads = np.array([])

    for  i in range(0,len(populations)):
        # concatenate the list of all grafted sites
        grafted_pop_ads = np.concatenate((grafted_pop_ads, populations[i][rand_delete[i]][:,0]))

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


def grafted_sites_model(graft_E_predicted, n_bins, time):
    
    graft_E_predicted = np.reshape(graft_E_predicted,(np.shape(graft_E_predicted)[0],1))

    histogram_graft, bins_graft = np.histogram(graft_E,n_bins)

    # bin index of graft_E elements
    bin_index_r = np.digitize(graft_E_predicted, bins_graft, right=False)
    bin_index_l = np.digitize(graft_E_predicted, bins_graft, right=True)
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

    # array combining grafting bin index, and grafting energy (sorted using bin index)
    combined = np.hstack((graft_E_predicted,bin_index))
    combined_sorted = combined[combined[:,1].argsort()]

    # set up populations = # of bins. This is the starting population i.e. all the sites are ungrafted at the moment
    populations = []
    for i in range(0,int(combined_sorted[np.argmax(combined_sorted[:,1])][2])):
        populations.append(combined_sorted[np.where(combined_sorted[:,1]==float(i+1))])

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

    return graft_E_modified



def grafting_learn(graft_E, local_coordinates, initial_pool_size):
   
    
    n_sites = len(graft_E)
    sampled_graft_sites = sample_sites(graft_E, initial_pool_size)
    sampled_grafting_barrier_heights = graft_E[sampled_graft_sites]
    sampled_local_coordinates = local_coordinates[sampled_graft_sites]
    #TRAINING
    M, model__grafting_barriers_LOO, residuals = train(sampled_local_coordinates, sampled_grafting_barrier_heights)
    
    # the function for predicting adsorption energies can be used to predict grafting barriers
    model_barrier_heights = predicted_adsorption_energies(sampled_local_coordinates, sampled_grafting_barrier_heights, M, local_coordinates, n_sites) 


    return model_barrier_heights

if __name__ == "__main__":
    initial_pool_size = 50    
    output = 'C:\\Users\\Salman\\Desktop\\Research\il-pedagogical\\logs\\results\\morse_potential_{}.json'.format(initial_pool_size)

    lattice_size = 200

    T = 300
    cov = 0.00022
    

    displacement_type = 'normal'
    disp_kwargs = {'covariance' : [[cov, 0], [0, cov]]}

    D_MO = 524.4
    r_eq_MO = 1.
    a_MO = 1.9
    # a_MO = 2.1

    D_M_O = 33.6 
    r_eq_M_O = 1.16
    a_M_O = 2.3
    # a_M_O = 3.5

    E_MA = 90

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

    graft_E, ads_E = grafting_energies(MO_Morse, siloxane_Morse, E_MA, T, graftable_sites, decorated_lattice)
    graft_E = 131.3+0.5*graft_E
    # print(graft_E)
    hist_data, bin_edges = np.histogram(graft_E, bins = 20)
    n_bins = 40
    n_samples = 10
    mid_hist = np.asarray([(bin_edges[i]+bin_edges[i+1])/2 for i in range(np.shape(bin_edges)[0]-1)])
    lattice_len = int(np.sqrt(len(graft_E)) + 2)
    sampled_sites = sample_sites(graft_E, n_samples)
    sampled_barrier_heights = graft_E[sampled_sites]
    sampled_local_coordinates = local_coordinates[sampled_sites]

    M, model_barriers_LOO, residuals = train(sampled_local_coordinates, sampled_barrier_heights)
    n_sites = np.shape(graft_E)[0]
    model_barrier_heights = predicted_adsorption_energies(sampled_local_coordinates, sampled_barrier_heights, M, local_coordinates, n_sites) 

    time = [0, 1, 10, 100, 1000, 10000]
    wid = mid_hist[2]-mid_hist[1]


    # Full grafting solution ( no training )
    for i in time:
        
        h_mod = histogram_graft_mod(hist_data, bin_edges, i, T)
        # plt.hist(h_mod,stacked=True, fill=False)
        plt.bar(mid_hist, height = h_mod, width = wid, alpha = 0.6)

    plt.xlabel(r'$G_{grafting}$, kJ/mol')
    plt.ylabel(r'$\rho(G_{grafting})$')
    plt.tight_layout()
    plt.show()
    
    fig = plt.figure(figsize=(12,12))

    ax = fig.add_subplot(1, 1,1)
    model_heights = model_barriers_LOO
    # initial_pool = il_gaussian.sampled_sites(0)
    true_barriers = sampled_barrier_heights
#     print(true_barriers[initial_pool])
    plt.title('Iteration 0, Initial Pool = {}'.format(n_samples))
    plot_trained(ax, model_heights, true_barriers, 2.5, n_samples)
    

    plt.show()


    fig, axs = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    local_coordinates = local_coordinates
    true_barriers = graft_E
    model_heights = model_barrier_heights

    sampled_local_coordinates = sampled_local_coordinates
    initial_pool = n_samples
    initial_pool_barriers = model_barriers_LOO


    axs[0].set_title('True Barriers')
    tri1 = axs[0].tricontourf(local_coordinates[:,0], local_coordinates[:, 1], true_barriers, 20, 
                    vmin=np.min(true_barriers), vmax=np.max(true_barriers))
    axs[0].set_xlabel('OH-OH distance')
    axs[0].set_ylabel('Siloxane-Siloxane distance')
    fig.colorbar(tri1, ax=axs[0])


    axs[1].set_title('Model Predicted Barriers, Iteration 30')
    tri2 = axs[1].tricontourf(local_coordinates[:,0], local_coordinates[:, 1], model_heights, 20, 
                    vmin=min(model_heights), vmax=max(model_heights))
    axs[1].scatter(sampled_local_coordinates[:,0], sampled_local_coordinates[:,1], s=50, c='blue', edgecolors='black', label='Initial Pool')
    # axs[1].scatter(sampled_local_coordinates[50:,0], sampled_local_coordinates[50:,1], s=50, c='w', edgecolors='black', label='Importance Sampled')
    axs[1].set_xlabel('OH-OH distance')
    # plt.legend()
    # axs[1].set_ylabel('Siloxane-Siloxane distance')
    fig.colorbar(tri2, ax=axs[1])

    # plt.colorbar([tri1, tri2], ax=axes.ravel().tolist())

    plt.tight_layout()
    plt.show()
