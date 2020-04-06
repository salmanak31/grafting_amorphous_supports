import warnings
import json
from decimal import Decimal

import numpy as np
import matplotlib.pyplot as plt
from metric_learn import MLKR

from visualization import *

def k_weighted_avg_activation_E(barrier_heights, T):
    """
    Computes k weighted mean from a distribtuion

    input -
    barrier_heights - nx1 numpy array of barrier heights in kJ/mol
    T - float of catalyst operation temperature

    returns - 
    k_weight_mean - k weighted mean of barrier heights in kJ/mol
    """
    B = 1000/8.314/T
    
    return np.sum(barrier_heights * np.exp(-B * barrier_heights))/(np.sum(np.exp(-B * barrier_heights)))

def sample_sites(barrier_distribution, n_samples, rand_seed, use_seed = False):
    """
    Draws random pairs of points that are unique.
    Points along the boundary, [0, :], [:, 0], [-1, :], [:, -1], are ignored 

    lattice_dims -- pair of lattice dimensions, e.g. (50, 50) draws x-y coordinates in the domain ([0, 50], [0, 50])
    n_samples -- number of unique sites to sample
    
    We dont want to sample the boundaries, but its easier to first sample the lower left corner, which is (lattice_dims - 2 x lattice_dims - 2)
    and then shift the samples to exclude the boundaries by adding the number 1

    o = sample-able region
    x = non-sample-able region

      sample      shift samples
      corner      to boundaries

    x x x x x     x x x x x
    x x x x x     x o o o x
    o o o x x --> x o o o x
    o o o x x     x o o o x
    o o o x x     x x x x x
    """
    # See docstring for what's happening here
    # lattice_dims = np.asarray([lattice_dims - 2, lattice_dims - 2])

    # pull from list of barrier distributions without replacement to get possible sites

    if use_seed:
        np.random.seed(rand_seed)
    st0 = np.random.get_state()
    idx = np.random.choice(np.arange(len(barrier_distribution)), n_samples, replace=False)

    return idx, st0

def inspect_sites(lattice_len, local_coordinates, barrier_heights, n_samples):
    """
    Like doing a DFT calculation on a site. 
    Inputs - 
    lattice - numpy array of lattice of sites (ensemble of sites), 
    nearest_neighbor_distances - n x n x 4 array of distances to nearest neighbors. Local coordinates
    barrier_heights - n x n  np array of barrier heights from model chemistry
    n_samples - number of sites to sample

    returns - 
    sampled_sites - n_samples x 2 array of randomly sampled site IDs
    sampled_barrier_heights - n_samples x 1 np array, true barrier heights of sampled sites
    sampled_NN_distances - n_samples x 4 np array, array of nearest neighbor distances of sampled sites
    """
    
    sampled_sites = sample_sites(barrier_heights, n_samples)
    sampled_barrier_heights = barrier_heights[sampled_sites]
    sampled_NN_distances = local_coordinates[sampled_sites]
    return sampled_sites, sampled_barrier_heights, sampled_NN_distances

def biased_sample(model_barrier_heights, T):
    """
    Samples sites with replacement using biased probabilities (exp(-BE))
    
    input - 
    predicted_adsorption_energies - n_test x 1 numpy array of predicted adsorption energies
    T - catalyst operation temperature

    returns - 
    test site - ID of chosen site
    """
    B = 1000/8.314/T
    exp_E = np.exp(-B * model_barrier_heights) / np.sum(np.exp(-B * model_barrier_heights))
    return np.random.choice(np.arange(len(model_barrier_heights)), replace=True, p = exp_E)

def biased_error(adsorption_energies):
    """
    Calculates the error of the computed k-weighted average if sites are sampled using the biased
    distribution using the central limit theorem. However, this doesn't take into account the error associated with the model.
    Inputs - 
    predicted_adsorption_energies - numpy array of adsoprtion energies sampled using the biased distribution

    returns - 
    biased sampling error - error in computing the k-weighted average on sampling sites using the biased distribution 
    """
    
    return np.var(adsorption_energies)/np.sqrt(len(adsorption_energies))


def unbiased_error(adsorption_energies,T):
    """
    Calculates the error of the computed k-weighted average if sites are sampled using random sampling.
    Inputs - 
    adsorption_energies - numpy array of adsortion energies sampled randomly
    temperature - float of  catalyst operation temperature

    returns - 
    sampling error - error in computing the k-weighted average on sampling sites randomly 
    """
    B = 1000/8.314/T
    Num = (np.exp(-B*adsorption_energies))*adsorption_energies
    Denom = np.sum(np.exp(-B*adsorption_energies))
    return np.sqrt(len(adsorption_energies)*(np.var(Num/Denom)))

def bootstrap_unbiased_error(barrier_distribution, n_samples, T):
    """
    Bootstrap barrier height distribution to get standard error estimates unbiased, k-weighted, activation energy. 
    Uncertainty returned is 95% confidence interval.
    """
    values, indices = np.histogram(barrier_distribution, bins=40)
    weights=values/np.sum(values)
    Ea_k = []
    for i in range(5000):
        bootstrapped_dist = np.random.choice(indices[1:], n_samples, p=weights)
        Ea_k.append(k_weighted_avg_activation_E(bootstrapped_dist, T))
    return np.average(Ea_k), np.std(Ea_k) * 1.96

def train(X_train, Y_train, is_verbose=False):
    """
    Trains the metric learning model on the input training sample.
    Inputs - 
    X_train - nx4 numpy array of site parameters
    Y_train - nx4 numpy array of activation barriers of sites
    is_verbose - boolean, whether to print training progress (i.e. cost as a function of training steps)

    returns - 
    M - 4x4 numpy array of the Mahalanobis distance matrix
    model_predicted_barriers - barriers of sites predicted by the metric learning. 
                               The barriers of the training set are predicted by excluding itself
    residuals - numpy array of true barrier - predicted barrier 
    """
    mlkr = MLKR(verbose=is_verbose)
    fit_final = mlkr.fit(X_train, Y_train)
    M = fit_final.metric()
    model_predicted_barriers = []
    Y_train =  [Decimal(Y_train[i]) for i in range(0,len(Y_train))]
    for i in range(0,len(Y_train)):
        test = X_train[i]
        temp = np.delete(X_train,i,axis=0)
        delta = temp-test
        dist_mat = np.diagonal(np.matmul(np.matmul(delta,M),np.transpose(delta)))
        dist_mat = np.transpose([Decimal(dist_mat[i]) for i in range(len(dist_mat))])
        k_mat = np.exp(-1*dist_mat)
        temp2 = np.delete(Y_train,i,axis=0)
        model_predicted_barriers.append(np.sum(np.multiply(temp2,k_mat))/(np.sum(k_mat)))
    model_predicted_barriers  = np.asarray(model_predicted_barriers)
    residuals = Y_train - model_predicted_barriers 
    return M, model_predicted_barriers, residuals

def predicted_adsorption_energies(training_local_coordinates, training_barrier_heights, M, local_coordinates, n_sites):
    """
    Prediction of adsorption energy with the trained metric learning model
    Inputs - 
    training_NN_distances - n_train x 4 numpy array of NN distances of sites on which the metric learning model has been trained
    training_barrier_heights - n_train x 1 numpy array of true adsoprtion energies of sites on which the metric learning model has been trained
    M - 4x4 numpy array of the trained Mahalanobis distance
    NN_distances - n_test x 4 numpy array of NN distances of sites for which adsorption energy prediction has to be made
    lattice_size - float of size lattice

    returns - 
    predicted_energies - n_test x 1 numpy array of adsorption energies of the test set predicted by the model
    """
    # This seems sketchy AF but is not needed because the warning is resolved. 
    # If a site is really far from training data in distance space, the Gaussian kernels will all be 0 because of lack of precision
    # In the sum / sum line at the end of the for loop, you get 0/0 = NaN and a raised runtime warning.
    # The NaN cases are addressed in the next for loop, thus the runtime warning is not needed
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    predicted_energies = np.zeros(n_sites)
    training_size = len(training_barrier_heights)
    training_barrier_heights = np.longdouble(training_barrier_heights)

    for i in range(n_sites):
        delta = training_local_coordinates-local_coordinates[i]
        d = np.diagonal(delta @ M @ delta.T)
        k_mat = np.exp(-1 * d)
        predicted_energies[i] = np.sum(np.multiply(training_barrier_heights, k_mat)) / (np.sum(k_mat))

    # If any sites are too far from the training data they give NaN b/c of not enough precision
    # This happens b/c sum(training_barrier_heights @ k_mat)/sum(kmat) = 0/0 = NaN
      
    # Find NaN sites and rerun them with infinite precision via Decimal()
    # This isn't done for every site because it's slow AF
    bad_list = np.argwhere(np.isnan(predicted_energies)).ravel().tolist()
    if len(bad_list) > 0:
        training_decimal = [Decimal(float(training_barrier_heights[i])) for i in range(training_size)]
        
        for i in bad_list:
            delta = training_local_coordinates-local_coordinates[i]
            dist_mat = np.diagonal(np.matmul(np.matmul(delta, M), np.transpose(delta)))
            dist_mat = np.transpose([Decimal(dist_mat[i]) for i in range(len(dist_mat))])
            k_mat = np.exp(-1*dist_mat)
            predicted_energies[i] = np.sum(np.multiply(training_decimal, k_mat)) / (np.sum(k_mat))

    return predicted_energies

def importance_learning(barrier_distribution, local_coordinates, T, initial_pool_size, n_iterations, plot_every=5, verbose=True):
    """
    Importance learning loop. 
    
    Inputs:
    barrier_distribution - Barrier height distribution, n x 1 np array
    NN_distances - n x 4 np array of ranked nearest neighbor distances,
    T - Temperature (K)
    initial_pool_size - int, how many sites to use to train the model before importance learning
    n_interations - int, number of importance learning iterations to perform
    verbose - boolean, whether to print and plot results
    plot_every - int, plot model fit and distribution of site estimates every multiple. Will not plot if verbose=False

    Returns: Dictionary of importance learning results
    """
    lattice_len = int(np.sqrt(len(barrier_distribution)) + 2)
    n_sites = len(barrier_distribution)
    IL = {}
    sampled_sites, sampled_barrier_heights, sampled_local_coordinates = inspect_sites(lattice_len, local_coordinates, barrier_distribution, initial_pool_size)
    if verbose:
        print("###########################################")
        print("#### ENTERING IMPORTANCE LEARNING LOOP ####")
        print("###########################################")
        print('\n')
        print('Number of sites: {}'.format(n_sites))
        print('True <Ea>k: {:02.1f} kJ/mol'.format(k_weighted_avg_activation_E(barrier_distribution, T)))
        print('\n')

    for i in range(n_iterations):
        # Excluding sites sampled more than once 
        indices_local_coords = np.unique(sampled_local_coordinates,return_index = True, axis = 0)[1]
        sampled_local_coords_unique  = np.asarray([sampled_local_coordinates[index] for index in sorted(indices_local_coords)])

        indices_sampled_barrier = np.unique(sampled_barrier_heights,return_index = True, axis = 0)[1]
        sampled_barrier_heights_unique = np.asarray([sampled_barrier_heights[index] for index in sorted(indices_sampled_barrier)])

        #TRAINING
        M, model_barriers_LOO, residuals = train(sampled_local_coords_unique, sampled_barrier_heights_unique)

        # Get model predicted barrier heights for all sites
        model_barrier_heights = predicted_adsorption_energies(sampled_local_coords_unique, sampled_barrier_heights_unique, M, local_coordinates, n_sites) 

        # Importance Sample
        # w_PDF, w_CDF, w_bin_edges = Distribution(model_barrier_heights, n_bins).weight_energy_PDF(T, False)
        # PDF, CDF, bin_edges = Distribution(model_barrier_heights, n_bins).make_distribution()

        # w_energy_midpoint = [0.5*(w_bin_edges[i]+w_bin_edges[i+1]) for i in range(n_bins)]
        # energy_midpoint = [0.5*(bin_edges[i]+bin_edges[i+1]) for i in range(n_bins)]

        # test_site = ImportanceSample(0, model_barrier_heights, T, n_bins).sample_distribution(w_CDF, w_bin_edges, sampled_sites)
        
        test_site = biased_sample(model_barrier_heights, T)
        # Compute <Ea>k by averaging importance sampled barrier heights
        if i == 0:
            avg_Ea_IS = barrier_distribution[test_site]
            importance_sampled_sites = np.asarray([test_site])
        else:
            importance_sampled_sites = np.append(importance_sampled_sites, test_site)
            avg_Ea_IS = np.mean(barrier_distribution[importance_sampled_sites])
        
        sampling_error = biased_error(barrier_distribution[importance_sampled_sites])
        unbiased_Ea_k, unbiased_error = bootstrap_unbiased_error(barrier_distribution, i + initial_pool_size, T)
        if verbose:
            print("####### Iteration {} #######".format(i))
            print("Importance sampled site: {}".format(test_site))
            print("Model Predicted Energy: {:04.1f} kJ/mol".format(model_barrier_heights[test_site]))
            print("True Energy: {:04.1f} kJ/mol".format(barrier_distribution[test_site]))
            print("NN Distances", local_coordinates[test_site])
            # print("<Ea>k (Model average) = {:02.1f} +/- {:02.1f} kJ/mol".format(Averages(T).compute_avg_activation_E(PDF, energy_midpoint), biased_error(sampled_barrier_heights)))
            print("<Ea>k (Importance sampled average) = {:02.1f} +/- {:02.1f} kJ/mol".format(avg_Ea_IS, sampling_error))
            print("<Ea>k (Random Sampling) = {:02.1f} +/- {:02.1f} kJ/mol".format(unbiased_Ea_k, unbiased_error))
            print('\n')

            if i % plot_every == 0:
                plt.rcParams['figure.figsize'] = 5, 4

                # Show training results
                fig, axes = plt.subplots(1, 1, figsize=(7,6))
                plot_trained(axes, model_barriers_LOO, sampled_barrier_heights, 2.5, initial_pool_size)
                plt.show()
                
                fig, axes = plt.subplots(1, 2, sharex=True, figsize=(14,6))
                histogram(axes[0], axes[1], model_barrier_heights, T, n_bins=20)
                plt.show()

                # biased_e_plot = [biased_error(np.asarray(IL[str(i)]["Sampled Barrier Heights"])) for i in range(i)]
                # if len(sampled_weighted_E) != 0 and len(sampled_weighted_E) != 1:
                # # plt.errorbar(np.arange(i), sampled_weighted_E, yerr=biased_e_plot,errorevery=1)
                # plt.show()

        # Store info in dict
        IL[str(i)] = {
            "Sampled Sites" : sampled_sites.tolist(),
            "Sampled Barrier Heights" : sampled_barrier_heights.tolist(),
            # "Sampled NN Distances" : sampled_NN_distances.tolist(),
            
            "Model Coefficients" : M.tolist(),
            # "Model Barrier Heights" : model_barrier_heights.tolist(),
            "Next Site" : int(test_site),
            "Predicted Barrier" : model_barrier_heights[test_site],
            "True Barrier" : barrier_distribution[test_site],
            "<Ea>k model" : k_weighted_avg_activation_E(model_barrier_heights,T),
            "<Ea>k importance sampled" : avg_Ea_IS,
            "Sampling Error" : sampling_error,
            "Unbiased Sampling Error" : bootstrap_unbiased_error(barrier_distribution, i + initial_pool_size, T)[1],
            "Training predicted barrier" : [float(x) for x in model_barriers_LOO]
        }
        # Append newly selected sites to sampled info
        sampled_sites = np.append(sampled_sites, test_site)
        sampled_barrier_heights = np.append(sampled_barrier_heights, barrier_distribution[test_site])
        sampled_local_coordinates = np.append(sampled_local_coordinates, [local_coordinates[test_site]], axis=0)
    return IL