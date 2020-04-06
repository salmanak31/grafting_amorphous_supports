import sys
sys.path.append('../')

from il_pedagogical import *
import numpy as np


lattice_size = 75

###### Gaussian #######
# a = 35 # ~ Potential width, dimensionless
# D = 60 # Potential depth, kJ/mol
# r_eq = 1.0 # Equilibrium distance of unperturbed lattice, dimensionless
# T = 300
# sigma = morse_std(r_eq, D, a, T)
# delta = 0.35
# gauss_pot_std = 0.9
# dissociation_type = 2
# potential_type = 2

####### Morse ########
lattice_size = 75
a = 30.
D = 35.
r_eq = 1.0 # Equilibrium distance of unperturbed lattice, dimensionless
T = 373
sigma = morse_std(r_eq, D, a, T)
delta = 1.75 * sigma
potential_type = 1
dissociation_type = 2

lower = 0.0
upper = 0.5 
# sigma = 0.1
mean = 0.0

n_iterations = 5
initial_pool_size = 50
displacement_type = 'uniform'


if displacement_type == 'trunc_norm':
    disp_kwargs = {'lower' : lower, 'upper' : upper, 'sigma' : delta, 'mean' : mean}
else:
    disp_kwargs = False

if potential_type == 1:
    potential_kwargs = {'r-eq': r_eq, 'D' : D}
else:
    potential_kwargs = {'D' : D, 'std' : gauss_pot_std}

lattice = make_quenched_disorder_lattice(lattice_size, delta, displacement_type, False, lower, upper, delta, mean)
# plot_lattice(lattice)
NN_distances = nearest_neighbor_distances(lattice)
# Ads_E = adsorption_energies(lattice, dissociation_type, potential_type, r_eq=r_eq, D=D, std=gauss_pot_std, T=T)
Ads_E = adsorption_energies(lattice, dissociation_type, potential_type, r_eq=r_eq, D=D, a=a, T=T)


NN_distances = nearest_neighbor_distances(lattice)[1:-1, 1:-1]
barrier_distribution = Ads_E + 50

### True Average ###
B = 1000/8.314/T
site_avg_E = np.mean(np.exp(-B*barrier_distribution)*(barrier_distribution))/np.mean(np.exp(-B*barrier_distribution))

# Flatten from 2D to 1D arrays
NN_distances = np.reshape(NN_distances, ((lattice_size - 2)**2, 4))
NN_distances.sort()

barrier_distribution = np.ravel(barrier_distribution)
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(14,6))
histogram(axes[0], axes[1], barrier_distribution, T, 35)
plt.show()
### Start Importance Learning ###
n_bins = 20

lattice_len = int(np.sqrt(len(barrier_distribution)) + 2)
IL = {}
sampled_sites, sampled_barrier_heights, sampled_NN_distances = inspect_sites(lattice_len, NN_distances, barrier_distribution, initial_pool_size)

M, model_barriers_LOO, residuals = train(sampled_NN_distances, sampled_barrier_heights)

def check_precision_accuracy(sampled_NN_distances, sampled_barrier_heights, M, NN_distances, lattice_size):
    """
    Check accuracy of model predicted sites computed by:
    (1.) longdouble precision + infinite precision where underflow happens
    (2.) All infinite precision (which is sloooow but exact)

    Returns L2 norm of the difference
    """
    standard_prec_barriers = predicted_adsorption_energies(sampled_NN_distances, sampled_barrier_heights, M, NN_distances, lattice_size)
    decimal_prec_barriers = predicted_adsorption_energies_d(sampled_NN_distances, sampled_barrier_heights, M, NN_distances, lattice_size)
    
    return(np.linalg.norm(standard_prec_barriers - decimal_prec_barriers))

def compare_time(sampled_NN_distances, sampled_barrier_heights, M, NN_distances, lattice_size, n_times=3):
    t_long = timeit.Timer(functools.partial(predicted_adsorption_energies, sampled_NN_distances, sampled_barrier_heights, M, NN_distances, lattice_size))
    t_decimal = timeit.Timer(functools.partial(predicted_adsorption_energies_d, sampled_NN_distances, sampled_barrier_heights, M, NN_distances, lattice_size))

    print('Long/arbitrary precision mix (s): {}'.format(t_long.timeit(n_times)))
    print('Arbitrary precision all sites (s): {}'.format(t_decimal.timeit(n_times)))
    pass

accuracy = check_precision_accuracy(sampled_NN_distances, sampled_barrier_heights, M, NN_distances, lattice_size)

print('||Ea(long) - Ea(arbitrary)|| = {}'.format(accuracy))

compare_time(sampled_NN_distances, sampled_barrier_heights, M, NN_distances, lattice_size)