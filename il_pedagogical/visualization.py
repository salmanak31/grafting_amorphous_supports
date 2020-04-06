import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
plt.rcParams.update({'font.size': 14})
sns.set(style="white", palette="deep", color_codes=True, font_scale=1.5)
sns.set(style="white", palette="deep", color_codes=True, font_scale=1.5)

def plot_lattice(lattice, plot_graftable, graftable_sites = None):
    """
    Plot the quenched disorder lattice. Lattice sites are shown as red dots and bonded to nearest neighbors (black lines).
    inputs:
    lattice - n x n x 2 numpy array of lattice sites and their Cartesian positions
    """
    fig, ax = plt.subplots()
    ax.plot([lattice[0:, 0:-1, 0].ravel(), lattice[0:, 1:, 0].ravel()],[lattice[0:, 0:-1, 1].ravel(), lattice[0:, 1:, 1].ravel()], c='k', zorder=1)
    ax.plot([lattice[0:-1, 0:, 0].ravel(), lattice[1:, 0:, 0].ravel()],[lattice[0:-1, 0:, 1].ravel(), lattice[1:, 0:, 1].ravel()], c='k', zorder=1)

    cmap = np.where(lattice[:, :, 2] == 0, 'white', (np.where(lattice[:, :, 2] == 1, 'red', 'green'))).ravel()

    ax.scatter(np.ravel(lattice[:,:,0]), np.ravel(lattice[:,:,1]), marker='.', c=cmap, s=200, zorder=2, edgecolors='k', linewidths=1)

    if not plot_graftable:
        legend_elements = [Line2D([0], [0], marker='.', color='k', label='Empty site', markersize=15, markerfacecolor='w'),
                        Line2D([0], [0], marker='.', color='k', label='-OH',  markersize=15, markerfacecolor='r'),
                        Line2D([0], [0], marker='.', color='k', label='Si-O-Si',  markersize=15, markerfacecolor='g')]
    else:
        legend_elements = [Line2D([0], [0], marker='.', color='k', label='Empty site', markersize=15, markerfacecolor='w'),
                Line2D([0], [0], marker='.', color='k', label='-OH',  markersize=15, markerfacecolor='r'),
                Line2D([0], [0], marker='.', color='k', label='Si-O-Si',  markersize=15, markerfacecolor='g'),
                Line2D([0], [0], marker='.', color='k', label='Graftable Site',  markersize=15, markerfacecolor='#FCC200'),]
        for i in graftable_sites:
            ax.scatter(lattice[i[0], i[1] , 0], lattice[i[0], i[1] , 1], marker='.', c='#FCC200', s=225, zorder=3, edgecolors='k', linewidths=1)
    ax.legend(handles=legend_elements)
    plt.show()
    pass

def histogram(ax1, ax2, barrier_distribution, T, n_bins = 35):
    """
    Plot biased and unbiased histograms with site avg kinetics
    Inputs -
    barrier_distribution - np array of barrier heights
    T - Temperature
    n_bins - number of bins to use, default is 35
    """
    from il import k_weighted_avg_activation_E
    site_avg_E = k_weighted_avg_activation_E(barrier_distribution, T)
    
    # Unbiased
    y, x, _ = ax1.hist(barrier_distribution, density=True,bins=n_bins, color='dodgerblue', alpha=0.75)
    ax1.plot([site_avg_E, site_avg_E], [0, max(y) + max(y)*0.05], linewidth=2, c='k')
    ax1.annotate(r'$\langle E_a \rangle_k$', ((site_avg_E + min(barrier_distribution))/ 2, 3 * max(y)/4))
    ax1.set_ylim([0, max(y) + max(y)*0.05])
    ax1.set_xlabel(r'$E_a$, kJ/mol')
    ax1.set_ylabel(r'$\rho(E_a)$')
    
    # Biased
    y, x, _ = ax2.hist(barrier_distribution, density=True, weights=np.exp(-(barrier_distribution*1000)/(8.314*T)), bins=n_bins, color='sandybrown')
    ax2.plot([site_avg_E, site_avg_E], [0, max(y) + max(y)*0.05], linewidth=2, c='k')
    ax2.annotate(r'$\langle E_a \rangle_k$', (site_avg_E + 1, 3 * max(y)/4))
    ax2.set_xlabel(r'$E_a$, kJ/mol')
    ax2.set_ylabel(r'$\rho_w(E_a)$')    
    ax2.set_ylim([0, max(y) + max(y)*0.05])

    return ax1, ax2

def plot_trained(ax, m_size, model_barriers, true_barriers, confidence_int, initial_pool_size, x_lab = True, y_lab = True):
    """
    Assess model fit by plotting model predicted barriers against true barriers against a parity line

    inputs:
    ax: matplotlib axes object
    model_barriers: np array, model predicted barrier heights
    true_barriers: np array, actual barrier heights
    confidence_int: float, amount offset additional lines above an below parity line
    initial_pool_size: int, size of initial pool

    """
    # If a site is selected twice, the model predicted barrier will only have one entry but the true barrier array will have 2 entries.
    # This if statement removes the duplicate in the true barrier array and maintains ordering 
    if len(true_barriers) != len(model_barriers):
       true_barriers = true_barriers[np.sort(np.unique(true_barriers, return_index=True)[1])]

    min_barrier = min(true_barriers)-5
    max_barrier = max(true_barriers)+5

    x_line = [min_barrier, max_barrier]
    y_line = [min_barrier, max_barrier]

    # y_above = [min_barrier + confidence_int, max_barrier + confidence_int]
    # y_below = [min_barrier - confidence_int, max_barrier - confidence_int]

    # Parity line
    ax.plot(x_line, y_line, color="Black")
    # + confidence_int kJ/mol
    # ax.plot(x_line, y_above, color="dodgerblue")
    # - confidence_int kJ/mol
    # ax.plot(x_line, y_below, color="dodgerblue", label=r'$\pm {}$ kJ/mol'.format(confidence_int))
    
    if initial_pool_size == len(true_barriers):
        # plot initial pool
        ax.plot(true_barriers, model_barriers,'ro', c="blue", markersize=m_size)
    else:
        # plot initial pool
        ax.plot(true_barriers[0:initial_pool_size], model_barriers[0:initial_pool_size],'ro', label='Initial pool', c="blue")
        ax.plot(true_barriers[initial_pool_size:], model_barriers[initial_pool_size:], 'bo', label='Importance Learning', c="red")        

    if x_lab == True:
        ax.set_xlabel(r'True $\Delta G^{\ddag}$ kJ/mol', fontsize = 25)
    if y_lab == True:
        ax.set_ylabel(r'Model  $\Delta G^{\ddag}$, kJ/mol', fontsize = 25)
    
    
    ax.set_ylim(y_line)
    ax.set_xlim(x_line)

    # ax.legend(prop={'size': 20})
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # ax.legend(prop={'size': 35})
    # ax.tick_params(axis='both', which='major', labelsize=30)

    return ax

def plot_convergence(ax, n_iterations, pred_Ea, sampling_error, true_avg_Ea, bars_every=1):
    """
    Plot estimated <Ea>k with error bars as function of iteration number. Also shows  exact <Ea>k for comparison.

    inputs:
    ax: pyplot axes object
    n_iterations: int, number of iterations
    pred_Ea: list of estimated <Ea>k values
    sampling_error: list of uncertainty associated with pred_Ea
    true_avg_Ea: float, <Ea>k computed from exact barrier heights of all sites
    
    returns:
    ax object with plots
    """
    iterations = np.arange(n_iterations)
    ax.errorbar(iterations, pred_Ea, yerr=sampling_error, errorevery=bars_every, c='red')
    ax.plot([0, n_iterations], [true_avg_Ea, true_avg_Ea], c='k', linestyle='--', label=r'True $\langle E_a \rangle$')
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r'$\langle E_a \rangle_k$, kJ/mol')
    # ax.set_ylim(bottom=(true_avg_Ea - 1.25*max([max(sampling_error) + true_avg_Ea, max(pred_Ea)])))
    ax.set_xlim([0, n_iterations])
    ax.legend(loc=1)

    return ax

def plot_residuals(ax, true_barriers, model_barriers, n_bins=40):
    """
    Plot the residuals, Ea,true - Ea,model, to see how they distributed
    """
    residuals = true_barriers - model_barriers
    sns.distplot(residuals, bins=n_bins, color="grey",label = "this")
    ax.set_xlabel(r'$G_{true}^{\ddag} - G_{model}^{\ddag}$, kJ/mol')
    ax.set_ylabel('Frequency')
    return ax