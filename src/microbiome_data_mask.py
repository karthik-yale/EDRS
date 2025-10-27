import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import cvxpy as cvx
from joblib import Parallel, delayed, cpu_count

import warnings

warnings.simplefilter('ignore', DeprecationWarning)

class ss_data():
    def __init__(self, M, S, C, R0, cov, death, num_data, inflow_sparsity=1, num_niches = 1, plot=False):
        self.M = M                  #Number of resources
        self.S = S                  #Number of species
        self.C = C                  #Consumer preference matrix
        self.R0 = R0                #Resource fixed point
        self.cov = cov
        self.inflow_sparsity = inflow_sparsity
        self.death = death          #death rate
        self.num_data = num_data    #Number of data points
        self.num_niches = num_niches

        self.make_inflow_list()   #make list of resource inflows
        self.make_abu_list_parallel()   #make list of abundances
        
        if plot:
            self.make_abu_plot()
            self.make_inv_cummulative_abundance_plot()
            self.make_shannon_list_parallel()
            self.make_taylors_law_plot()
            self.make_prevalence_plot()

    def get_normalized(self, Data):
        data = np.copy(Data)
        data /= np.sum(data, axis=1)[:, np.newaxis]
        
        data = data[:, np.mean(data, axis=0) > 1e-3]
        data /= np.sum(data, axis=1)[:, np.newaxis]
        return data
        
    def make_inflow_list(self):
        self.inflow_list = np.abs(np.random.default_rng().multivariate_normal(self.R0, self.cov, self.num_data))
        if self.num_niches > 1:
            masks = sp.sparse.random(self.num_niches, self.M, density=self.inflow_sparsity).toarray()
            masks[masks > 0] = 1
            ratios = np.random.rand(self.num_data, self.num_niches)
            self.mask = np.matmul(ratios, masks)
            self.mask = self.mask/np.average(self.mask, axis=1)[:, np.newaxis]        

            self.inflow_list = self.mask*self.inflow_list
    
    def abu(self, inflow):
        wR = cvx.Variable(shape=(self.M))
        wR0 = inflow
        obj = cvx.Minimize(cvx.sum(cvx.kl_div(wR0, wR)))
        constraints = [self.C@wR <= self.death]
        prob = cvx.Problem(obj, constraints)
        prob.solver_stats
        prob.solve(solver=cvx.ECOS,abstol=1e-8,reltol=1e-8,warm_start=True,verbose=False,max_iters=5000)
        Nf=constraints[0].dual_value[0:self.S].reshape(self.S)
        Rf=wR.value.reshape(self.M)

        return Nf, Rf
    
    def make_abu_list_parallel(self):
        abus = Parallel(n_jobs=-1)(delayed(self.abu)(inflow) for inflow in self.inflow_list)
        self.s_abu = np.array([Nf for Nf, Rf in abus])
        self.s_abu_unnormalized = np.copy(self.s_abu)
        row_sums = self.s_abu.sum(axis=1)
        self.s_abu = self.s_abu / row_sums[:, np.newaxis]
        self.r_abu = np.array([Rf for Nf, Rf in abus])
        self.mult_data = np.array([np.random.multinomial(10000, abus) for abus in self.s_abu])/10000
        self.data_normalized = self.get_normalized(self.mult_data)
    
    def make_abu_plot(self):
        positive_data = self.data_normalized[self.data_normalized > 0].reshape(-1)
        if len(positive_data) == 0:
            print("No positive abundance data to plot.")
            return
        num_bins = 20
        min_val = np.min(positive_data)
        max_val = np.max(positive_data)

        log_bins = np.logspace(np.log10(min_val), np.log10(max_val), num_bins)

        # Plot the histogram with log-spaced bins
        plt.hist(positive_data, bins=log_bins, edgecolor='black', alpha=0.7)

        # Set the x-axis to a logarithmic scale
        plt.xscale('log')

        plt.title('Species abundance distribution')
        plt.xlabel('Species abundance (log scale)')
        plt.ylabel('Frequency')
        plt.grid(True, which="both", ls="-", alpha=0.2) # Add a grid for better readability
        plt.show()



    def make_inv_cummulative_abundance_plot(self):
        print('Multinomial data used')
        x_data = np.sort(self.data_normalized[np.nonzero(self.data_normalized)], axis=None)
        inv_cum = np.cumsum(x_data)
        inv_cum = inv_cum[-1] - inv_cum
        inv_cum = inv_cum / inv_cum[0]
        plt.plot(x_data, inv_cum)
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Inverse cumulative abundance distribution')
        plt.xlim(1e-4,1)
        plt.ylim(1e-4,10)
        plt.show()


    def shannon_entropy(self, probabilities):
        non_zero_probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(non_zero_probabilities * np.log(non_zero_probabilities))
        return entropy
    
    def make_shannon_list_parallel(self):
        self.shannon_list = Parallel(n_jobs=-1)(delayed(self.shannon_entropy)(probabilities) for probabilities in self.s_abu)
        plt.hist(self.shannon_list, bins=20, density=True)
        plt.xlabel('Shannon entropy')
        plt.ylabel('Frequency')
        plt.title('Shannon entropy distribution')
        plt.show()   
    
    def make_taylors_law_plot(self):
        variance = np.var(self.data_normalized, axis=0)
        mean = np.mean(self.data_normalized, axis=0)
        plt.scatter(mean, variance)
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Taylors law plot')
        plt.xlabel('Mean')
        plt.ylabel('Variance')
        plt.show()

    def fit_laplace(self, d):
        d = d.reshape(-1)
        mu = np.median(d)
        b = np.average(np.abs(d - mu))
        return mu, b

    def get_abu_change(self):
        self.delta = np.zeros((self.num_data - 1, self.data_normalized.shape[1]))
        for i in range(self.num_data - 1):
            self.delta[i] = np.log10(self.data_normalized[i+1]/self.data_normalized[i])
        
    
    def make_abu_change_plot(self):
        mu, b = self.fit_laplace(self.delta)
        x=np.linspace(np.min(self.delta),np.max(self.delta),1000)
        y1 = 1/(2*b)*np.exp(-np.abs(x-mu)/b)
        self.delta = self.delta[np.abs(self.delta) != np.inf]
        plt.hist(self.delta.reshape(-1), bins=100, density=True, alpha=0.5, color='green')
        plt.plot(x,y1, color='blue', linewidth=1)
        plt.yscale('log')
        plt.xlabel('\u0394L')
        plt.ylabel('P(\u0394L)')
        plt.title('sigma = {}'.format(self.sigma))
        plt.show()

    def make_prevalence_plot(self):
        num_non_zero = np.zeros(self.data_normalized.shape[1])
        for i in range(self.data_normalized.shape[1]):
            num_non_zero[i] = np.count_nonzero(self.data_normalized[:,i])
        plt.scatter(np.mean(self.data_normalized, axis=0), num_non_zero/self.num_data)
        plt.xscale('log')
        plt.xlabel('Average abundance')
        plt.ylabel('Prevalence')
        plt.title('Prevalence Plot')
        plt.xlim(1e-6, 1e-0)
        plt.show()