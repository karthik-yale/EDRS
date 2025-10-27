import numpy as np
from microbiome_data_mask import ss_data
import scipy as sp
import cvxpy as cvx
from joblib import Parallel, delayed, cpu_count
import warnings
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
from tqdm import tqdm
from datetime import datetime

warnings.simplefilter('ignore')

class Annealing():
    def __init__(self, fixed_params, params_initial, params_bounds, data_original, num_steps=500, inv_temp=1, 
                 make_plots=False, model=None):
       
        self.S, self.M, self.R_max, self.C_sparsity, self.num_niches = fixed_params
        
        self.model = model
        # if self.model != 'wingreen':
        #     assert self.S <= self.M, "Not enough resources to assign a dedicated resource to a species"
        
        self.C_max, self.inflow_sparsity, self.M_over_C, self.var_coeff = params_initial
        self.params_initial = np.copy(params_initial)

        self.data_original = data_original
        self.num_species_original = data_original.shape[1]
        self.num_data = 100
        self.num_steps = num_steps
        self.inv_temp = inv_temp
        self.bounds = params_bounds
        self.make_plots = make_plots
        

        self.alpha_mean_original = self.calc_alpha(self.data_original)
        self.beta_mean_original = self.calc_beta(self.data_original)
        

        check_initial_params = self.check_bounds(self.params_initial)
        if not check_initial_params:
            raise ValueError('Initial parameters must be within bounds')

        self.distance_list = []
        self.params_list = []

    def make_C(self):
        if self.model == 'wingreen':
            if self.C_sparsity <= 0:
                raise ValueError("C_sparsity must be > 0 for 'wingreen' model")
            max_attempts = 100
            for attempt in range(max_attempts):
                C_temp = sp.sparse.rand(self.S, self.M, density=self.C_sparsity).toarray()
                if not np.any(C_temp.sum(axis=1) == 0):
                    C_temp /= C_temp.sum(axis=1)[:, np.newaxis]
                    return C_temp
            # fallback: ensure at least one nonzero per row deterministically
            C_temp = sp.sparse.rand(self.S, self.M, density=self.C_sparsity).toarray()
            for i in range(self.S):
                if C_temp[i].sum() == 0:
                    j = np.random.randint(self.M)
                    C_temp[i, j] = 1.0
            C_temp /= C_temp.sum(axis=1)[:, np.newaxis]
            return C_temp
        
        elif self.model == 'rng':
            # assert self.S == self.M, "For model without tradeoffs, no. of species must be equal to number of resources"
            C_temp = sp.sparse.rand(self.S, self.M, density=self.C_sparsity).toarray()
            # C_temp_diagonal = np.random.rand(self.M)
            # np.fill_diagonal(C_temp, C_temp_diagonal)
        
            return C_temp
        
        elif self.model == 'high_comp':
            for i in range(100):
                with warnings.catch_warnings(record=True) as w:
                    resource_s = np.random.exponential(self.C_sparsity/2, size=self.M)
                    species_s = np.random.exponential(self.C_sparsity/2, size=self.S)
                    resource_s[::-1].sort()
                    species_s[::-1].sort()
                    resource_s = np.tile(np.array([resource_s]).T, (1, self.S))
                    species_s = np.tile(species_s, (self.M,1))
                    S_mat = resource_s + species_s
                    original_matrix = (S_mat >= np.random.rand(self.M, self.S)).astype(int)
                    C_temp = np.where(original_matrix==1, np.random.rand(self.M,self.S), original_matrix).T

                    # col_sums = np.sum(C_temp, axis=1)
                    # C_temp = C_temp/col_sums[:, np.newaxis]

                    if not w:
                        return C_temp 
            if i == 99:
                raise RuntimeError("Could not find CR matrix with the specifications provided")
        
        raise ValueError("Model class has to be either wingreen or rng")
        
    def create_data_obj(self, new_params):
        C_max, inflow_sparsity, M_over_C, var_coeff = new_params
        C_temp = self.make_C()
        self.C = C_max*C_temp
        death = M_over_C*C_max*np.ones(self.S)
        R0 = self.R_max*np.random.rand(self.M)
        var = var_coeff*(R0**2)
        cov = np.diag(var)
        obj = ss_data(self.M, self.S, self.C, R0, cov, death, self.num_data, inflow_sparsity=inflow_sparsity, num_niches=self.num_niches, plot=False)
        return obj
    
    def create_data(self, new_params, max_counter=None):
        obj = self.create_data_obj(new_params)
        self.metadata = obj.r_abu
        self.inflow_list = obj.inflow_list
        avg_abu_1 = np.average(obj.mult_data, axis=0)
        survival_idx = avg_abu_1>=1e-3
        high_abu_species = obj.mult_data[:, avg_abu_1>=1e-3]
        row_sum = np.sum(high_abu_species, axis=1)
        high_abu_species = high_abu_species/row_sum[:, np.newaxis]
        self.survival_idx = survival_idx
        self.s_abu_unnormalized = obj.s_abu_unnormalized
        self.C_survived = self.C[survival_idx, :]

        return high_abu_species

    def calc_alpha(self, data):
        alpha = sp.stats.entropy(data, axis=1).mean()
        return alpha

    def calc_beta(self, data):
        beta = sp.spatial.distance.pdist(data, metric='jensenshannon').mean()
        return beta
    
    def calc_dist(self, data):
        alpha_mean_data = self.calc_alpha(data)
        beta_mean_data = self.calc_beta(data)
        num_species_data = data.shape[1]

        alpha_mean_dist = np.abs(alpha_mean_data - self.alpha_mean_original)/self.alpha_mean_original
        beta_mean_dist = np.abs(beta_mean_data - self.beta_mean_original)/self.beta_mean_original
        num_species_dist = np.abs(num_species_data - self.num_species_original)/self.num_species_original

        return alpha_mean_dist + beta_mean_dist + num_species_dist

    def calc_prob_transition(self, dist_i, dist_f):
        prob_transition = min(1, np.exp(-self.inv_temp*(dist_f - dist_i)))
        return prob_transition

    def change_params(self, current_params):
        new_params = current_params.copy()

        '''
        If there is only one niche then we don't need to change niche sparsity
        So we'll assign the last index to changing that
        Hence, if num_niches is 1 we sample indices from [0, num of changeable parameters - 1)
        '''
        if self.num_niches == 1:
            params_change_idx = np.random.randint(0, 3)
        else:
            params_change_idx = np.random.randint(0, 4)

        if params_change_idx == 0:
            C_max_new = self.C_max*1.3**np.random.randn()
            new_params[0] = C_max_new
        if params_change_idx == 1:
            M_over_C_new = self.M_over_C*1.3**np.random.randn()
            new_params[2] = M_over_C_new
        if params_change_idx == 2:
            var_coeff_new = self.var_coeff*1.3**np.random.randn()
            new_params[3] = var_coeff_new
        if params_change_idx == 3:
            inflow_sparsity_new = self.inflow_sparsity*1.3**np.random.randn()
            new_params[1] = inflow_sparsity_new
        
        return new_params
    
    def check_bounds(self, numbers):
        for i, num in enumerate(numbers):
            lower_bound, upper_bound = self.bounds[i]
            if not (lower_bound <= num <= upper_bound):
                return False
        return True

    def anneal(self):
        print('Annealing...')
        current_params = [self.C_max, self.inflow_sparsity, self.M_over_C, self.var_coeff]
        current_data = None
        while_loop_counter = 0
        while current_data is None:
            if while_loop_counter >= 100:
                return None
            while_loop_counter += 1
            try:
                current_data = self.create_data(current_params, max_counter=1)
            except Exception as e:
                print(e)
                continue            
        current_dist = self.calc_dist(current_data)    

        step = 0
        while step < self.num_steps:            
            new_params = self.change_params(current_params)
            if not self.check_bounds(new_params):
                continue

            num_species_loop_counter = 0
            while num_species_loop_counter<10: 
                num_species_loop_counter += 1
                try:     
                    new_data = self.create_data(new_params)
                    num_species_diff = np.abs(new_data.shape[1]-self.num_species_original)
                except Exception as e:
                    continue
                break
            
            if num_species_loop_counter == 10:
                continue
            
            if step % 50 == 0:
                current_datetime = datetime.now()
                print(f'Num steps: {step}, Date-time: {current_datetime}')
                print(f'Distance: {current_dist:.3f}, Num species diff: {num_species_diff}, Inv temp: {self.inv_temp:.3f}')
                print(current_params)
            new_dist = self.calc_dist(new_data)
            prob_transition = self.calc_prob_transition(current_dist, new_dist)

            if np.random.rand() < prob_transition:
                self.C_max, self.inflow_sparsity, self.M_over_C, self.var_coeff = new_params
                current_params = [self.C_max, self.inflow_sparsity, self.M_over_C, self.var_coeff]
                self.distance_list.append(new_dist)
                current_dist = new_dist
                self.params_list.append(current_params)
            
            if current_dist < 0.3:
                break
                
            # if current_dist < 0.4:
            #     print(current_dist)
            #     print(current_params)
            step += 1 
            self.inv_temp /= 0.99

        self.distance_list = np.array(self.distance_list)
        self.params_list = np.array(self.params_list)
        
        self.final_params = [self.C_max, self.inflow_sparsity, self.M_over_C, self.var_coeff]
        
        if self.make_plots:
            self.plot_history()
            
        return 1
    
    def plot_history(self):
        plt.plot(np.arange(self.distance_list.shape[0]), self.distance_list)
        plt.title('History of distance from original data')
        plt.ylabel('Distance')
        plt.xlabel('Number of steps')
        plt.show()
        
    





        

    


    
