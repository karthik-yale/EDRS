import numpy as np
from gradientDescent import ztheta
import scipy as sp
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})


class Predict():
    def __init__(self, data, target=None, num_latents=10, eta=0.01, num_steps=1000, plot_loss=True):
        self.data = data
        self.target = target
        self.num_latents = num_latents
        self.eta=eta
        self.num_steps=num_steps
        self.plot_loss = plot_loss
    
    def lvm_train(self):
        self.lvm_obj = ztheta(self.data, eta=self.eta, num_steps=self.num_steps, num_latents=self.num_latents, plot=self.plot_loss)

    def train_microbiome_lasso_model(self, test_size=0.2, plot_performance=True, target_sparsity=0.1, tol=0.0001, max_iter=1000000):
        def safe_pearsonr(x, y):
            """Calculate Pearson correlation coefficient with handling for constant inputs."""
            if np.all(x == x[0]) or np.all(y == y[0]):
                return np.nan
            return sp.stats.pearsonr(x, y)[0]

        # Copy and standardize the data
        X = np.copy(self.data)
        X -= np.mean(X, axis=0)[np.newaxis, :]
        X /= np.std(X, axis=0)[np.newaxis, :]
        y = self.target - np.mean(self.target, axis=0)[np.newaxis, :]
        y /= np.std(y, axis=0)[np.newaxis, :]

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Binary search for alpha
        lower_bound = 1e-8
        upper_bound = 1.0
        best_alpha = None
        best_sparsity = None

        counter = 0
        while lower_bound <= upper_bound and counter<50:
            counter += 1
            alpha = (lower_bound + upper_bound) / 2
            model = Lasso(alpha=alpha, max_iter=max_iter)
            model.fit(self.X_train, self.y_train)

            # Calculate the sparsity (percentage of non-zero coefficients)
            sparsity = np.mean(np.abs(model.coef_) > 1e-5)

            if abs(sparsity - target_sparsity) < tol:
                best_alpha = alpha
                best_sparsity = sparsity
                break

            if sparsity > target_sparsity:
                lower_bound = alpha
            else:
                upper_bound = alpha

        # If no exact match found, use the closest alpha
        if best_alpha is None:
            best_alpha = (lower_bound + upper_bound) / 2

        # Train the Lasso model with the selected alpha
        model = Lasso(alpha=best_alpha, max_iter=max_iter)
        model.fit(self.X_train, self.y_train)
        self.lasso_sparsity = np.mean(np.abs(model.coef_) > 1e-5)
        # Predict and compute in-sample (train) R^2 score
        self.y_train_pred = model.predict(self.X_train)
        self.microbiome_r2_train = np.array([safe_pearsonr(self.y_train[:, i], self.y_train_pred[:, i])**2 for i in range(self.y_train.shape[1])])

        # Predict and compute out-of-sample (test) R^2 score
        self.y_test_pred = model.predict(self.X_test)
        self.microbiome_r2_test = np.array([safe_pearsonr(self.y_test[:, i], self.y_test_pred[:, i])**2 for i in range(self.y_test.shape[1])])

        # Compute overall performance
        self.microbiome_performance = np.nanmean(self.microbiome_r2_test)
        self.microbiome_in_sample_performance = np.nanmean(self.microbiome_r2_train)

        # Plot performance if requested
        if plot_performance:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.bar(range(self.microbiome_r2_test.shape[0]), self.microbiome_r2_test[np.argsort(self.microbiome_r2_test)[::-1]])
            plt.ylabel('R2 score (Test Set)')
            plt.ylim(-1, 1)

            plt.subplot(1, 2, 2)
            plt.bar(range(self.microbiome_r2_train.shape[0]), self.microbiome_r2_train[np.argsort(self.microbiome_r2_train)[::-1]])
            plt.ylabel('R2 score (Train Set)')
            plt.ylim(-1, 1)
            
            plt.show()

    def train_microbiome_linear_model(self, test_size=0.2, plot_performance=True):
        X = np.copy(self.data)
        X -= np.mean(X, axis=0)[np.newaxis, :]
        X /= np.std(X, axis=0)[np.newaxis, :]
        y = self.target - np.mean(self.target, axis=0)[np.newaxis, :]
        y /= np.std(y, axis=0)[np.newaxis, :]


        out_r2_list = []
        in_r2_list = []
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)
            out_r2 = np.array([sp.stats.pearsonr(y_test[:, i], y_test_pred[:, i])[0]**2 for i in range(y_test.shape[1])])
            in_r2 = np.array([sp.stats.pearsonr(y_train[:, i], y_train_pred[:, i])[0]**2 for i in range(y_train.shape[1])])    
            out_r2_list.append(out_r2)
            in_r2_list.append(in_r2)
        
        self.out_r2 = np.mean(out_r2_list, axis=0)
        self.in_r2 = np.mean(in_r2_list, axis=0)

        self.out_performance = np.mean(self.out_r2)
        self.in_performance = np.mean(self.in_r2)
        
        self.out_median_performance = np.median(self.out_r2)
        self.in_median_performance = np.median(self.in_r2)



        if plot_performance:
            plt.bar(range(self.r2.shape[0]), self.r2[np.argsort(self.r2)[::-1]])
            plt.ylabel('R2 score')
            plt.ylim(-1,1)
            plt.show()
    
    def calc_edrs(self):
        average_data = np.ones(self.data.shape)*np.average(self.data, axis=0)
        kl_avg = sp.stats.entropy(self.data.T, average_data.T).sum()

        self.kl_list = []
        for i in range(1, 6):
            kl_obj = ztheta(self.data, eta=self.eta, num_steps=self.num_steps, num_latents=i, plot=self.plot_loss)
            assert kl_obj.dcdl_list[-1] + kl_obj.dcdy_list[-1] <= 1e-2, "Gradient descent for latent variable model did not converge"

            data_reconstruct = np.exp(np.matmul(-kl_obj.z, kl_obj.theta))
            data_reconstruct /= np.sum(data_reconstruct, axis=1)[:, np.newaxis]
            kl_div = sp.stats.entropy(self.data.T, data_reconstruct.T).sum()

            self.kl_list.append(kl_div)
        
        self.kl_list = np.array(self.kl_list)/kl_avg
        model = LinearRegression()
        model.fit(np.arange(len(self.kl_list)).reshape(-1, 1), np.log(self.kl_list))
        self.edrs = -1/model.coef_[0]

    def calc_edrs_mean_retain_null(self):
        kl_list_null_list = []
        for i in range(3):
            permuted_data = np.array([np.random.permutation(x) for x in self.data.T]).T
            permuted_data /= np.sum(permuted_data, axis=1)[:, np.newaxis]
            average_data = np.ones(permuted_data.shape)*np.average(permuted_data, axis=0)
            kl_avg = sp.stats.entropy(permuted_data.T, average_data.T).sum()

            kl_list_null = []
            for i in range(1, 6):
                kl_obj = ztheta(permuted_data, eta=self.eta, num_steps=self.num_steps, num_latents=i, plot=self.plot_loss)
                assert kl_obj.dcdl_list[-1] + kl_obj.dcdy_list[-1] <= 1e-2

                data_reconstruct = np.exp(np.matmul(-kl_obj.z, kl_obj.theta))
                data_reconstruct /= np.sum(data_reconstruct, axis=1)[:, np.newaxis]
                kl_div = sp.stats.entropy(permuted_data.T, data_reconstruct.T).sum()

                kl_list_null.append(kl_div)
            
            kl_list_null = np.array(kl_list_null)/kl_avg
            kl_list_null_list.append(kl_list_null)
        
        self.kl_list_null = np.mean(kl_list_null_list, axis=0)
        model = LinearRegression()
        model.fit(np.arange(len(self.kl_list_null)).reshape(-1, 1), np.log(self.kl_list_null))
        self.edrs_null = -1/model.coef_[0]