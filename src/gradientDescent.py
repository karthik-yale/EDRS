import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

class ztheta():
    def __init__(self, data, eta, num_steps, num_latents, plot=False, warm_start=False, restrict=False, fig_name=None):
        self.data = data
        self.eta = eta
        self.num_steps = num_steps
        self.num_latents = num_latents
        self.restrict = restrict #Restricts z to be negative and theta to be positive
        self.fig_name = fig_name
        if warm_start:
            log_data = np.log((self.data+1e-3)/np.mean(self.data+1e-3, axis=1)[:, np.newaxis])
            u, s, vh = np.linalg.svd(log_data, full_matrices=False)
            u_latent = u[:, :self.num_latents]
            s_latent = s[:self.num_latents]
            vh_latent = vh[:self.num_latents]
            self.z = -np.matmul(u_latent, np.sqrt(np.diag(s_latent)))
            self.theta = np.matmul(np.sqrt(np.diag(s_latent)), vh_latent)
            if self.restrict:
                self.z[self.z > 0] = 0
                self.theta[self.theta < 0] = 0
            data_reconstruction = self.q_calc(self.z, self.theta)
            plt.scatter(self.data, data_reconstruction)
            plt.show()
        else:
            self.z = -np.random.default_rng().uniform(0.2,0.5,(self.data.shape[0], self.num_latents))
            self.theta = np.random.default_rng().uniform(0.2,0.5,(self.num_latents,self.data.shape[1]))

        self.loss_list, self.dcdl_list, self.dcdy_list, self.z, self.theta = self.grad_descent(self.z, self.theta)
        
        self.converged = (self.dcdl_list[-1] + self.dcdy_list[-1]) < 1e-2
        
        self.theta -= self.theta.min(axis=1).reshape(-1,1)
        if plot:
            self.make_plots()
    
    def partition_func(self, l, y):
        return np.sum(np.exp(-np.dot(l, y)),axis=1)

    def q_calc(self, l,y):
        part_func = self.partition_func(l,y)
        return np.exp(-np.dot(l,y))/(part_func[:,np.newaxis])

    def loss(self, l, y):
        z = self.partition_func(l,y)
        term_1 = np.sum(np.log(z))
        term_2 = np.sum(np.multiply(self.data,np.dot(l,y)))
        return term_1 + term_2

    def grad_descent_step(self, l, Y):
        pred = self.q_calc(l,Y)
        dcdl = np.dot(self.data,Y.T) - np.dot(pred,Y.T)
        dcdy = np.dot(l.T,self.data) - np.dot(l.T,pred)
        dcdlnorm = norm(dcdl)/norm(l)
        dcdynorm = norm(dcdy)/norm(Y)
        l = l - self.eta*dcdl
        Y = Y - self.eta*dcdy
        if self.restrict:
            l[l>0] = 0
            Y[Y<0] = 0
        return l, Y, dcdlnorm, dcdynorm

    def grad_descent(self, l, Y):
        loss_list = []
        dcdl_list = []
        dcdy_list = []
        l_grad = np.inf
        y_grad = np.inf
        current_loss = np.inf
        self.counter = 0
        while l_grad + y_grad > 1e-2 and self.counter <= self.num_steps:
            l_new, Y_new, dcdl, dcdy = self.grad_descent_step(l, Y)
            new_loss = self.loss(l, Y)
            # if new_loss > current_loss:
            #     self.eta = self.eta*0.1
            #     if self.eta < 1e-15:
            #         break
            #     continue
            l = np.copy(l_new)
            Y = np.copy(Y_new)
            current_loss = new_loss
            dcdl_list.append(dcdl)
            dcdy_list.append(dcdy)
            loss_list.append(current_loss)
            l_grad = dcdl
            y_grad = dcdy
            self.counter += 1
        return loss_list, dcdl_list, dcdy_list, l, Y
    
    def make_plots(self):
        x = np.linspace(1,self.counter,self.counter)
        fig, axis = plt.subplots(1,2,figsize=(10,5))
        axis[0].plot(x,self.loss_list)
        axis[0].set_title('loss')
        axis[0].set_yscale('log')
        axis[1].plot(x,self.dcdl_list, label='dcdl')
        axis[1].set_title('normalized gradient')
        axis[1].plot(x,self.dcdy_list, label='dcdy')
        axis[1].set_yscale('log')
        plt.legend()
        if self.fig_name is not None:
            plt.savefig('/home/ks2823/my_CRM/pictures/{}'.format(self.fig_name))
        plt.show()
