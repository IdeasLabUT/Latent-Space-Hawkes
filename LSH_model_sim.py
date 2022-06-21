#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 14:32:49 2021

This is the latent space Hawkes model with baseline mu depend on latent space

intensity:
\lambda_{uv}^*(t) &= \mu_{uv} +   \sum_{t_{uv} < t}\sum_b^B C_b \alpha_1 \beta_b e^{-\beta_b(t-t_{uv})} \\
&+   \sum_{t_{vu} < t}\sum_b^B C_b \alpha_2 \beta_b e^{-\beta_b(t-t_{vu})}, \quad \forall u \neq v

For simulation, there is no slope term

"""


from LSH_model_fit import LSHM_mle
import Latent_Space_Hawkes_utils as lsh_utils
import numpy as np
import time
from scipy.spatial import procrustes
from math import e
import pickle


class MHP:
    # The simulation is referred from Steven Morse
    #  https://stmorse.github.io/docs/orc-thesis.pdf
    #  https://stmorse.github.io/docs/6-867-final-writeup.pdf
    def __init__(self, alpha=[[0.5]], mu=[0.1], C=[1], beta=[[1.0]], seed = 1):
        '''params should be of form:
        alpha: numpy.array((u,u)), mu: numpy.array((,u)), omega: float'''
        
        self.data = []
        self.alpha, self.mu, self.C ,self.beta =  np.array(alpha), np.array(mu), np.array(C), np.array(beta)
        self.dim = self.mu.shape[0]
        self.check_stability()
        self.seed = seed

    def check_stability(self):
        ''' check stability of process (max alpha eigenvalue < 1)'''
        w,v = np.linalg.eig(self.alpha)
        me = np.amax(np.abs(w))
        #print('Max eigenvalue: %1.5f' % me)
        if me >= 1.:
            print('(WARNING) Unstable.')

    def generate_seq(self, horizon):
        '''Generate a sequence based on mu, alpha, omega values. 
        Uses Ogata's thinning method, with some speedups, noted below'''
        np.random.seed(self.seed)
        self.data = []
        # Istare : hold maximum intensity (initially equal sum of baselines)
        Istar = np.sum(self.mu)
        # s : hold new event time
        # intertimes follow exponential distribution
        s = np.random.exponential(scale=1. / Istar)
        # attribute (weighted random sample, since sum(mu)==Istar)
        # n0 is process with first event
        n0 = np.random.choice(np.arange(self.dim), 1, p=(self.mu / Istar))
        n0 = int(n0)
        self.data.append([s, n0])

        # last_rates : (M,) np.array
        # holds values of lambda(t_k) where k is most recent event for each process
        # starts with just the base rate
        last_rates = self.mu.copy()
        Q = len(self.beta)
        exp_sum_Q_last = np.zeros((Q, self.dim))

        # decrease I* (I* decreases if last event was rejected)
        decIstar = False
        while True:
            tj, uj = self.data[-1][0], self.data[-1][1]

            if decIstar:
                # if last event was rejected, decrease Istar
                Istar = np.sum(rates)
                decIstar = False
                # print(f"rejected - I* = {Istar}")
            else:
                # just had an event, increase Istar
                # by summing over column of process uj
                #Istar = np.sum(last_rates) + np.sum(self.alpha[:, uj])
                Istar = np.sum(last_rates) + np.sum(self.alpha[:, uj])*np.sum(self.beta*self.C)
                # print(f"not rejected - I* = {Istar}")

            # generate new event
            s += np.random.exponential(scale=1. / Istar)

            # calc rates at time s (use trick to take advantage of rates at last event)
            # rates : (M,) np.array holds intensity of each process at t=s
            #rates = self.mu + np.exp(-self.beta[:,uj] * (s - tj)) * (self.alpha[:,uj].flatten() + lastrates - self.mu)
            exp_term = np.exp(-self.beta * (s - tj)).reshape((Q, 1))
            exp_term_repeat = np.tile(exp_term, (1, self.dim))
            C_alpha = (self.C * self.beta).reshape(Q, 1) @ self.alpha[:, uj].reshape(1, self.dim)
            exp_sum_Q = exp_term_repeat * (C_alpha + exp_sum_Q_last)
            rates = self.mu + np.sum(exp_sum_Q, axis=0)
            # print(f"trick  = {rates}")
            # print(f"detail = {self.calc_rates(s)}\n")

            # attribution/rejection test
            # handle attribution and thinning in one step as weighted random sample
            diff = Istar - np.sum(rates)
            try:
                n0 = np.random.choice(np.arange(self.dim + 1), 1,
                                      p=(np.append(rates, diff) / Istar))
                n0 = int(n0)
            except ValueError:
                # by construction this should not happen
                print('Probabilities do not sum to one.')
                #self.data = np.array(self.data)
                #return self.data

            if n0 < self.dim:
                # s is accepted
                self.data.append([s, n0])
                # update last_rates
                last_rates = rates.copy()
                exp_sum_Q_last = exp_sum_Q
            else:
                decIstar = True

            # if past horizon, done
            if s >= horizon:
                self.data = np.array(self.data, dtype=np.float)
                self.data = self.data[self.data[:, 0] < horizon]
                return self.data

def lantet_hawkes_simulation(end_time, N, beta, z, C, theta, delta, seed):
    """
    simulate a latent sapce Hawkes process network with mu depend on latent space

    Parameters
    ----------
    end_time : int
        durantion time of HP.
    N : int
        # of nodes in the network.
    beta : array (1*3)
        decays for different kernel.
    z : array (n*dim)
        latent dimensions.
    C : array (1*3)
        scalling parameters for jump size.
    theta : array (1*3)
        intercept parameter, jump size for self-exciting term and jump size for reciprocal term.
    delta : array ((N-1)*2)
        sender and receive effect.
    seed : int
        random seed.

    Returns
    -------
    P_tot : array 
        simulated timestamp for all pair of nodes .

    """ 
    xp = np.tile(np.sum(np.power(z,2), 1), [N,1]).T  
    dis = xp + xp.T - 2*np.matmul(z, z.T)

    alpha_1 = theta[1]
    alpha_2 = theta[2]
    
    delta_1 = delta[0:N-1]
    delta_2 = delta[N-1:]
    
    delta_1 = np.append(delta_1, -sum(delta_1))
    delta_2 = np.append(delta_2, -sum(delta_2))
    
    delta_1 = npresize(delta_1,N, 1)
    delta_2 = npresize(delta_2,1, N)
    
    logit = -dis + theta[0] + delta_1 + delta_2
    # prevent mu = 0
    mu = e**logit + 0.00000000001

    P_tot = np.empty([N,N], dtype=np.object)
      
    for i in range(mu.shape[0]):
        for j in range(mu.shape[1]):
            #if j == 99:
                #print(j)
            if i!=j:
                alpha = np.array([[alpha_1, alpha_2], [alpha_2, alpha_1]])
                baselines = [mu[i,j], mu[j,i]]
                P = MHP(mu=baselines, alpha=alpha, C = C, beta=beta, seed = seed)
                sim_process = P.generate_seq(end_time)
                p1 = []
                p2 = []
                for s in sim_process:
                    if s[1] == 0:
                        p1.append(s[0])
                    else: p2.append(s[0])
                P_tot[i,j], P_tot[j,i] = np.array(p1), np.array(p2) 
            else:
                P_tot[i,j]=[]
                P_tot[j,i]=[] 
    return P_tot



def count_process(P):
    """
    

    Parameters
    ----------
    P : a numpy array where each entry is a list
        Timstamps of a network.

    Returns
    -------
    count : array
        hwo many events for each pair of node in the network.

    """
    # count number of process
    count = np.zeros([P.shape[0],P.shape[1]])
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            count[i,j] = len(P[i,j])
    return count

def npresize(a, u, v):
    a0 = np.array([a[0:v]])
    for i in range(2, u+1):
        a0 = np.concatenate((a0, np.array([a[v*(i-1):i*v]])))
    return a0

def yang_dict_to_adjacency_list(num_nodes, event_dicts, dtype=np.float64):
    """
    Converts event dict to weighted/aggregated adjacency matrix

    :param num_nodes: (int) Total number of nodes
    :param event_dicts: Edge dictionary of events between all node pair. Output of the generative models.
    :param dtype: data type of the adjacency matrix. Float is needed for the spectral clustering algorithm.

    :return: np array (num_nodes x num_nodes) Adjacency matrix where element ij denotes the number of events between
                                              nodes i an j.
    """
    
    # intialize a 2D matrix with all elements are empty list. This is a stupid method, and it could have better way
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.object)
    for u in range(num_nodes):
        for v in range(num_nodes):
            if adjacency_matrix[u,v] == 0:
                adjacency_matrix[u,v] = []
                
    for (u, v, event_times) in event_dicts:
        adjacency_matrix[u, v].append(event_times)
    
    return adjacency_matrix


def plotlsp(z1, N, nodes, i):
    import matplotlib.pyplot as plt
    plt.figure(i, figsize=(4, 3))
    off_set_x = -0.012
    off_set_y = 0.012
    size = 12
    x_pos = z1[:,0]
    y_pos = z1[:,1]

    for i in range(N):
        plt.scatter(x_pos[i],y_pos[i], marker = 'x', color = 'black')
        plt.text(x_pos[i]+off_set_x, y_pos[i]+off_set_y, nodes[i])
    plt.axis('equal')
    plt.tight_layout(0.1)
    #plt.xticks(fontsize=12)
    #plt.yticks(fontsize=12)
    
    #pl.legend(loc='upper right')
    plt.show()

# test simulation
if __name__ == "__main__":
    
    np.random.seed(2001)
    end_time = [100]
    #n_nodes = [10, 20, 50]
    n_nodes= [20]
    dim = 2
    beta = [0.1, 1, 10]
    C = [1/3, 1/3, 1/3]
    
    # intercept, alpha1, alpha2
    theta = [-3.2, 0.01, 0.02]
    
    n0 = len(n_nodes)
    for n_num in range(n0):
        print("Num of nodes are:", n_nodes[n_num])
        
        z = np.random.normal(0,1, [n_nodes[n_num], dim]) 
        # sender and receive effect
        delta = np.random.normal(0,1,size=[(n_nodes[n_num]-1)*2,1])
        ## actual params should include n*dim z, 3 thetas, n-1 delta_1, n-1 delta_2
        p_1 = np.vstack((np.resize(z,(n_nodes[n_num]*dim, 1)), np.resize(theta,(3, 1))))
        params_actual = np.vstack((p_1, delta))
        
        theta_delta = np.vstack((np.resize(theta,(3, 1)), delta))
        
        delta_1 = delta[:n_nodes[n_num]-1]
        delta_2 = delta[n_nodes[n_num]-1:]
        
        if abs(sum(delta_1)) > 2: delta_1 = np.append(delta_1, 0)
        else: delta_1 = np.append(delta_1, -sum(delta_1))
        if abs(sum(delta_1)) > 2: delta_2 = np.append(delta_1, 0)
        else: delta_2 = np.append(delta_2, -sum(delta_2))
        
        delta_1 = npresize(delta_1,n_nodes[n_num], 1)
        delta_2 = npresize(delta_2,1, n_nodes[n_num])
        
        xp = np.tile(np.sum(np.power(z,2), 1), [n_nodes[n_num],1]).T  
        dis = xp + xp.T - 2*np.matmul(z, z.T)
        np.fill_diagonal(dis, 0.0)
        
        logit = -dis + theta[0] + delta_1 + delta_2
        mu_actual = e**logit
        np.fill_diagonal(mu_actual, 0.0)
        
        # number of T
        n1 = len(end_time)

        # number of simulation for each T
        n2 = 1
        RMSE_delta_1_array = np.zeros((n1, n2))
        RMSE_delta_2_array = np.zeros((n1, n2))
        RMSE_z_array = np.zeros((n1, n2))
        RMSE_mu_array = np.zeros((n1, n2))
        RMSE_theta_1_array = np.zeros((n1, n2))
        RMSE_alpha_1_array = np.zeros((n1, n2))
        RMSE_alpha_2_array = np.zeros((n1, n2))
        
        for T in range(n1):
            print("Duration time is:", end_time[T])
            
            seed = 2000
            for i in range(n2):
                #print("sum of z is:", sum(z))
                print("--------------------------random params-------------------------------")
                P_sim = lantet_hawkes_simulation(end_time[T], n_nodes[n_num], beta, z, C, theta, delta, seed)
                count_full = count_process(P_sim)
                print("Simulation complete, the number of events are: ", sum(sum(count_full)))
            
                # Estimating the simulation data
                start_fit_time = time.time()   
    
                z_est, theta_est = LSHM_mle(P_sim, count_full, end_time[T], dim, beta, verbose=False)
                end_fit_time = time.time()  
                print("fitting seed is:", seed)
                print("fitting time is:", (end_fit_time - start_fit_time))
                
                z_est = np.resize(z_est,(z_est.shape[0],1))
                theta_est = np.resize(theta_est,(theta_est.shape[0],1))
                params_est = np.vstack((z_est, theta_est))
                
                # procrustes transform for actual z and estimate z, z should multiplied by sqrt(theta_est[0])
                z_est_single = np.resize(z_est,(n_nodes[n_num],dim))*np.sqrt(theta_est[0])
                mtx1,mtx2,disparity = procrustes(z, z_est_single)
                                      
                print('theta are:', theta_est[:4])
                #RMSE_delta = np.sqrt(np.mean((delta - theta_est[4:])**2))
                RMSE_z = np.sqrt(np.mean((mtx1.flatten() - mtx2.flatten())**2))
                
                nodes_label =np.linspace(0, 19, 20).astype(int).astype('str')
                import matplotlib.pyplot as plt
                plotlsp(mtx1, 20, nodes_label,1) 
                plt.savefig('./storage/results/' + 'simulate_actual.pdf') 
                plotlsp(mtx2, 20, nodes_label,2) 
                plt.savefig('./storage/results/' +'simulate_estimate.pdf') 
                
                
                print('RMSE for z:', RMSE_z)
    
                # compute estimate mu
                delta_1_est = theta_est[4:n_nodes[n_num]+3]
                delta_2_est = theta_est[n_nodes[n_num]+3:]
                
                delta_1_est = np.append(delta_1_est, -sum(delta_1_est))
                delta_2_est = np.append(delta_2_est, -sum(delta_2_est))
                
                rmse_delta_1 = np.sqrt(np.mean((delta_1_est.flatten() - delta_1.flatten())**2))
                rmse_delta_2 = np.sqrt(np.mean((delta_2_est.flatten() - delta_2.flatten())**2))
                print('RMSE for delta_1:', rmse_delta_1)
                print('RMSE for delta_2:', rmse_delta_2)
                
                delta_1_est = npresize(delta_1_est,n_nodes[n_num], 1)
                delta_2_est = npresize(delta_2_est,1, n_nodes[n_num])
                 
                xp_est = np.tile(np.sum(np.power(z_est_single,2), 1), [n_nodes[n_num],1]).T  
                dis_est = xp_est + xp_est.T - 2*np.matmul(z_est_single, z_est_single.T)
                np.fill_diagonal(dis_est, 0.0)
         
                logit_est = -theta_est[0] * dis_est + theta_est[1] + delta_1_est + delta_2_est
                mu_est = e**logit_est
                rmse_mu = np.sqrt(np.mean((mu_est - mu_actual)**2))
                
                print("RMSE for mu:", rmse_mu)
    
                RMSE_delta_1_array[T,i] = rmse_delta_1
                RMSE_delta_2_array[T,i] = rmse_delta_2
                RMSE_z_array[T,i] = RMSE_z
                RMSE_mu_array[T,i] = rmse_mu
                RMSE_theta_1_array[T,i] = np.abs(theta_est[1].T-theta[0])
                RMSE_alpha_1_array[T,i] = np.abs(theta_est[2].T-theta[1])
                RMSE_alpha_2_array[T,i] = np.abs(theta_est[3].T-theta[2])
                
                # set seed for another run                   
                seed+=1
            
    # save files
    saved_name = 'simulation_T'+ str(end_time[-1])+ '_N'+ str(n_nodes[-1])+ '.pickle'
    with open(saved_name, 'wb') as handle:
        pickle.dump([RMSE_delta_1_array,RMSE_delta_2_array, RMSE_z_array, RMSE_mu_array, 
                     RMSE_theta_1_array, RMSE_alpha_1_array, RMSE_alpha_2_array], handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open(filename, 'rb') as f:
        #[RMSE_delta_array, RMSE_z_array, RMSE_mu_array, RMSE_theta_1_array, RMSE_alpha_1_array, RMSE_alpha_2_array] = pickle.load(f)
    
