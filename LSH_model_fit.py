#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 14:32:49 2021

This is the latent space Hawkes model with baseline mu depend on latent space

intensity:
\lambda_{uv}^*(t) &= \mu_{uv} +   \sum_{t_{uv} < t}\sum_b^B C_b \alpha_1 \beta_b e^{-\beta_b(t-t_{uv})} \\
&+   \sum_{t_{vu} < t}\sum_b^B C_b \alpha_2 \beta_b e^{-\beta_b(t-t_{vu})}, \quad \forall u \neq v

"""


import dataset_utils
import Latent_Space_Hawkes_utils as lsh_utils
import autograd.numpy as np
from autograd import grad
import time
import pickle
from math import e
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import networkx as nx
from sklearn.manifold import MDS
import argparse
import sys


def BivariateHawkes_LL(params_array, events_list, end_time, M=2):
    """
    Calculate multivariate Hawkes process log-likelihood(M=2 for bivariate)

    :param params_array: contains all parameters needed for calculating Hawkes process
    :param events_lists: contains two list of timestamps
    :param end_time: end duration time
    :param M: M = 2 indicate bivariate Hawkes process

    :return: log-likelihood of the multivariate Hawkes process
    """ 
    
    
    mu_array, alpha_array, beta, q = params_array
    
    alpha_array = np.resize(alpha_array, [2,2])
    # first term
    first = - np.sum(mu_array)*end_time
    # second term
    second = 0
    for m in range(M): 
        for v in range(M):
            for b in range(len(q)):
                if len(events_list[v]) == 0:
                    continue
                second -= alpha_array[m, v] * q[b] * np.sum(1 - np.exp(-beta[b] * (end_time - events_list[v])))
    # third term
    third = 0
    for m in range(M):
        for k in range(len(events_list[m])):
            tmk = events_list[m][k]
            inter_sum = 0
            for v in range(M):
                for b in range(len(q)):
                    if len(events_list[v]) == 0:
                        continue
                    v_less = events_list[v][events_list[v] < tmk]
                    Rmvk = np.sum(np.exp(-beta[b] * (tmk - v_less)))
                    inter_sum += alpha_array[m, v] * beta[b] * q[b] * Rmvk
            if mu_array[m] + inter_sum == 0:
                third += 0
            else: third += np.log(mu_array[m] + inter_sum)
    return (first+second+third)

def LSHM_mle_am(t, count, endtime, dim, decays, neg_slop = False, verbose=False):
    """
    MLE for LSH model

    :param t: contains all timestamps for all node pairs
    :param count: count matrix for the network where each entry denotes the number of event
    :param endtime: end duration time
    :param dim: latent dimension for the model
    :param decays: decays for the kernel

    :return: estimated parameters for LSH model
    """ 
    n_nodes = count.shape[0]
    
    # 5 conversion - unweigthed adj, shortest path
    # For uniweighted, since the all_pairs_shortest_path_length() gives the unweighted shortest path
    # it no longer need to convet count values greater than 1 to 1.
    initdis = -np.ones((n_nodes,n_nodes))
    #count[count>=1] = 1
    net = nx.Graph(count)
    iLengths = nx.all_pairs_shortest_path_length(net)
    for source,lengths in iLengths:
        for target in lengths.keys():
            initdis[source,target] = lengths[target]
    initdis[initdis==-1] = 2*np.max(initdis)
     
    # make it Symmetrical
    initdis = (initdis+initdis.T)/2.0  
    # fill all diagonal entries to 0
    if neg_slop:
        initdis = np.max(initdis)-initdis
    np.fill_diagonal(initdis, 0.0)
    
    # MDS initilization
    netMDS = MDS(n_components=dim,dissimilarity='precomputed')
    initPos = netMDS.fit_transform(initdis)
    initPos = np.resize(initPos,(n_nodes*dim, 1))
    
    # random initilize theta(slope and intercept for z, and two jump size) and delta ((n-1)*2)
    #np.random.seed(1002)
    theta = np.random.uniform(0,2,size=[4,1])
    delta = np.random.normal(0,1,size=[(n_nodes-1)*2,1])


    # stack theta and delta together for alternating minimization, temporary values for theta, z, ll
    theta_temp = np.vstack((theta, delta))
    z_temp = initPos
    ll_temp = 0
    
    # number of iterations for alternating minimization
    opt = {"maxiter": 1}
    
    # set initial log-likelihood
    ll1 = 1000000
    # set tolerance error
    TOL = 1e-6
    # contraint slope, and two jump size > 0

    if neg_slop:
        bnds =((None, -0.001), (None, None), (0.001, None), (0.001, None))  
    else: bnds =((0.001, None), (None, None), (0.001, None), (0.001, None)) 
    for i in range(n_nodes-1):
        bnds += ((None, None), (None, None))
    
    c = 0
    while(np.abs(ll1 - ll_temp)/max(np.abs(ll1), np.abs(ll_temp), 1) > TOL):
        
        c+=1
        #print(np.abs(ll1 - ll_temp)/max(np.abs(ll1), np.abs(ll_temp), 1))
        
        z_temp1 = np.resize(z_temp,(n_nodes*dim, 1))
        theta_temp1 = np.resize(theta_temp,((n_nodes*2+2), 1))
        # stack all unknown parameters
        param_est = np.vstack((z_temp1, theta_temp1))
        
        
        # update likelihood
        ll_temp = ll1
        # likelihood for the model
        ll1 = LSHM_ll(param_est, t, n_nodes, endtime, dim, decays)
        #print("likelihood temp:",ll_temp)
        
        # alternativly optimize z and the other parameters
        grad_ll_1 = grad(LSHM_ll_fix_z)
        res_theta = minimize(LSHM_ll_fix_z, theta_temp, args=(z_temp, t, n_nodes, endtime, dim, decays, verbose), method="L-BFGS-B",
                 jac = grad_ll_1,bounds = bnds, options=opt)   
        theta_temp = res_theta.x
        grad_ll_2 = grad(LSHM_ll_fix_theta)
        res_z = minimize(LSHM_ll_fix_theta, z_temp, args=(theta_temp, t, n_nodes, endtime, dim, decays, verbose), method="L-BFGS-B",
             jac = grad_ll_2,options=opt)   
        z_temp = res_z.x
     
    print('num of iterations:', c*2)
    #param_est = np.vstack((z_temp, theta_temp))
    print('ll at convergence:', ll1)
    
    return z_temp, theta_temp

def LSHM_mle(t, count, endtime, dim, decays, neg_slop = False, verbose=False):
    """
    MLE for LSH model

    :param t: contains all timestamps for all node pairs
    :param count: count matrix for the network where each entry denotes the number of event
    :param endtime: end duration time
    :param dim: latent dimension for the model
    :param decays: decays for the kernel

    :return: estimated parameters for LSH model
    """ 
    n_nodes = count.shape[0]
    
    # conversion - unweigthed adj, shortest path
    # For uniweighted, since the all_pairs_shortest_path_length() gives the unweighted shortest path
    # it no longer need to convet count values greater than 1 to 1.
    initdis = -np.ones((n_nodes,n_nodes))
    #count[count>=1] = 1
    net = nx.Graph(count)
    iLengths = nx.all_pairs_shortest_path_length(net)
    for source,lengths in iLengths:
        for target in lengths.keys():
            initdis[source,target] = lengths[target]
    initdis[initdis==-1] = 2*np.max(initdis)
     
    # make it Symmetrical
    initdis = (initdis+initdis.T)/2.0  
    # fill all diagonal entries to 0
    from scipy.special import expit
    if neg_slop:
        #initdis = np.max(initdis)-initdis
        initdis = 1-expit(initdis)
    np.fill_diagonal(initdis, 0.0)
    
    # MDS initilization
    netMDS = MDS(n_components=dim,dissimilarity='precomputed')
    initPos = netMDS.fit_transform(initdis)
    initPos = np.resize(initPos,(n_nodes*dim, 1))
    
    # random initilize theta(slope and intercept for z, and two jump size) and delta ((n-1)*2)
    #np.random.seed(see)
    theta = np.random.uniform(0,2,size=[4,1])
    delta = np.random.normal(0,1,size=[(n_nodes-1)*2,1])


    # stack theta and delta together for alternating minimization, temporary values for theta, z, ll
    theta_temp = np.vstack((theta, delta))
    z_temp = initPos

        
    # optimize all parameters together
        
    # bnds
    bnds =[]
    if dim % 2 == 0:
        num_iter = int(n_nodes*(dim/2))
        for i in range(num_iter):
            bnds += ((None, None), (None, None))
        if neg_slop:
            bnds +=((None, -0.001), (None, None), (0.001, None), (0.001, None))  
        else: bnds +=((0.001, None), (None, None), (0.001, None), (0.001, None))  
    elif dim % 2 == 1:
        for i in range(n_nodes):
            bnds += ((None, None), (None, None), (None, None))
        num_iter = int(n_nodes*((dim-1)/2-1))
        for i in range(num_iter):
            bnds += ((None, None), (None, None))  
        if neg_slop:
            bnds +=((None, -0.001), (None, None), (0.001, None), (0.001, None))  
        else: bnds +=((0.001, None), (None, None), (0.001, None), (0.001, None))  
     
    for i in range(n_nodes-1):
        bnds += ((None, None), (None, None))
    
    from itertools import count
    c = count()
    def callback(x):
        #print(c)
        next(c)
    
    z_temp1 = np.resize(z_temp,(n_nodes*dim, 1))
    theta_temp1 = np.resize(theta_temp,((n_nodes*2+2), 1))
    param_est = np.vstack((z_temp1, theta_temp1))
    grad_ll = grad(LSHM_ll)
    res = minimize(LSHM_ll, param_est, args=(t, n_nodes, endtime, dim, decays, verbose), method="L-BFGS-B", callback=callback,
             jac = grad_ll, bounds = bnds, options = {'ftol': 1e-6 })  
    
    #print('num of iterations: %f', c)
    #print('ll at convergence: %f', LSHM_ll(res.x, t, n_nodes, endtime, dim, decays, verbose))
    
    z_temp, theta_temp = res.x[:n_nodes*dim], res.x[n_nodes*dim:]
    print(res.message)
    print(res.success)
        
    return z_temp, theta_temp



def LSHM_ll(params, timestamp, n_nodes, end_time, dim, decays, verbose=False):
    """
    Log-likelihood for LSH model

    :param params: contains all unknown parameters
    :param timestamp: contains all timestamps for all node pairs
    :param n_nodes: number of nodes in the network
    :param end_time: end duration time
    :param dim: latent dimension for the model
    :param decays: decays for the kernel

    :return: negative log-likelihood for LSH model
    """ 
    z = params[0:n_nodes*dim]
    z = np.ravel(z)
    theta = params[n_nodes*dim:n_nodes*dim+4]
    q = [1/3, 1/3, 1/3]
    
    delta_1 = params[n_nodes*dim+4:n_nodes*(1+dim)+3]
    delta_2 = params[n_nodes*(1+dim)+3:]
    
    delta_1 = np.append(delta_1, -sum(delta_1))
    delta_2 = np.append(delta_2, -sum(delta_2))
    
    delta_1 = npresize(delta_1,n_nodes, 1)
    delta_2 = npresize(delta_2,1, n_nodes)
    
    beta = decays

    
    z = npresize(z,n_nodes,dim)
    
    # distance squared
    xp = np.tile(np.sum(np.power(z,2), 1), [n_nodes,1]).T  
    dis = xp + xp.T - 2*np.matmul(z, z.T)
    #print("test")
    
    logit = -theta[0] * dis + theta[1] + delta_1 + delta_2
    lam = e**logit
    #np.fill_diagonal(lam, 0.0001)
    alpha_1 = theta[2]
    alpha_2 = theta[3]


    sum_ll = 0
    for u in range(n_nodes):
        for v in range(u):
            t = [timestamp[u,v], timestamp[v,u]]
            mu = [ lam[u,v], lam[v,u] ]
            alpha = [alpha_1, alpha_2, alpha_2, alpha_1]
            alpha = np.resize(alpha, [2,2])
            p = [mu, alpha, beta, q]    
            ll = BivariateHawkes_LL(p, t, end_time, 2)
            sum_ll += ll 

    return -sum_ll

def LSHM_ll_fix_z(params, pos_temp, timestamp, n_nodes, end_time, dim, decays, verbose=False):
    """
    Log-likelihood for LSH model with fixed latent positions

    :param params: contains all unknown parameters
    :param pos_temp: fixed latent positions values
    :param timestamp: contains all timestamps for all node pairs
    :param n_nodes: number of nodes in the network
    :param end_time: end duration time
    :param dim: latent dimension for the model
    :param decays: decays for the kernel

    :return: negative log-likelihood for LSH model with fixed latent positions
    """ 
    z = np.ravel(pos_temp)
    
    theta = params[:4]
    
    delta_1 = params[4:n_nodes+3]
    delta_2 = params[n_nodes+3:]
    
    delta_1 = np.append(delta_1, -sum(delta_1))
    delta_2 = np.append(delta_2, -sum(delta_2))
    
    delta_1 = npresize(delta_1,n_nodes, 1)
    delta_2 = npresize(delta_2,1, n_nodes)
    
    q = [1/3, 1/3, 1/3]

    beta = decays

    z = npresize(z,n_nodes,dim)
    
    # distance squared
    xp = np.tile(np.sum(np.power(z,2), 1), [n_nodes,1]).T  
    dis = xp + xp.T - 2*np.matmul(z, z.T)

    
    logit = -theta[0] * dis + theta[1] + delta_1 + delta_2
    lam = e**logit
    #np.fill_diagonal(lam, 0.0001)
    alpha_1 = theta[2]
    alpha_2 = theta[3]
    
    sum_ll = 0
    
    for u in range(n_nodes):
        for v in range(u):
            t = [timestamp[u,v], timestamp[v,u]]
            mu = [ lam[u,v], lam[v,u] ]
            alpha = [alpha_1, alpha_2, alpha_2, alpha_1]
            alpha = np.resize(alpha, [2,2])
            p = [mu, alpha, beta, q]    
            ll = BivariateHawkes_LL(p, t, end_time, 2)
            sum_ll += ll    
    return -sum_ll

def LSHM_ll_fix_theta(params, theta_temp, timestamp, n_nodes, end_time, dim, decays, verbose=False):  
    """
    Log-likelihood for LSH model with fixed parameters other than latent positions

    :param params: contains all unknown parameters
    :param theta_temp: fixed parameters other than latent positions values
    :param timestamp: contains all timestamps for all node pairs
    :param n_nodes: (int) number of nodes in the network
    :param end_time: (int) end duration time
    :param dim: (int) latent dimension for the model
    :param decays: decays for the kernel

    :return: negative log-likelihood for LSH model with fixed parameters other than latent positions
    """ 
    
    z = params
    
    theta = theta_temp[0:4]
    q = [1/3, 1/3, 1/3]
    
    delta_1 = theta_temp[4:n_nodes+3]
    delta_2 = theta_temp[n_nodes+3:]
    
    delta_1 = np.append(delta_1, -sum(delta_1))
    delta_2 = np.append(delta_2, -sum(delta_2))
    
    delta_1 = npresize(delta_1,n_nodes, 1)
    delta_2 = npresize(delta_2,1, n_nodes)
    
    beta = decays
    
    z = npresize(z,n_nodes,dim)
    # distance squared
    xp = np.tile(np.sum(np.power(z,2), 1), [n_nodes,1]).T  
    dis = xp + xp.T - 2*np.matmul(z, z.T)
    
    #distance
    #dis = distance_matrix(z, z)

    logit = -theta[0] * dis + theta[1] + delta_1 + delta_2
    lam = e**logit
    #np.fill_diagonal(lam, 0.0001)
    alpha_1 = theta[2]
    alpha_2 = theta[3]
    
    sum_ll = 0
    for u in range(n_nodes):
        for v in range(u):
            t = [timestamp[u,v], timestamp[v,u]]
            mu = [ lam[u,v], lam[v,u] ]
            alpha = [alpha_1, alpha_2, alpha_2, alpha_1]
            alpha = np.resize(alpha, [2,2])
            p = [mu, alpha, beta, q]    
            ll = BivariateHawkes_LL(p, t, end_time, 2)
            sum_ll += ll      
    return -sum_ll




def npresize(a, u, v):
    """
    to resize an array in (u,v). The purpose of doing this is the np.resize is not compatible with autograd.

    :param a: array you would like to resize
    :param w: (int) first dimension
    :param v: (int) second dimension

    :return: new array with size (u,v)
    """ 
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



# Test real dataset
if __name__ == "__main__":
    
    ### Argument
    parser = argparse.ArgumentParser('Latent space Hawkes model training')
    parser.add_argument('--data', type=str, help='Dataset name (eg. reality, enron, MID, Enron-Yang or fb-forum)',
                    default='reality')
    parser.add_argument('-d', '--dim', nargs='+', type=int, help='latent dimensions(can enter multiple values)',
                    default= [2])
    parser.add_argument('-n', '--negative', type=bool, help='whether to use negative slope for latent space model',
                    default= False)
    parser.add_argument('-a', '--alternating', type=bool, help='whether to use alternating minimization for latent space model',
                    default= False)
    parser.add_argument('-c', '--continent', type=bool, help='whether to plot 2D continent plot for MID',
                    default= False)

    try:
      args = parser.parse_args()
    except:
      parser.print_help()
      sys.exit(0)
      
    seed = 2000
    #np.random.seed(seed)
    dataset_name = args.data
    dim_input = args.dim
    continent = args.continent
    neg = args.negative
    am = args.alternating


    if dataset_name == 'reality':
        train_tuple, test_tuple, all_tuple, nodes_not_in_train = dataset_utils.load_reality_mining_test_train(remove_nodes_not_in_train=True)
        events_dict_train, n_nodes_train, end_time_train = train_tuple
        events_dict_all, n_nodes_all, end_time_all = all_tuple
        events_dict_test, n_nodes_test, end_time_test = test_tuple
        # convert the data from dictionary to adjacency list
        P_train = lsh_utils.event_dict_to_adjacency_list(n_nodes_train, events_dict_train)
        P_all = lsh_utils.event_dict_to_adjacency_list(n_nodes_all, events_dict_all)
        decays = [1/0.25,1/6,1/45]
        
    elif dataset_name == 'enron':
        train_tuple, test_tuple, all_tuple, nodes_not_in_train = dataset_utils.load_enron_train_test(remove_nodes_not_in_train=True)
        events_dict_train, n_nodes_train, end_time_train = train_tuple
        events_dict_all, n_nodes_all, end_time_all = all_tuple
        events_dict_test, n_nodes_test, end_time_test = test_tuple
        P_train = lsh_utils.event_dict_to_adjacency_list(n_nodes_train, events_dict_train)
        P_all = lsh_utils.event_dict_to_adjacency_list(n_nodes_all, events_dict_all)
        decays = [1/0.6,1/16,1/114]
    
    elif dataset_name == 'fb-forum':
        train_tuple, test_tuple, all_tuple, nodes_not_in_train = dataset_utils.load_fb_forum_train_test(remove_nodes_not_in_train=True)
        events_dict_train, n_nodes_train, end_time_train = train_tuple
        events_dict_all, n_nodes_all, end_time_all = all_tuple
        events_dict_test, n_nodes_test, end_time_test = test_tuple
        P_train = lsh_utils.event_dict_to_adjacency_list(n_nodes_train, events_dict_train)
        P_all = lsh_utils.event_dict_to_adjacency_list(n_nodes_all, events_dict_all)
        # total of 165 days
        decays = [1/0.25,1/6, 1/42]
        
    elif dataset_name =="MID":
        # load MID dataset
        '''
        timestamp_scale = 1000
        dnx_pickle_file_name = './storage/datasets/MID/MID_std1hour.p'
        #dnx_pickle_file_name = './storage/datasets/MID/MID_std1sec.p'
        train_tup, all_tup, nodes_not_in_train = dataset_utils.load_MID_data_train_all(dnx_pickle_file_name, split_ratio=0.8,
                                                                     scale=timestamp_scale, remove_small_comp=True,remove_node_not_in_train=True)
        events_dict_train, end_time_train, n_nodes_train, n_events_train, id_node_map_train = train_tup
        events_dict_all, end_time_all, n_nodes_all, n_events_all, id_node_map_all = all_tup
        P_train = lsh_utils.event_dict_to_adjacency_list(n_nodes_train, events_dict_train)
        P_all = lsh_utils.event_dict_to_adjacency_list(n_nodes_all, events_dict_all)
        
        '''
        train_tuple, test_tuple, all_tuple, nodes_not_in_train = dataset_utils.load_mid_train_test(remove_nodes_not_in_train=True)
        events_dict_train, n_nodes_train, end_time_train = train_tuple
        events_dict_all, n_nodes_all, end_time_all = all_tuple
        events_dict_test, n_nodes_test, end_time_test = test_tuple
        P_train = lsh_utils.event_dict_to_adjacency_list(n_nodes_train, events_dict_train)
        P_all = lsh_utils.event_dict_to_adjacency_list(n_nodes_all, events_dict_all)
        with open('./storage/datasets/MID/country_label.csv') as f:
            id_node_map_all = f.read().splitlines()
        decays = [0.00497, 0.119, 0.835]

        
    elif dataset_name == "Enron-Yang":
        # code to read Yang's Enron dataset
        train_tuple, test_tuple, all_tuple, nodes_not_in_train = dataset_utils.load_enron_yang_train_test(remove_nodes_not_in_train=True)
        events_dict_train, n_nodes_train, end_time_train = train_tuple
        events_dict_all, n_nodes_all, end_time_all = all_tuple
        events_dict_test, n_nodes_test, end_time_test = test_tuple
        P_train = lsh_utils.event_dict_to_adjacency_list(n_nodes_train, events_dict_train)
        P_all = lsh_utils.event_dict_to_adjacency_list(n_nodes_all, events_dict_all)
        decays = [24,1,1/7]
        
    # count how many events in each entry
    count_train = lsh_utils.count_process(P_train)
    count_full = lsh_utils.count_process(P_all)

    
    for dim in dim_input:
        print("{} Dataset - Latent Space Hawkes Process: {}".format(dataset_name, dim))
        print("fitting seed is:", seed)     
        start_fit_time = time.time()  
        
        if am:
            z_est, theta_est = LSHM_mle_am(P_train, count_train, end_time_train, dim, decays, neg, verbose=False)
        else: 
            z_est, theta_est = LSHM_mle(P_train, count_train, end_time_train, dim, decays, neg, verbose=False)
        
        filename = ('LSH_' + str(dim) + 'd_'+ dataset_name+ '.pickle')
        
        with open(filename, 'wb') as handle:
            pickle.dump([z_est, theta_est], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(filename, 'rb') as f:
            [z_est, theta_est] = pickle.load(f)
        end_fit_time = time.time()  
        
        #theta_est[0] =  -theta_est[0]
        
        print("fitting time is:", (end_fit_time - start_fit_time))
        
        # stack z and theta. Theta includes all parameters other than z (sender, recieive, jump size, alpha, theta)
        z_est = np.resize(z_est,(z_est.shape[0],1))
        theta_est = np.resize(theta_est,(theta_est.shape[0],1))
        params_est = np.vstack((z_est, theta_est))
        #print("estimated slope is:", theta_est[0])

        # compute log-likelihood
        logll_train =  LSHM_ll(params_est, P_train, n_nodes_train, end_time_train, dim, decays)
        logll_full =  LSHM_ll(params_est, P_all, n_nodes_train, end_time_all, dim, decays) 
        logll_test = logll_full - logll_train
        
        print("log likelihood per event:")
        print("full: %f", -logll_full/(np.sum(count_full)))
        print("training:", -logll_train/(np.sum(count_train)))
        print("test:", -logll_test/(np.sum(count_full) - np.sum(count_train)))
        
        # assign labels to each node
        if dataset_name != 'MID':
            nodes_label =np.linspace(0,n_nodes_train-1, n_nodes_train).astype(int).astype('str')
        else: nodes_label = id_node_map_all
        
        # plot 2D latent sapce 
        if dim == 2 and dataset_name == 'MID':
            lsh_utils.plotlsp(np.resize(z_est,(n_nodes_train,dim)), n_nodes_train, nodes_label, count_full ,1, "Estimate plot", dataset_name, True) 
            plt.savefig('./storage/results/' + dataset_name +'/LSP_'+ dataset_name +'.pdf') 

        # plot MID with continent colored plot
        if dataset_name == 'MID' and continent and dim == 2:
            continent_txt = np.genfromtxt("./storage/datasets/MID/MID_country.csv",np.object, delimiter=',')
            lsh_utils.plotlspmid(np.resize(z_est,(n_nodes_train,dim)), n_nodes_train, nodes_label, count_full ,2, "Estimate plot", continent_txt, dataset_name) 
            plt.savefig('./storage/results/' + dataset_name +'/LSP_'+ dataset_name +'_cont.pdf') 
        
        # save nodal effects
        '''
        delta_1 = theta_est[4: 3+n_nodes_train]
        delta_2 = theta_est[3+n_nodes_train:]
        delta_1 = np.append(delta_1, -sum(delta_1))
        delta_2 = np.append(delta_2, -sum(delta_2))
        
        with open('nodal effect.csv', 'w') as f:
            for i in range(n_nodes_train):
                f.write(nodes_label[i])
                f.write(',')
                f.write(str(delta_1[i]))
                f.write(',')
                f.write(str(delta_2[i]))
                f.write('\n')'''
