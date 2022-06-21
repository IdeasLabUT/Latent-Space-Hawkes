#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 13:07:57 2021

This is the dynamic link prediction for LSH model. Including plot ROC curves and get mean of AUC values over 100 time interval

"""
import numpy as np
from scipy import integrate
from sklearn import metrics
import dataset_utils
import matplotlib.pyplot as plt
import Latent_Space_Hawkes_utils as lsh_utils
import pickle
from math import e
from LSH_model_fit import yang_dict_to_adjacency_list
import argparse
import sys

def predict_probs(n_nodes, t0, delta, timestamp, mu, alpha, beta, C):
    """
    Computes the predicted probability that a link from u to v appears in
    [t, t + delta) based only on the events data from [0, t)
    for all combinations of u and v.
    """
    prob_dict = np.zeros((n_nodes, n_nodes))   # Predicted probs that link exists
    for u in range(n_nodes):
        for v in range(n_nodes):
            t1 = timestamp[u,v][np.where(timestamp[u,v] < t0)]
            t2 = timestamp[v,u][np.where(timestamp[v,u] < t0)]
            prob_dict[u,v] = 1 - np.exp(-integrate.quad(univariate_cif, t0, t0+delta, args=([t1, t2], mu[u,v], alpha, beta, C),limit=100)[0])

    return prob_dict
    
def univariate_cif(t, times, mu, alpha, beta, C):
    " Conditional intensity function of a univariate Hawkes process with exponential kernel.       "
    "                                                                                              "
    " Parameters:                                                                                  "
    " - mu corresponds to the baseline intensity of the HP.                                        "
    " - alpha corresponds to the jump intensity, representing the jump in intensity upon arrival.  "
    " - beta is the decay parameter, governing the exponential decay of intensity.                 "
    " - C is the scalling parameter of the jump size                                               "
    #times = np.array(times, dtype=object)
    t_uv = np.array(times[0])
    t_vu = np.array(times[1])
    lam = mu
    for b in range(3):
        lam += C[b]*sum(alpha[0]*beta[b]*np.exp(-beta[b]*np.subtract(t, t_uv[np.where(t_uv<t)]))) \
        + sum(alpha[1]*beta[b]*np.exp(-beta[b]*np.subtract(t, t_vu[np.where(t_vu<t)])))
    return lam

def actual_y(n_nodes, t0, delta, timestamp):
    """
    calculate the actual labels in the time window [t0, t0+ delta]

    :param n_nodes: number of nodes in the network.
    :param t0: start time of the wondow.
    :param delta: the window size.
    :param timestamp: contains timestamps for all node pairs

    :return: array list that contains the actual label for each pair of nodes in the time window 
    """
    y = np.zeros((n_nodes, n_nodes)) 
    
    for u in range(n_nodes):
        for v in range(n_nodes):
            timestamp[u,v] = np.array(timestamp[u,v])
            if len(timestamp[u,v][np.logical_and(timestamp[u,v]>=t0, timestamp[u,v]<=t0+delta)]) > 0:
                y[u,v] = 1
    return y

def calculate_auc(n_nodes, t0, delta, timestamp, mu, alpha, beta, C, show_figure = False):
    """
    calculate the probabilities in the time window [t0, t0+ delta]

    :param n_nodes: number of nodes in the network.
    :param t0: start time of the wondow.
    :param delta: the window size.
    :param timestamp: contains timestamps for all node pairs
    :param mu: corresponds to the baseline intensity of the HP. 
    :param alpha: corresponds to the jump intensity, representing the jump in intensity upon arrival
    :param beta: is the decay parameter, governing the exponential decay of intensity
    :param C: is the scalling parameter of the jump size  

    :return: array list that contains the auc values for each pair of nodes in the time window 
    """
    y = actual_y(n_nodes, t0, delta, timestamp)
    scores = predict_probs(n_nodes, t0, delta, timestamp, mu, alpha, beta, C)
    fpr, tpr, thresholds = metrics.roc_curve(y.flatten(), scores.flatten(), pos_label=1)
    roc_auc = metrics.roc_auc_score(y.flatten(), scores.flatten())
    if show_figure == True:
        plt.figure(1, figsize=(5, 4))
        plt.plot(fpr, tpr, color='darkorange', lw=2)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate',fontsize=12)
        plt.tight_layout()
        plt.show()
    return roc_auc

    

if __name__ == "__main__":

    
    ### Argument
    parser = argparse.ArgumentParser('Latent space Hawkes model training')
    parser.add_argument('--data', type=str, help='Dataset name (eg. reality, enron, MID or Enron-Yang)',
                    default='Enron-Yang')
    parser.add_argument('-d', '--dim', type=int, help='latent dimensions)',
                    default= 4)
    try:
      args = parser.parse_args()
    except:
      parser.print_help()
      sys.exit(0)
      
    seed = 2000
    #np.random.seed(seed)
    dataset_name = args.data
    dim = args.dim
    

    if dataset_name == 'reality':
        train_tuple, test_tuple, all_tuple, nodes_not_in_train = dataset_utils.load_reality_mining_test_train(remove_nodes_not_in_train=True)
        events_dict_train, n_nodes_train, end_time_train = train_tuple
        events_dict_all, n_nodes_all, end_time_all = all_tuple
        events_dict_test, n_nodes_test, end_time_test = test_tuple
        decays = [1/0.25,1/6,1/45]
        # two weeks window
        P_train = lsh_utils.event_dict_to_adjacency_list(n_nodes_train, events_dict_train)
        P_all = lsh_utils.event_dict_to_adjacency_list(n_nodes_all, events_dict_all)
        w_size = 60
        
    elif dataset_name == 'enron':
        train_tuple, test_tuple, all_tuple, nodes_not_in_train = dataset_utils.load_enron_train_test(remove_nodes_not_in_train=True)
        events_dict_train, n_nodes_train, end_time_train = train_tuple
        events_dict_all, n_nodes_all, end_time_all = all_tuple
        events_dict_test, n_nodes_test, end_time_test = test_tuple
        decays = [1/0.6,1/16,1/114]
        # convert the data from dictionary to adjacency list
        P_train = lsh_utils.event_dict_to_adjacency_list(n_nodes_train, events_dict_train)
        P_all = lsh_utils.event_dict_to_adjacency_list(n_nodes_all, events_dict_all)
        # two weeks window
        #w_size = 233
        w_size = 125
    
    elif dataset_name == 'fb-forum':
        train_tuple, test_tuple, all_tuple, nodes_not_in_train = dataset_utils.load_fb_forum_train_test(remove_nodes_not_in_train=True)
        events_dict_train, n_nodes_train, end_time_train = train_tuple
        events_dict_all, n_nodes_all, end_time_all = all_tuple
        events_dict_test, n_nodes_test, end_time_test = test_tuple
        P_train = lsh_utils.event_dict_to_adjacency_list(n_nodes_train, events_dict_train)
        P_all = lsh_utils.event_dict_to_adjacency_list(n_nodes_all, events_dict_all)
        # total of 165 days
        w_size = 80;
        decays = [1/0.25,1/6, 1/42]
        
    elif dataset_name =="MID":
        # load MID dataset
        timestamp_scale = 1000
        dnx_pickle_file_name = './storage/datasets/MID/MID_std1hour.p'
        #dnx_pickle_file_name = './storage/datasets/MID/MID_std1sec.p'
        train_tup, all_tup, nodes_not_in_train = dataset_utils.load_MID_data_train_all(dnx_pickle_file_name, split_ratio=0.8,
                                                                     scale=timestamp_scale, remove_small_comp=True,remove_node_not_in_train=True)
        events_dict_train, end_time_train, n_nodes_train, n_events_train, id_node_map_train = train_tup
        events_dict_all, end_time_all, n_nodes_all, n_events_all, id_node_map_all = all_tup
        decays = [0.00497, 0.119, 0.835]
            # convert the data from dictionary to adjacency list
        P_train = lsh_utils.event_dict_to_adjacency_list(n_nodes_train, events_dict_train)
        P_all = lsh_utils.event_dict_to_adjacency_list(n_nodes_all, events_dict_all)
        # two month window
        w_size = 7.15
        
    elif dataset_name == "Enron-Yang":
        # code to read Yang's Enron dataset
        with open('enron-events.pckl', 'rb') as f:
            n_nodes, end_time_all, enron_train = pickle.load(f)
        
        P_all = yang_dict_to_adjacency_list(n_nodes, enron_train)
        P_train, P_test, end_time_train, end_time_test = lsh_utils.split_train_test(P_all, end_time_all, 0.7)
        n_nodes_train = n_nodes
        n_nodes_test = n_nodes
        n_nodes_all = n_nodes
        decays = [24,1,1/7]
        w_size = 14
    
    # count how many events in each entry
    count_train = lsh_utils.count_process(P_train)
    count_full = lsh_utils.count_process(P_all)
    print("{} Dataset - Latent Space Hawkes Process: {}".format(dataset_name, dim))
    
    filename = ('LSH_' + str(dim) + 'd_'+ dataset_name +'.pickle')
    with open(filename, 'rb') as f:
        [z_est, theta_est] = pickle.load(f)
        
    z_est = np.resize(z_est,(z_est.shape[0],1))
    theta_est = np.resize(theta_est,(theta_est.shape[0],1))
    
    params_est = np.vstack((z_est, theta_est))
    z_est_0 = np.resize(z_est,(n_nodes_train,dim)) * np.sqrt(theta_est[0])
    delta_1_est = theta_est[4:n_nodes_train+3]
    delta_2_est = theta_est[n_nodes_train+3:]
    delta_1_est = np.append(delta_1_est, -sum(delta_1_est))
    delta_2_est = np.append(delta_2_est, -sum(delta_2_est))
    delta_1_est = np.resize(delta_1_est,(n_nodes_train, 1))
    delta_2_est = np.resize(delta_2_est,(1, n_nodes_train))
    xp_est = np.tile(np.sum(np.power(z_est_0,2), 1), [n_nodes_train,1]).T
    dis_est = xp_est + xp_est.T - 2*np.matmul(z_est_0, z_est_0.T)
    np.fill_diagonal(dis_est, 0.0)
    logit_est = -dis_est + theta_est[1] + delta_1_est + delta_2_est
    mu_est = e**logit_est
        
    # pre-stored window size.
    t0_file = './storage/dynamic_t0/'+ dataset_name + '_t0.csv'
    t = np.loadtxt(t0_file, np.float, delimiter=',', usecols=(1))

    runs = 2
    C = [1/3,1/3,1/3]
    auc = np.zeros(runs) 
    for i in range(runs):
        #t0 = np.random.uniform(low=end_time_train, high=end_time_all-w_size, size=None)
        t0 = t[i]
        auc[i] = calculate_auc(n_nodes_train, t0, w_size, P_all, mu_est, theta_est[2:4], decays, C, show_figure = True)
        #print("auc is:", auc[i])
    print("the average auc is:", np.mean(auc))
    print("The standard deviation of aus is:", np.std(auc))
    #print(auc)
    plt.savefig('./storage/results/LSH_ROC_'+dataset_name+'.pdf') 
    # save AUC vlaues
    aucname = ('LSH_' + str(dim) + 'd_'+ dataset_name +'_auc.pickle')
    with open(aucname, 'wb') as handle:
        pickle.dump([auc], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(aucname, 'rb') as f:
        a = pickle.load(f)
