# -*- coding: utf-8 -*-
"""
This is to generative test file

"""


import numpy as np
import dataset_utils
import matplotlib.pyplot as plt
import Latent_Space_Hawkes_utils as lsh_utils
import pickle
import networkx as nx
from LSH_model_sim import lantet_hawkes_simulation
import argparse
import sys
from LSH_model_fit import yang_dict_to_adjacency_list


def degree_histogram_directed(G, in_degree=False, out_degree=False):
    """Return a list of the frequency of each degree value.

    Parameters
    ----------
    G : Networkx graph
       A graph
    in_degree : bool
    out_degree : bool

    Returns
    -------
    hist : list
       A list of frequencies of degrees.
       The degree values are the index in the list.

    Notes
    -----
    Note: the bins are width one, hence len(list) can be large
    (Order(number_of_edges))
    """
    nodes = G.nodes()
    if in_degree:
        in_degree = dict(G.in_degree())
        degseq=[in_degree.get(k,0) for k in nodes]
    elif out_degree:
        out_degree = dict(G.out_degree())
        degseq=[out_degree.get(k,0) for k in nodes]
    else:
        degseq=[v for k, v in G.degree()]
    dmax=max(degseq)+1
    freq= [ 0 for d in range(dmax) ]
    for d in degseq:
        freq[d] += 1
    return freq

def cal_recip_trans(P,  dataset="", save=False):
    """

    Parameters
    ----------
    P : The timestamps of all pair of nodes.
    motif_delta : int
        motif windown size.
    dataset : string, optional
        dataset name. The default is "".


    Returns
    -------
    recip : float
        reciprocity values of the network.
    trans : float
        transitivity values of the network..
    num_events : int
        numebr of events in the network..
    avg_cluster : float
        average clustering coefficients values of the network..
    dataset_motif : array
        motif counts of the network..
    degrees_in : float
        in degree values of the network..
    degrees_out : float
        out degree values of the network..
    degrees : float
        degree values of the network..

    """
    count_sim = lsh_utils.count_process(P)
    count_sim[count_sim>0]=1
    net = nx.DiGraph(count_sim) 
    recip = nx.overall_reciprocity(net)
    trans = nx.transitivity(net)
    avg_cluster = nx.average_clustering(net)
    degrees_in = [net.in_degree(n) for n in net.nodes()]
    degrees_out = [net.out_degree(n) for n in net.nodes()]
    degrees = [net.degree(n) for n in net.nodes()]

    num_events = np.sum(lsh_utils.count_process(P))
    if save:
        results_dict = {}
        results_dict["dataset_recip"] = recip
        results_dict["dataset_trans"] = trans
        results_dict["dataset_n_events"] = num_events
        with open(f"{dataset}_counts.p", 'wb') as fil:
            pickle.dump(results_dict, fil)

    return recip, trans, num_events, avg_cluster, degrees_in, degrees_out, degrees


def avg_run_length(P):
    run_len = []
    for i in range(P.shape[0]):
        for j in range(i):
            if len(P[i,j]) > 0 and len(P[j,i]) > 0:
                r = run_length(P[i,j], P[j,i])
                run_len+=r
    return np.mean(run_len)

def run_length(u, v):
    t = np.concatenate((u,v))
    t_sort = np.sort(t)
    for i in range(len(t_sort)):
        if t_sort[i] in u:
            t_sort[i] = 1
        elif t_sort[i] in v:
            t_sort[i] = -1
        else: print("There is error happened in the order list!")
    run = []
    
    if t_sort[0] > 0: pos = 1
    else: pos = -1
    
    r = 1
    for i in range(1, len(t_sort)):
        if t_sort[i] == pos: 
            r+=1
        else: 
            pos = -pos
            run.append(r)
            r = 1
        
    return run
    
            


def plot_degree_dist(G):
    """
    Plot degree distribution

    Parameters
    ----------
    G : networkx
        Graph.

    Returns
    -------
    None.

    """
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees)
    plt.show()
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser('Latent space Hawkes model training')
    parser.add_argument('--data', type=str, help='Dataset name (eg. reality, enron, MID or Enron-Yang)',
                    default='reality')
    parser.add_argument('-d', '--dim', type=int, help='latent dimensions)',
                    default= 2)
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
        # convert the data from dictionary to adjacency list
        P_train = lsh_utils.event_dict_to_adjacency_list(n_nodes_train, events_dict_train)
        P_all = lsh_utils.event_dict_to_adjacency_list(n_nodes_all, events_dict_all)
        decays = [1/0.25,1/6,1/45]
        
    elif dataset_name == 'enron':
        train_tuple, test_tuple, all_tuple, nodes_not_in_train = dataset_utils.load_enron_train_test(remove_nodes_not_in_train=True)
        events_dict_train, n_nodes_train, end_time_train = train_tuple
        events_dict_all, n_nodes_all, end_time_all = all_tuple
        events_dict_test, n_nodes_test, end_time_test = test_tuple
            # convert the data from dictionary to adjacency list
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

    
    # calculate the actual properties
    recip_acutal, trans_actual, num_events_actual, avg_cluster_actual , degree_in_actual, degree_out_actual, degree_actual = cal_recip_trans(P_all,  dataset="", save=False)
    
    run_length_actual = avg_run_length(P_all)
    
    
    nRuns = 15
    num_events = np.zeros(nRuns)
    recipSim = np.zeros(nRuns)
    transSim = np.zeros(nRuns)
    avg_cluster = np.zeros(nRuns)
    dataset_motif = np.zeros((nRuns,6,6))
    degree_in = np.zeros((nRuns, n_nodes_train))
    degree_out = np.zeros((nRuns, n_nodes_train))
    degree = np.zeros((nRuns,n_nodes_train))
    deg = np.zeros((nRuns,n_nodes_train))
    count_actual = lsh_utils.count_process(P_all)
    deg_actual = np.sum(count_actual,axis= 0)
    run_len = np.zeros(nRuns)
    
    
    seed_sim = 2000
    for run in range(nRuns):
        print(run)
        P_sim = lantet_hawkes_simulation(end_time_all, n_nodes_train, decays, z_est_0, [1/3,1/3,1/3], theta_est[1:4].flatten(), theta_est[4:].flatten(), seed_sim)
        recipSim[run], transSim[run], num_events[run],avg_cluster[run], degree_in[run], degree_out[run], degree[run] = cal_recip_trans(P_sim,  dataset="", save=False)
        count = lsh_utils.count_process(P_sim)
        deg[run] = np.sum(count,axis= 0)
        run_len[run] = avg_run_length(P_sim)
        seed_sim +=1
    
    plt.figure(0, figsize=(5, 4))
    plt.hist(recipSim)
    plt.axvline(x= run_length_actual,color = 'b')
    plt.axvline(x= np.mean(run_len),color = 'r')
    plt.tight_layout()
    plt.show()
    plt.savefig('./storage/results/'+ dataset_name +'/LSH_run_length_'+dataset_name+'.pdf') 
    print("Actual reciprocity:", run_len)
    print("Estimated reciprocity:", np.mean(run_len))
    
    '''
    plt.figure(1, figsize=(5, 4))
    plt.hist(recipSim)
    plt.axvline(x= recip_acutal,color = 'b')
    plt.axvline(x= np.mean(recipSim),color = 'r')
    plt.tight_layout()
    plt.show()
    plt.savefig('./storage/results/'+ dataset_name +'/LSH_reciprocity_'+dataset_name+'.pdf') 
    print("Actual reciprocity:", recip_acutal)
    print("Estimated reciprocity:", np.mean(recipSim))
    
    plt.figure(2, figsize=(5, 4))
    plt.hist(transSim)
    plt.axvline(x= trans_actual,color = 'b')
    plt.axvline(x= np.mean(transSim),color = 'r')
    plt.tight_layout()
    plt.show()
    plt.savefig('./storage/results/'+ dataset_name +'/LSH_trans_'+dataset_name+'.pdf') 
    print("Actual transitivity:", trans_actual)
    print("Estimated transitivity:", np.mean(transSim))
    
    plt.figure(3, figsize=(5, 4))
    plt.hist(avg_cluster)
    plt.axvline(x= avg_cluster_actual,color = 'b')
    plt.axvline(x= np.mean(avg_cluster),color = 'r')
    plt.tight_layout()
    plt.show()
    plt.savefig('./storage/results/'+ dataset_name +'/LSH_coef_'+dataset_name+'.pdf') 
    print("Actual Average clustering coefficients:", avg_cluster_actual)
    print("Estimated Average clustering coefficients:", np.mean(avg_cluster))
    
    plt.figure(4, figsize=(5, 4))
    plt.hist(num_events)
    plt.axvline(x= num_events_actual,color = 'b')
    plt.axvline(x= np.mean(num_events),color = 'r')
    plt.tight_layout()
    plt.show()
    plt.savefig('./storage/results/'+ dataset_name +'/LSH_events_'+dataset_name+'.pdf') 
    print("Actual number of events:", num_events_actual)
    print("Estimated number of events:", np.mean(num_events))

    fig, axs = plt.subplots(1, 2,figsize=(5,4))
    axs[0].hist(degree_actual)
    plt.tight_layout()

    axs[1].hist(np.mean(degree,axis = 0))
    plt.tight_layout()
    plt.savefig('./storage/results/'+ dataset_name +'/LSH_degree_distribution_'+dataset_name+'.pdf') 
    print("Actual mean degree:", np.mean(degree_actual))
    print("Estimated mean degree:", np.mean(degree))
    '''

