#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:02:27 2020

This is Latent space Hawkes process model util functions

"""


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2020)
    
def plotlsp(z1, N, nodes, count_full, i, lsptitle='', dataset = '', top_ten = False):
    """    
    Parameters
    ----------
    z1 : (N, dim) np.array 
        the latent positives.
    N : int
        numer of nodes in the network.
    nodes : string
        label/text for each node.
    count_full : (N, N) np.array
        Each entry stores the number of events for each pair of nodes.
    i : int
        figiure index.
    lsptitle : string, optional
        the title of the plot.
    dataset : string, optional
        which dataset to plot. The default is ''.

    Returns
    -------
    None.

    """
    send = np.sum(count_full, axis = 1)
    rev = np.sum(count_full, axis = 0)
    top_5_send = np.argsort(send)[-5:]
    top_5_rev = np.argsort(rev)[-5:]
    x_pos = z1[:,0]
    y_pos = z1[:,1]
    plt.figure(i, figsize=(10, 8))
    off_set_x = 0
    off_set_y = 0
    size = 9
    if dataset == 'MID':
        off_set_x = -0.09
        off_set_y = 0.07
        size = 12
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
    #plt.scatter(x_pos[0:size],y_pos[0:size], marker = 'x', color = 'red', label = 'node')
    for i in range(N):
        if (i in top_5_send and i in top_5_rev) and dataset == 'MID':   
            #plt.scatter(x_pos[0:size],y_pos[0:size], marker = 'x', color = 'red', label = 'node')
            plt.text(x_pos[i]+off_set_x, y_pos[i]+off_set_y, nodes[i], fontsize = size, color = 'red')
            plt.scatter(x_pos[i],y_pos[i], marker = '*', color = 'red', label = 'node', s = 50)
        elif (i in top_5_send) and dataset == 'MID':
            plt.text(x_pos[i]+off_set_x, y_pos[i]+off_set_y, nodes[i], fontsize = size, color = 'b')
            plt.scatter(x_pos[i],y_pos[i], marker = 'o', color = 'blue', label = 'node', s = 50)
        elif i in top_5_rev and dataset == 'MID':
            plt.text(x_pos[i]+off_set_x, y_pos[i]+off_set_y, nodes[i], fontsize = size, color = 'green')
            plt.scatter(x_pos[i],y_pos[i], marker = 's', color = 'green', label = 'node', s = 50)
        else:
            plt.scatter(x_pos[i],y_pos[i], marker = 'x', color = 'black', alpha=.5)
            plt.text(x_pos[i]+off_set_x, y_pos[i]+off_set_y, nodes[i], fontsize = size, alpha=.5)
    #plt.title(lsptitle)
    if top_ten:
        count_list = count_full.flatten().tolist()
        for i in range(10):
            don_index = np.where(count_full == max(count_list))[0]
            rec_index = np.where(count_full == max(count_list))[1]
            count_list.remove(max(count_list))
            x_value = [x_pos[don_index], x_pos[rec_index]]
            y_value = [y_pos[don_index], y_pos[rec_index]]
            #plt.plot(x_value, y_value, 'c--', alpha = 0.5)
            ax = plt.axes()
            dx = x_pos[rec_index][0]-x_pos[don_index][0]
            dy= y_pos[rec_index][0]-y_pos[don_index][0]
            ax.arrow(x_pos[don_index][0], y_pos[don_index][0], dx, dy, head_width=0.08, head_length=0.08, fc='lightskyblue', ec='lightskyblue')
    plt.axis('equal')
    #plt.tight_layout(0.1)
    #pl.legend(loc='upper right')
    plt.show()


def plotlspmid(z1, N, nodes, count_full, i, lsptitle, continent, dataset = ''):
    """    
    For plot MID on colored by different continents.
    Parameters
    ----------
    z1 : (N, dim) np.array 
        the latent positives.
    N : int
        numer of nodes in the network.
    nodes : string
        label/text for each node.
    count_full : (N, N) np.array
        Each entry stores the number of events for each pair of nodes.
    i : int
        figiure index.
    lsptitle : string, optional
        the title of the plot.
    dataset : string, optional
        which dataset to plot. The default is ''.

    Returns
    -------
    None.

    """
    x_pos = z1[:,0]
    y_pos = z1[:,1]
    plt.figure(i, figsize=(10, 8))
    Asia = True
    Europe = True
    Americas = True
    Oceania = True
    Africa = True
    for i in range(N):
        if continent[i,1].decode('UTF-8') == 'Asia':   
            #plt.scatter(x_pos[0:size],y_pos[0:size], marker = 'x', color = 'red', label = 'node')
            plt.text(x_pos[i]-0.09, y_pos[i]+0.07, nodes[i], fontsize = 12, color = 'red')
            if Asia:
                plt.scatter(x_pos[i],y_pos[i], marker = '*', color = 'red', label = 'Asia', s = 50)
                Asia = False
            else: plt.scatter(x_pos[i],y_pos[i], marker = '*', color = 'red', s = 50)
        elif continent[i,1].decode('UTF-8') == 'Europe':
            plt.text(x_pos[i]-0.09, y_pos[i]+0.07, nodes[i], fontsize = 12, color = 'blue')
            if Europe:
                plt.scatter(x_pos[i],y_pos[i], marker = 'o', color = 'blue', label = 'Europe', s = 50)
                Europe = False
            else: plt.scatter(x_pos[i],y_pos[i], marker = 'o', color = 'blue', s = 50)
        elif continent[i,1].decode('UTF-8') == 'Americas':
            plt.text(x_pos[i]-0.09, y_pos[i]+0.07, nodes[i], fontsize = 12, color = 'black')
            if Americas:
                plt.scatter(x_pos[i],y_pos[i], marker = '2', color = 'black', label = 'Americas', s = 50)
                Americas = False
            else: plt.scatter(x_pos[i],y_pos[i], marker = '2', color = 'black', s = 50)
        elif continent[i,1].decode('UTF-8') == 'Oceania':
            plt.text(x_pos[i]-0.09, y_pos[i]+0.07, nodes[i], fontsize = 12, color = 'green')
            if Oceania:
                plt.scatter(x_pos[i],y_pos[i], marker = 's', color = 'green', label = 'Oceania', s = 50)
                Oceania = False
            else: plt.scatter(x_pos[i],y_pos[i], marker = 's', color = 'green', s = 50)
        elif continent[i,1].decode('UTF-8') == 'Africa':
            plt.text(x_pos[i]-0.09, y_pos[i]+0.07, nodes[i], fontsize = 12, color = 'purple')
            if Africa:
                plt.scatter(x_pos[i],y_pos[i], marker = '+', color = 'purple', label = 'Africa', s = 50)
                Africa = False
            else: plt.scatter(x_pos[i],y_pos[i], marker = '+', color = 'purple', s = 50)
        else:
            plt.scatter(x_pos[i],y_pos[i], marker = 'x', color = 'black')
            plt.text(x_pos[i]-0.09, y_pos[i]+0.07, nodes[i], fontsize = 12)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.axis('equal')
    plt.tight_layout(0.1)
    plt.legend(fontsize=15, loc='lower right')
    plt.show()
    
    
def count_process(P):
    """
    

    Parameters
    ----------
    P : (N,N) matrix
        each entry is a list contains the events in the pair of node.

    Returns
    -------
    count : (N,N) array of object
        the count for every pairs of node.

    """
    # count number of process
    count = np.zeros([P.shape[0],P.shape[1]])
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            count[i,j] = len(P[i,j])
    return count

def split_train_test(P, T, rate):
    """
    Hold last 20% of data as testing set
    
    Parameters
    ----------
    P : Array of object
        The HP of all pair of nodes, including the timestamp of
        events for all pair of nodes.

    Returns
    -------
    train : Array of object
        The first 80% of HP of all pair of nodes as the training set
    test : Array of object
        The last 20% of HP of all pair of nodes as the testing set.

    """
    
    end_time_train = T*rate
    end_time_test = T - end_time_train
    
    train = np.empty([P.shape[0],P.shape[1]], dtype=np.object)
    test = np.empty([P.shape[0],P.shape[1]], dtype=np.object)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if len(P[i,j]) == 0:
                train[i,j] = []
                test[i,j] = []
            else:
                P[i,j] = np.array(P[i,j])
                train[i,j] = P[i,j][P[i,j] <= end_time_train]
                test[i,j] = P[i,j][P[i,j] > end_time_train]
    return train, test, end_time_train, end_time_test

def event_dict_to_adjacency_list(num_nodes, event_dicts, dtype=np.float):
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
                adjacency_matrix[u,v] = np.array([])
                
    for (u, v), event_times in event_dicts.items():
        adjacency_matrix[u, v] = np.array(event_times)
    
    return adjacency_matrix

def adjaceny_to_events_dict(P, N):
    """
    

    Parameters
    ----------
    P : Array of object
        The HP of all pair of nodes, including the timestamp of
        events for all pair of nodes.
    N : int
        numer of dimensions.

    Returns
    -------
    events_dict : TYPE
        DESCRIPTION.

    """
    events_dict = {}
    for u in range(N):
        for v in range(N):
            if len(P[u,v]) != 0:
                events_dict[(u,v)] = P[u,v]
    return events_dict