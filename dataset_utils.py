# -*- coding: utf-8 -*-
"""
@author: Makan Arastuie
This code is refered from: https://github.com/IdeasLabUT/CHIP-Network-Model

"""

import os
import sys

import numpy as np
from os.path import join
import pickle

def get_script_path():
    """
    :return: the path of the current script
    """
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def load_reality_mining_test_train(remove_nodes_not_in_train=False):
    """
    Loads Reality Mining dataset.

    :param remove_nodes_not_in_train: if True, removes the nodes that do not appear in the training set.

    :return: Three tuples one for each train, test and combined datasets. Each Tuple contains:
             ((dict) with (caller_id, receiver_id): [unix_timestamps] (event dict structure),
             (int) number of nodes,
             (float) duration)
             (list) nodes_not_in_train
    """
    train_file_path = join(get_script_path(), 'storage', 'datasets', 'reality-mining', 'train_reality.csv')
    test_file_path = join(get_script_path(), 'storage', 'datasets', 'reality-mining', 'test_reality.csv')

    # Timestamps are adjusted to start from 0 and go up to 1000.
    combined_duration = 1000.0

    return load_train_test(train_file_path, test_file_path, combined_duration, remove_nodes_not_in_train)

def load_fb_train_test(remove_nodes_not_in_train=False):
    """
    Loads FB dataset.

    :param remove_nodes_not_in_train: if True, removes the nodes that do not appear in the training set.

    :return: Three tuples one for each train, test and combined datasets. Each Tuple contains:
             ((dict) with (caller_id, receiver_id): [unix_timestamps] (event dict structure),
             (int) number of nodes,
             (float) duration)
             (list) nodes_not_in_train
    """
    train_file_path = join(get_script_path(), 'storage', 'datasets', 'facebook-wallposts', 'train_FB_event_mat.csv')
    test_file_path = join(get_script_path(), 'storage', 'datasets', 'facebook-wallposts', 'test_FB_event_mat.csv')

    # Timestamps are adjusted to start from 0 and go up to 8759.9.
    combined_duration = 8759.9

    return load_train_test(train_file_path, test_file_path, combined_duration, remove_nodes_not_in_train)

def load_email_eu_core_train_test(remove_nodes_not_in_train=False):
    """
    Loads FB dataset.

    :param remove_nodes_not_in_train: if True, removes the nodes that do not appear in the training set.

    :return: Three tuples one for each train, test and combined datasets. Each Tuple contains:
             ((dict) with (caller_id, receiver_id): [unix_timestamps] (event dict structure),
             (int) number of nodes,
             (float) duration)
             (list) nodes_not_in_train
    """
    train_file_path = join(get_script_path(), 'storage', 'datasets', 'email_eu_core', 'email_eu_core_train.csv')
    test_file_path = join(get_script_path(), 'storage', 'datasets', 'email_eu_core', 'email_eu_core_test.csv')

    # Timestamps are adjusted to start from 0 and go up to 8759.9.
    combined_duration = 1000.0

    return load_train_test(train_file_path, test_file_path, combined_duration, remove_nodes_not_in_train)


def load_fb_forum_train_test(remove_nodes_not_in_train=False):
    """
    Loads FB dataset.

    :param remove_nodes_not_in_train: if True, removes the nodes that do not appear in the training set.

    :return: Three tuples one for each train, test and combined datasets. Each Tuple contains:
             ((dict) with (caller_id, receiver_id): [unix_timestamps] (event dict structure),
             (int) number of nodes,
             (float) duration)
             (list) nodes_not_in_train
    """
    train_file_path = join(get_script_path(), 'storage', 'datasets', 'fb_forum', 'fb_forum_train.csv')
    test_file_path = join(get_script_path(), 'storage', 'datasets', 'fb_forum', 'fb_forum_test.csv')

    # Timestamps are adjusted to start from 0 and go up to 8759.9.
    combined_duration = 1000.0

    return load_train_test(train_file_path, test_file_path, combined_duration, remove_nodes_not_in_train)

def load_mid_train_test(remove_nodes_not_in_train=False):
    """
    Loads FB dataset.

    :param remove_nodes_not_in_train: if True, removes the nodes that do not appear in the training set.

    :return: Three tuples one for each train, test and combined datasets. Each Tuple contains:
             ((dict) with (caller_id, receiver_id): [unix_timestamps] (event dict structure),
             (int) number of nodes,
             (float) duration)
             (list) nodes_not_in_train
    """
    train_file_path = join(get_script_path(), 'storage', 'datasets', 'MID', 'MID_train.csv')
    test_file_path = join(get_script_path(), 'storage', 'datasets', 'MID', 'MID_test.csv')

    # Timestamps are adjusted to start from 0 and go up to 8759.9.
    combined_duration = 1000.0

    return load_train_test(train_file_path, test_file_path, combined_duration, remove_nodes_not_in_train)

def load_enron_yang_train_test(remove_nodes_not_in_train=False):
    """
    Loads FB dataset.

    :param remove_nodes_not_in_train: if True, removes the nodes that do not appear in the training set.

    :return: Three tuples one for each train, test and combined datasets. Each Tuple contains:
             ((dict) with (caller_id, receiver_id): [unix_timestamps] (event dict structure),
             (int) number of nodes,
             (float) duration)
             (list) nodes_not_in_train
    """
    train_file_path = join(get_script_path(), 'storage', 'datasets', 'enron_yang', 'enron_yang_train.csv')
    test_file_path = join(get_script_path(), 'storage', 'datasets', 'enron_yang', 'enron_yang_test.csv')

    # Timestamps are adjusted to start from 0 and go up to 8759.9.
    combined_duration = 452.325

    return load_train_test(train_file_path, test_file_path, combined_duration, remove_nodes_not_in_train)


def load_MID_data_train_all(dnx_pickle_file_name, split_ratio=0.8, scale=7 * 24 * 60 * 60 ,remove_small_comp=False, remove_node_not_in_train=True):
    incident_dnx_list = pickle.load(open(dnx_pickle_file_name, 'rb'))
    digraph1 = incident_dnx_list[0]

    small_comp_countries_train = ['GUA', 'BLZ', 'GAM', 'SEN', 'SAF', 'LES', 'SWA', 'MZM', 'GNB']
    small_comp_countres_full = ['BLZ', 'GUA', 'MZM', 'SWA', 'SAF', 'LES']
    nodes_not_in_train = ['PAN','SSD']

    if remove_node_not_in_train:
        nodes_before = set(digraph1.nodes())
        for country in nodes_not_in_train:
            for node in nodes_before:
                digraph1.remove_edge(country, node)
                digraph1.remove_edge(node, country)
        

    if remove_small_comp:
        # print("n_events before removing small components: ", len(digraph1.edges()))
        nodes_before = set(digraph1.nodes())
        for country in small_comp_countries_train:
            for node in nodes_before:
                digraph1.remove_edge(country, node)
                digraph1.remove_edge(node, country)

    # find train splitting point
    n_events_all = len(digraph1.edges())
    split_point = int(n_events_all * split_ratio)
    timestamp_last_train = digraph1.edges()[split_point - 1][2]  # time of last event included in train dataset
    timestamp_last_all = digraph1.edges()[-1][2]  # time of last event in all dataset
    timestamp_first = digraph1.edges()[0][2]
    n_events_train = len(digraph1.edges(end=timestamp_last_train))
    duration = timestamp_last_all - timestamp_first
    print("duration = ", int(duration/(60*60*24)), " days")

    # get train and all nodes id map
    node_set_all = set(digraph1.nodes(end=timestamp_last_all))
    n_nodes_all = len(node_set_all)
    node_id_map_all, id_node_map_all = get_node_id_maps(node_set_all)
    node_set_train = set(digraph1.nodes(end=timestamp_last_train))
    n_nodes_train = len(node_set_train)
    node_id_map_train, id_node_map_train = get_node_id_maps(node_set_train)

    # create event dictionary of train and all dataset
    event_dict_all = {}
    event_dict_train = {}
    for edge in digraph1.edges():
        sender_id, receiver_id = node_id_map_all[edge[0]], node_id_map_all[edge[1]]
        if scale == 1000: # scale timestamp in range [0 : 1000]
            timestamp = (edge[2] - timestamp_first) / duration * scale
        else:
            timestamp = (edge[2] - timestamp_first) / scale
        if timestamp < 0:
            print(edge)
        if (sender_id, receiver_id) not in event_dict_all:
            event_dict_all[(sender_id, receiver_id)] = []
        event_dict_all[(sender_id, receiver_id)].append(timestamp)
        if edge[2] <= timestamp_last_train:
            sender_id_t, receiver_id_t = node_id_map_train[edge[0]], node_id_map_train[edge[1]]
            if (sender_id_t, receiver_id_t) not in event_dict_train:
                event_dict_train[(sender_id_t, receiver_id_t)] = []
            event_dict_train[(sender_id_t, receiver_id_t)].append(timestamp)
    # train and all end time
    if scale == 1000:
        T_all = (timestamp_last_all - timestamp_first) / duration * scale
        T_train = (timestamp_last_train - timestamp_first) / duration * scale
    else:
        T_all = (timestamp_last_all - timestamp_first) / scale
        T_train = (timestamp_last_train - timestamp_first) / scale
    # node not in train list
    nodes_not_in_train = []
    for n in (node_set_all - node_set_train):
        nodes_not_in_train.append(node_id_map_all[n])
        
        

    tuple_train = event_dict_train, T_train, n_nodes_train, n_events_train, id_node_map_train
    tuple_all = event_dict_all, T_all, n_nodes_all, n_events_all, id_node_map_all
    return tuple_train, tuple_all, nodes_not_in_train

def get_node_id_maps(node_set):
    nodes = list(node_set)
    nodes.sort()

    node_id_map = {}
    id_node_map = {}
    for i, n in enumerate(nodes):
        node_id_map[n] = i
        id_node_map[i] = n

    return node_id_map, id_node_map


def load_train_test(train_file_path, test_file_path, combined_duration, remove_nodes_not_in_train):
    """
    Loads datasets already split into train and test, such as Enron and FB.

    :param train_file_path: path to the train dataset.
    :param test_file_path: path to the test dataset.
    :param combined_duration: Entire duration of the network, train + test.
    :param remove_nodes_not_in_train: if True, removes the nodes that do not appear in the training set.

    :return: Three tuples one for each train, test and combined datasets. Each Tuple contains:
             ((dict) with (caller_id, receiver_id): [unix_timestamps] (event dict structure),
             (int) number of nodes,
             (float) duration)
             (list) nodes_not_in_train
    """

    combined_node_id_map, train_node_id_map, test_node_id_map, nodes_not_in_train = \
        load_and_combine_nodes_for_test_train(train_file_path, test_file_path, remove_nodes_not_in_train)

    train_event_dict, train_duration = load_test_train_data(train_file_path, train_node_id_map)
    test_event_dict, test_duration = load_test_train_data(test_file_path, test_node_id_map)
    combined_event_dict = load_test_train_combined(train_file_path, test_file_path, combined_node_id_map)

    return ((train_event_dict, len(train_node_id_map), train_duration),
            (test_event_dict, len(test_node_id_map), test_duration),
            (combined_event_dict, len(combined_node_id_map), combined_duration),
            nodes_not_in_train)


def load_and_combine_nodes_for_test_train(train_path, test_path, remove_nodes_not_in_train):
    """
    Loads the set of nodes in both train and test datasets and maps all the node ids to start form 0 to num total nodes

    :param train_file_path: path to the train dataset.
    :param test_file_path: path to the test dataset.
    :param remove_nodes_not_in_train: if True, all the nodes in test and combined that are not in train, will be removed
    :return `full_node_id_map` dict mapping node id in the entire dataset to a range from 0 to n_full
            `train_node_id_map` dict mapping node id in the train dataset to a range from 0 to n_train
            `test_node_id_map` dict mapping node id in the test dataset to a range from 0 to n_test
            `nodes_not_in_train` list of mapped node ids that are in test, but not in train.
    """

    # load dataset. caller_id,receiver_id,unix_timestamp

    # Train data
    train_nodes = np.genfromtxt(train_path, np.int, delimiter=',', usecols=(0, 1),encoding='utf-8',skip_header=1)
    train_nodes_set = set(train_nodes.reshape(train_nodes.shape[0] * 2))
    #print(train_nodes_set)
    train_node_id_map = get_node_map(train_nodes_set)

    # Test data
    test_nodes = np.genfromtxt(test_path, np.int, delimiter=',', usecols=(0, 1),encoding='utf-8',skip_header=1)
    test_nodes_set = set(test_nodes.reshape(test_nodes.shape[0] * 2))
    #print(test_nodes_set)
    if remove_nodes_not_in_train:
        test_nodes_set = test_nodes_set - test_nodes_set.difference(train_nodes_set)
    test_node_id_map = get_node_map(test_nodes_set)

    # Combined
    if remove_nodes_not_in_train:
        full_node_id_map = train_node_id_map
    else:
        all_nodes = list(train_nodes_set.union(test_nodes_set))
        full_node_id_map = get_node_map(all_nodes)
        all_nodes.sort()

    nodes_not_in_train = []
    for n in test_nodes_set.difference(train_nodes_set):
        nodes_not_in_train.append(full_node_id_map[n])

    return full_node_id_map, train_node_id_map, test_node_id_map, nodes_not_in_train


def get_node_map(node_set):
    """
    Maps every node to an ID.

    :param node_set: set of all nodes to be mapped.
    :return: dict of original node index as key and the mapped ID as value.
    """
    nodes = list(node_set)
    nodes.sort()

    node_id_map = {}
    for i, n in enumerate(nodes):
        node_id_map[n] = i

    return node_id_map


def load_test_train_data(file, node_id_map, prev_event_dict=None, if_fb_forum = False):
    """
    Loads a train or test dataset based on the node_id_map.

    :param file: path to the dataset or a loaded dataset.
    :param node_id_map: (dict) dict of every node to its id.
    :param prev_event_dict: (dict) Optional. An event dict to add the dataset to

    :return: event_dict, duration
    """
    # File can be both the file path or an ordered event_list
    if isinstance(file, str):
        # load the core dataset. sender_id,receiver_id,unix_timestamp
        data = np.genfromtxt(file, np.float, delimiter=',', usecols=(0, 1, 2),encoding='utf-8',skip_header=1)
        # Sorting by unix_timestamp
        data = data[data[:, 2].argsort()]
    else:
        data = file

    duration = data[-1, 2] - data[0, 2]

    event_dict = {} if prev_event_dict is None else prev_event_dict

    for i in range(data.shape[0]):
        # This step is needed to skip events involving nodes that were not in train, in case they were removed.
        if np.int(data[i, 0]) not in node_id_map or np.int(data[i, 1]) not in node_id_map:
            continue

        sender_id = node_id_map[np.int(data[i, 0])]
        receiver_id = node_id_map[np.int(data[i, 1])]

        if (sender_id, receiver_id) not in event_dict:
            event_dict[(sender_id, receiver_id)] = []

        event_dict[(sender_id, receiver_id)].append(data[i, 2])

    return event_dict, duration





def load_test_train_combined(train, test, node_id_map):
    """
    Combines train and test dataset to get the full dataset.

    :param train: path to the train dataset or the loaded dataset itself.
    :param test: path to the test dataset or the loaded dataset itself.
    :param node_id_map: (dict) dict of every node to its id.

    :return: combined_event_dict
    """
    combined_event_dict, _ = load_test_train_data(train, node_id_map)
    combined_event_dict, _ = load_test_train_data(test, node_id_map, combined_event_dict)

    return combined_event_dict


def split_event_list_to_train_test(event_list, train_percentage=0.8, remove_nodes_not_in_train=False):
    """
    Given an event_list (list of [sender_id, receiver_id, timestamp]) it splits it into train and test,
    ready for model fitting.

    :param event_list: a list of all events [sender_id, receiver_id, timestamp].
    :param train_percentage: (float) top `train_percentage` of the event list will be returned as the training data
    :param remove_nodes_not_in_train: if True, all the nodes in test and combined that are not in train, will be removed

    :return: Three tuples one for each train, test and combined datasets. Each Tuple contains:
         ((dict) with (caller_id, receiver_id): [unix_timestamps] (event dict structure),
         (int) number of nodes,
         (float) duration)
         (list) nodes_not_in_train
    """
    # sort by timestamp
    event_list = event_list[event_list[:, 2].argsort()]
    # make the dataset to start from time 0
    event_list[:, 2] = event_list[:, 2] - event_list[0, 2]

    combined_duration = event_list[-1, 2] - event_list[0, 2]

    split_point = np.int(event_list.shape[0] * train_percentage)

    # Train data
    train_event_list = event_list[:split_point, :]
    train_nodes_set = set(train_event_list[:, 0]).union(train_event_list[:, 1])
    train_node_id_map = get_node_map(train_nodes_set)

    # Test data
    test_event_list = event_list[split_point:, :]
    test_nodes_set = set(test_event_list[:, 0]).union(test_event_list[:, 1])
    if remove_nodes_not_in_train:
        test_nodes_set = test_nodes_set - test_nodes_set.difference(train_nodes_set)
    test_node_id_map = get_node_map(test_nodes_set)

    # Combined
    if remove_nodes_not_in_train:
        combined_node_id_map = train_node_id_map
    else:
        all_nodes = list(train_nodes_set.union(test_nodes_set))
        combined_node_id_map = get_node_map(all_nodes)
        all_nodes.sort()

    nodes_not_in_train = []
    for n in test_nodes_set.difference(train_nodes_set):
        nodes_not_in_train.append(combined_node_id_map[n])

    train_event_dict, train_duration = load_test_train_data(train_event_list, train_node_id_map)
    test_event_dict, test_duration = load_test_train_data(test_event_list, test_node_id_map)
    combined_event_dict = load_test_train_combined(train_event_list, test_event_list, combined_node_id_map)

    return ((train_event_dict, len(train_node_id_map), train_duration),
            (test_event_dict, len(test_node_id_map), test_duration),
            (combined_event_dict, len(combined_node_id_map), combined_duration),
            nodes_not_in_train)

