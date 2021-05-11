"""
This module implements randomized algorithms for computation of HeteSim and Pruned HeteSim.
It is designed to work with the datastructure HetGraph given in offline.py
"""

# imports
import math
import random
import numpy as np
import pandas as pd




def restricted_random_walk_on_metapath(graph, start_node, metapath, bad_nodes, walk_forward=True):
    """
    take a random walk in graph along a specified metapath, backtracking if you find a dead end

    Inputs:
    ______
        graph: Hetgraph
            graph to walk on

        start_node: str
            cui of node to start from

        metapath: list of strs
            metapath that constrains the node and edge types in the walk
            format [node_type, edge_type, node_type, ... , edge_type, node_type]

        walk_forward: boolean
            if True, walk forward along (meta)path edges, starting from position 0 in metapath
            if False, walk backward on path edges, starting from end of metapath
            
        bad_nodes: list of sets
            bad_nodes[i] is a set giving all nodes that are dead-ends at step i

    Returns:
    ________
        (bad_nodes, node): (list of lists, str)
        bad_nodes: updated list of dead end nodes    
        node: the node arrived at when the end of the metapath is reached, or the dead end node
        OR returns (0,0) if no path exists
    """

    path_len = int((len(metapath)-1)/2)
    #print(metapath)
    #print(path_len)
    i=1
    node_stack=[]
    cur_node = start_node

    while i>0:
        #print("i:  " + str(i))
        #print("cur_node: " + cur_node)
        #print("node_stack: " + str(node_stack))
        #print("bad_nodes: " + str(bad_nodes))
        if i==path_len+1:
            return (bad_nodes, cur_node)

        if walk_forward:
            neighbors = list( graph.outgoing_edges[cur_node][metapath[2*i -1]][metapath[2*i]] - bad_nodes[i-1] ) # set of neighbors of cur_node under the next relation in the metapath, except those in bad_nodes
        else:
            neighbors = list( graph.incoming_edges[cur_node][metapath[2*path_len + 1 - 2*i]][metapath[2*path_len - 2*i]] - bad_nodes[i-1] )
        #print("neighbors: " + str(neighbors))
        
        #step forward if we can
        if len(neighbors) >0:
            if walk_forward:
                edge_weights = [graph.outgoing_edge_weights[cur_node][metapath[2*i - 1]][y] for y in neighbors]
            else:
                edge_weights = [graph.incoming_edge_weights[cur_node][metapath[2*path_len + 1 - 2*i]][y] for y in neighbors]
    
            #print(edge_weights)
            node_stack.append(cur_node)
            cur_node = random.choices(neighbors, weights=edge_weights)[0] 
            i+=1
        else: #cur_node is dead end
            bad_nodes[i-2].add(cur_node)
            cur_node = node_stack.pop()
            i-=1
    return (0,0)


def randomized_pruned_hetesim(graph, start_nodes, end_nodes, metapaths, k_max, epsilon, r):
    """
    computes a randomized approximation to pruned hetesim, using a just-in-time pruning strategy
    Let PH_e be the estimate returned by this functior.
    Let PH be the true pruned HeteSim value.
    Then, |PH_e - PH| < epsilon with probability at least r

    Inputs:
    _______
        graph: HetGraph
            underlying graph
        
        start_nodes: list of str
            node 1, type must match start of metapath

        end_nodes: list of str    
            node 2, type must match end of metapath

        metapaths: list of list of str
            metapaths on which to compute pruned hetesim
            format [node_type, edge_type, node_type, ... , edge_type, node_type]
            All metapaths must have the same length and length must be even
        
        k_max: int
            maximum number of reachable center layer nodes, over all metapaths and start/end nodes

        epsilon: float
            error tolerance

        r: float
            probability of being within error tolerance
    """
    
    c = (5 + 2*math.sqrt(5))/2
    C = 2*(c + math.sqrt(c**2+4*epsilon))**2 + epsilon*(c+math.sqrt(c**2+4*epsilon))
    N = math.ceil(math.ceil(C/(epsilon**2))*k_max*math.log(4*k_max/(1-r)))
    
    # figure out what the set of first halves of metapaths is
    path_len = int((len(metapaths[0])-1)/2)
    left_halves = []
    for mp in metapaths:
        left_mp = mp[0:path_len + 1]
        if not left_mp in left_halves:
            left_halves.append(left_mp)

    node_prob_left = {}
    # compute vectors from left    
    for lh in left_halves:
        fixed_mp_dict = {}
        for  s in start_nodes:
            fixed_mp_dict[s] =  _compute_approx_pruned_hs_vector_from_left(graph, s, lh, N)
            #print(fixed_mp_dict[s])
        node_prob_left[str(lh)] = fixed_mp_dict

    # figure out what the set of second halves of metapaths is
    right_halves= []
    for mp in metapaths:
        right_mp = mp[path_len:]
        if not right_mp in right_halves:
            right_halves.append(right_mp)

    node_prob_right = {}
    # compute vectors from right
    for rh in right_halves:
        fixed_mp_dict = {}
        for  t in end_nodes:
            fixed_mp_dict[t] =  _compute_approx_pruned_hs_vector_from_right(graph, t, rh, N)
            #print(fixed_mp_dict[t])
        node_prob_right[str(rh)] = fixed_mp_dict

    # create output dict phs[mp][s][t] 
    phs =  {}
    for mp in metapaths:
        left_half = str(mp[0:path_len+1])
        right_half =  str(mp[path_len:])
        fixed_mp_dict = {}
        for s in start_nodes:
            fixed_s_dict = {}
            for t in end_nodes:
                fixed_s_dict[t] = _cos_similarity(node_prob_left[left_half][s], node_prob_right[right_half][t])
            fixed_mp_dict[s] = fixed_s_dict
        phs[str(mp)] = fixed_mp_dict
        
    return phs


def _compute_approx_pruned_hs_vector_from_left(graph, start_node, metapath, N):
    """
    computes an approximation to the probability vector used in computing pruned hetesim,
    using N iterations that end at the end of the metapath (NOT dead ends)
    
        Inputs:
        graph: HetGraph
            underlying graph
            
        start_node: str
            cui of start node
        
        metapath: tbd
            metapath for which to approximate probability vector
            
        N: int
            number of random walks which must make it to the end of the metapath
        
    Outputs:
        approx_hs_vector: dict mapping center-layer nodes to probabilities
           approximate pruned hetesim probability vector for random walks along given metapath from start_node
        
    """
    path_len = int((len(metapath)-1)/2)
    
    # set up dictionary to hold frequencies of encountering each node
    node_freqs = {}    

    # set up list of bad / dead end nodes, for each step of the path
    bad_nodes = [set() for i in range(path_len)]

    num_successes = 0
    while num_successes < N:
        # take a walk, avoiding bad nodes
        (new_bad_nodes, node) = restricted_random_walk_on_metapath(graph, start_node, metapath, bad_nodes, walk_forward=True)
        if new_bad_nodes == 0 and node == 0: # there is no path to the center
            return {} #return an empty dictionary because there are no reachable center layer nodes
        num_successes += 1
        if node in node_freqs:
            node_freqs[node] += 1
        else:
            node_freqs[node] = 1
        bad_nodes=new_bad_nodes

    for node in node_freqs:
        node_freqs[node]/=N

    return node_freqs

def _cos_similarity(vec_1, vec_2):
    #print('vec_1' + str(vec_1))
    #print('vec_2' + str(vec_2))
    # compute length of the two vectors
    vec_1_len = math.sqrt(math.fsum([j**2 for j in vec_1.values()]))
    vec_2_len = math.sqrt(math.fsum([j**2 for j in vec_2.values()]))

    # compute the dot product
    dot_prod = 0
    for k in vec_1.keys():
        if k in vec_2:
            dot_prod += vec_1[k] * vec_2[k]
    
    return dot_prod / (vec_1_len * vec_2_len)
        




def _compute_approx_pruned_hs_vector_from_right(graph, end_node, metapath, N):
    """
    computes an approximation to the probability vector used in computing pruned hetesim,
    using N iterations that end at the end of the metapath (NOT dead ends)
    Walks backward along metapath, starting the end_node
    
        Inputs:
        graph: HetGraph
            underlying graph
            
        end_node: str
            cui of end node
        
        metapath: list of str
            metapath for which to approximate probability vector
            format [node_type, edge_type, node_type, ... , edge_type, node_type]
            
        N: int
            number of random walks which must make it to the end of the metapath
        
    Outputs:
        approx_hs_vector: pandas df
            approximate pruned hetesim probability vector for random walks along reverse of given metapath 
            from end_node
        
    """
    
    path_len = int((len(metapath)-1)/2)
    
    # set up dictionary to hold frequencies of encountering each node
    node_freqs = {}    

    # set up list of bad / dead end nodes, for each step of the path
    bad_nodes = [set() for i in range(path_len)]

    num_successes = 0
    while num_successes < N:
        # take a walk, avoiding bad nodes
        (new_bad_nodes, node) = restricted_random_walk_on_metapath(graph, end_node, metapath, bad_nodes, walk_forward=False)
        if new_bad_nodes == 0 and node == 0: # there is no path to the center
            return {} #return an empty dictionary because there are no reachable center layer nodes
        num_successes += 1
        if node in node_freqs:
            node_freqs[node] += 1
        else:
            node_freqs[node] = 1
        bad_nodes = new_bad_nodes

    for node in node_freqs:
        node_freqs[node]/=N
    return node_freqs

