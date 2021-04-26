"""
This module implements randomized algorithms for computation of HeteSim and Pruned HeteSim.
It is designed to work with the datastructure given in offline.py
"""

# imports
import numpy as np

from semnet.offline import HetGraph

def random_walk_on_metapath(graph, start_node, metapath, walk_forward=True):
    """
    take a random walk in graph along a specified metapath

    Inputs:
    ______
        graph: Hetgraph
            graph to walk on

        start_node: str
            cui of node to start from

        metapath: tbd
            metapath that constrains the node and edge types in the walk

        walk_forward: boolean
            if True, walk forward along (meta)path edges, starting from position 0 in metapath
            if False, walk backward on path edges, starting from end of metapath

    Returns:
    ________
        (dead_end, node): (bool, str)
            dead_end: True if the path hit a dead end; false if it made it to the end of the metapath
            node: the node arrived at when the end of the metapath is reached, or the dead end node
    """

def randomized_hetesim(graph, start_rode, end_node, metapath, epsilon, r, g):
    """
    randomized implementation of HeteSim. 
    Returns an estimate for HeteSim H_e so that |H-H_e|<epsilon with probability at least r,
    where H is the true HeteSim score.
    g is a parameter used in estimating the dead-end probability to determine how many iterations are needed.
    If this initial estimate of dead end probability is too far off, the algorithm will fail and will return -1.
    Increasing g will decrease the chance of this failure, but will increase runtime.
    """

    Inputs:
    _______
        graph: HetGraph
            graph on which to compute HeteSim

        start_node: str
            cui of node 1, should have type matching the first node of the metapath

        end_node: str
            cui of node 2, should have type matching last node of metapath

        metapath: tbd
            metapath on which to compute hetesim

        epsilon: float
            error tolerance

        r: float
            probability that the estimate is vithin the error tolerance (must have 0<r<1)

        g: integer
            number of iterations to use in initial estimation of dead end probability (must have g>0)

    Returns:
    ________
        H_s: float
            estimated value of HeteSim
    """

def _compute_approx_hs_vector_from_left(graph, start_node, metapath, epsilon, r, g):
    """
    computes an approximation to the probability vector usec in computing hetesim
    guarantees that Pr(|true ith entry - estimated ith entry| < delta*g / (2*n_L*N) for all i) is at least 1- 4*n_L^2 / (N*delta^2*g^2)
    for delta <= epsilon / ((2+sqrt(2))*k) 
    and we guarantee N > 2*n_L^2 / ((1-r)*delta^2*g^2)
    When combined with a vector of probabilities from the right, we can conclude that the error in the
    estimated hetesim value is less than epsilon with probability at least r
    Returns -1 if the selected N wasn't big enough.
    """

def _compute_approx_hs_vector_from_right(graph, end_node, metapath, epsilon, r, g):
    """
    computes an approximation to the probability vector used in computing hetesim
    guarantees that Pr(|true ith entry - estimated ith entry| < delta*g / (2*n_R*N) for all i) is at least 1- 4*n_R^2 / (N*delta^2*g^2)
    for delta <= epsilon / ((2+sqrt(2))*k) 
    and we guarantee N > 2*n_R^2 / ((1-r)*delta^2*g^2)
    When combined with a vector of probabilities from the right, we can conclude that the error in the
    estimated hetesim value is less than epsilon with probability at least r
    Returns -1 if the selected N wasn't big enough.
    """

def randomized_pruned_hetesim(graph, start_node, end_node, metapath, epsilon, r):
    """
    computes a randomized approximation to pruned hetesim, using a just-in-time pruning strategy
    Let PH_e be the estimate returned by this functior.
    Let PH be the true pruned HeteSim value.
    Then, |PH_e - PH| < epsilon with probability at least r

    Inputs:
    _______
        graph: HetGraph
            underlieing graph
        
        start_node: str
            node 1, type must match start of metapath

        end_node: str    
            node 2, type must match end of metapath

        metapath: tbd
            metapath on which to compute pruned hetesim

        epsilon: float
            error tolerance

        r: float
            probability of being within error tolerance
    """
