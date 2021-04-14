'''
This file implements an offline graph datastructure for SemNet to speed up computation
'''
# Imports
import numpy as np 
import torch
import logging

# Functions and submodules
from collections import defaultdict as dd
from tqdm.auto import tqdm, trange

# Set up logging
logger = logging.getLogger('__file__')

class HetGraph():
    def __init__(edgelist=None):
        '''
        Init heterogeneous graph from a weighted edge list

        Format: 
            Initialize two edge dicts for incoming and outgoing edges.
                * Dict keys are node CUIs
                * Dict values are dictionaryies
        '''
        self.outgoing_edges = dd(dd(set))
        self.incoming_edges = dd(dd(set))
        self.type2nodes = dd(set)
        self.type_counts = dd(dd(int))
        self.node2type = {}


    def construct_graph(self, edgelist):
        '''
        Construct graph from list of edges.
        We expect that each element of edge_list is a dict with the following attributes:
            * start_node:   CUI of starting node
            * start_type:   Node type of starting node
            * end_node:     CUI of ending node
            * end_type:     Node type of ending node
            * relation:    Relation between starting and ending node
            * weight:  Weight of edge (i.e. number of papers in which it appears)

        TODO:
            * Batch group nodes by type
        '''
        # Create dicts of outgoing and incoming edges
        logger.info("Constructing edge lists")
        for e in tqdm(edgelist):
            start_node = e['start_node']
            start_type = e['start_type']
            end_node = e['end_node']
            end_type = e['end_type']
            relation = e['relation']
            weight = e['weight']
            
            self.outgoing_edges[start_node][relation][end_type].add(end_node)
            self.incoming_edges[end_node][relation][start_type].add(start_node)

            # Get counts of how often node appears as each type (since it may have multiple categories)
            self.type_counts[start_node][start_type] += weight
            self.type_counts[end_node][end_type] += weight

        # Label each nodetype
        logger.info("Storing node type information")
        for node, count_dict in tqdm(type_counts.items()):
            types = [k for k in count_dict.keys()]
            counts = np.array([c for c in count_dict.values()])
            node_type = types[counts.argmax()]
            self.node2type[node] = node_type
            self.type2nodes[node_type].add(node)


    def add_inverse_edges(self, rel2inv):
        '''
        Add inverse edges for every relation in graph

        Params:
        -------
            rel2inv: dict
                Dict mapping each relation to its inverse
        '''
        # Inverse edges from outgoing
        logger.info("Adding inverse outgoing edges")
        for node, d in tqdm(outgoing_edges.items()):
            for rel, neighbors in d:
                # Note: Neighbors is a dict of format {nodetype:set(nodes)}
                incoming_edges[node][rel2inv[rel]].add(neighbors)

        # Inverse edges from incoming
        logger.info("Adding inverse outgoing edges")
        for node, d in tqdm(incoming_edges.items()):
            for rel, neighbors in d:
                # Note: Neighbors is a dict of format {nodetype:set(nodes)}
                outgoing_edges[node][rel2inv[rel]].add(neighbors)


    def compute_fixed_length_paths(self, start_node, end_node, length=2):
        '''
        Compute all paths of a fixed length

        NOTE: There are easy ways to memoize/parallelize this when using multiple 
                start/end nodes 

        TODO: Possible improvements:
            * Batch path enumeration by metapaths using DFS style search to speed up Hetesim
                This will require us to compute elements to loop over, then run loop in reverse order
            * Compute metapaths at same time as paths
            * Parallelization
            * Optimization for using multiple sets of source/target nodes
        '''
        # Compute fan out and fan in
        fan_out_depth = length // 2
        fan_in_depth = length // 2
        if length % 2 == 1:
            fan_out_depth += 1

        for out_dict, out_path in tqdm(self._fan_out(start_node, depth=fan_out_depth)):
            for in_dict, in_path in tqdm(self._fan_in(end_node, depth=fan_in_depth)):
                joint_types = set(out_dict.keys()).intersection(in_dict.keys())
                for t in joint_types:
                    middle_set = out_dict[t].intersection(in_dict[t])
                    for node in middle_set:
                        yield self._merge_paths(out_path, node, in_path)


    def _fan_out(self, node, curr_path=[], depth=1):
        '''
        Recursively compute all reachable target nodes by path of length 
            $DEPTH from $NODE.

        Only follows outgoing edges

        Inputs:
        -------
            node: str
                CUI string of node

            curr_path: list
                Path traversed so far to reach node

            depth: int >= 0
                Number of additional path segments to compute before returning

        Returns:
        --------
            Iterator of:
                next_nodes: Dict of terminal nodes at end of path
                current_path: Path used to reach each node in next_nodes
        '''
        # If depth is 0, just return current node
        # TODO: Update this to match output form of rest of function
        if depth == 0:
            return node, []

        # If depth is 1, return each set of neighbors and the path used to get there
        elif depth == 1:
            for (next_nodes, next_path) in self._get_edges_to_nbhrs(node):
                current_path = self._merge_paths(curr_path, curr_node, next_path)
                yield (next_nodes, current_path)

        # Otherwise, recursively travel down edges until we reach depth 1
        else:
            for (next_dict, next_path) in self._get_edges_to_nbhrs(node):
                current_path = self._merge_paths(curr_path, curr_node, next_path)
                for node_type in next_dict:
                    for node in next_dict[node_type]:
                        yield from _fan_out(node, curr_path=current_path, depth=depth-1)


    def _fan_in(self, node, curr_path=[], depth=1):
        '''
        Recursively compute all nodes $DEPTH steps away that feed into $NODE

        Similar to `_fan_out()` but looks at incoming edges instead of outgoing edges.
        '''
        raise NotImplementedError


    def _get_edges_to_nbhrs(self, node):
        '''
        Create an iterator of the neighbors of $NODE with paths that lead to each

        Inputs:
        -------
            node: str
                CUI string of node

        Returns:
        --------
            node_group: set
                Set of nodes at end of edge segment

            edge_type: list of str
                Edge leading to next node group
        '''
        # Get edges to next node
        for edge_type, node_group in self.outgoing_edges[node].items():
            # for node_type, node_set in node_group:
            yield node_group, [edge_type]


    def _merge_paths(self, curr_path, curr_node, next_path):
        '''
        Merge path segments together 
        '''
        return curr_path + [curr_node] + next_path


    def _path_to_metapath(self, path):
        '''
        Get metapath from a path segment
        '''
        return [self.x2type[x] for x in path]


    def _metapath_to_string(self, metapath):
        '''
        Turn metapath from list into string
        '''
        return '->'.join(metapath)


    def _score_metapath(self, metapath, path_instances=None):
        '''
        Get HeteSim score of individual metapath
        '''
        raise NotImplementedError

        
