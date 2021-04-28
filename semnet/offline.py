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
    def __init__(self, edgelist=None, rel2inv=None):
        '''
        Init heterogeneous graph from a weighted edge list

        Format: 
            Initialize two edge dicts for incoming and outgoing edges.
                * Dict keys are node CUIs
                * Dict values are dictionaryies
        '''
        # Edges stored as nested default dicts
        self.outgoing_edges = dd( # Source node
                                lambda: dd(  # Edge type
                                    lambda: dd( # Target type
                                        set # Target nodes
                                            )))
        self.incoming_edges = dd(
                                lambda: dd(
                                    lambda: dd(set)))

        self.outgoing_edge_weights = dd( #source nodes
                                    lambda: dd( #Edge type
                                            lambda: dd( # Target node
                                                float #edge weight
                                                    )))

        self.incoming_edge_weights = dd( # target node
                                    lambda: dd( #Edge type
                                            lambda: dd( # source node
                                                float #edge weight
                                                    )))
        self.type2nodes = dd(set)
        self.type_counts = dd(lambda: dd(int))
        self.node2type = {}
        self.relations = set([])

        # Construct graph if edelist if provided
        if edgelist is not None:
            self.construct_graph(edgelist, rel2inv)

        


    def construct_graph(self, edgelist, rel2inv=None, add_inverses=True):
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
        if add_inverses:
            if rel2inv is None:
                print("Cannot add inverse edges without mapping of edge to inverse!")
                print("Please add inverses manually using HetGraph.add_inverse_edges()")

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
            self.outgoing_edge_weights[start_node][relation][end_node] = weight
            self.incoming_edges[end_node][relation][start_type].add(start_node)
            self.incoming_edge_weights[end_node][relation][start_node] = weight
            
            self.relations.add(relation)

            if rel2inv is not None and add_inverses==True:
                inverse_relation = rel2inv[relation]
                self.incoming_edges[start_node][inverse_relation][end_type].add(end_node)
                self.incoming_edge_weights[start_node][inverse_relation][end_node] = weight
                self.outgoing_edges[end_node][inverse_relation][start_type].add(start_node)
                self.outgoing_edge_weights[end_node][inverse_relation][start_node] = weight
                self.relations.add(inverse_relation)

            # Get counts of how often node appears as each type (since it may have multiple categories)
            self.type_counts[start_node][start_type] += weight
            self.type_counts[end_node][end_type] += weight

        # Label each nodetype
        logger.info("Storing node type information")
        for node, count_dict in tqdm(self.type_counts.items()):
            types = [k for k in count_dict.keys()]
            counts = np.array([c for c in count_dict.values()])
            node_type = types[counts.argmax()]
            self.node2type[node] = node_type
            self.type2nodes[node_type].add(node)

        rel2self = {rel:rel for rel in self.relations}
        self.x2type = {**self.node2type, **rel2self}

        


    def add_inverse_edges(self, rel2inv):
        '''
        Add inverse edges for every relation in graph

        Inputs:
        -------
            rel2inv: dict
                Dict mapping each relation to its inverse
        '''
        # Make sure we didn't leave anything out of rel2inv
        inv_rel2inv = {val:key for key, val in rel2inv.items()}
        rel2inv = {**inv_rel2inv, **rel2inv}

        # Inverse edges from outgoing
        logger.info("Adding inverse outgoing edges")
        for node, d in tqdm(self.outgoing_edges.items()):
            for rel, neighbors in d.items():
                # Note: Neighbors is a dict of format {node_type:set(nodes)}
                if rel2inv[rel] in self.incoming_edges[node]:
                    for node_type, node_set in neighbors.items():
                        # print()
                        # print("Node type:", node_type)
                        # print("Node set:", node_set)
                        # print("Node:", node)
                        # print("rel2inv[rel]:", rel2inv[rel])
                        # print()
                        self.incoming_edges[node][rel2inv[rel]][node_type] |= node_set
                else:
                    self.incoming_edges[node][rel2inv[rel]] = neighbors

        # Inverse edges from incoming
        logger.info("Adding inverse outgoing edges")
        for node, d in tqdm(self.incoming_edges.items()):
            for rel, neighbors in d.items():
                
                # Note: Neighbors is a dict of format {node_type:set(nodes)}
                if rel2inv[rel] in self.outgoing_edges[node]:
                    for node_type, node_set in neighbors.items():
                        self.outgoing_edges[node][rel2inv[rel]][node_type] |= node_set
                else:
                    self.outgoing_edges[node][rel2inv[rel]] = neighbors

        # Add inverse relations to relation list
        for key, val in rel2inv.items():
            if key in self.relations:
                self.relations.add(val)
                self.x2type[key] = key


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
                joint_types = set(out_dict.keys()).intersection(set(in_dict.keys()))
                for t in joint_types:
                    # Excude nodes that we have already visited
                    middle_set = out_dict[t].intersection(in_dict[t]) - set(out_path + in_path)
                    for node in middle_set:
                        yield self._merge_paths(out_path, node, in_path)


    def _fan_out(self, node, depth=1, curr_path=[]):
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
        if depth == 0:
            # yield ({node}, [])
            yield ({self.node2type[node]:{node}}, [])

        # If depth is 1, return each set of neighbors and the path used to get there
        elif depth == 1:
            for (next_path, next_dict) in self.outgoing_edges[node].items():
                current_path = self._merge_paths(curr_path, node, next_path)
                # for node_type, node_set in next_dict.items():
                    # yield (node_set - set(current_path), current_path)
                yield (next_dict, current_path)

        # Otherwise, recursively travel down edges until we reach depth 1
        else:
            for (next_path, next_dict) in self.outgoing_edges[node].items():
                current_path = self._merge_paths(curr_path, node, next_path)
                for node_type, node_set in next_dict.items():
                    for next_node in node_set:
                        if next_node not in current_path:
                            yield from self._fan_out(next_node, curr_path=current_path, depth=depth-1)


    def _fan_in(self, node, curr_path=[], depth=1):
        '''
        Recursively compute all nodes $DEPTH steps away that feed into $NODE

        Similar to `_fan_out()` but looks at incoming edges instead of outgoing edges.
        '''
        # If depth is 0, just return current node
        if depth == 0:
            # yield ({node}, [])
            yield ({self.node2type[node]:{node}}, [])

        # If depth is 1, return each set of neighbors and the path used to get there
        elif depth == 1:
            for (next_path, next_dict) in self.incoming_edges[node].items():
                current_path = self._merge_paths(next_path, node, curr_path)
                # for node_type, node_set in next_dict.items():
                    # yield (node_set - set(current_path), current_path)
                yield (next_dict, current_path)

        # Otherwise, recursively travel down edges until we reach depth 1
        else:
            for (next_path, next_dict) in self.incoming_edges[node].items():
                current_path = self._merge_paths(next_path, node, curr_path)
                for node_type, node_set in next_dict.items():
                    for next_node in node_set:
                        if next_node not in current_path:
                            yield from self._fan_in(next_node, curr_path=current_path, depth=depth-1)


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
        for edge_type, node_group in self.incoming_edges[node].items():
            # for node_type, node_set in node_group:
            yield node_group, [edge_type]

    # def _get_edges_from_nbhrs(self, node)


    def _merge_paths(self, curr_path, curr_node, next_path):
        '''
        Merge path segments together 
        '''
        if type(curr_path) == str:
            if len(curr_path) == 0:
                curr_path = []
            else:
                curr_path = [curr_path]
        if type(curr_node) == str:
            if len(curr_node) == 0:
                curr_node = []
            else:
                curr_node = [curr_node]
        if type(next_path) == str:
            if len(next_path) == 0:
                next_path = []
            else:
                next_path = [next_path]
        
        if (type(curr_path) != list 
            or type(curr_node) != list
            or type(next_path) != list):
            raise TypeError("Arguments must be of type 'str' or 'list'")

        return curr_path + curr_node + next_path


    def _path_to_metapath(self, path):
        '''
        Get metapath from a path segment
        '''
        return [self.x2type[x] for x in path]


    def _path_to_string(self, path):
        '''
        Turn (meta)path from list into string
        '''
        return '->'.join(path)


    def _score_metapath(self, metapath, path_instances=None):
        '''
        Get HeteSim score of individual metapath
        '''
        raise NotImplementedError

        
