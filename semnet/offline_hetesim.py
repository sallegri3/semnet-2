"""
This module implements deterministic hetesim for the datastructure HetGraph given in offline.py
"""

def hetesim(graph, start_nodes, end_nodes, metapaths):
    """
    computes all hetesim scores between elements of start_nodes and end_nodes, under elements of metapaths


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
    """
    
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
            fixed_mp_dict[s] =  _compute_hs_vector_from_left(graph, s, lh)
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
            fixed_mp_dict[t] =  _compute_hs_vector_from_right(graph, t, rh)
            #print(fixed_mp_dict[t])
        node_prob_right[str(rh)] = fixed_mp_dict

    # create output dict hs[mp][s][t] 
    hs =  {}
    for mp in metapaths:
        left_half = str(mp[0:path_len+1])
        right_half =  str(mp[path_len:])
        fixed_mp_dict = {}
        for s in start_nodes:
            fixed_s_dict = {}
            for t in end_nodes:
                fixed_s_dict[t] = _cos_similarity(node_prob_left[left_half][s], node_prob_right[right_half][t])
            fixed_mp_dict[s] = fixed_s_dict
        hs[str(mp)] = fixed_mp_dict
        
    return hs
    
def _compute_hs_vector_from_left(graph, start_node, metapath):
    """
    computes the left-hand-side probability vector used to compute hetesim
    
        Inputs:
        graph: HetGraph
            underlying graph
            
        start_node: str
            cui of start node
        
        metapath: list of strs
            metapath for which to compute probability vector
            
        
    Outputs:
        hs_vector: dict mapping center-layer nodes to probabilities
           hetesim probability vector for random walks along given metapath from start_node
    """
    path_len = int((len(metapath)-1)/2)
    
    # set up a list of dictionaries, one for each step of the metapath
    node_probs = [{} for _ in range(path_len + 1)]
    
    # populate the first layer probabilities
    node_probs[0][start_node] = 1
    
    # iterate through the steps of the metapath
    for i in range(path_len):
        current_node_type = metapath[2*i]
        relation = metapath[2*i + 1]
        next_node_type = metapath[2*i + 2]
        
        #print("current_node_type: " + current_node_type)
        #print("relation: " + relation)
        #print("next_node_type: " + next_node_type)

        for cur_node in node_probs[i].keys():
            neighbors = graph.outgoing_edges[cur_node][relation][next_node_type]
            weighted_degree = sum([graph.outgoing_edge_weights[cur_node][relation][n] for n in neighbors])
            
            for n in neighbors:
                new_prob = node_probs[i][cur_node] * graph.outgoing_edge_weights[cur_node][relation][n] / weighted_degree
                if n in node_probs[i+1]:
                    node_probs[i+1][n] += new_prob
                else:
                    node_probs[i+1][n] = new_prob
        #print(str(node_probs))

    
    return node_probs[path_len]
    
def _compute_hs_vector_from_right(graph, end_node, metapath):
    """
    computes the right-hand-side probability vector used to compute hetesim
    
        Inputs:
        graph: HetGraph
            underlying graph
            
        end_node: str
            cui of end node
        
        metapath: list of strs
            metapath for which to compute probability vector
            
        
    Outputs:
        hs_vector: dict mapping center-layer nodes to probabilities
           hetesim probability vector for random walks along given metapath from end_node
    """
    
    path_len = int((len(metapath)-1)/2)
    
    # set up a list of dictionaries, one for each step of the metapath
    node_probs = [{} for i in range(path_len + 1)]
    
    # populate the first layer probabilities
    node_probs[0][end_node] = 1
    
    # iterate through the steps of the metapath
    for i in range(path_len):
        current_node_type = metapath[2*path_len - 2*i]
        relation = metapath[2*path_len - 2*i-1]
        next_node_type = metapath[2*path_len - 2*i -2]
        
        for cur_node in node_probs[i].keys():
            neighbors = graph.incoming_edges[cur_node][relation][next_node_type]
            weighted_degree = sum([graph.incoming_edge_weights[cur_node][relation][n] for n in neighbors])
            
            for n in neighbors:
                new_prob = node_probs[i][cur_node] * graph.incoming_edge_weights[cur_node][relation][n] / weighted_degree
                if n in node_probs[i+1]:
                    node_probs[i+1][n] += new_prob
                else:
                    node_probs[i+1][n] = new_prob
    
    return node_probs[path_len]

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
    
def hetesim_all_metapaths(graph, source_nodes, target_nodes, path_len):
    """
        computes hetesim for all metapaths of specified length between the source nodes and the target node

        Input:
            graph: HetGraph
                graph where hetesim is to be computed

            source_nodes: list of strs
                list of source node cuis, all source nodes must have same type

            target_node: list of str
                list of target node cui, all target nodes must have same type

            path_len: int
                path length, must be even

        Outputs:
            hetesim_scores: dict of dicts
                accessed as hetesim_scores[metapath][source][target]
    """
    #find all metapaths
    metapaths = []
    for s in source_nodes:
        for t in target_nodes:
            paths = graph.compute_fixed_length_paths(s, t, length=path_len)
            for mp in [graph._path_to_metapath(p) for p in paths]:
                if not mp in metapaths:
                    metapaths.append(mp)
    
    # compute hetesim
    return hetesim(graph, source_nodes, target_nodes, metapaths)