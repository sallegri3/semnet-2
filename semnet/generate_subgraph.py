from py2neo import Graph
import pandas as pd
import pickle
import json
import numpy as np
import sys
import logging
import re
pd.set_option('display.max_rows', 1000)
graph = Graph(password='Mitch-Lin')

# Load list of generic concepts
GENERIC_CONCEPTS = pickle.load(open('data/generic_concepts.pkl', 'rb'))
abbr2node = pickle.load(open('data/abbr2nodetype.pkl','rb'))

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p', level=logging.WARNING)

logger = logging.getLogger(__file__)

def get_node_nbhrs(node_cuis, concept_types, concepts_to_exclude=GENERIC_CONCEPTS):
    '''
    Function to get neighbors of a particular family of nodes
    '''
    # Cypher Query
    query = """
            MATCH (a)-[r]-(b)
            WHERE a.identifier IN {0}
            AND b.kind IN {1}
            AND NOT b.identifier IN {2}
            RETURN a.name as start_node, a.kind as start_type, r.predicate as relationship, r.weight as weight, b.name as end_node, b.kind as end_type, b.identifier as cui LIMIT 40000 
            """.format(node_cuis, concept_types, concepts_to_exclude)

    # Run query and get results
    cursor = graph.run(query)
    res = cursor.data()

    # Drop duplicates and make sure nodes have correct type
    data = pd.DataFrame(res).drop_duplicates()
    data['weight'] = data['weight'].astype(int)
    return data


def check_is_reversed(start_type, end_type, edge):
    '''
    Determine direction of an edge
    '''
    node_pat = re.compile('[A-Z]+')
    endpoint_types = [abbr2node[n] for n in node_pat.findall(edge.split('_')[-1])]
    reversed = False
    if start_type != endpoint_types[0]:
        if start_type != endpoint_types[1]:
            raise Exception("Types don't match up with edge")
        else:
            reversed = True
    return reversed


def get_top_nbhrs(data, 
                    nodes_to_skip=[], 
                    downweight_node_types=[],
                    downweight_edge_types=['ASSOCIATED'], 
                    node_downweight_factor=.5,
                    edge_downweight_factor=.5,
                    num_nodes=20,
                    **kwargs):
    '''
    Clean dataframe of concepts 

    Inputs:
    -----------------
        nodes_to_skip: list of str
            Names (not CUIs) of nodes that you want excluded from final results 

        downweight_edge_types: list of str
            Edge types you want downweighed so they appear less frequently in final results

        edge_downweight_factor: float in (0,1)
            How much to downweight certain edge types
    '''
    # Standardize end_node and relationship names
    data['end_node'] = data['end_node'].map(lambda x: x.split('|')[-1])
    data['end_node'] = data['end_node'].map(lambda x: x.title() if x[0].islower() else x)

    # Downweight overly common node types
    if len(downweight_node_types) > 0:
        downweight_mask = data.end_type.map(lambda x: np.any([x.startswith(t) for t in downweight_node_types]))
        data.loc[downweight_mask, 'weight'] *= node_downweight_factor

    # Group into edges, drop generic values, get top terms
    grouping_terms = ['end_node','start_node']
    nodes = data.groupby(grouping_terms)
    nodes = nodes.sum().sort_values(by='weight', ascending=False).head(num_nodes).reset_index()
    node_list = nodes.query("end_node not in {}".format(nodes_to_skip)).end_node.tolist()

    # Get good sampling of nodes types and relationships
    limited = data.query(f"end_node in {node_list}").copy()

    # Downweight overly common edge types
    if len(downweight_edge_types) > 0:
        downweight_mask = limited.relationship.map(lambda x: np.any([x.startswith(t) for t in downweight_edge_types]))
        limited.loc[downweight_mask, 'weight'] *= edge_downweight_factor

    

    # Get highest weighted relationship after downweighting
    limited = limited.loc[limited.groupby(grouping_terms)['weight'].idxmax()]
    return limited

def standardize_nbhr_data(data, 
                          rels_to_drop=['HIGHER_THAN',
                                        'LOWER_THAN',
                                        'COMPARED',
                                        'NEG'],
                          downweight_node_types=[],
                          downweight_edge_types=['ASSOCIATED'],
                          node_downweight_factor=.5, 
                          edge_downweight_factor=.5,
                          **kwargs):
    '''
    Clean up data of neighbors of starting concepts

    Inputs:
    ------------------
        data: pandas.DataFrame
            Dataframe of data to clean

        rels_to_drop: list of str
            Relations to drop from dataframe
    '''
    # Standardize end_node and relationship names
    data['end_node'] = data['end_node'].map(lambda x: x.split('|')[-1])
    data['end_node'] = data['end_node'].map(lambda x: x.title() 
                                            if x[0].islower() else x)
    data['start_node'] = data['start_node'].map(lambda x: x.split('|')[-1])
    data['start_node'] = data['start_node'].map(lambda x: x.title() 
                                                if x[0].islower() else x)

    # Get rid of meaningless relations
    for rel in rels_to_drop:
        data = data[~data.relationship.map(lambda x: x.startswith(rel))]

    # Downweight overly common edge types
    if len(downweight_edge_types) > 0:
        downweight_mask = data.relationship.map(lambda x: np.any([x.startswith(t) for t in downweight_edge_types]))
        data.loc[downweight_mask, 'weight'] *= edge_downweight_factor

    # Downweight overly common node types
    if len(downweight_node_types) > 0:
        downweight_mask = data.end_type.map(lambda x: np.any([x.startswith(t) for t in downweight_node_types]))
        data.loc[downweight_mask, 'weight'] *= node_downweight_factor

    # Limit DF to relationships with highest overall weight
    limited = data.loc[data.groupby(['end_node','start_node'])['weight'].idxmax()]
    return limited

def remove_covid_nodes(df):
    '''
    Remove edges added from CORD-19 data
    '''
    covid_edge_mask = ( (df.start_type == df.start_type.map(lambda x: x.lower())) 
                      & (df.end_type == df.end_type.map(lambda x: x.lower()))
                      )
    df = df[~covid_edge_mask]
    return df


def reverse_edges(data):
    '''
    Reverse edges in data that are listed as pointing in the wrong direction
    '''
    # Check for edges with incorrect direction
    rev = data[['start_type','end_type','relationship']].apply(lambda x: check_is_reversed(*x), axis=1)
    
    # Swap start and end types
    temp = data.loc[rev, 'start_type']
    data.loc[rev, 'start_type'] = data.loc[rev, 'end_type']
    data.loc[rev,'end_type'] = temp
    
    # Swap start and end values
    temp = data.loc[rev, 'start_node']
    data.loc[rev, 'start_node'] = data.loc[rev, 'end_node']
    data.loc[rev,'end_node'] = temp
    data.loc[:,'relationship'] = data.loc[:,'relationship'].map(lambda x: ' '.join(x.split('_')[:-1]))
    return data


def get_subgraph(node_cuis, 
                   min_weight=1, 
                   total_nodes=25, 
                   nodes_to_skip=[], 
                   downweight_node_types=[],
                   downweight_edge_types=['ASSOCIATED'], 
                   node_downweight_factor=.5,
                   edge_downweight_factor=.5,
                   concept_types=['GeneOrGenome'],
                   **kwargs):
    '''
    Create final list of nodes and edges for graph

    Inputs:
    ------------------
        node_cuis: list of str
            CUIs of central nodes to be included in subgraph

        min_weight: int
            Minimum edge weight to be included in final graph (can be useful for pruning)

        total_nodes: int
            Total nodes in final graph

        nodes_to_skip: list of str
            Names (not CUIs) of nodes you want to manually exclude

        downweight_edge_types: list of str
            Edge types to downweight in considerations
    '''
    

    # Specify number of neighbors to retrieve
    num_nbhrs = total_nodes - len(node_cuis)

    # Get nbhrs of central nodes
    logger.info("Getting neighbors")
    node_nbhrs = get_node_nbhrs(node_cuis, concept_types, GENERIC_CONCEPTS)
    limited = get_top_nbhrs(node_nbhrs, 
                    num_nodes=num_nbhrs,
                    nodes_to_skip=nodes_to_skip, 
                    downweight_node_types=downweight_node_types, 
                    downweight_edge_types=downweight_edge_types, 
                    node_downweight_factor=node_downweight_factor, 
                    edge_downweight_factor=edge_downweight_factor)

    # Make final list of nodes
    neighbor_list = limited.cui.tolist() + node_cuis

    # Get final graph
    query = """
        MATCH (a)-[r]-(b)
        WHERE a.identifier IN {0}
        AND b.identifier IN {0}
        RETURN a.name as start_node, a.kind as start_type, r.predicate as relationship, r.weight as weight, b.name as end_node, b.kind as end_type, b.identifier as cui LIMIT 40000 
        """.format(neighbor_list)
    cursor = graph.run(query)
    res = cursor.data()

    # Standardize data in final graph
    nbhr_data = pd.DataFrame(res).drop_duplicates()
    nbhr_data['weight'] = nbhr_data['weight'].astype(int)
    nbhr_edges = standardize_nbhr_data(nbhr_data,
                    downweight_node_types=downweight_node_types, 
                    downweight_edge_types=downweight_edge_types, 
                    node_downweight_factor=node_downweight_factor, 
                    edge_downweight_factor=edge_downweight_factor)

    # Assemble final dataframe and prune by weight
    final_data = remove_covid_nodes(nbhr_edges.dropna())
    final_data = reverse_edges(final_data.drop('cui',axis=1))
    final_data = final_data.query(f'weight >= {min_weight}')


    # Limit node to meaningful columns and return
    final_data = final_data[['start_node','start_type','relationship','end_node','end_type','weight']]
    final_data = final_data.sort_values(by=['start_type', 'start_node'])

    return final_data


if __name__=='__main__':
    config_filepath = sys.argv[1]
    config = json.load(open(config_filepath, 'r'))
    subgraph = get_subgraph(**config)
    subgraph.to_csv(config['save_filepath'], index=False)

    
    
