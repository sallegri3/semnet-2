# import pytest
import sys
import pandas as pd 
import networkx as nx

from tqdm.auto import tqdm, trange
# from semnet.offline import HetGraph

sys.path.insert(0,'/nethome/akirkpatrick3/semnet/semnet')
from offline import HetGraph

# Create toy graph with 6 nodes, 2 node types, 2 relation types
toy_graph = pd.DataFrame([['a','t1','b','t1','r1',1],
                          ['a','t1','c','t2','r1',1],
                          ['c','t2','d','t1','r1',1],
                          ['b','t1','e','t2','r1',1],
                          ['a','t1','e','t2','r1',1],
                          ['c','t2','f','t2','r1',1],
                          ['d','t1','b','t1','r2',1],
                          ['e','t2','c','t2','r2',1],
                          ['a','t1','f','t2','r2',1],
                          ['f','t2','d','t1','r2',1],
                          ['b','t1','a','t1','r2',1],
                          ['a','t1','a','t1','r1',1]], 
                         columns=['start_node',
                                  'start_type',
                                  'end_node',
                                  'end_type',
                                  'relation',
                                  'weight'])

num_start_nodes = len(toy_graph.start_node.unique())
num_end_nodes = len(toy_graph.end_node.unique())

# Inverse relations
# r1 is a symmetric relation; r2 is not
rel2inv = {'r1': 'r1', 'r2':'r2_inv'}

def test_constructor():
    '''
    Test HetGraph constructor

    Inputs:
    -------
        toy_graph: A toy example knowledge graph to test graph construction
    '''
    edgelist = toy_graph.to_dict(orient='records')
    hg = HetGraph(edgelist)
    assert len(hg.outgoing_edges) == num_start_nodes
    assert len(hg.incoming_edges) == num_end_nodes
    assert len(hg.type2nodes) == 2
    # assert len(hg.node2type)
    assert len(hg.outgoing_edges['a']) == 2
    assert len(hg.incoming_edges['b']) == 2
    assert hg.outgoing_edge_weights['a']['r1']['b'] == 1
    assert len(hg.schema_outgoing_edges) == 2
    assert len(hg.schema_outgoing_edges['t1']['r1']) == 2


def test_inverse_edges():
    '''
    Test addition of inverse edges

    Inputs:
    -------
        hg: HetGraph 
            HetGraph object constructed from toy_graph

        rel2inv: Dict
            Dictionary containing mapping from edges to their inverses
    '''
    edgelist = toy_graph.to_dict(orient='records')
    hg = HetGraph(edgelist)
    hg.add_inverse_edges(rel2inv)
    assert len(hg.outgoing_edges) == num_start_nodes
    assert len(hg.incoming_edges) == num_end_nodes
    assert max([len(x) for _, x in hg.outgoing_edges.items()]) == 3
    assert max([len(x) for _, x in hg.incoming_edges.items()]) == 3
    assert len(hg.outgoing_edges['a']) == 3
    assert len(hg.incoming_edges['b']) == 3



def test_fan_out_0(hg):
    node, path = next(hg._fan_out('a', depth=0))
    assert node == {'t1':{'a'}}
    assert path == []

def test_schema_fan_out_0(hg):
    node, path = next(hg._schema_fan_out('t1', depth=0))
    assert node == {'t1'}
    assert path == []

def test_fan_out_1(hg):
    '''
    Test HetGraph._fan_out function with depth 1
    '''
    nbhrs = [(hg._path_to_string(x[1]), x[0]) for x in hg._fan_out('a', depth=1)]
    # print(nbhrs)
    nbhr_nodes = set([])
    for path, node_dict in nbhrs:
        for nodes in node_dict.values():
            nbhr_nodes |= nodes
    print(nbhr_nodes)
    # assert ('a->r1', ('t2', {'c', 'e'}) in nbhrs
    # assert ('a->r2_inv', {'t1':{'b'}}) in nbhrs
    assert nbhr_nodes == {'a','b','c','e','f'}

def test_schema_fan_out_1(hg):
    '''
    Test HetGraph._fan_out function with depth 1
    '''
    nbhrs = [(hg._path_to_string(x[1]), x[0]) for x in hg._schema_fan_out('t1', depth=1)]
    #print(nbhrs)
    nbhr_nodes = set([])
    for path, nodes in nbhrs:
        nbhr_nodes |= nodes
    #print(nbhr_nodes)
    # assert ('a->r1', ('t2', {'c', 'e'}) in nbhrs
    # assert ('a->r2_inv', {'t1':{'b'}}) in nbhrs
    assert nbhr_nodes == {'t1', 't2'}


def test_fan_out_2(hg):
    '''
    Test HetGraph._fan_out function with depth 2
    '''
    nbhrs = [(hg._path_to_string(x[1]), x[0]) for x in hg._fan_out('a', depth=2)]
    nbhr_nodes = set([])
    for path, node_dict in nbhrs:
        for nodes in node_dict.values():
            nbhr_nodes |= nodes
    # assert ('a->r1->b->r2_inv', {'t1':{'d'}}) in nbhrs
    # assert ('a->r1->c->r1', {'t1':{'d'}}) in nbhrs
    assert nbhr_nodes == {'a','b','c','d','e','f'}
    
def test_schema_fan_out_2(hg):
    '''
    Test HetGraph._schema_fan_out function with depth 2
    '''
    nbhrs = [(hg._path_to_string(x[1]), x[0]) for x in hg._schema_fan_out('t1', depth=2)]
    # print(nbhrs)
    nbhr_nodes = set([])
    for path, node_set in nbhrs:
        nbhr_nodes |= node_set
    # assert ('a->r1->b->r2_inv', {'t1':{'d'}}) in nbhrs
    # assert ('a->r1->c->r1', {'t1':{'d'}}) in nbhrs
    assert nbhr_nodes == {'t1','t2'}


def test_fan_in_0(hg):
    node, path = next(hg._fan_in('a', depth=0))
    assert node == {'t1':{'a'}}
    assert path == []

def test_fan_in_1(hg):
    nbhrs = [(x[0], hg._path_to_string(x[1])) for x in hg._fan_in('a', depth=1)]
    # print(nbhrs)
    nbhr_nodes = set([])
    for node_dict, path in nbhrs:
        for nodes in node_dict.values():
            nbhr_nodes |= nodes
    assert nbhr_nodes == {'a','b','c','e','f'}
    # assert ({'t2':{'f'}},'r2_inv->a') in nbhrs
    # assert ({'t1':{'b'}}, 'r1->a') in nbhrs

def test_fan_in_2(hg):
    nbhrs = [(x[0], hg._path_to_string(x[1])) for x in hg._fan_in('a', depth=2)]
    nbhr_nodes = set([])
    for node_dict, path in nbhrs:
        for nodes in node_dict.values():
            nbhr_nodes |= nodes
    assert nbhr_nodes == {'a','b','c','d','e','f'}
    # assert ({'t2':{'c'}},'r1->f->r2_inv->a') in nbhrs
    # assert ({'t1':{'d'}}, 'r2->b->r1->a') in nbhrs


def test_merge_paths(hg):
    merged = hg._merge_paths('a',['b'], 'c')
    assert merged ==['a','b','c']


def test_fixed_length_paths(hg):
    paths = hg.compute_fixed_length_paths('a','c',length=2)
    path_strings = [hg._path_to_string(p) for p in paths]
    assert 'a->r1->e->r2->c' in path_strings
    assert 'a->r2->f->r1->c' in path_strings
    assert all([x.startswith('a') for x in path_strings])
    assert all([x.endswith('c') for x in path_strings])
    
def test_fixed_length_schema_walks(hg):
    paths = hg.compute_fixed_length_schema_walks('t1','t2',length=2)
    path_strings = [hg._path_to_string(p) for p in paths]
    print(path_strings)
    assert 't1->r1->t1->r2->t2' in path_strings

def test_compute_metapath_reachable_nodes(hg):
    mp = ['t1', 'r1', 't1']
    reachable_nodes = hg.compute_metapath_reachable_nodes('a', mp)
    assert 'a' in reachable_nodes
    assert 'b' in reachable_nodes
    assert not 'd' in reachable_nodes
    assert not 'c' in reachable_nodes
    
    mp2 = ['t1', 'r1', 't2', 'r1', 't1'] 
    reachable_nodes_2 = hg.compute_metapath_reachable_nodes('a', mp2)
    assert 'a' in reachable_nodes_2
    assert 'b' in reachable_nodes_2
    assert 'd' in reachable_nodes_2
    assert not 'e' in reachable_nodes_2 


def test_compute_fixed_length_metapaths(hg):
    mps = [hg._path_to_string(mp) for mp in hg.compute_fixed_length_metapaths('a', 'b', length=2)]
    # _print(mps)
    assert 't1->r1->t2->r1->t1' in mps
    assert 't1->r1->t1->r1->t2' in mps
    
    
if __name__ == '__main__':
    print(toy_graph)
    edgelist = toy_graph.to_dict(orient='records')
    hg = HetGraph(edgelist, rel2inv)
    # test_constructor()
    # test_inverse_edges()
    test_fan_out_0(hg)
    test_fan_out_1(hg)
    test_fan_out_2(hg)
    test_fan_in_0(hg)
    test_fan_in_1(hg)
    test_fan_in_2(hg)
    test_fixed_length_paths(hg)
    # test_merge_paths(hg)
    test_compute_metapath_reachable_nodes(hg)
    test_schema_fan_out_0(hg)
    test_schema_fan_out_1(hg)    
    test_schema_fan_out_2(hg)
    test_fixed_length_schema_walks(hg)