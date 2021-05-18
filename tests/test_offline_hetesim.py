import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import math

sys.path.insert(0,'/nethome/akirkpatrick3/semnet/semnet')
from offline import HetGraph
from offline_hetesim import hetesim, hetesim_all_metapaths, mean_hetesim_scores


def test_hetesim(graph, mp, true_value):
    hs = hetesim(graph, ['s'], ['t'], [mp])[str(mp)]['s']['t']
    print("Computed hetesim: " + str(hs))
    print("True hetesim: " + str(true_value))
    assert( true_value - hs < 0.001 and hs - true_value < 0.001)
    
def test_hetesim_all_metapaths(graph, path_len, metapath, true_hs_value):
    assert(abs(hetesim_all_metapaths(graph, ['s'],['t'], path_len)[str(metapath)]['s']['t'] -true_hs_value)< 0.001)

def test_mean_hetesim_scores(graph, path_len, true_mean_hs_value):
    print(mean_hetesim_scores(graph, ['s'], 't', path_len))
    assert(abs( mean_hetesim_scores (graph, ['s'], 't', path_len)['s'] - true_mean_hs_value) < 0.0001) 
    
if __name__ == '__main__':

    # load toy graphs
    toy_graph_1_df = pd.read_csv('toy_graph_1.tsv', sep="\t", header=0)
    toy_graph_1 = HetGraph(toy_graph_1_df.to_dict(orient='records'))
    toy_graph_2_df = pd.read_csv('toy_graph_2.tsv', sep="\t", header=0)
    toy_graph_2 = HetGraph(toy_graph_2_df.to_dict(orient='records'))
    toy_graph_3_df = pd.read_csv('toy_graph_3.tsv', sep="\t", header=0)
    toy_graph_3 = HetGraph(toy_graph_3_df.to_dict(orient='records'))
    toy_graph_4_df = pd.read_csv('toy_graph_4.tsv', sep="\t", header=0)
    toy_graph_4 = HetGraph(toy_graph_4_df.to_dict(orient='records'))

    # some basic tests
    #print("Number of nodes with a least 1 outgoing edge: " + str(len(toy_graph_1.outgoing_edges)))
    #print("Number of nodes with a least 1 incoming edge: " + str(len(toy_graph_1.incoming_edges)))
    
    mp1 = ['t1', 'r1', 't2', 'r2', 't3', 'r3', 't1', 'r1', 't4']
    mp2 = ['t1', 'r1', 't2', 'r1', 't3', 'r1', 't4', 'r1', 't5', 'r1', 't6', 'r1', 't7']
    mp3 = ['t1', 'r1', 't2', 'r1', 't1', 'r2', 't1', 'r3', 't1']
    test_hetesim(toy_graph_1, mp1, 0.5774)
    test_hetesim(toy_graph_2, mp2, 0.8437)
    test_hetesim(toy_graph_3, mp3, 0.8333)

    test_hetesim_all_metapaths(toy_graph_1, 4, mp1, 0.5774)

    test_mean_hetesim_scores(toy_graph_4, 4, 0.4173)
