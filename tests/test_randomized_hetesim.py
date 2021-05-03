import sys
import pandas as pd 

sys.path.insert(0,'/nethome/akirkpatrick3/semnet/semnet')
from offline import HetGraph
from randomized_hetesim import randomized_pruned_hetesim

def test_randomized_pruned_hetesim(tg1, tg2, tg3):
    epsilon = 0.1
    r = 0.95
    # for each graph, run the algorithm 100 times
    mp1 = ['t1', 'r1', 't2', 'r2', 't3', 'r3', 't1', 'r1', 't4']
    for i in range(100):
        print(randomized_pruned_hetesim(tg1, ['s'], ['t'],  [mp1], 3, epsilon, r))   



if __name__ == '__main__':

    # load toy graphs
    toy_graph_1_df = pd.read_csv('toy_graph_1.tsv', sep="\t", header=0)
    toy_graph_1 = HetGraph(toy_graph_1_df.to_dict(orient='records'))
    toy_graph_2_df = pd.read_csv('toy_graph_2.tsv', sep="\t", header=0)
    toy_graph_2 = HetGraph(toy_graph_2_df.to_dict(orient='records'))
    toy_graph_3_df = pd.read_csv('toy_graph_3.tsv', sep="\t", header=0)
    toy_graph_3 = HetGraph(toy_graph_3_df.to_dict(orient='records'))

    # some basic tests
    #print("Number of nodes with a least 1 outgoing edge: " + str(len(toy_graph_1.outgoing_edges)))
    #print("Number of nodes with a least 1 incoming edge: " + str(len(toy_graph_1.incoming_edges)))

    test_randomized_pruned_hetesim(toy_graph_1, toy_graph_2, toy_graph_3)
