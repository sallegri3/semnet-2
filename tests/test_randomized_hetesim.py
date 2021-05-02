import sys
import pandas as pd 

sys.path.insert(0,'/nethome/akirkpatrick3/semnet/semnet')
from offline import HetGraph

if __name__ == '__main__':

    # load toy graphs
    toy_graph_1_df = pd.read_csv('toy_graph_1.tsv', sep="\t", header=0)
    print(toy_graph_1_df)
    toy_graph_1 = HetGraph(toy_graph_1_df.to_dict(orient='records'))

    # some basic tests
    print("Number of nodes with a least 1 outgoing edge: " + str(len(toy_graph_1.outgoing_edges)))
    print("Number of nodes with a least 1 incoming edge: " + str(len(toy_graph_1.incoming_edges)))
