import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

sys.path.insert(0,'/nethome/akirkpatrick3/semnet/semnet')
from offline import HetGraph
from randomized_hetesim import randomized_pruned_hetesim, restricted_random_walk_on_metapath, randomized_pruned_hetesim_all_metapaths, approximate_mean_pruned_hetesim

def test_restricted_random_walk_on_metapath(tg1):
    mp1 = ['t1', 'r1', 't2', 'r2', 't3', 'r3', 't1', 'r1', 't4']
    mp1l = mp1[:5]
    bad_nodes = [set() for i in range(2)]
    for i in range(100):
        print(restricted_random_walk_on_metapath(tg1, 's', mp1l , bad_nodes))
    

def test_randomized_pruned_hetesim(graph, mp, epsilon, k, r, true_value, filename, N, plot_title):
    # run the algorithmt N times
    results = []
    for i in range(N):
       results.append(randomized_pruned_hetesim(graph, ['s'], ['t'], [mp], k, epsilon, r)[str(mp)]['s']['t'])
    results_df = pd.DataFrame(results)
    results_df.columns = ["approximate pruned hetesim"]
    results_df.to_csv(filename + ".csv")
    num_within_epsilon = len([x for x in results if true_value - epsilon <= x and true_value + epsilon >= x])
    percent_within_epsilon = num_within_epsilon / N * 100
    print("Of " + str(N) + " iterations, " + str(num_within_epsilon) + " (" + str(percent_within_epsilon) + "% ) had error less than epsilon.")

    print(results_df)
    #hist = results_df.hist(column='approximate pruned hetesim', bins=30)
    #hist.savefig(filename + ".png")

    
def test_randomized_pruned_hetesim_all_metapaths(graph, mp, path_len, epsilon, r, true_value, N):
    # run the algorithmt N times
    results = []
    for i in range(N):
       results.append(randomized_pruned_hetesim_all_metapaths(graph, ['s'], ['t'], path_len, epsilon, r)[str(mp)]['s']['t'])
    num_within_epsilon = len([x for x in results if true_value - epsilon <= x and true_value + epsilon >= x])
    percent_within_epsilon = num_within_epsilon / N * 100
    print("Of " + str(N) + " iterations, " + str(num_within_epsilon) + " (" + str(percent_within_epsilon) + "% ) had error less than epsilon.")


def test_approximate_mean_pruned_hetesim(graph, path_len, epsilon, r, true_value, N):
    # run the algorithmt N times
    results = []
    for i in range(N):
        approx_mean_hs =approximate_mean_pruned_hetesim(graph, ['s'], 't',  path_len, epsilon, r)['s']
        results.append(approx_mean_hs)
        print("Approximate mean hs is " + str(approx_mean_hs))

    num_within_epsilon = len([x for x in results if true_value - epsilon <= x and true_value + epsilon >= x])
    percent_within_epsilon = num_within_epsilon / N * 100
    print("Of " + str(N) + " iterations, " + str(num_within_epsilon) + " (" + str(percent_within_epsilon) + "% ) had error less than epsilon.")


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
    test_randomized_pruned_hetesim(toy_graph_1, mp1, 0.05, 4, 0.95, 0.5774, "toy_graph_1_test", 1, "Computed approximate pruned HeteSim values for toy graph 1")
    test_randomized_pruned_hetesim(toy_graph_2, mp2, 0.05, 3, 0.95, 0.8944, "toy_graph_2_test", 1, "Computed approximate pruned HeteSim values for toy graph 2")
    test_randomized_pruned_hetesim(toy_graph_3, mp3, 0.05, 3, 0.95, 0.8333, "toy_graph_3_test", 1, "Computed approximate pruned HeteSim values for toy graph 3")

    #test_randomized_pruned_hetesim_all_metapaths(toy_graph_1, mp1, 4, 0.05, 0.95, 0.5774, 5)
    #test_randomized_pruned_hetesim_all_metapaths(toy_graph_2, mp2, 6, 0.05, 0.95, 0.8944, 5)
    #test_approximate_mean_pruned_hetesim(toy_graph_4, 4, 0.05, 0.95, 0.6007, 5)

    #test_restricted_random_walk_on_metapath(toy_graph_1)
