from matplotlib import pyplot as plt
import xarray as xr
import gzip
from py2neo import Graph
import pandas as pd
import re
import os
import pickle

# Load Graph
graph = Graph(password='Mitch-Lin')

# Load dict to convert node CUI to node type
_ROOT = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(_ROOT, 'data/cui2type.pkl.gz')
with gzip.open(path, 'rb') as file:
    convert2type = pickle.load(file)

abbr2node = pickle.load(open(os.path.join(_ROOT, 'data/abbr2nodetype.pkl'),'rb'))

temp = {key.upper():val for key, val in abbr2node.items()}
abbr2node.update(temp)

cui2name = pickle.load(open(os.path.join(_ROOT, 'data/cui2name.pkl'),'rb'))
# import seaborn as sns
# sns.set()

def get_nbhr_and_edge_types(target):
    """
    Get types of neighbors of target node, weighted by the connection strength to each one
    """

    # Make param dict
    t_type = convert2type[target]
    param_dict = {'target':target, 't_type':t_type}

    # Make query to get vis results
    q = """
    MATCH (a:{t_type} {{identifier: '{target}'}}) - [r] - (b)
    RETURN labels(b) as nodetype, r.weight as count, r.predicate as relationship
    """.format(**param_dict)
    print(q)

    # Pull data
    cursor = graph.run(q)
    results = pd.DataFrame(cursor.data())
    cursor.close()

    # Clean data with pandas
    pattern = re.compile(r'[A-Z]+')
    results['count'] = results['count'].astype(int)
    results['nodetype'] = results['nodetype'].map(lambda x: x[0])
    results['predicate'] = results['relationship'].apply(lambda x: '_'.join(x.split('_')[:-1]).lower())
    results['rel_abbr'] = results['relationship'].apply(lambda x: x.split('_')[-1])
    results['endpoints'] = results['rel_abbr'].map(lambda x: pattern.findall(x))
    results['num_endpoints'] = results['endpoints'].map(len)
    results['node_start'] = results['endpoints'].map(lambda x: x[0])
    results['node_stop'] = results['endpoints'].map(lambda x: x[1])
    for col in ['node_start','node_stop']:
        results[col] = results[col].map(abbr2node)

    # Get counts of each neighbor type
    nbhr_counts = results[['count','nodetype']].groupby('nodetype').sum()
    nbhr_counts['proportion'] = nbhr_counts['count']/nbhr_counts['count'].sum()

    # Assign predicate direction and get counts of each edge type
    outgoing_edges = results[results.nodetype == results.node_stop]
    outgoing_edge_type_counts = outgoing_edges[['predicate','count']].groupby('predicate').sum()
    outgoing_edge_type_counts['proportion'] = outgoing_edge_type_counts['count']/outgoing_edge_type_counts['count'].sum()

    incoming_edges = results[results.nodetype == results.node_start]
    incoming_edge_type_counts = incoming_edges[['predicate','count']].groupby('predicate').sum()
    incoming_edge_type_counts['proportion'] = incoming_edge_type_counts['count']/incoming_edge_type_counts['count'].sum()

    return nbhr_counts, outgoing_edge_type_counts, incoming_edge_type_counts


def plot_nbhd(target_list):
    """
    Plot visualization of neighborhood around a given set of target nodes

    Will produce 3 plots:
        * Plot comparing incoming edge types
        * Plot comparing outgoing edge types
        * Plot comparing node types of neighbors
    """
    names = [cui2name[target] for target in target_list]
    nbhr_dfs, out_dfs, in_dfs = [], [], []

    for target in target_list:
        nbhr_counts, out_counts, in_counts = get_nbhr_and_edge_types(target)
        name = cui2name[target]
        nbhr_dfs.append(nbhr_counts.rename({'count':name+'_count', 'proportion':name+'_proportion'}, axi=1))
        out_dfs.append(out_counts.rename({'count':name+'_count', 'proportion':name+'_proportion'}, axi=1))
        in_dfs.append(in_counts.rename({'count':name+'_count', 'proportion':name+'_proportion'}, axi=1))

    count_cols = [name+'_count' for name in names]
    proportion_cols = [name+'_proportion' for name in names]

    nbhrs = pd.concat(nbhr_dfs, axis=1)
    nbhrs['avg_proportion'] = nbhrs[proportion_cols]
    outs = pd.concat(out_dfs, axis=1)
    ins = pd.concat(in_dfs, axis=1)

    fig, ax = plt.subplots()


def plot_cosine_cluters():
    """
    Plot seaborn heatmap of hierarchical clustering with cosine similarity
    """
    pass

def plot_metapath_distribution():
    """
    Create seaborn violin plots of distribution of metapath counts between target node and each source node
    """
    pass







if __name__ == '__main__':

    # target = 'C0302600'
    for target in
    name =
    a, b, c = get_nbhr_and_edge_types(target)
    print(a)
    print(b)
    print(c)
    nbhrs = a.sort_values(by='count', ascending=False)
    nbhrs.head(10).plot.barh()
    plt.title("Angiogenesis Neighbor Types")
    plt.savefig("ang_nbhr.png")
    plt.show()

    outgoing = b.sort_values(by='count', ascending=False)
    outgoing.head(10).plot.barh()
    plt.title("Angiogenesis Outgoing Edge Types")
    plt.savefig("ang_out.png")
    plt.show()

    incoming = c.sort_values(by='count', ascending=False)
    incoming.head(10).plot.barh()
    plt.title("Angiogenesis Incoming Edge Types")
    plt.savefig(f"{name}_in.png")
    plt.show()
