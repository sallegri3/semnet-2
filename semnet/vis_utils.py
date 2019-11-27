from matplotlib import pyplot as plt
import xarray as xr
import gzip
from py2neo import Graph
import pandas as pd
import re
import os
import pickle
import seaborn as sns
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import numpy as np
from semnet.utils import metapath_to_english
pd.options.display.max_colwidth = 100

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


def plot_nbhd(target_list, filepath=None):
    """
    Plot visualization of neighborhood around a given set of target nodes

    Will produce 3 plots:
        * Plot comparing incoming edge types
        * Plot comparing outgoing edge types
        * Plot comparing node types of neighbors
    """
    names = [cui2name[target] for target in target_list]
    nbhr_dfs, out_dfs, in_dfs = [], [], []

    # Get metrics for each target node
    for target in target_list:
        nbhr_counts, out_counts, in_counts = get_nbhr_and_edge_types(target)
        name = cui2name[target]
        nbhr_dfs.append(nbhr_counts.rename({'count':name+'_count', 'proportion':name+'_proportion'}, axis=1))
        out_dfs.append(out_counts.rename({'count':name+'_count', 'proportion':name+'_proportion'}, axis=1))
        in_dfs.append(in_counts.rename({'count':name+'_count', 'proportion':name+'_proportion'}, axis=1))

    # Visualize count and proportion
    count_cols = [name+'_count' for name in names]
    proportion_cols = [name+'_proportion' for name in names]

    # Concatenate count and proportion data together and get metric to find most important cols
    nbhrs = pd.concat(nbhr_dfs, axis=1)
    nbhrs['avg_proportion'] = nbhrs[proportion_cols].mean(axis=1)
    outs = pd.concat(out_dfs, axis=1)
    outs['avg_proportion'] = outs[proportion_cols].mean(axis=1)
    ins = pd.concat(in_dfs, axis=1)
    ins['avg_proportion'] = ins[proportion_cols].mean(axis=1)

    for data, name in zip([nbhrs, outs, ins], ['Neighbors','Outgoing Edges','Incoming Edges']):
        fig,(ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(30,10))

        # Plot counts
        sorted_data = data.sort_values(by='avg_proportion', ascending=False)
        sorted_data[count_cols].head(12).rename({col:col.split('_')[0] for col in count_cols}, axis=1).plot.barh(ax=ax1)
        ax1.set_title(f"Count of {name} by Type")

        # Plot proportions
        sorted_data[proportion_cols].rename({col:col.split('_')[0] for col in proportion_cols}, axis=1).head(12).plot.barh(ax=ax2)
        ax2.set_title(f"Proportion of {name} by Type")

        # Plot total count
        data[count_cols].rename({col:col.split('_')[0] for col in count_cols}, axis=1).sum().plot.bar(ax=ax0)
        ax0.set_title(f"Total {name}")

        plt.suptitle(f"Summary of {name} for Target Nodes")

        if filepath is not None:
            plt.savefig(f"{filepath}_{name.lower()}.png")

    plt.show()



def plot_cosine_clusters(metapath_data, target, metric, source_subset=None, filepath=None, agg='concat'):
    """
    Plot seaborn heatmap of hierarchical clustering with cosine similarity.
    """
    # Reformat data to get desired vars and plot
    if source_subset:
        metapath_data = metapath_data.loc[{'source':source_subset}]
    data = metapath_data.loc[{'target':target, 'metric':metric}]

    if type(target) == list:
        if agg=='sum':
            data = data.sum(dim='target')
        else:
            data = data.stack(features=('target','metapath'))
    # else:
    #     data = data.stack(features=('metapath'))
    df = pd.DataFrame(data.values, index = [cui2name[i] for i in data.get_index('source')])

    # Precompute distances for clustering
    dist_matrix = 1 - pairwise_distances(df, metric='cosine')
    np.fill_diagonal(dist_matrix, 0)
    dist_array = pdist(df.values)
    links = linkage(dist_array, optimal_ordering=True, method='complete')
    dist_df = pd.DataFrame(dist_matrix, index=df.index, columns=df.index)


    # Plot clustering
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})
    sns.clustermap(dist_df, row_linkage=links, col_linkage=links, vmin=0, vmax=.75, xticklabels=True, yticklabels=True)
    if type(target) == list:
        plt.title("Multitarget Feature Correlation")
    else:
        plt.title(f"Feature Correlation for {cui2name[target]}")
    # plt.gcf().subplots_adjust(bottom=0.15)

    # Save figure if desired
    if filepath is not None:
        if type(target) == list:
            plt.savefig(f"{filepath}_multitarget_{metric}_cluster_heatmap.png")
        else:
            plt.savefig(f"{filepath}_{cui2name[target]}_{metric}_cluster_heatmap.png")

    plt.show()


def vertex_pair_relationship(targets, source, filepath=None, signed_subset=False):
    """
    Function to summarize the relationships between a target-source pair
    """
    meaningful_predications = ['treats','prevents','predisposes','disrupts','causes','augments','complicates','stimulates','inhibits']
    all_results = []
    missed_targets = []
    for target in targets:
        # Make param dict
        t_type = convert2type[target]
        s_type = convert2type[source]
        param_dict = {'target':target,
                      't_type':t_type,
                      'source':source,
                      's_type':s_type}

        # Make query to get vis results
        q = """
        MATCH (a:{t_type} {{identifier: '{target}'}}) - [r] - (b:{s_type} {{identifier: '{source}'}})
        RETURN r.weight as count, r.predicate as relationship
        """.format(**param_dict)
        # print(q)

        # Pull data
        cursor = graph.run(q)
        results = pd.DataFrame(cursor.data())
        cursor.close()
        # print(results)

        if results.shape[0] > 0:

            # Clean data with pandas
            pattern = re.compile(r'[A-Z]+')
            results['count'] = results['count'].astype(int)
            results[cui2name[target]] = results['count']
            results['predicate'] = results['relationship'].apply(lambda x: '_'.join(x.split('_')[:-1]).lower())
            results = results.drop(['relationship', 'count'], axis=1)

            grouped = results.groupby('predicate').sum()
            if signed_subset:
                mask = np.array([i in meaningful_predications for i in grouped.index])
                grouped = grouped.loc[mask,:]
            # grouped /= grouped.sum()
            all_results.append(grouped)
        else:
            missed_targets.append(target)

    results_agg = pd.concat(all_results, axis=1)
    for target in missed_targets:
        results_agg[cui2name[target]] = 0
    results_agg.plot.barh()
    plt.title(f"Relationships for {cui2name[source]}")

    # Write results to file
    if filepath:
        if signed_subset:
            filepath = filepath + '_signed'
        else:
            filepath = filepath + '_full'
        plt.savefig(f'{filepath}_{cui2name[source]}_bar.png')

    # results_agg.plot.pie(subplots=True, figsize=(24,8))
    # if filepath:
    #     plt.savefig(f'{filepath}_{cui2name[source]}_pie.png')
    # plt.show()


def plot_metapath_distribution(metapath_data, metric='count', filepath=None):
    """
    Create seaborn violin plots of distribution of metapath counts between target node and each source node
    """
    # Aggregate by source and target node
    agg_data = (metapath_data.loc[{'metric':metric}] > 0).sum(dim='metapath')
    data = agg_data.to_dataframe(name=metric).reset_index()
    data = data.drop(['metric','source'], axis=1).query(f"{metric} > 0")
    data['target'] = data['target'].map(cui2name)

    # Get metrics to adjust plot axes to make more visually appealing
    data_max = data[metric].max().flatten()[0]
    data_99 = data[metric].quantile(.99).flatten()[0]

    # Plot data
    sns.boxplot(x='target', y=metric, data = data)
    # if data_max > 1:
    #     plt.ylim([0, data_99])
    plt.title(f"{metric.title()} of Metapaths")
    plt.xlabel('Target')
    plt.ylabel(metric.title())

    # Save figure if desired
    if filepath:
        plt.savefig(f"{filepath}_{metric}_metapath_distribution.png")
    plt.show()


def plot_key_metapaths(weights, targets, source, metric='hetesim'):
    '''
    Create a bar plot of the weights of the most important metapaths
    between a target-source pair.
    '''
    subset = weights.sel(target=targets,
                         source=source).squeeze()

    data = pd.DataFrame(subset.transpose('metapath','target').values,
                        index=subset.get_index('metapath'),
                        columns=[cui2name[t] for t in targets])

    # Get metapaths for all sources with max path weight
    mask = (data.max(axis=1) == data.values.max())

    # Also get metapaths for top ranked in each category
    top_rank_masks = [(data[col].rank(method='min', ascending=False) <= 5)
                      for col in data.columns]

    # Combine into overall mask that is elementwise OR of individuals
    all_masks = pd.concat([mask]+top_rank_masks, axis=1)
    overall_mask = all_masks.max(axis=1)

    top_data = data[overall_mask]
    top_data.index = top_data.index.map(metapath_to_english)
    # print(top_data.index.map(metapath_to_english))

    print(cui2name[source])
    for col in top_data.columns:
        m = top_data[col] > 0
        display(top_data.loc[m, col].reset_index())

    # Style data for display
    # top_data.style.set_caption(f'{cui2name[source]}')
    # d = top_data.style.background_gradient(cmap='Greens')
    # print(source)
    # print(top_data.sum(axis=1).max())
    # display(d)
    # top_data.plot.barh()
    # plt.title(f"Top Metapaths for {cui2name[source]}")
    # plt.show()

# def highlight_nonzero()





if __name__ == '__main__':
    # targets = ['C0302600', 'C0016059', 'C0489482']
    # plot_nbhd(targets,filepath='/home/dkartchner3/research/semnet_applications/cvd/figures/cvd_target')

    from glob import glob
