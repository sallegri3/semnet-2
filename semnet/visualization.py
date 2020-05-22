from matplotlib import pyplot as plt
from matplotlib_venn import venn2
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
    # print(q)

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


def plot_nbhd(target_list, filepath=None, detailed=False, ntypes=12):
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
    nbhrs = pd.concat(nbhr_dfs, axis=1, sort=True)
    nbhrs['avg_proportion'] = nbhrs[proportion_cols].mean(axis=1)
    outs = pd.concat(out_dfs, axis=1, sort=True)
    outs['avg_proportion'] = outs[proportion_cols].mean(axis=1)
    ins = pd.concat(in_dfs, axis=1, sort=True)
    ins['avg_proportion'] = ins[proportion_cols].mean(axis=1)

    if detailed:
        for data, name in zip([nbhrs, outs, ins], ['Neighbors','Outgoing Edges','Incoming Edges']):
            fig,(ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(30,10))

            # Plot counts
            sorted_data = data.sort_values(by='avg_proportion', ascending=False)
            sorted_data[count_cols].head(ntypes).rename({col:col.split('_')[0] for col in count_cols}, axis=1).plot.barh(ax=ax1)
            ax1.set_title(f"Count of {name} by Type")

            # Plot proportions
            sorted_data[proportion_cols].rename({col:col.split('_')[0] for col in proportion_cols}, axis=1).head(ntypes).plot.barh(ax=ax2)
            ax2.set_title(f"Proportion of {name} by Type")

            # Plot total count
            data[count_cols].rename({col:col.split('_')[0] for col in count_cols}, axis=1).sum().plot.bar(ax=ax0)
            ax0.set_title(f"Total {name}")

            plt.suptitle(f"Summary of {name} for Target Nodes")

            if filepath is not None:
                plt.savefig(f"{filepath}_{name.lower()}.png")
    else:
        fig,(ax0, ax1, ax2) = plt.subplots(1,3, figsize=(24,7))
        plt.suptitle(f"Summary of Node Neighborhood")
        sns.set()
        for data, name, ax in zip([nbhrs, ins, outs], ['Neighbors','Incoming Edges', 'Outgoing Edges'], [ax0, ax1, ax2]):
            # Plot proportions
            sorted_data = data.sort_values(by='avg_proportion', ascending=False)
            sorted_data[proportion_cols].rename({col:col.split('_')[0] for col in proportion_cols}, axis=1).head(ntypes).plot.barh(ax=ax)
            ax.set_title(f"Proportion of {name} by Type")
    plt.show()



def plot_cosine_clusters(metapath_data, target, metric, source_subset=None, filepath=None, agg='concat'):
    """
    Plot seaborn heatmap of hierarchical clustering with cosine similarity.
    """
    # sns.set_palette("GnBu_d")

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

    new_index = df.index.map(lambda x: x.split('|')[-1])
    dist_df = pd.DataFrame(dist_matrix, index=new_index, columns=new_index)

    # Plot clustering
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})
    # with sns.color_palette("PuBuGn_d"):
    sns.clustermap(dist_df, 
                   row_linkage=links, 
                   col_linkage=links, 
                   vmin=0, 
                   vmax=.75, 
                   xticklabels=True, 
                   yticklabels=True, 
                   cmap='GnBu'
                   )
    # if type(target) == list:
    #     plt.title("Multitarget Feature Correlation")
    # else:
    #     plt.title(f"Feature Correlation for {cui2name[target]}")
    # plt.gcf().subplots_adjust(bottom=0.15)

    # Save figure if desired
    if filepath is not None:
        if type(target) == list:
            plt.savefig(f"{filepath}_multitarget_{metric}_cluster_heatmap.png", bbox_inches='tight')
        else:
            plt.savefig(f"{filepath}_{cui2name[target]}_{metric}_cluster_heatmap.png", bbox_inches='tight')

    plt.show()


def vertex_pair_relationship(targets, source, filepath=None, signed_subset=False):
    """
    Function to summarize the relationships between a target-source pair
    """
    # Make list of which predications we care about
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

        # Pull data
        cursor = graph.run(q)
        results = pd.DataFrame(cursor.data())
        cursor.close()

        if results.shape[0] > 0:

            # Clean data with pandas
            pattern = re.compile(r'[A-Z]+')
            results['count'] = results['count'].astype(int)
            results[cui2name[target]] = results['count']
            results['predicate'] = results['relationship'].apply(lambda x: '_'.join(x.split('_')[:-1]).lower())
            results = results.drop(['relationship', 'count'], axis=1)

            # Gropu targets by predicate
            grouped = results.groupby('predicate').sum()
            if signed_subset:
                mask = np.array([i in meaningful_predications for i in grouped.index])
                grouped = grouped.loc[mask,:]
            all_results.append(grouped)
        else:
            missed_targets.append(target)

    # Aggregate results together
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
        plt.savefig(f"{filepath}_{metric}_metapath_distribution.png", bbox_inches='tight')
    plt.show()


def get_key_metapaths(weights, targets, source, metric='hetesim'):
    '''
    Create a bar plot of the weights of the most important metapaths
    between a target-source pair.
    '''
    subset = weights.sel(target=targets,
                         source=source).squeeze()

    data = pd.DataFrame(subset.transpose('metapath','target').values,
                        index=subset.get_index('metapath'),
                        columns=[t for t in targets])

    # Get metapaths for all sources with max path weight
    mask = (data.max(axis=1) == data.values.max())

    # Also get metapaths for top ranked in each category
    top_rank_masks = [(data[col].rank(method='min', ascending=False) <= 5)
                      for col in data.columns]

    # Combine into overall mask that is elementwise OR of individuals
    all_masks = pd.concat([mask]+top_rank_masks, axis=1)
    overall_mask = all_masks.max(axis=1)

    top_data = data[overall_mask]
    
    top_dict = {}
    for t in targets:
        top_dict[t] = top_data.index[top_data[t] > 0].tolist()
    return top_dict

    # display(top_data)
    top_data.index = top_data.index.map(metapath_to_english)
    # print(top_data.index.map(metapath_to_english))

    # print(cui2name[source])
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


def ranking_correlation(rankings, method='kendall', savepath=None):
    '''
    Compute correlation among rankings for various target nodes and 
    plot correlation heatmap.

    Inputs:
    -----------------------
        rankings: pandas.DataFrame
            Dataframe of rankings to be correlated.  Should not have other
            extraneous columns

        method: str
            Type of correlation to compute, defaults to Kendall-Tau

        savepath: str
            Path to save output heatmap if desired
    '''
    # Calculate correlation
    corr = rankings.corr(method=method)

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, 
                mask=mask, 
                center=0, 
                # cmap='GnBu',
                # cmap='viridis', 
                # vmin=.4, 
                square=True, 
                linewidths=.5, 
                cbar_kws={"shrink": .5}
                )

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')

    plt.show()


def rankings_venn_diagram_dict(rankings, 
                                t1, 
                                t2, 
                                max_size=10, 
                                ranking_type='raw',
                                return_annotation=False):
    '''
    Generate venn diagram of rankings to differentiate nodes that are mutually relevant
    from those that are disease specific

    Inputs:
    ------------------------------
        rankings: pandas.DataFrame
            Rankings for a set of target nodes

        t1, t2: string
            CUIs of targets for which to make venn diagram

        max_size: int
            Maximum number of elements to include in section of Venn diagram

    Returns:
    -----------------------------
        concept_subsets: dict
            Dictionary with keys for each target and values that are  
            sets of top concepts for each target
    '''
    if t1 in cui2name.keys():
        t1 = cui2name[t1]

    if t2 in cui2name.keys():
        t2 = cui2name[t2]

    assert t1 in cui2name.values()
    assert t2 in cui2name.values()

    # Get node specific rankings
    n = rankings.shape[0]
    r1 = rankings[f'{t1}_{ranking_type}_rank'] / n
    r2 = rankings[f'{t2}_{ranking_type}_rank'] / n
    

    # Determine which sources are concept specific
    diff = (r1 - r2)
    inv_diff = (r2 - r1) 
    # display(diff.sort_values())
    # display(inv_diff.sort_values())

    t1_concepts = diff[(r1 < r1.quantile(.2))].sort_values().head(max_size)
    t2_concepts = inv_diff[(r2 < r2.quantile(.2))].sort_values().head(max_size)
    # t1_concepts = (r1 + diff).sort_values().head(max_size)
    # t2_concepts = (r2 - diff).sort_values().head(max_size)
    display(t1_concepts, t2_concepts)


    # And which are in the intersection
    # n = r1.shape[0]
    # intersection = (np.abs(diff) + r1 + r2)
    intersection = r1 + r2
    intersection_concepts = intersection[intersection < n/5].sort_values().head(max_size)
    display(intersection_concepts)
    

    concept_dict = {t1: set(t1_concepts.index.tolist()+ intersection_concepts.index.tolist()),
                    t2: set(t2_concepts.index.tolist() + intersection_concepts.index.tolist())}
        
    # display(diff.sort_values().head().map(np.abs).to_dict())
    # display(inv_diff.sort_values().head().map(np.abs).to_dict())
                     
    if return_annotation:
        annotation_dict = {'left':t1_concepts.map(np.abs).to_dict(), 
                        'right': t2_concepts.map(np.abs).to_dict(), 'intersection':intersection}
        return concept_dict, annotation_dict

    return concept_dict


def venn_diagram(concept_dict, savepath=None, offset_label=False, annotation_dict=None):
    '''
    Make Venn diagram of overlap between different rankings

    Inputs:
    ------------------
        concept_dict: dict
            Dictionary of venn diagram contents output by 
            rankings_venn_diagram_dict

        savepath: string
            Filepath to save output figure, if desired

        offset_label: bool
            Whether to move diagram labels outside of plot for added readability
    '''
    # Make venn diagram with concept dict
    (k1, v1), (k2, v2) = concept_dict.items()
    v = venn2([v1, v2], (k1, k2))

    # Plot middle labels inside of diagram
    ppp = v.get_label_by_id('11').set_text('\n'.join(v1 & v2))

    # If offset is desired, plot annotations in bubbles outside of plot
    if offset_label:
        v.get_label_by_id('10').set_text('')
        v.get_label_by_id('01').set_text('')

        # Add proportion of difference to concepts
        if annotation_dict:
            sorted_left = sorted(list(v1 - v2), key=lambda x: annotation_dict['left'][x])[::-1]
            sorted_right = sorted(list(v2-v1), key=lambda x: annotation_dict['right'][x])[::-1]

            left_concepts = [f'{name}, {round(annotation_dict["left"][name], 3)}' for name in sorted_left]
            right_concepts = [f'{name}, {round(annotation_dict["right"][name], 3)}' for name in sorted_right]

            left_annotation = '\n'.join(left_concepts)
            right_annotation = '\n'.join(right_concepts)
        
        else:
            left_annotation = '\n'.join(v1-v2)
            right_annotation = '\n'.join(v2-v1)

        # Add annotations
        plt.annotate(left_annotation, xy=v.get_label_by_id('10').get_position() +
                np.array([0, 0.2]), xytext=(-40,40), ha='center',
                textcoords='offset points', 
                bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),
                arrowprops=dict(arrowstyle='->',              
                                connectionstyle='arc',color='gray'))
        plt.annotate(right_annotation, xy=v.get_label_by_id('01').get_position() +
                np.array([0, 0.2]), xytext=(40,40), ha='center',
                textcoords='offset points', 
                bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),
                arrowprops=dict(arrowstyle='->',              
                                connectionstyle='arc',color='gray'))

    # Plot other labels inside of diagram if desired
    else:
        v.get_label_by_id('10').set_text('\n'.join(v1 - v2))
        v.get_label_by_id('01').set_text('\n'.join(v2 - v1))

    
    
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()





