"""
This module implements an unsupervised rank aggregation method [1]_ for use with the :mod:`semnet.feature_extraction` and :mod:`semnet.hetesim` modules. It is based
on the idea that each metapath score is an independent ranker of the source
nodes. The metapath rankers are weighted based on their agreement with the
consensus ranking, and then used to compute an aggregate score and corresponding
ranking for each source node.

.. [1] Klementiev, Alexandre, Dan Roth, and Kevin Small. "An unsupervised learning algorithm for rank aggregation." European Conference on Machine Learning. Springer, Berlin, Heidelberg, 2007.
"""

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import os
import pickle
from scipy import stats

_ROOT = os.path.abspath(os.path.dirname(__file__))
cui2name = pickle.load(open(os.path.join(_ROOT, 'data/cui2name.pkl'),'rb'))

class UnsupervisedRankAggregator(object):
    """
    This class implements the rank aggregation methods described by Klementiev
    [1]_. It wraps a xr.DataArray object for data management.

    Parameters
    ----------
        data_array: xarray.DataArray
            A dataarray output from one of the feature exraction outputs.

        cui2name: dict
            A dictionary that converts CUI in the dataarray back into verbose descriptions.
    """

    def __init__(self, data_array, cui2name):
        """
        Initializes the URA using characteristics of the xr.DataArray.
        """
        assert isinstance(data_array, xr.DataArray), 'Constructor requires xr.DataArray.'

        self.data_array = data_array
        self.data = self.data_array.values
        self.sources = self.data_array.source.values
        self.targets = self.data_array.target.values
        self.metrics = self.data_array.metric.values
        self.metapaths = self.data_array.metapath.values

        # Fetch the verbose names
        self.source_names = np.array([cui2name[source] for source in self.sources])


    def aggregate(self, metric, target, lambd=1, theta=500):
        """
        Learn weights that prioritize features that agree with the mean rank. Implements
        the additive aggregation method described by Klementiev.

        .. note:: This is the main function in this module.

        Parameters
        ----------
            metric: 'count', 'dwpc', or 'hetesim'
                The metric to be used for rank aggregation.

            target: str
                The CUI of the ranking target.

            lambd:
                The learning rate.

            theta:
                A hyperparameter. "For some items, only a small fraction of the ranking functions return a valid rank. If less than :math:`\\theta` rankers, as defined by the user, return a valid rank for an item, the information is deemed untrustworthy and no updates are made for this item" [1]_.
        Returns
        -------
            mp_wts:
                Metapath weights for the learned ranker.
        """

        # Check that target and metric are valid and store them in memory
        assert metric in self.metrics, 'Metric is invalid: %r' % metric
        #assert target in self.targets, 'Target is invalid: %r' % target


        # # TODO: Accomodate multi-target aggregations
        if type(target) != str:
            assert isinstance(target, list)
            for t in  target:
                assert  t in  self.data_array.get_index('target'), 'One or more targets are invalid: %r' % target
        else:
            assert target in self.targets, 'Target is invalid %r' % target
        #
        #     if len(target) > 1:
        #         self.agg_target = "_".join(target)
        #         data_array = self.data_array
        #         #.assign_coords(target=self.agg_target).expand_dims('target')
        #     else:
        #         self.agg_target = target[0]
        #         data_array = self.data_array



        self.agg_target = target
        self.agg_metric = metric


        self.rankings, self.thresholds = rank_by_feature(self.data_array, self.agg_target, self.agg_metric)

        r = self.rankings
        k = self.thresholds

        w = [np.zeros(len(self.thresholds))]
        # Compute the number of within-threshold ranking functions associated with each item
        below_thresh = r <= k
        nx_vec = below_thresh.sum(axis=1)

        # Iterating through the items (x), use "good" ranking functions and their rankings
        for nx, r_i in zip(nx_vec, r*below_thresh):
            if nx >= theta:
                mu = sum(r_i) / nx
                # Compute the update for each feature (i) based on whether or not r_ix
                # is above threshold
                update = np.array([k_i + 1 if r_ix==0 else r_ix for r_ix, k_i in zip(r_i, k)])
                delta = (update - mu)**2

                w.append(w[-1] + lambd * delta)
        # Normalize each w_t to sum to 1
        w = np.array([w_t / np.linalg.norm(w_t, ord=1) for w_t in w[1:]])
        # Keep the final set of weights
        try:
            self.weights = w[-1]
        except:
            #return nx
            raise Exception('no features above threshold')

        return {mp: wt for mp, wt in zip(self.metapaths, self.weights)}

    def get_scores(self, cui_key=False):
        """ Returns a dictionary of source scores based on the learned ranking model. """

        scores = np.dot(self.rankings, self.weights)
        ranked_indices = np.argsort(scores)
        if cui_key:
            ranked_sources = self.sources[ranked_indices]
        else:
            ranked_sources = self.source_names[ranked_indices]
        ranked_scores = sorted(scores)
        self.scores = {source:score for source, score in zip(ranked_sources, ranked_scores)}

        return self.scores



def rank_by_feature(all_data, target, metric):
    """
    This helper function assigns each item a natural number ranking by feature,
    based on feature values. Ties are resolved by giving both numbers the highest
    possible rank. Thresholds are set to exclude items with zero-valued features.

    Parameters
    ----------
        all_data: xarray.DataArray
            A dataarray containing feature extractor data

        target: str or list
            The CUI (or list of CUIs) of the node(s) we would like to rank
            with respect to.

        metric: 'count', 'dwpc', or 'hetesim'
            The metric to be used for rank aggregation.

    Returns
    -------
        rankings: np.array
            A 2-D ``numpy`` array of integer rankings of source nodes with respect to the target.

        thresholds: list
            A list of the lowest ranks.
    """
    rankings = []
    thresholds = []
    n_sources = all_data.get_index('source').shape[0]
    features = all_data.loc[:, target, :, metric].values.T.reshape((-1, n_sources))
    for row in features:
        ranks = np.array(pd.Series(row).rank(method='dense', ascending=False)).astype(int)

        """
        # Generate randomized rankings to resolve ties
        import numpy.random as random
        random.seed(seed)
        randomized_tiebreakers = dict()
        for rank, count in zip(*np.unique(ranks, return_counts=True)):
            randomized_tiebreakers[rank] = list(rank + random.permutation(np.arange(count)))
        permuted_ranks = np.array([randomized_tiebreakers[rank].pop() for rank in ranks])
        rankings.append(permuted_ranks)
        """

        rankings.append(ranks)
        if min(row) == 0:
            thresholds.append(max(ranks)-1)
        else:
            thresholds.append(max(ranks))
    rankings = np.array(rankings).T

    return rankings, thresholds


# Define function to get rankings w.r.t. a list of items
def get_all_scores(features, cuis, metric='hetesim',lambd=1, theta=500, return_overall_ranking=False):
    '''
    Function to get rankings with respect to a list of targets

    Inputs:
    ----------------------------------------------------------
        features: xarray.DataArray
            DataArray containing feature extractor data

        cuis: list of string
            List of CUIs of targets for which we want rankings
    '''
    # Make rank aggregator
    ura = UnsupervisedRankAggregator(features, cui2name)

    # Get rankings in terms of each CUI
    n_feats = features.metapath.shape[0]
    rankings = []
    score_cols = []
    for cui in tqdm(cuis):
        name = cui2name[cui]+ '_raw_score'
        # score_cols.append(name)

        try:
            ura.aggregate(metric, cui, lambd=lambd, theta=theta)
        except Exception as e:
            print(e)
            print(f"Removing cui for {cui2name[cui]} from list of targets")
            cuis.remove(cui)
            continue
        curr_rankings = pd.Series(ura.get_scores(cui_key=True), name=name)
        # curr_rankings -= curr_rankings.min()
        curr_rankings /= curr_rankings.max()
        curr_min = curr_rankings.min()
        curr_rankings = (1 - curr_rankings) + curr_min
        rankings.append(curr_rankings)

    rankings_df = pd.concat(rankings, axis=1, sort=False)

    # Get rankings for all of the individual target columns
    for col in [cui2name[cui] for cui in cuis]:
        rankings_df[col+'_raw_rank'] = rankings_df[col+'_raw_score'].rank(ascending=False)

    # Get average to create overall ranking
    if return_overall_ranking:
        rankings_df = rankings_df[sorted(rankings_df.columns.tolist())]
        rankings_df['overall_raw_score'] = rankings_df[score_cols].mean(axis=1)
        rankings_df['overall_raw_rank'] = rankings_df['overall_raw_score'].rank(ascending=False)

    # Get rankings for all of the individual target columns
    # rankings_df[col+' rank'] = rankings_df[col].rank()
    # for col in [cui2name[cui] for cui in cuis]:

    # Get overall ranking and add to df
    # if len(cuis) > 1:
    #     ura.aggregate(metric, cuis, lambd=lambd, theta=len(cuis)*theta_frac*n_feats)
    #     overall_score = pd.Series(ura.get_scores(cui_key=True), name='overall_score')
    #     overall_score -= overall_score.min()
    #     overall_score /= overall_score.max()
    #     rankings_df = pd.concat([rankings_df, overall_score], axis=1, sort=False)
    #     rankings_df['overall_rank'] = rankings_df['overall_score'].rank()

    # Add medical names for each term and reorder so this is first column
    rankings_df['Name'] = rankings_df.index.map(cui2name)
    cols = rankings_df.columns.tolist()
    rankings_df = rankings_df[['Name'] + cols[:-1]]
    rankings_df.index.name = 'CUI'
    return rankings_df



def high_importance_low_count(rankings, counts, max_path_length=2, eps=.01):
    """
    Get list of nodes prioritized by the geometric mean of their rank and count of metapaths
    We generally consider metapaths of length 1 to indicate relationships present in the literature

    Inputs:
        ranking: pandas.DataFrame
            Dataframe of rankings produced by :get_all_scores:

        Counts: xarray.DataArray
            targets x sources x metapaths x metrics dataarray containing metapath counts for each source target pair

        max_path_length: int
            Max length of metapath to consider when reranking based on path connectivity

    Returns:

    """
    # Make sure we only have count data
    counts = counts.loc[{'metric':'count'}]

    # Get lengths of metapaths and mask of paths that work
    metapath_names = list(counts.get_index('metapath'))
    metapath_lengths =np.array([s.count('>') + s.count('<') for s in metapath_names])
    mask = (metapath_lengths <= max_path_length)

    novelty_weights = []

    # Get prioritized rankings for each target individually
    for cui in list(counts.get_index('target')):
        name = cui2name[cui]
        if f'{name}_raw_score' not in rankings.columns:
            print(f'{name} not found in rankings, moving to next target')
            continue
        path_counts = counts.loc[{'target':cui,
                                  'metapath':mask}].sum(dim=['metapath']).to_dataframe(name=f'{name}_path_count')
        # print(path_counts)
        path_counts = path_counts.drop(['target','metric'], axis=1)
        path_counts.index.name = 'CUI'

        # Get path counts and normalize
        normalized_path_counts = (1/(path_counts + 1))**.5
        # normalized_path_counts = path_counts / path_counts.max()
        # path_min = normalized_path_counts.min()
        # normalized_path_counts = (1 - normalized_path_counts) + path_min
        # normalized_path_counts[normalized_path_counts < eps] += (normalized_path_counts[normalized_path_counts < eps] - eps)/eps
        normalized_path_counts = normalized_path_counts.rename(columns={f'{name}_path_count':f'{name}_normalized_path_count'})
        # print(normalized_path_counts)

        # Add extra weight to sources directly connected to target
        # Weight is $(1 - nbhr_weight) is disconnected, 1 if connected
        nbhr_mask = (metapath_lengths == 1)
        nbhr_weight = .5
        nbhr_flag = counts.loc[{'target':cui,
                                  'metapath':mask}].sum(dim=['metapath']).to_dataframe(name=f'{name}_nbhr_flag')
        nbhr_flag = nbhr_flag.drop(['target','metric'], axis=1)
        nbhr_flag = nbhr_weight * (nbhr_flag > 0).astype(float) + (1 - nbhr_weight)
        nbhr_flag.index.name = 'CUI'

        if 'CUI' in rankings.columns:
            rankings = rankings.set_index('CUI')
        rankings = rankings.join(path_counts)
        rankings = rankings.join(normalized_path_counts)
        rankings = rankings.join(nbhr_flag)

        novelty_weights.append(normalized_path_counts)

        rankings[f'{name}_newness_score'] = stats.gmean(rankings[[f'{name}_normalized_path_count', f'{name}_nbhr_flag']], axis=1)
        rankings[f'{name}_novelty_score'] = stats.gmean(rankings[[f'{name}_raw_score', f'{name}_normalized_path_count', f'{name}_nbhr_flag']], axis=1)
        # print(rankings)
        rankings[f'{name}_novelty_rank'] = rankings[f'{name}_novelty_score'].rank(ascending=False).astype(int)

        rankings = rankings.drop([f'{name}_normalized_path_count', f'{name}_nbhr_flag'], axis=1)
        # rankings = rankings.drop([f'{name}_path_count'], axis=1)


    # Get overall novelty score/rank
    mean_novelty = pd.concat(novelty_weights, axis=1).mean(axis=1)
    novelty_agg = pd.concat([mean_novelty, rankings['overall_raw_score']], axis=1)
    novelty_agg['overall_novelty_score'] = stats.gmean(novelty_agg, axis=1)
    novelty_agg['overall_novelty_rank'] = novelty_agg['overall_novelty_score'].rank(ascending=False)
    # display(novelty_agg.head())
    rankings = rankings.join(novelty_agg[['overall_novelty_score','overall_novelty_rank']])
    # rankings['overall_path_count'] = rankings.filter(regex='path_count$', axis=1).sum(axis=1)

    cols = rankings.columns.tolist()
    rankings = rankings[['Name'] + sorted(cols[1:])]

    return rankings


def rankings_venn_diagram(rankings, 
                          t1, 
                          t2, 
                          max_size=10, 
                          ranking_type='raw'):
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
    # Get node specific rankings
    r1 = rankings[f'{cui2name[t1]}_{ranking_type}_rank']
    r2 = rankings[f'{cui2name[t2]}_{ranking_type}_rank']

    # Determine which sources are concept specific
    diff = r1 - r2

    # t1_concepts = diff[(r1 < r1.quantile(.2))].rank().sort_values().head(10)
    t1_concepts = (r1 + diff).sort_values().head(max_size)
    t2_concepts = (r2 - diff).sort_values().head(max_size)
    display(t1_concepts, t2_concepts)


    # And which are 
    intersection = (2 * np.abs(diff) + r1 + r2)
    intersection_concepts = intersection[intersection < 200].sort_values().head(max_size)
    display(intersection_concepts)

    concept_dict = {cui2name[t1]: set(t1_concepts.index.tolist() + intersection_concepts.index.tolist()),
                    cui2name[t2]: set(t2_concepts.index.tolist() + intersection_concepts.index.tolist())}

    return concept_dict



