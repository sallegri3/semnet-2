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
        ranks = np.array(pd.Series(row).rank(method='dense', ascending=False))

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
def get_all_scores(features, cuis, metric='hetesim',lambd=1,theta_frac=.05):
    # Make rank aggregator
    ura = UnsupervisedRankAggregator(features, cui2name)

    # Get rankings in terms of each CUI
    n_feats = features.metapath.shape[0]
    rankings = []
    for cui in tqdm(cuis):
        ura.aggregate(metric, cui, lambd=lambd, theta=theta_frac*n_feats)
        curr_rankings = pd.Series(ura.get_scores(cui_key=True), name=cui2name[cui])
        curr_rankings -= curr_rankings.min()
        curr_rankings /= curr_rankings.max()
        rankings.append(curr_rankings)

    rankings_df = pd.concat(rankings, axis=1, sort=False)
    for col in [cui2name[cui] for cui in cuis]:
        rankings_df[col+' rank'] = rankings_df[col].rank()
    rankings_df = rankings_df[sorted(rankings_df.columns.tolist())]
    rankings_df['avg_score'] = rankings_df[[cui2name[cui] for cui in cuis]].mean(axis=1)
    rankings_df['avg_rank'] = rankings_df['avg_score'].rank()

    # Get rankings for all of the individual target columns
    for col in [cui2name[cui] for cui in cuis]:
        rankings_df[col+' rank'] = rankings_df[col].rank()

    # Get overall ranking and add to df
    if len(cuis) > 1:
        ura.aggregate(metric, cuis, lambd=lambd, theta=len(cuis)*theta_frac*n_feats)
        overall_score = pd.Series(ura.get_scores(cui_key=True), name='overall_score')
        overall_score -= overall_score.min()
        overall_score /= overall_score.max()
        rankings_df = pd.concat([rankings_df, overall_score], axis=1, sort=False)
        rankings_df['overall_rank'] = rankings_df['overall_score'].rank()

    # Add medical names for each term and reorder so this is first column
    rankings_df['Name'] = rankings_df.index.map(cui2name)
    cols = rankings_df.columns.tolist()
    rankings_df = rankings_df[['Name'] + cols[:-1]]
    rankings_df.index.name = 'CUI'
    return rankings_df
