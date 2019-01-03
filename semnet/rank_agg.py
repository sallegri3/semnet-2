"""
This module implements a modified version of rank aggregation described in:

Klementiev, Alexandre, Dan Roth, and Kevin Small. "An unsupervised learning 
algorithm for rank aggregation." European Conference on Machine Learning. 
Springer, Berlin, Heidelberg, 2007.
"""

import numpy as np
import pandas as pd
import xarray as xr

class UnsupervisedRankAggregator(object):
	"""
	This class implements the rank aggregation methods described by 
	Klementiev et al. 2007. It wraps a xr.DataArray object for data 
	management.
	"""
	def __init__(self, data_array, cui2name):
		""" Initializes the URA using characteristics of the xr.DataArray """
		assert isinstance(data_array, xr.DataArray), 'Constructor requires xr.DataArray.'

		self.data_array = data_array
		self.data = self.data_array.values
		self.sources = self.data_array.source.values
		self.targets = self.data_array.target.values
		self.metrics = self.data_array.metric.values
		self.metapaths = self.data_array.metapath.values

		# Fetch the verbose names
		#int_rf_df = pd.read_csv('data/impd_cogn_sources.tsv', delimiter='\t')
		#cui2name = {cui:name for cui, name in zip(int_rf_df.identifier, int_rf_df.name)} 
		self.source_names = np.array([cui2name[source] for source in self.sources])


	def aggregate(self, metric, target, lambd=1, theta=500):
		""" Learn weights that prioritize features that agree with the mean rank. Implements
		the additive aggregation method described by Klementiev. """

		# Check that target and metric are valid and store them in memory
		assert metric in self.metrics, 'Metric is invalid: %r' % metric
		#assert target in self.targets, 'Target is invalid: %r' % target
		
		"""
		TODO: Accomodate multi-target aggregations
		
		assert isinstance(target, list)
		assert target.all() in self.targets, 'One or more targets are invalid: %r' % target
		
		if len(target) > 1:
			self.agg_target = "_".join(target)
			data_array = self.data_array
			#.assign_coords(target=self.agg_target).expand_dims('target')
		else:
			self.agg_target = target
			data_array = self.data_array
		"""
		
		
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

	def get_scores(self):
		""" Returns a dictionary of source scores based on the learned ranking model. """

		scores = np.dot(self.rankings, self.weights)
		ranked_indices = np.argsort(scores)
		ranked_sources = self.source_names[ranked_indices]
		ranked_scores = sorted(scores)
		self.scores = {source:score for source, score in zip(ranked_sources, ranked_scores)}

		return self.scores



def rank_by_feature(all_data, target, metric):
	""" This function assigns each item a natural number ranking by feature, 
	based on feature values. Ties are resolved by giving both numbers the highest
	possible rank. Thresholds are set to exclude items with zero-valued features. """

	rankings = []
	thresholds = []
	for row in all_data.loc[:, target, :, metric].values.T:
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