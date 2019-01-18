import xarray as xr
import numpy as np
from collections import Counter

import hetio.readwrite
import hetio.neo4j

from py2neo import Graph

from semnet.neo4j import build_metapath_query, execute_multithread_query
from semnet.conversion import get_metapath_abbrev

"""
This module implements three feature extractors that wrap the py2neo.Graph class.
When objects are constructed, they initiate a connection to the locally hosted 
Neo4j instance. Given pairs of nodes, feature extractors compute a representation
of the metapaths between them.
"""

class BaseFeatureExtractor(object):
	""" Defines the basic functions of a feature extractor """
	def __init__(self):
		self.graph = Graph(password='j@ck3t5_m1tch311')
		
	def results_to_dataarray(self, sources, targets, results, metric):
		""" Converts the results array of dicts into the structured xr.DataArray format """
		
		s2ix = {s:ix for ix, s in enumerate(sorted(set(sources)))}
		t2ix = {t:ix for ix, t in enumerate(sorted(set(targets)))}
		mp2ix = {mp:ix for ix, mp in enumerate(sorted({metapath for r in results for metapath in r.keys()}))}
		
		data = np.zeros((len(s2ix), len(t2ix), len(mp2ix), 1))
		
		for s, t, mps in zip(sources, targets, results):
			for mp, value in mps.items():
				data[s2ix[s], t2ix[t], mp2ix[mp], 0] = value
		
		s_type = list(results[0].keys())[0][:4]
		t_type = list(results[0].keys())[0][-4:]
		
		data = xr.DataArray(data,
							coords=[sorted(s2ix.keys()), sorted(t2ix.keys()), sorted(mp2ix.keys()), [metric]],
							dims=['source', 'target', 'metapath', 'metric'],
							attrs={'s_type':s_type, 't_type':t_type})

		return data
		

class CountExtractor(BaseFeatureExtractor):
	""" Extracts metapath counts between a pair of nodes """

	def get_metapath_counts(self, source, target, d):
		""" Gets metapath counts from a source node to a target node"""

		query = build_metapath_query(source, target, d)
		cursor = self.graph.run(query)
		query_results = cursor.data()
		cursor.close()

		return Counter([get_metapath_abbrev(r) for r in query_results])


	def get_all_metapath_counts(self, sources, targets, d, workers=40):
		""" Runs Cypher queries to count metapaths for all examples """
		
		# Retrieve the results from Neo4j
		params = []
		for i, (s, t) in enumerate(zip(sources, targets)):
			params.append({'source': s, 'target': t, 'd': d})
			
		result = execute_multithread_query(self.get_metapath_counts, params=params, workers=workers)
		
		# Remembering which metapaths are nonzero helps with computational efficiency for dwpc
		self.metapath_counts = result
		
		return self.results_to_dataarray(sources, targets, result, 'count')
	
	
	
class DwpcExtractor(BaseFeatureExtractor):
	"""
	Degree-Weighted Path Counts (DWPC) are an alternative to simple 
	metapath counts that downweights paths with highly connected nodes.
	-------
	Reference:
	Himmelstein, Daniel S., and Sergio E. Baranzini. "Heterogeneous 
	network edge prediction: a data integration approach to prioritize 
	disease-associated genes." PLoS computational biology 11.7 (2015): 
	e1004259.
	"""
	
	def __init__(self):
		""" Load the metagraph and connect to Neo4j """
		
		path = '../semnet/data/sem-net-mg_hetiofmt.json.gz'
		self.metagraph = hetio.readwrite.read_metagraph(path)
		super(DwpcExtractor, self).__init__()
	
	def compute_dwpc(self, source, target, metapath, damping):
		""" Performs a DWPC calculation for a source/target pair along a
		single metapath """

		metapath = self.metagraph.get_metapath(metapath)
		query = hetio.neo4j.construct_dwpc_query(metapath, 'identifier')

		params = {
			'source': source,
			'target': target,
			'w': damping
		}

		cursor = self.graph.run(query, params)
		query_results = cursor.data()
		cursor.close()

		return query_results


	def  compute_example_dwpc(self, source, target, metapaths, damping):
		""" Performs all DWPC calculations between a given pair of nodes """

		dwpcs = dict()
		for mp in metapaths:
			result = self.compute_dwpc(source, target, mp, damping)
			dwpcs[mp] = result[0]['DWPC']

		return dwpcs



	def get_all_dwpc(self, sources, targets, d, damping, metapath_counts=None, workers=40):
		""" Performs all DWPC calculations for all example pairs """
		
		if not metapath_counts.any():
			metapath_counts = get_all_metapath_counts(sources, targets, d, workers)
		assert isinstance(metapath_counts, xr.DataArray)
		
		#metapaths = [list(mps.keys()) for mps in metapath_counts]
		metapaths = list()
		
		params = []
		for i,(s, t) in enumerate(zip(sources, targets)):
			nz_metapath_ix = metapath_counts.loc[s, t, :, 'count'].values.nonzero()
			nz_metapaths = metapath_counts.metapath.values[nz_metapath_ix]
			params.append({'source': s, 'target': t, 'metapaths': nz_metapaths, 'damping': damping})

		result = execute_multithread_query(self.compute_example_dwpc, params=params, workers=workers)

		return self.results_to_dataarray(sources, targets, result, 'dwpc')
	
	
#class HetesimExtractor(BaseFeatureExtractor):
#	"""
#	"""
