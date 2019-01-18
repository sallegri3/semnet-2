import pickle
import threading
import concurrent.futures
import gzip
from tqdm import tqdm_notebook
# Avoid set size change warning
tqdm_notebook.monitor_interval=0
import copy

from semnet.conversion import get_metapath_abbrev

"""
Constructs queries compatible with neo4j and submits multithreaded jobs

Neo4j Data Fields and Example Formatting
Nodes:
- type: DiseaseOrSyndrome
- alt_counts: 184391,1,1
- alt_kinds: DSYN,PATF,ANAB
- identifier: C0002395
- kind: DiseaseOrSyndrome
- name: Alzheimer's Disease

Edges:**
- type: ASSOCIATED_WITH_AAPPaswtDSYN
- pmid: 8725894,8725894,8725894
- predicate: ASSOCIATED_WITH_AAPPaswtDSYN
- weight: 3
- SOURCE: glycogen synthase
- TARGET: Alzheimer's Disease
"""

def build_metapath_query(source, target, d):
	""" Generates a Cypher query string for all metapaths of length
	less than or equal to d """

	q = """
		MATCH path = (m:`{s_type}` {{identifier: '{source}'}})-
		[*..{d}]-(n:`{t_type}` {{identifier: '{target}'}}) 
		RETURN extract(a in nodes(path) | a.kind) as nodes, 
		extract(b in relationships(path) | b.predicate ) as edges 
		"""
    
	with gzip.open('../semnet/data/cui2type.pkl.gz', 'rb') as file:
		convert2type = pickle.load(file)

	s_type = convert2type[source]
	t_type = convert2type[target]

	format_dict = {'source': source, 
					'target': target,
					's_type': s_type,
					't_type': t_type,
					'd': d}

	return q.format(**format_dict).replace('\n', '').replace('\t', '')


def execute_multithread_query(func, params, workers=40):
	""" Executes a large number of Cypher queries simultaneously """

	# Transform params for mapping
	#transformed_params = copy.deepcopy([list(col) for col in zip(*[param.values() for param in params])])
	transformed_params = [list(col) for col in zip(*[param.values() for param in params])]

	# Submit jobs with ThreadPoolExecutor
	with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
		results = list(tqdm_notebook(executor.map(func, *transformed_params), total=len(params)))
	return results
