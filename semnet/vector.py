import numpy as np

"""
These functions turn sets of counters and weighted counts into vectors
"""

def vectorize_example(counts, mp2ix):
	""" Turns a Counter of metapaths from an example query into a 
	unit vector """
	vec = np.zeros(len(mp2ix))
	for mp, ct in counts.items():
		ix = mp2ix[mp]
		vec[ix] = ct

	norm = np.linalg.norm(vec) if np.linalg.norm(vec) != 0 else 1

	return vec / norm

def vectorize_results(counters_list):
	""" Turns a list of metapath counters into a numpy array """
	mp_count_mat = [None]*len(counters_list)
	mp_vocab = get_mp_vocab(counters_list)
	mp2ix = {mp:ix for ix, mp in enumerate(mp_vocab)}
	for i, ex in enumerate(counters_list):
		mp_count_mat[i] = vectorize_example(ex, mp2ix)

	return np.array(mp_count_mat), mp2ix

def get_mp_vocab(query_data):
	""" Returns an ordered list of metapaths that occurred in all queries """
	mp_vocab = {mp for ex in query_data for mp in list(ex.keys())}

	return sorted(list(mp_vocab))
