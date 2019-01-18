import re
import hetio.readwrite


"""
Converts data from the neo4j database into a format compatible with 
manipulation in hetio
"""

""" Load the metagraph. """
path = '../semnet/data/sem-net-mg_hetiofmt.json.gz'
metagraph = hetio.readwrite.read_metagraph(path)


def get_metapath_abbrev(query_result):
	""" Creates a string abbreviation to label query results """

	node_types = query_result['nodes']
	edge_types = query_result['edges']
	mp = neo4j_rels_as_metapath(edge_types, node_types)

	return str(mp)

def neo4j_rels_as_metapath(edge_types, node_types):
	""" Converts a list of typed relationship formats from Neo4j
	(e.g. 'TREATS_ORCHtrtsDSYN') to a MetaPath (e.g. ORCHtrts>DSYN).
	Note, directionality may be compromised when two sequential 
	nodes have the same type."""

	# Capture the abbreviation at the end of the edge type
	abbrevs = [re.split('_(?=[A-Z])', e)[-1] for e in edge_types]
	# Insert the directionality symbol
	all_matches = [re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|$)', a) for a in abbrevs]
	std_metaedge_abbrevs = ('>'.join([m.group(0) for m in matches]) for matches in all_matches)
	# Convert to MetaEdges
	metaedges = [metagraph.get_metaedge(e) for e in std_metaedge_abbrevs]

	# Check if the relationships need to be inverted - note that 
	# relationships are passed through by neo4j in the correct order,
	# but sometimes subject may need to be swapped with predicate 
	# for the metapath.
	for i, metaedge in enumerate(metaedges):
		if i==0:
			# Match node type to database formatted string
			metaedge_source = str(metaedge.source).title().replace(' ', '')
			if metaedge_source != node_types[0]: 
				metaedges[i] = metaedge.inverse
		else:
			if metaedge.source != metaedges[i-1].target:
				metaedges[i] = metaedge.inverse

	return metagraph.get_metapath(tuple(metaedges))