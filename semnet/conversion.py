"""
Converts data from the Neo4j database into a format compatible with manipulation
in hetio. Hetio provides simple operations on the metagraph.
"""

import re
import pickle
import hetio.readwrite
from hetio.hetnet import MetaEdge, direction_to_inverse



""" Load the metagraph. """
import os
_ROOT = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(_ROOT, 'data/sem-net-mg_hetiofmt.json.gz')
metagraph = hetio.readwrite.read_metagraph(path)

# Load file to convert abbreviated nodetypes to long-form nodetypes
# Necessary for fix after updating the graph with CORD-19 data
nodeabbr_path = os.path.join(_ROOT, 'data/abbr2nodetype.pkl')
abbr2nodetype = pickle.load(open(nodeabbr_path, 'rb'))


def get_metapath_abbrev(query_result):
    """ 
    Creates a string abbreviation to label query results.

    Converts a dictionary of an individual Neo4j query results into a simplified
    and unique string representation.

    Parameters
    ----------
    query_result: dict
        The result of a query for metapaths between a source and target node.

    Returns
    -------
    metapath: str
        A string representation of the metapath, which we use to refer to it 
        throughout semnet.
    """

    node_types = query_result['nodes']
    edge_types = query_result['edges']
    mp = neo4j_rels_as_metapath(edge_types, node_types)

    return str(mp)

def neo4j_rels_as_metapath(edge_types, node_types):
    """ 
    Converts lists of edge and node types to a ``hetio.hetnet.MetaPath`` object.

    Uses regular expressions to capture the appreviation at the end of the edge
    type and insert directionality symbol. Using this method, a list of strings
    formatted as ``TREATS_ORCHtreatsDSYN`` in Neo4j become MetaPath objects.
    Note, directionality may become compromised when two sequential nodes have
    the same type.

    Parameters
    ----------
    edge_types: array_like 
        A sequence of edge type strings. 
    
    node_types: array_like 
        A sequence of node type strings.

    Returns
    -------
    metapath: hetio.hetnet.MetaPath 
        A MetaPath object that represents the sequence of nodes and edges.
    """
    node_types = [abbr2nodetype[n.upper()] if n.upper() in abbr2nodetype else n for n in node_types]
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

    # output_str = "Node Types: " + str(node_types) + '\n'
    # output_str += "Metaedges:" + "\n\t".join([str(i) for i in metaedges]) + '\t'
    for i, metaedge in enumerate(metaedges):
        if i==0:
            # Match node type to database formatted string
            metaedge_source = str(metaedge.source).title().replace(' ', '')
            if metaedge_source != node_types[0]:
                # output_str += "First block\n"
                # output_str += f'{metaedge}\t{metaedge.inverse}\n'.replace(' ', '')
                metaedges[i] = metaedge.inverse
                # metaedges[i] = MetaEdge(metaedge.source, metaedge.target, )
        else:
            # Make sure we align endpoints of each path segment
            s = str(metaedge.source).title().replace(' ', '')
            t = str(metaedges[i-1].target).title().replace(' ', '')
            if s != t:
                # output_str += 'Second block\n'
                # output_str += f'{metaedge}\t{metaedge.inverse}\n'.replace(' ', '')
                metaedges[i] = metaedge.inverse
        metapath = metagraph.get_metapath(tuple(metaedges))
        metapath_str = str(metapath)
    # output_str += metapath_str +'\n'
    # output_str += str(tuple([str(i).replace(' ','') for i in metaedges]))
    # output_str += '\n******\n'
    # if not metapath_str.startswith('BACS') or not metapath_str.endswith('DSYN'):
    #     with open("../debug/metaedge_output.txt", 'a') as f:
    #         f.write(output_str)
    return metapath