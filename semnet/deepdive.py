import numpy as np
import pandas as pd
from py2neo import Graph
import xarray as xr
import os
from semnet.utils import parse_metapath, merge_path_segments, filter_pmids
import requests
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import re
from pandarallel import pandarallel

# Init pandas parallel processing
pandarallel.initialize(progress_bar=True)

# Load Graph
graph = Graph(password='Mitch-Lin')
key = 'da79e005a4869dfef15528a82feb069ee908'

# Load dict to convert node CUI to node type
_ROOT = os.path.abspath(os.path.dirname(__file__))

def get_paths_in_metapath(target, source, metapath, max_pmids=25, impact_factor_cutoff=1):
    '''
    Get list of all paths making up a particular metapath, along with PMIDs 
        from which each segment was taken.

    Inputs:
        target: str
            CUI of target node that starts metapath

        source: str
            CUI of source node at end of metapath

        metapath:
            String representation of metapath from source to target

        max_pmids: int
            Maximum number of PMIDs to return for a given edge

        impact_factor_cutoff: float
            Minimum impact factor for articles.  Articles published in journals with 
                impact factor below this cutoff are not returned.       

    Returns:
        paths: list
            List of paths making up metapath

        pmids: list
            List of PMIDs from which each path was extracted
    '''

    # Get info from metapath
    node_types, edge_types, directions, relationships = parse_metapath(metapath,
                                                        return_directions=True,
                                                        return_relationships=True)
    

    # Query graph to get all paths
    # Get string of all relationships
    m = len(edge_types)
    rels = f"[r0:{relationships[0]}]"

    if m > 1:
        for i in range(1, m):
            rels += f" - (n{i}:{node_types[i]}) - [r{i}:{relationships[i]}]"

    # Get string of all things that need to be returned
    ret_nodes = ", ".join([f"n{i}.name as n{i}" for i in range(m+1)])
    ret_pmid = ", ".join([f"r{i}.pmid[0] as pmid{i}" for i in range(m)])

    # Assemble into query
    q = (
        f"MATCH (n0:{node_types[0]} {{identifier: '{source}'}}) - {rels} - (n{m}:{node_types[-1]} {{identifier: '{target}'}}) "
        f"RETURN {ret_nodes}, {ret_pmid}"
    )


    # Get query results
    cursor = graph.run(q)
    results = pd.DataFrame(cursor.data())
    cursor.close()


    # Add in edge info
    metapath = merge_path_segments(node_types,
                                   edge_types,
                                   directions)
    pattern = re.compile("['n\d+']")
    cols = [c for c in results.columns if bool(pattern.match(c))]

    results['relationship'] = results[cols].apply(merge_path_segments, args=(edge_types, directions), axis=1)
    results['meta_relationship'] = metapath
    results['endpoint'] = results[f'n{m}']
    results['gene_or_aapp'] = results[f'n0']

    # Get top PMIDs for each segment
    for i in range(m):
        results[f'seg_{i+1}_pmids'] = results[f'pmid{i}'].apply(lambda x: filter_pmids(x.split(','), max_return=max_pmids, impact_cutoff=impact_factor_cutoff))
    
    cols = ['endpoint','gene_or_aapp','meta_relationship','relationship'] + [f'seg_{i}_pmids' for i in range(1, m+1)]
    return results[cols]