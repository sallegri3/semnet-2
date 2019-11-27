import numpy as np
import pandas as pd
from py2neo import Graph
import xarray as xr
from utils import parse_metapath, merge_path_segments
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

def get_paths_in_metapath(target, source, metapath):
    '''
    Get list of all paths making up a particular metapath

    Inputs:
        target: str
            CUI of target node that starts metapath

        source: str
            CUI of source node at end of metapath

        metapath:
            String representation of metapath from source to target

    Returns:
        paths: list
            List of paths making up metapath

        pmids: list
            List of PMIDs from which each path was extracted
    '''

    # Get info from metapath
    node_types, edge_types, directions = parse_metapath(metapath,
                                                        return_directions=True)

    # Query graph to get all paths
    # Get string of all relationships
    m = len(edge_types)
    rels = f"[r0:{edge_types[0]}]"

    if m > 1:
        for i in range(1, m):
            rels += f" - (n{i}:{node_types[i]}) - [r{i}:{edge_types[i]}]"

        # Get string of all things that need to be returned
        ret_nodes = ", ".join([f"n{i}.name as n{i}" for i in range(m+1)])
        ret_pmid = ", ".join([f"r{i}.pmid[0] as pmid{i}" for i in range(m)])

    # Assemble into query
    q = (
        f"MATCH (n0:{node_types[0]} {{identifier: '{target}'}}) - {rels} - (n{m}:{node_types[-1]} {{identifier: '{source}'}}) "
        f"RETURN {ret_nodes}, {ret_pmid}"
    )

    # Get query results
    cursor = graph.run(q)
    results = pd.DataFrame(cursor.data())
    cursor.close()

    # Add in edge info
    metapath = merge_path_segments([target, node_types[1:-1], source],
                                   edge_types,
                                   directions)
    pattern = re.compile(['n\d+'])
    cols = [c for c in results.columns if bool(pattern.match(c))]
    paths = results[cols].apply(merge_path_segments, args=(edge_types, directions))
