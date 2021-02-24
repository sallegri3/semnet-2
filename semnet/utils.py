import xarray as xr
import gzip
import pandas as pd
import re
import os
import pickle
import seaborn as sns
import numpy as np
import requests

from py2neo import Graph
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from tqdm.auto import tqdm

# Load Graph
graph = Graph(password='Mitch-Lin')

# Load dict to convert node CUI to node type
_ROOT = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(_ROOT, 'data/cui2type.pkl.gz')
with gzip.open(path, 'rb') as file:
    convert2type = pickle.load(file)

abbr2node = pickle.load(open(os.path.join(_ROOT, 'data/abbr2nodetype.pkl'),'rb'))
abbr2edge = pickle.load(open(os.path.join(_ROOT, 'data/abbr2edgetype.pkl'),'rb'))


temp = {key.upper():val for key, val in abbr2node.items()}
abbr2node.update(temp)

cui2name = pickle.load(open(os.path.join(_ROOT, 'data/cui2name.pkl'),'rb'))

all_journals = pickle.load(open(os.path.join(_ROOT, 'data/alljournals.pkl.gz'),'rb'))

journal2impactfactor = pickle.load(open(os.path.join(_ROOT, 'data/journal2impactfactor.pkl.gz'),'rb'))
# print(all_journals)


key = 'da79e005a4869dfef15528a82feb069ee908'

def metapath_to_english(metapath):
    nodes, edges, directions = parse_metapath(metapath, return_directions=True)
    return merge_path_segments(nodes, edges, directions)


def check_is_reversed(start_type, end_type, edge):
    '''
    Determine direction of an edge
    '''
    node_pat = re.compile('[A-Z]+')
    endpoint_types = [abbr2node[n] for n in node_pat.findall(edge.split('_')[-1])]
    reversed = False
    if start_type != endpoint_types[0]:
        if start_type != endpoint_types[1]:
            raise Exception("Types don't match up with edge")
        else:
            reversed = True
    return reversed


def parse_metapath(metapath, return_directions=False, return_relationships=False):
    '''
    Extract metapath details from metapath abbreviation that can be used to
    query Neo4j Graph

    Inputs:
        metapath: str
            String abbreviation of metapath

    Returns:
        node_types: list
            List of node types of metapath.  Has same order as in metapath.

        edge_types: list
            List of edge types in metapath.  Has same order as in metapath.
    '''
    # Patterns to find desired pieces
    node_pat = re.compile('[A-Z]+')
    edge_pat = re.compile('[a-z]+')
    dir_pat = re.compile('[<>]')

    # Pieces parsed from metapath
    nodes = node_pat.findall(metapath)
    edges = edge_pat.findall(metapath)
    directions = dir_pat.findall(metapath)

    # Get long-form representation of each node and edge
    node_types = [abbr2node[n] for n in nodes]
    edge_types = [abbr2edge[e] for e in edges]

    m = len(edges)
    relationships = [f'{edge_types[i]}_{nodes[i]}{edges[i]}{nodes[i+1]}'  if directions[i] == '>' else f'{edge_types[i]}_{nodes[i+1]}{edges[i]}{nodes[i]}' for i in range(m)]

    results = [node_types, edge_types]

    if return_directions:
        results.append(directions)

    if return_relationships:
        results.append(relationships)

    return results


def merge_path_segments(nodes, edges, directions):
    path = nodes[0]
    for (n, e, d) in zip(nodes[1:], edges, directions):
        # print(d, e, n)
        path += d + e + d + n
    return path


def get_pmid_info(pmid):
    # This is simpler with entrez module
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {"db": 'pubmed', 'id': pmid, 'retmode':'json', 'api_key': key}
    r = requests.get(url=url, params=params)

    data = r.json()
    return data

def find_journal_match(journal_name, all_journals=all_journals, thresh=90):
    match, score = process.extractOne(journal_name, all_journals, scorer=fuzz.ratio)
    if score > thresh:
        return match
    else:
        return None

def get_article_info(pmid, relationship_str=None):
    '''
    Pulls metadata for a given PMID
    '''
    
    # Get metadata for each PMID
    if pmid == 'covid':
        return None

    data = get_pmid_info(pmid)['result'][str(pmid)]

    # Get publication info
    try:
        date = data['pubdate'][:4] # epubdate or pubdate?
        title = data['title']

        # Get journal name and do some postprocessing
        journal = data['fulljournalname']
        journal = journal.replace("&amp;", '&')

        # Find impactfactor
        journal_match = find_journal_match(journal)

        # Notate relationship to which these articles correspond
        if relationship_str:
            curr_results = [relationship_str]
        else:
            curr_results = []

        if journal_match:
            impact_factor = journal2impactfactor[journal_match]
            return curr_results + [pmid, title, journal, date, impact_factor]
        else:
            return None
    except:
        print(data)
        return None


def filter_pmids(pmids, impact_cutoff=2, date_cutoff=1970, max_return=None):

    # Get info on each article
    info = [get_article_info(pmid) for pmid in pmids]
    info = [x for x in info if x is not None]

    # Filter by journals that are relatively recent and credible
    data = pd.DataFrame(info, columns=['pmid','title','journal','date','impact_factor'])
    data['date'] = data['date'].astype(int)
    # data['impact_factor'] = data['impact_factor'].astype(float)
    filtered = data.query(f'(date >= {date_cutoff}) & (impact_factor >= {impact_cutoff})')
    sorted_pmids = filtered.sort_values(by='impact_factor',ascending=False)

    # Return number top n pmids, joined with semicolons
    if max_return is not None:
        return ';'.join(sorted_pmids.head(max_return).pmid.tolist())

    else:
        return ';'.join(sorted_pmids.pmid.tolist())


def filtered_pmid_dataframe(pmids, impact_cutoff=2, date_cutoff=1970, sort_col='date'):
    '''
    Create dataframe of info on PMIDs for articles satisfying impact factor 
    and publication date cutoffs
    '''
    # Get info on each article
    info = [get_article_info(pmid) for pmid in tqdm(pmids)]
    filtered_info = [x for x in info if x is not None]

    # Filter by journals that are relatively recent and credible
    data = pd.DataFrame(filtered_info, columns=['pmid','title','journal','date','impact_factor'])
    data['date'] = data['date'].astype(int)
    # data['impact_factor'] = data['impact_factor'].astype(float)
    filtered = data.query(f'(date >= {date_cutoff}) & (impact_factor >= {impact_cutoff})')
    sorted_pmids = filtered.sort_values(by=sort_col,ascending=False)

    return sorted_pmids


def get_pmids_for_node(target, source=None):
    '''
    Get PMIDs corresponding to a particular node (or node pair) in Neo4j

    Calculates this as union of PMIDs for all edges connected to a node, or 
    union of PMIDs between two nodes for a pair
    '''
    
    # Make param dict
    t_type = convert2type[target]

    if source is not None:
        s_type = convert2type[source]
        param_dict = {'target':target,
                        't_type':t_type,
                        'source':source,
                        's_type':s_type}

        # Make query to get vis results
        q = """
        MATCH (a:{t_type} {{identifier: '{target}'}}) - [r] - (b:{s_type} {{identifier: '{source}'}})
        RETURN r.weight as count, r.predicate as relationship, r.pmid as pmid
        """.format(**param_dict)
    
    else:
        param_dict = {'target':target,
                        't_type':t_type}

        # Make query to get vis results
        q = """
        MATCH (a:{t_type} {{identifier: '{target}'}}) - [r] - (b)
        RETURN r.weight as count, r.predicate as relationship, r.pmid as pmid
        """.format(**param_dict)

    # Pull data
    cursor = graph.run(q)
    results = pd.DataFrame(cursor.data())
    cursor.close()

    all_pmids = list(set([pmid for pmid_list in results.pmid.map(lambda x: 
                 x[0].split(',')) for pmid in pmid_list]))

    return all_pmids
