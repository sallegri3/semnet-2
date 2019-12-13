import xarray as xr
import gzip
from py2neo import Graph
import pandas as pd
import re
import os
import pickle
import seaborn as sns
import numpy as np
import requests
from fuzzywuzzy import process
from fuzzywuzzy import fuzz

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
    # Get metadata for each PMID
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