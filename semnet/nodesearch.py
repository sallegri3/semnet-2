'''
author: @davidkartchner
created April 20, 2020
'''
import lxml.html as lh
from lxml.html import fromstring
import requests
import json
from py2neo import Graph
from tqdm import tqdm
import pandas as pd
import numpy as np

# Get UMLS url
uri="https://utslogin.nlm.nih.gov"
auth_endpoint = "/cas/v1/api-key"

# Connect to Neo4j
graph = Graph(password='Mitch-Lin')

class Authentication:
    '''
    Class to handle authentication with the UMLS api and give you a ticket for each query 
    '''
    def __init__(self, apikey):
        self.apikey=apikey
        self.service="http://umlsks.nlm.nih.gov"

    def gettgt(self):
        params = {'apikey': self.apikey}
        h = {"Content-type": "application/x-www-form-urlencoded", 
             "Accept": "text/plain", 
             "User-Agent":"python" }
        r = requests.post(uri+auth_endpoint,
                          data=params,
                          headers=h)
        response = fromstring(r.text)
        ## extract the entire URL needed from the HTML form (action attribute) returned - looks similar to https://utslogin.nlm.nih.gov/cas/v1/tickets/TGT-36471-aYqNLN2rFIJPXKzxwdTNC5ZT7z3B3cTAKfSc5ndHQcUxeaDOLN-cas
        ## we make a POST call to this URL in the getst method
        tgt = response.xpath('//form/@action')[0]
        return tgt

    def getst(self, tgt):
        params = {'service': self.service}
        h = {"Content-type": "application/x-www-form-urlencoded", 
             "Accept": "text/plain", 
             "User-Agent":"python" }
        r = requests.post(tgt,
                          data=params,
                          headers=h)
        st = r.text
        return st

def get_cuis_for_concept(concept_name, num_results=1000, return_names=False):
    '''
    Get CUIs matching a query concept in UMLS
    
    Parameters:
    --------------------
        concept_name: str
            String of concept you want to search for.  Case insensitive 
            and resilient to misspelling.
        
        num_results: int
            Maximum number of possible matches you want to return.
            Default 1000.
            
        return_names: bool
            Whether to return names with CUIs of concept.  Default False.
            
    Returns:
    --------------------
        cuis: list or str
            List of CUIs for each potential UMLS concept match
            
        names: list of str (optional)
            List of names for each potential UMLS concept match.  Only 
            returned if return_names == True.
    '''
    # Authenticate
    API_KEY = '784e30e6-1938-47c6-8479-d1c9b654c704'
    VERSION = 'current'
    AuthClient = Authentication(API_KEY)

    # Get ticket and connect to API endopint
    tgt = AuthClient.gettgt()
    url = "https://uts-ws.nlm.nih.gov/rest/"
    content_endpoint = f'search/{VERSION}'

    # Set params for API query
    query = {'ticket':AuthClient.getst(tgt), 'string':concept_name, 'pageSize':num_results}
    
    # Grab results
    r = requests.get(url=url+content_endpoint, params=query)
    r.encoding = 'utf-8'
    items = json.loads(r.text)
    
    # Turn results into neat list
    if return_names:
        cuis = [res['ui'] for res in items['result']['results']]
        names = [res['name'] for res in items['result']['results']]
        return cuis, names
    else:
        cuis = [res['ui'] for res in items['result']['results']]
        return cuis

def filter_results_in_graph(cuis, return_dataframe=False):
    '''
    Filter UMLS concepts by what is in current knowledge graph
    
    Parameters:
    --------------------
        cuis: list of str
            List of CUIs you want to search for in the graph
            
        return_dataframe: bool
            Whether to return pandas dataframe of results. Default True.
            If false, returns a list of dicts.
    
    Returns:
    --------------------
        results: list of dict
            List of nodes contained in graph sorted by degree
            
        -- OR --
            
        data: pandas.DataFrame
            Datraframe of nodes contained in graph sorted by degree
    '''
    # Get only unique CUIs
    cuis = list(np.unique(cuis))

    # Query to match CUIs in graph
    query = """
    MATCH (a)
    WHERE a.identifier IN {}
    RETURN a.identifier as cui, a.name as name, a.kind as kind, size((a)-[]-()) as degree
    """.format(cuis)
    
    # Run query and get results with nonzero degree
    cursor = graph.run(query)
    results = sorted([res for res in cursor.data() if res['degree'] > 0], key=lambda x: x['degree'])[::-1]
    cursor.close()
    
    # Return output
    if return_dataframe:
        data = pd.DataFrame(results, columns=['cui','name','kind','degree'])
        return data
    else:
        return results
    
def match_concepts_by_name(name, return_dataframe=True):
    '''
    Get CUIs matching a query concept in UMLS
    
    Parameters:
    --------------------
        name: str
            String of concept you want to search for.  Case insensitive 
            and resilient to misspelling.
            
    Returns:
    --------------------
        results: list of dict
            List of nodes contained in graph sorted by degree
            
        -- OR --
            
        data: pandas.DataFrame
            Datraframe of nodes contained in graph sorted by degree
    '''
    cuis = get_cuis_for_concept(name, return_names=True)
    results = filter_results_in_graph(cuis, return_dataframe=True)
    return results