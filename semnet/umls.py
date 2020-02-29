'''
Created by: David Kartchner
Date: Jan 16, 2020
Contact: david.kartchner@gatech.edu

Utilities to connect to UMLS REST API and extract relevant information
'''

from Authentication import *
import requests
import json

def get_cuis_for_concept(concept_name):
    '''
    Get CUIs matching a query concept in UMLS
    '''
    API_KEY = '784e30e6-1938-47c6-8479-d1c9b654c704'
    VERSION = 'current'
    AuthClient = Authentication(API_KEY)

    tgt = AuthClient.gettgt()
    url = "https://uts-ws.nlm.nih.gov/rest/"

    content_endpoint = f'search/{VERSION}'

    query = {'ticket':AuthClient.getst(tgt), 'string':concept_name, 'retmode':'json'}

    params = {"db": 'pubmed', 'id': pmid, 'retmode':'json', 'api_key': key}
    r = requests.get(url=url, params=query)

    data = r.json()
    return data
