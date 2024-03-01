#################################################################################
# usage of the script
# usage: python search-terms.py -k APIKEY -v VERSION -s STRING
# see https://documentation.uts.nlm.nih.gov/rest/search/index.html for full docs
# on the /search endpoint
#################################################################################

from __future__ import print_function
from Authentication import *
import requests
import json
import argparse

def search_term(string, page_limit = 2, check = False):
    apikey = '6119ca85-39ff-4649-aa79-6c1b1d281a02'
    version = '2018AA'
    string = string
    uri = "https://uts-ws.nlm.nih.gov"
    content_endpoint = "/rest/search/"+version
    ##get at ticket granting ticket for the session
    AuthClient = Authentication(apikey)
    tgt = AuthClient.gettgt()
    pageNumber=0
    res = []

    while True:
        ##generate a new service ticket for each page if needed
        ticket = AuthClient.getst(tgt)
        pageNumber += 1
        query = {'string':string,'ticket':ticket, 'pageNumber':pageNumber}
        r = requests.get(uri+content_endpoint,params=query)
        r.encoding = 'utf-8'
        items  = json.loads(r.text)
        jsonData = items["result"]
        #print (json.dumps(items, indent = 4))s

        #print("Results for page " + str(pageNumber)+"\n")
        if jsonData["results"][0]["ui"] == "NONE" or pageNumber > page_limit:
            break
        for result in jsonData["results"]:
            
            if check:
                try:
                    res.append((result["ui"],result["name"]))
                except:
                    NameError
            else:
                try:
                    res.append(result["ui"])
                except:
                    NameError
        
    
                ##Either our search returned nothing, or we're at the end
        
    return res
    
    
    
    
    

