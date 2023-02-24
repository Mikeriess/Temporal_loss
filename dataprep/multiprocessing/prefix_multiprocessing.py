#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 22:35:04 2022

@author: mikeriess
"""

def flatten(l):
    return [item for sublist in l for item in sublist]

import pandas as pd
import random


caseid = []
eventno = []

for i in range(0,3):
    
    n_events = random.randint(0, 10)
    
    caseid.append([i]*n_events)
    eventno.append(list(range(0,n_events)))
    

df = pd.DataFrame({"caseid":flatten(caseid),
                  "eventno":flatten(eventno)})


def make_prefix(df):
    import pandas as pd
    
    def flatten(l):
        return [item for sublist in l for item in sublist]
    
    #list of results
    res = []
    
    #get ids
    caseids = df.caseid.unique().tolist()
    
    pfx_caseids = []
    pfx_eventids = []
    
    for idx in caseids:
        #do something
        print("case",idx)
        
        #subset on trace
        trace = df.loc[df.caseid == idx]
        
        #get ids
        eventids = list(range(0,len(trace)))
        
        #progress list
        progress = []
        
        #for each event
        for eventid in eventids:
            print(eventid)
            
            #placeholder
            to_append = []
            
            #update progress: values to add
            progress.append(eventid)
            
            #variables
            pfx_caseids.append([idx])
            pfx_eventids.append(progress)
            
            # if eventid > 0:
            #     pfx_eventids.append()
            # else:
            #     pfx_eventids.append(eventid)
            
    
    #result df
    pfx_caseids = flatten(pfx_caseids)
    #pfx_eventids = flatten(pfx_eventids)
    
    print(len(pfx_caseids))
    print(len(pfx_eventids))
    
    df_prefix = pd.DataFrame({"caseid":pfx_caseids,
                      "eventno":pfx_eventids})
    
    return df_prefix

df_prefix = make_prefix(df)
print(df_prefix)