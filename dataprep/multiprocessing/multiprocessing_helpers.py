#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 22:35:04 2022

@author: mikeriess
"""
    
import time
import pandas as pd
import numpy as np

def flatten(l):
    return [item for sublist in l for item in sublist]
    
def make_eventlog(n_cases=5, min_trace_length=3, max_trace_length=10):
    def flatten(l):
        return [item for sublist in l for item in sublist]

    import pandas as pd
    import random
        
    caseid = []
    eventno = []
    
    for i in range(0,n_cases):
        
        n_events = random.randint(min_trace_length, max_trace_length)
        
        caseid.append([i]*n_events)
        eventno.append(list(range(0,n_events)))
            
    df = pd.DataFrame({"id":flatten(caseid),
                      "eventno":flatten(eventno)})
    return df


def make_prefix(df):
    import pandas as pd
    
    def flatten(l):
        return [item for sublist in l for item in sublist]
        
    #get ids
    caseids = df.id.unique().tolist()
    
    #placeholders for columns
    pfx_caseids = []
    pfx_eventids = []
    
    for idx in caseids:
        #do something
        #print("case",idx)
        
        #subset on trace
        trace = df.loc[df.id == idx]
        
        #get ids
        eventids = list(range(0,len(trace)))
        
        #progress list: reset for each case
        progress = []
        
        #for each event
        for eventid in eventids:
            #print(eventid)
            
            #placeholder
            to_append = []
            
            #update progress: values to add
            progress.append(eventid)
            
            caseids_to_add = [idx]*len(progress)
            eventids_to_add = progress.copy()
            
            #Debug
            #print("caseid:",len(caseids_to_add),"eventid:",len(eventids_to_add))
            
            #variables
            pfx_caseids.append(caseids_to_add)
            pfx_eventids.append(eventids_to_add)
            
            
    #Create result df
    pfx_caseids = flatten(pfx_caseids)
    pfx_eventids = flatten(pfx_eventids)
        
    df_prefix = pd.DataFrame({"id":pfx_caseids,
                      "eventno":pfx_eventids})
    
    return df_prefix


def make_df_dict(data):
    
    #create unique list of names
    UniqueNames = data.caseid.unique()
    
    #create a data frame dictionary to store your data frames
    DataFrameDict = {elem : pd.DataFrame() for elem in UniqueNames}
    
    for key in DataFrameDict.keys():
        DataFrameDict[key] = data[:][data.caseid == key]
    
    return DataFrameDict


def make_df_list(df, idvar="id"):
    
    dflist = [v for k, v in df.groupby(idvar)]
    
    return dflist


def process_single_df(df):
    """
    Function that processes a single df.
    """
    # for column_name in need_to_change_column_name:
    #     # some column name changes
    #     ...

    df.set_index('id', inplace=True)

    ## dropping any na
    df = df.dropna()
    ...

    df['cost'] = df['eventno'] * 100

    return df
    
# """
# ##############################################################################
# #       Run the experiment
# ##############################################################################
# """

# df = make_eventlog(n_cases=10, min_trace_length=3, max_trace_length=10)
# #dfdict = make_df_dict(df)
# dflist = make_df_list(df)

# """
# Without multiprocessing
# """
# start_time = time.time()
# df_prefix = make_prefix(df)
# print((time.time() - start_time),"seconds")
# #print(df_prefix.head())

# """
# Multiprocessing
# """

# # from multiprocessing import Pool

# # with Pool() as pool:
# #     results = pool.imap_unordered(make_prefix, dflist)

# #     for df_i, duration in results:
# #         print(duration)

# import multiprocessing
# pool = multiprocessing.Pool(5)

# #list_of_results = pool.map(make_prefix, (df for df in dflist))

# #### list_of_results = pool.map(make_prefix, dflist)


# list_of_results = pool.map(process_single_df, dflist)