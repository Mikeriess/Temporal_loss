#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 22:35:04 2022

@author: mikeriess
"""
if __name__ == '__main__':
    
    import time
    import pandas as pd
    
    #from multiprocessing_helpers import make_eventlog, make_df_list, process_single_df
    import multiprocessing_helpers as fxs

    """
    ##############################################################################
    #       Run the experiment
    ##############################################################################
    """
    
    print("single core")
    
    df = fxs.make_eventlog(n_cases=10000, min_trace_length=3, max_trace_length=100)
    #dfdict = make_df_dict(df)
    dflist = fxs.make_df_list(df)
    
    print("n_rows:",len(df))
    """
    Without multiprocessing
    """
    start_time = time.time()
    df_prefix = fxs.make_prefix(df)
    
    df_prefix.to_csv("single_core.csv",index=False)

    print("n_output_rows:",len(df_prefix))
    
    print((time.time() - start_time),"seconds")
    print(df_prefix.tail())
    
    """
    Multiprocessing
    """
    print("multicore")
    print("n_rows:",len(df))
    
    import multiprocessing
    pool = multiprocessing.Pool(16)
            
    #print("input length",len(df))
    
    #list_of_results = pool.imap_unordered(fxs.make_prefix, dflist)
    list_of_results = pool.map(fxs.make_prefix, dflist)
    
    df_prefix = pd.concat(list_of_results)
    
    start_time = time.time()
    df_prefix.to_csv("multi_core.csv",index=False)
    print("n_output_rows:",len(df_prefix))
    
    print((time.time() - start_time),"seconds")
    print(df_prefix.tail())
    
    
    #print("output length",len(pd.concat(list_of_results)))
    