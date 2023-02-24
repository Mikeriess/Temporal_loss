# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:18:08 2022

@author: Mike
"""

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
    import dataprep.multiprocessing.multiprocessing_helpers as fxs

    """
    ##############################################################################
    #       Run the experiment
    ##############################################################################
    """
    
    print("make data")
    
    df = fxs.make_eventlog(n_cases=10000, min_trace_length=3, max_trace_length=100)
    #dfdict = make_df_dict(df)
    dflist = fxs.make_df_list(df)
    
    
    """
    Multiprocessing
    """
    print("multicore")
    print("n_rows:",len(df))
    
    
    
    #load multiprocessing module
    import multiprocessing
    pool = multiprocessing.Pool(16)
    
    # Run multiprocessing on a list of dataframes
    list_of_results = pool.map(fxs.make_prefix, dflist)
    
    #concatenate the dataframes into one
    df_prefix = pd.concat(list_of_results)
    
    #time and save results for inspection
    start_time = time.time()
    df_prefix.to_csv("multi_core.csv",index=False)
    print("n_output_rows:",len(df_prefix))
    
    print((time.time() - start_time),"seconds")
    print(df_prefix.tail())
    
    #print("output length",len(pd.concat(list_of_results)))
    