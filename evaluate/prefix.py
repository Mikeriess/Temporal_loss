# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:29:16 2022

@author: Mike
"""

def prefix_pivot(Inference_test, metric="AE", events=5):
    import numpy as np
    import pandas as pd
    
    inf_results = Inference_test.loc[Inference_test.event_number < events+1]
    
    if metric =="AE":
        #convert to days
        inf_results[metric] = inf_results[metric] /(24.0*3600)
        
    
    # if metric =="TC_t":
    #     #convert to days
    #     inf_results[metric] = inf_results[metric] /(24.0*3600)
        
    if metric =="AE_CUMSUM_t":
        #convert to days
        inf_results[metric] = inf_results[metric] /(24.0*3600)

    pivottable = pd.pivot_table(inf_results, 
                           values=metric,
                        columns=['event_number'], aggfunc=np.mean)

    newcols = []
    for colno in range(0,len(pivottable.columns)):
        colno = colno + 1
        name = metric+"_"+str(colno)+""
        newcols.append(name)

    pivottable.columns = newcols
    pivottable.index = [0]
            
    return pivottable
