# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:48:28 2022

@author: Mike
"""
import pandas as pd
df = pd.read_csv("data/traffic_fines.csv")
df.shape

def make_prefix(df):
    import pandas as pd
    
    def flatten(l):
        return [item for sublist in l for item in sublist]
        
    #get ids
    caseids = df.id.unique().tolist()
    
    #placeholders for columns
    pfx_caseids = []
    pfx_eventids = []
    pfx_ids = []
    
    for idx in caseids:
        #do something
        #print("case",idx)
        
        #subset on trace
        trace = df.loc[df.id == idx]
        
        #get ids
        eventids = trace.activity_no.values.tolist() #list(range(0,len(trace)))
        
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
            prefixids_to_add = [str(idx)+"_"+str(eventid)]*len(progress)
            
            #Debug
            #print("caseid:",len(caseids_to_add),"eventid:",len(eventids_to_add))
            
            #variables
            pfx_caseids.append(caseids_to_add)
            pfx_eventids.append(eventids_to_add)
            pfx_ids.append(prefixids_to_add)
            
    #Create result df
    pfx_caseids = flatten(pfx_caseids)
    pfx_eventids = flatten(pfx_eventids)
    pfx_ids = flatten(pfx_ids)
        
    df_prefix = pd.DataFrame({"id":pfx_caseids,
                      "eventno":pfx_eventids,
                      "SEQ_ID":pfx_ids})
    
    return df_prefix



df_prefix = make_prefix(df.loc[:120])
df_prefix.head(12)


def join_data(df,X):
    import pandas as pd
    
    #create the ID variable
    X["SEQ_ID"] = X["id"].astype(str) +"_"+ X["activity_no"].astype(str)
    
    #drop redundant variables
    X = X.drop(["id"],axis=1)
    
    #join original data on the new prefix table
    joined_df = pd.merge(left=df, right=X, on="SEQ_ID",how="left")
    
    return joined_df

df_prefix.head(12)
df.head(12)

df_joined = join_data(df_prefix, df)
df_joined.head(12)


def padding(df):
    from tensorflow.keras.sequence import padding
    
    
    
    return padded_df