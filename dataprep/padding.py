# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 13:27:34 2022

@author: Mike
"""


def pad_case(j, i, subset, maxlen, cols, padding):
    import pandas as pd
    import numpy as np
    ##################
    # Changing this to the event number, rather than type
    #j = int(row.eventno)
    ##################
    
    #Get current timestep, and all earlier timesteps
    EV = subset.loc[subset["event_number"] < j+1]
    
    #Figure out how many rows to pad
    rowstoadd = int(maxlen - len(EV))
    
    
    ### Padding: pre-event-padding ###
    zeros = np.zeros((rowstoadd, EV.shape[1]),dtype="float32")
    zeros = pd.DataFrame(zeros, columns=cols)
    
    if padding == "leading":
        #Add the zeros before the actual data
        EV = pd.concat([zeros, EV], ignore_index=True, axis=0)
        
    if padding == "trailing": 
        #Add the zeros after the actual data
        EV = pd.concat([EV, zeros], ignore_index=True, axis=0)
        
    #Set an ID for the sequence
    EV["SEQID"] = str(i)+"_"+str(j)
    EV["caseid"] = str(i)
    return EV



def pad_cases(caseids, df, max_prefix_len=3, padding="leading", verbose=True):    
    import pandas as pd
    import numpy as np
    import time
        
    print("\n\nInput length:",len(df))
    
    def padding_subroutine(df, i, max_prefix_len, padding):
        #from dataprep import pad_case
        
        #Get only events from case i
        subset = df.loc[df["caseid"] == i]
        # print("subset:")
        # print(subset)
        
        events = subset["event_number"].unique().tolist() #event
        cols = subset.columns    
               
        # list comprehension implementation: transform a case
        res_i = [pad_case(j, i, subset, max_prefix_len, cols, padding) for j in events]
        res_i = pd.concat(res_i)
        
        return res_i
    
    # Run the subroutine
    dataset = [padding_subroutine(df, i, max_prefix_len, padding) for i in caseids]
    
    """
    Future version:
        1) drop everything else than ids
        2) pad
        3) join everything back onto the padded df
        4) fillna if problems arise
    """
    
    #convert to dataframe    
    dataset = pd.concat(dataset)
     
    print("Output length:",len(dataset))
    return dataset



def pad_cases_w_merge(caseids, df, max_prefix_len=3, padding="leading", verbose=True):    
    import pandas as pd
    import numpy as np
    import time
    from dataprep.dataprep_helperfunctions import convert_to_float32
    
    """
    Reduce size of data to pad
    """
    
    df_pre_padding = df[["caseid","event_number"]]
    
    #add id to both dataframes so they can be merged later
    df_pre_padding["temp_SEQ_ID"] = df["caseid"].astype(str) + "" + df["event_number"].astype(str)
    df["temp_SEQ_ID"] = df["caseid"].astype(str) + "" + df["event_number"].astype(str)
    
        
    print("\n\nInput length:",len(df))
    
    def padding_subroutine(df, i, max_prefix_len, padding):
        #from dataprep import pad_case
        
        #Get only events from case i
        subset = df.loc[df["caseid"] == i]
        # print("subset:")
        # print(subset)
        
        events = subset["event_number"].unique().tolist() #event
        cols = subset.columns    
               
        # list comprehension implementation: transform a case
        res_i = [pad_case(j, i, subset, max_prefix_len, cols, padding) for j in events]
        res_i = pd.concat(res_i)
        
        return res_i
    
    # Run the subroutine
    padded_df = [padding_subroutine(df_pre_padding, i, max_prefix_len, padding) for i in caseids]
    
    #convert to dataframe    
    padded_df = pd.concat(padded_df)
    
    print("convert_to_float32:")
    padded_df = convert_to_float32(padded_df)
    #df = convert_to_float32(df)
    
    #make sure case id and event number is int
    padded_df["caseid"] = padded_df["caseid"].astype(int)
    padded_df["event_number"] =padded_df["event_number"].astype(int)
    
    print("padded_df shape")
    print(padded_df.shape)
    print("df shape")
    print(df.shape)
    
    print("merging features to padded df")
    padded_df = pd.merge(padded_df, df.drop(['caseid','event_number'], axis=1), on="temp_SEQ_ID", how="left", copy=False)   
    
    #ensure encoding is float 32
    #padded_df = convert_to_float32(padded_df)
    
    #make sure case id and event number is int
    padded_df["caseid"] = padded_df["caseid"].astype(int)
    padded_df["event_number"] =padded_df["event_number"].astype(int)
    
    #drop the temp_SEQ_ID again
    padded_df.drop("temp_SEQ_ID",axis=1, inplace=True)
    df.drop("temp_SEQ_ID",axis=1, inplace=True)
    
    print("padded_df shape")
    print(padded_df.shape)
    print(padded_df.columns)
    
    #fill na with zeros: all execept id variables
    padded_df = padded_df.fillna(0) 
    
    #reset the index
    padded_df.index = list(range(0,len(padded_df)))
    
    #move the SEQID to the end
    padded_df = padded_df[[c for c in padded_df if c not in ['SEQID']] + ['SEQID']]
    
    print(padded_df)
        
    print("Output length:",len(padded_df))
    
    return padded_df