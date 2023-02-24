# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:43:27 2019

@author: mikeriess
"""

def Sample(df, maxcases):
    import random
    
    #get cases
    cases = df["id"].unique().tolist() #[0:maxcases]
    
    if maxcases is not None:
        maxcases = int(maxcases)
        
        if maxcases < len(cases):
            print("/"*100)
            print("Sub-sampling:",maxcases," cases from event-log")            
            #sample
            casestoload = random.sample(cases, maxcases)
            
            #subset    
            df = df.loc[df["id"].isin(casestoload)]
    
    return df


def convert_to_float32(df):
    import pandas as pd
    # Select columns with 'float64' dtype  
    float64_cols = list(df.select_dtypes(include='float64'))
    
    # The same code again calling the columns
    df[float64_cols] = df[float64_cols].astype('float32')
    return df

def create_new_caseids(df):
    import pandas as pd
    
    #pd.factorize convert categories into int codes instead
    df["id"] = pd.factorize(df["id"].values.tolist())[0]
    
    # current_ids = list(set(df.id))    
    # new_ids = list(range(1,len(current_ids)))    
    # newdf = []    
    # for i in new_ids:
    #     #get the case
    #     subset = df.loc[df.id == current_ids[i]]
        
    #     #replace id
    #     subset["id"] = i        
    #     newdf.append(subset)
    
    # newdf = pd.concat(newdf)    
    return df


def drop_cases(df, min_len=3):
    import pandas as pd
    import numpy as np
    
    #list of ids to keep
    keep = []
    
    #loop over all ids
    for i in df.id.unique():
        
        #get number of events for current case i
        length = len(df.loc[df.id == i])
        
        if length >= min_len:
            keep.append(i)
            
    #subset df based on the generated list of ids
    df = df.loc[df.id.isin(keep)]
    
    return df


def GetFileInfo(df):
    print("Number of cases in log:",len(df["id"].unique()))
    import numpy as np
    import pandas as pd
    
    #Get the maximal trace length, for determining prefix length
    max_length = np.max(df['id'].value_counts())
    print("longest trace is:",max_length)
    
    #Look at the time format:
    print("Time format:",df["time"].loc[0])
    print("Std. format: %Y-%m-%d %H:%M:%S")
    
    print(df.head())
    return max_length


def Reshape(padded_df, prefixlength):
    import pandas as pd
    import numpy as np
        
    #prepare for join
    x = padded_df.reset_index(drop=True)
    
    #print("cols before reshaping;", x.columns)
    
    #drop system variables
    x = x.drop(["caseid","drop","SEQID"], axis=1)

    ############################# 
    X_train = x.values
    
    #Reshape:
    print("reshaping..")
    
    #time, n, k
    timesteps = prefixlength
    observations = len(padded_df["SEQID"].unique()) #y_train.shape[0] #int(X.shape[0]/prefixlength)
    k = X_train.shape[1]
    print(observations, timesteps, k)
    
    #Reshape the data
    X_train = X_train.reshape(observations, timesteps, k)
    
    print("Inference sample size (with prefixes of ",prefixlength,"):",X_train.shape[0])
    print("==========================================")
    #Check the shapes
    print("X_train: observations, timesteps, vars")
    print(X_train.shape)
    
    return X_train