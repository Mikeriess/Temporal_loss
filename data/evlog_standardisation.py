# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 13:09:29 2022

@author: Mike
"""


def prepare_csv(file):
    import pandas as pd
    
    print(file["name"])
    df = pd.read_csv("data/"+file["name"]+".csv")
    print(df.columns)
    
    #sort by caseid, then timestamp
    df = df.sort_values(['id', 'time'], ascending=[True, True])
    df.index = list(range(0,len(df)))
    
    #generate activity numbers
    df["activity_no"] = df.groupby('id').cumcount()
    
    df = df[['id', 'event', 'time', 'resource','activity_no']+file["cat_features"]+file["num_features"]]
    
    return df









def convert_to_standardised_csv(file):
    import pandas as pd
    
    if file["type"] == "csv":
        df = pd.read_csv(file["dest"],sep=file["sep"])
        
    if file["type"] == "xes":
        from pm4py.objects.conversion.log import converter as xes_converter
        from pm4py.objects.log.importer.xes import importer as xes_importer
        xeslog = xes_importer.apply(file["dest"])
        df = xes_converter.apply(xeslog, variant=xes_converter.Variants.TO_DATA_FRAME)
    
    #drop unused columns
    df = df[file["keep_columns"]]
    
    #rename colnames
    df.columns = file["new_colnames"]
    
    #format    
    if file["utc"] == True:    
        #convert to string, for later truncation
        df["timestamp"] = df["timestamp"].astype(str)
    
        #get length of datetime
        dt_len = len(df.loc[0]["timestamp"])
    
        #remove the last chars, containing utc
        df['timestamp'] = df['timestamp'].str.slice(0, dt_len-6)
    
    if file["utc"] == False: 
        # convert to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], format=file["timeformat"])#"%Y-%m-%d %H:%M:%S")
    
    
    
    #convert string case ids into integer codes
    df["caseid"] = df["caseid"].astype('category')
    df["caseid"] = df["caseid"].cat.codes
        
    #sort by caseid, then timestamp
    df = df.sort_values(['caseid', 'timestamp'], ascending=[True, True])
    df.index = list(range(0,len(df)))
    
    #generate activity numbers
    df["activity_no"] = df.groupby('caseid').cumcount()
     
    """
    further processing
    """
    
    # temp drop of unused cols
    #df = df[['caseid', 'activity', 'activity_no', 'timestamp']]
    
    # rename
    df = df.rename({"caseid":"id", "timestamp":"time","activity":"event"}, axis='columns')
        
    #store to CSV
    df.to_csv("data/"+file["name"]+".csv",index=False)
    print("saving result to data/"+file["name"]+".csv")
    return df