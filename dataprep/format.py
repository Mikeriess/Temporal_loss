# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 13:28:38 2022

@author: Mike
"""

def InitialFormatting(df, train=True, verbose=False):
    import pandas as pd
    
    # find cases to drop due to length
    print("Cases before dropping len=1:",len(set(df.id)),"cases",len(df),"rows")
    
    # Make function to apply to groups
    def func(sub):
        
        out = None
        
        keepid = min(sub.id)
        
        if len(sub) > 1:
            out = keepid
        
        return out
    
    # Make list of cases above length 1
    df_grp = df.groupby('id').apply(func)
    
    #Remove NaNs from the list
    keepers = df_grp.values
    
    if train == True:
        # Drop cases with only one event:
        df = df.loc[df["id"].isin(keepers)]
        
        print("Cases after dropping len=1:",len(keepers),"cases",len(df),"rows")  
    
    ##########################################################################
    print("Sorting by id, date (chronological order)")
    #generate new ID column:
    df = df.assign(id=(df['id']).astype('category').cat.codes)
    
    df["id"] = df.id.astype('int32')
    
    # Ensure ID starts at 1
    if min(df.id) == 0:
        df.id = df.id +1
    
    # Sort the DF baed on caseid, and the date of the event
    df = df.sort_values(['id',"time"], ascending=[True, True]) 
    
    # drop sorting variable again
    #df = df.drop("parsed_date",axis=1)
    
    from pandas.api.types import is_datetime64_any_dtype as is_datetime
    
    if is_datetime(df["time"]):
        print("converting datetime back to string in framework standard format")
        df['time'] = df['time'].dt.strftime("%Y-%m-%d %H:%M:%S")
        
    return df