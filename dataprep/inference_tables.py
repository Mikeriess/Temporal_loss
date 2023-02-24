# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:53:11 2022

@author: Mike
"""



def CaseData(df):
    print("Generating case data")
    # step 0: Get case aggregate stats:
    import pandas as pd
    CaseData = pd.DataFrame(df['id'].value_counts())
    CaseData.rename(columns={"id":"num_events"}, inplace=True)
    CaseData["caseid"] = CaseData.index
    CaseData.sort_values('caseid', inplace=True)
    
    distinct_events = df.groupby('id').event.nunique()
    CaseData["distinct_events"] = distinct_events
    
    mindate = df[df.groupby('id').time.transform('min') == df['time']]
    mindate = mindate.drop_duplicates(subset="id",keep="first")[["id","time"]]
    mindate.rename(columns={"time":"start"}, inplace=True)
    
    maxdate = df[df.groupby('id').time.transform('max') == df['time']]
    maxdate = maxdate.drop_duplicates(subset="id",keep="first")[["id","time"]]
    maxdate.rename(columns={"time":"stop"}, inplace=True)
    
    Dates = pd.merge(left=mindate,right=maxdate, on="id")
    Dates["start"] = pd.to_datetime(Dates["start"])
    Dates["stop"] = pd.to_datetime(Dates["stop"])
    
    import datetime as dt
    Dates["caseduration_days"] = (Dates['stop'] - Dates['start']).dt.days
    Dates["caseduration_seconds"] = (Dates['stop'] - Dates['start']).dt.seconds
    Dates.rename(columns={"id":"caseid"}, inplace=True)
    
    CaseData = pd.merge(left=CaseData,right=Dates, on="caseid")
    print("done")
    return CaseData



def process_seq_id(i, lastseq, df, padded_df, CaseData, drop_last_ev, prefixwindow, dateformat, verbose):
    import pandas as pd
    import numpy as np
    
    if verbose == True:
        print("Making casestats for SEQ:",i,"of",lastseq)
        
    prefix = padded_df.loc[padded_df["SEQID"]==i]
            
    SEQID = i

    #get case number and prefix number
    caseno = int(prefix["SEQID"].loc[0].split("_")[0])
    eventno = int(prefix["SEQID"].loc[0].split("_")[1])
    
    if drop_last_ev == False:
        #Add 1 to the index in the case all events are present
        eventno = eventno - 1
    
    #Get event-specific data:
    case = df.loc[df["id"]==caseno]
    event = case.loc[case["event_number"]==eventno]
    eventtime = event["time_parsed"].dt.strftime(dateformat).tolist()[0]
    
    #get case stats for current seuqnce/prefix:
    casestats = CaseData.loc[CaseData["caseid"]==caseno]
    
    casestats["prefix_date"] = eventtime        
    casestats["prefixes"] = casestats["num_events"]-1
    casestats["prefixwindow"] = prefixwindow
    casestats["prefix_number"] = eventno
    casestats["SEQID"] = SEQID
    casestats["event_number"] = eventno +1
    
    #select what is interesting:
    casestats = casestats[["SEQID",
                           "caseid",
                           "num_events",
                           "prefix_number",
                           "event_number", #added
                           "prefixes",
                           "prefixwindow",
                           "prefix_date",
                           "distinct_events",
                           "caseduration_days"]]
    return casestats




def GetCaseStats(df, padded_df, CaseData, y, prefixwindow=0, dateformat="%Y-%m-%d %H:%M:%S", drop_last_ev=True, verbose=False):
    import pandas as pd
    """  
    Outcome is a table that can be linked with the 
    observations in the training data. 
    
    This table has to be ordered, and then permuted the 
    exact same way as the X/y output of this pipeline.
    
    Table:
        - SEQID (level)
            - Input table (prefixes):
                - Event number (same as number of events in curr. SEQID)
                - number of events in parent case (trace length)
                - (other interesting attributes could be added here)
                
            - Target values
                - y
    
        - Dataset level (aggregated):
            - number of prefixes in dataset
            - number of unique cases
            - number of events 
            - average number of events per case
            - (other stats from the survey paper)
    """

    #Get all SEQIDs
    SEQIDS = padded_df["SEQID"].unique().tolist()
    lastseq = SEQIDS[len(SEQIDS)-1]
    
    print("Making casestats..")
    
    from dataprep.inference_tables import process_seq_id
    
    ##### Reduce problem size: Drop duplicate rows
    padded_df = padded_df.drop_duplicates(subset='SEQID', keep="first")
        
    # list comprehension for case processing
    output = [process_seq_id(i, lastseq, df, padded_df, CaseData, drop_last_ev, prefixwindow, dateformat, verbose) for i in SEQIDS]
    output = pd.concat(output)
        
    print("Done.")
    output["y"] = y.tolist()
    #output["y_t"] = y_t.tolist()
    output = output.reset_index(drop=True)
    
    return output