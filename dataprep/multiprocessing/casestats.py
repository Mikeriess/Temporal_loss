# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 21:43:12 2022

@author: Mike
"""

            
            
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
    event = case.loc[case["activity_no"]==eventno]
    eventtime = event["time_parsed"].dt.strftime(dateformat).tolist()[0]
    
    #get case stats for current seuqnce/prefix:
    casestats = CaseData.loc[CaseData["caseid"]==caseno]
    
    casestats["prefix_date"] = eventtime        
    casestats["prefixes"] = casestats["num_events"]-1
    casestats["prefixwindow"] = prefixwindow
    casestats["prefix_number"] = eventno
    casestats["SEQID"] = SEQID
    
    #select what is interesting:
    casestats = casestats[["SEQID","caseid","num_events",
                           "prefix_number","prefixes","prefixwindow","prefix_date",
                           "distinct_events","caseduration_days"]]
    return casestats



def get_case_stats(df):
    import pandas as pd
    import pickle
    """  
    What we want is a table that can be linked with the 
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
                - y_t
                - y_a
    
        - Dataset level (aggregated):
            - number of prefixes in dataset
            - number of unique cases
            - number of events 
            - average number of events per case
            - (other stats from the survey paper)
    """

    
    with open('results/casestats_data.pickle', 'rb') as handle:
        casestats_objects = pickle.load(handle)
    
    padded_df = casestats_objects["padded_df"]
    CaseData = casestats_objects["CaseData"]
    y_t = casestats_objects["y_t"]
    y_a = casestats_objects["y_a"]
    y = casestats_objects["y"]
    prefixwindow  = casestats_objects["prefixwindow"]
    dateformat = casestats_objects["dateformat"]
    drop_last_ev = casestats_objects["drop_last_ev"]
    verbose = casestats_objects["verbose"]
    
    #Get all SEQIDs
    SEQIDS = padded_df["SEQID"].unique().tolist()
    
    allseqs = len(SEQIDS)
    lastseq = SEQIDS[len(SEQIDS)-1]
    #logic to build output table
    
    print("Making casestats..")
    
    from dataprep.inference_tables import process_seq_id
        
    # list comprehension for case processing
    output = [process_seq_id(i, lastseq, df, padded_df, CaseData, drop_last_ev, prefixwindow, dateformat, verbose) for i in SEQIDS]
    output = pd.concat(output)
    
    print("Done.")
    output["y"] = y.tolist()
    output["y_t"] = y_t.tolist()
    output = pd.concat([output.reset_index(drop=True), 
                        y_a.reset_index(drop=True).drop("caseid",axis=1)], axis=1)
    return output