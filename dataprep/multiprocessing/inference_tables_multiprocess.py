def get_case_stats(df):

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

    import pandas as pd
    import pickle

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

    ################################################# for i in SEQIDS: #### see old code

    
    print("Making casestats..")
    
    #from dataprep.inference_tables_multiprocess import process_seq_id
    #from inference_tables_multiprocess import process_seq_id
        
    # list comprehension for case processing
    output = [process_seq_id(i, lastseq, df, padded_df, CaseData, drop_last_ev, prefixwindow, dateformat, verbose) for i in SEQIDS]
    output = pd.concat(output)
    
    print("Done.")
    output["y"] = y.tolist()
    output["y_t"] = y_t.tolist()
    output = pd.concat([output.reset_index(drop=True), 
                        y_a.reset_index(drop=True).drop("caseid",axis=1)], axis=1)
    return output


def get_case_stats_slow(df):
    import pandas as pd
    import pickle

    #load data from disk.............
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
    SEQIDS
    
    allseqs = len(SEQIDS)
    #logic to build output table
    counter = 0
    print("Making casestats..")
    for i in SEQIDS:
        if verbose == True:
            print("Making casestats for SEQ:",counter,"of",allseqs)
        counter = counter +1    
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
        
        #select what is interesting:
        casestats = casestats[["SEQID","caseid","num_events",
                               "prefix_number","prefixes","prefixwindow","prefix_date",
                               "distinct_events","caseduration_days"]]

        if counter == 1:
            output = casestats
        if counter > 1:
            output = pd.concat([output,casestats],axis=0)
    
    # step 2: Match the new table with target variables
    #"""    
    print("Done.")
    output["y"] = y.tolist()
    output["y_t"] = y_t.tolist()
    output = pd.concat([output.reset_index(drop=True), 
                        y_a.reset_index(drop=True).drop("caseid",axis=1)], axis=1)

    output.to_csv("results/casestats_data.csv",index=False, mode='a', append=True)
    return output


#read data
import pickle
with open('results/casestats_data.pickle', 'rb') as handle:
        casestats_objects = pickle.load(handle)

#load the main data
df = casestats_objects["df"]

import multiprocessing
pool = multiprocessing.Pool(14)


def make_df_list(df, idvar="id"):
    dflist = [v for k, v in df.groupby(idvar)]
    return dflist


dflist = make_df_list(df, idvar="id")

print("/"*100)
print("Multiprocessing")

#from dataprep.multiprocessing.casestats import get_case_stats

list_of_results = pool.map(get_case_stats_slow, dflist)

#concatenate to final DF
CaseStats = pd.concat(list_of_results)
print(CaseStats.head(20))
time.sleep(5)

print("/"*100)
print("pd.merge")
# make inference tables
Inference = pd.merge(left=CaseStats, right=split_criterion, on="caseid",how="left")

Inference_train = Inference.loc[Inference.trainset==True].drop("trainset",axis=1)
Inference_test = Inference.loc[Inference.trainset==False].drop("trainset",axis=1)
print("Inference train:",len(Inference_train))
print("Inference test: ",len(Inference_test))


Inference_data = {"Inference_train":Inference_train,
            "Inference_test":Inference_test}

#store results
with open('results/inference_tables.pickle', 'wb') as handle:
    pickle.dump(Inference_data, handle, protocol=pickle.HIGHEST_PROTOCOL)