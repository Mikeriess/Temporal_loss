# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 13:35:31 2022

@author: Mike
"""

def GenerateTrainData(df,
                      category_cols=[],
                      numeric_cols=[],
                      droplastev=True,
                      get_case_features = True,
                      dummify_time_features = True, 
                      max_prefix_length = 2,
                      window_position="first_k",
                      verbose=True,
                      predefined_dummies={"train":True}):
    
    import time as tm
    from datetime import datetime
    import pandas as pd
    import numpy as np
    import time
    
    #copy dummy dict with correct name
    dummy_dict = predefined_dummies
    
    #Make copy of df
    data = df.copy()    
    
    #Subset only relevant variables
    all_inputs = ["id","time","event"] + category_cols + numeric_cols
    print("features:")
    print(all_inputs)
    print("============================")
    
    df = df[all_inputs]
    
    #Standard dateformat to use
    dateformat = "%Y-%m-%d %H:%M:%S"
    
    # Make new case ids: ##############################
    cases = data["id"].unique().tolist()
    num_cases = len(cases) #make a counter to keep status
    newcaseids = list(range(0,len(cases)))
    
    dictdf = pd.DataFrame([cases,newcaseids]).T
    dictdf.columns =["id","newid"]
    
    newdata = pd.merge(left=data,right=dictdf,on="id")
    newdata.rename(columns={'id':'dropme',
                            'newid':'id'}, 
                            inplace=False).drop("dropme",axis=1)
    
    # List all cases by their new id:
    cases = data["id"].unique().tolist()
    
    # Make new event ids: ##############################
    evids = []
    for i in cases:
        subset = data.loc[data["id"] == i]
        evids = evids + list(range(0,len(subset)))
    
    evids = [x+1 for x in evids] # + 1 ###################################################### added +1
    
    #set the new eventids
    data["eventid"] = evids 
    
    #placeholder for all the cases
    out = []
    
    def process_case_i(data, i, max_prefix_length, verbose=False):
        # import pandas as pd
        # import numpy as np
        from dataprep.target import calculate_time_target
        
        """
        ##################################################
        """
        
        #iteration = iteration +1
        if verbose==True:  
            print("case:",i, "of",num_cases)
    
        #Look only at one caseid at a time
        subset = data.loc[data["id"] == i]
        subset.index = subset.eventid
        
        """
        #######################################################################
        PREFIX:
        #######################################################################
        """
                                
        #if trace is smaller than desired prefix len, just pick the full trace
        if max_prefix_length > len(subset):
            start = 1 #0
            stop = len(subset)
        
        #If the the max prefix len is smaller than the actual trace, 
        #take the K FIRST events (truncation approach)
        if max_prefix_length < len(subset):
            start = 1
            stop = max_prefix_length

        #If max prefix is identical to trace len, start from one        
        if max_prefix_length == len(subset):
            start = 1 #0
            stop = len(subset)
        if verbose==True:
            print("start",start,"stop",stop)
        
        #Prefix capability: Subset k last events from trace
        subset = subset.loc[start:stop]
            
        """
        #######################################################################
        #######################################################################
        """
        
        #Get feature table
        features = subset.copy()
        
        #Get list of events
        eventlist = subset.eventid.tolist()
        
        #store the case id
        caseid = str(i)
        
        from dataprep.target import calculate_time_target
        res_i = [calculate_time_target(event, i, start, stop, subset, dateformat, caseid) for event in eventlist]
        
        #convert to DF
        res_i = pd.DataFrame(res_i)
        res_i.columns = ['caseid',
                        'event_number',
                        'event_activity',
                        'timesincemidnight',
                        'dayofweek',
                        'hourofday',
                        'timesincestart',
                        'timesincelastev',
                        'y_timetofinish',
                        'drop']
        
        res_i.index = list(range(0,len(res_i)))
        features.index = list(range(0,len(res_i)))
        
        # print("res_i shape",res_i.shape)
        # print("features shape",features.shape)
        
        out = pd.concat([res_i, features], axis=1)
        
        # print("Nans after concatenation:")
        # print("out",out.isna().sum().sum())
        
        return out
    
    print("Generating time features")
    #list comprehenseion: do case I
    out = [process_case_i(data, i, max_prefix_length) for i in cases]
    print("done")
    
    #convert to df
    out = pd.concat(out, axis=0)
    out = out[out['caseid'].notna()]
    #out.index = list(range(0,len(out)))
 
    # print("Nans in out:")
    # #print("out",np.count_nonzero(np.isnan(out.values)))
    # print("out",out.isna().sum().sum())
    
    # print("shape of out:")
    # print("out",out.shape)
    # print(stophere)   
 
    #Get system variables
    cols = ['caseid',
            'event_number',
            'event_activity',
            'timesincemidnight',
            'dayofweek',
            'hourofday',
            'timesincestart',
            'timesincelastev',
            'y_timetofinish',
            'drop']
    
    print("creating two subsets of X: datasub, out")        
    # Create datasub: X without system variables
    datasub = out.copy()
    datasub = datasub[datasub['id'].notna()]
    datasub.index = list(range(0,len(datasub)))
    datasub["id"] = datasub["id"].astype(int)
    datasub["eventid"] = datasub["eventid"].astype(int)
    datasub["activity_no"] = datasub["activity_no"].astype(int)
    datasub = datasub[datasub.columns[~datasub.columns.isin(cols)]]
    

    
    # Subset out: X system variables only
    out = out[cols]
    out.index = list(range(0,len(out)))
    out["caseid"] = out["caseid"].astype(int)
    out["event_number"] = out["event_number"].astype(int)
    out["drop"] = out["drop"].astype(int)
    
    print(out.columns)
    
    print("*"*100)
    # print(datasub.head())
    # print(out.head())
    
    
    print("============================")
    print("Feature generation:")
    print("============================")
    #####################################
    # One-hot encoding
    
    #store original labels:
    
    #convert event into numerical codes
    out['event_activity'] = out['event_activity'].astype('category')
    out['event_activity'] = out['event_activity'].cat.codes
    out['event_activity'] = out['event_activity']+1
    
    #do the same for the subset of categorical features
    datasub['event_activity'] = datasub['event'].astype('category')
    datasub['event_activity'] = datasub['event_activity'].cat.codes
    datasub['event_activity'] = datasub['event_activity']+1

    """
    Activity/event features: Always included in the model
    """


        
    if predefined_dummies["train"] == True:        
        from dataprep.features import get_event_activity_features_train
        out, dummy_dict, observed_vals = get_event_activity_features_train(out)
            
    if predefined_dummies["train"] == False:        
        from dataprep.features import get_event_activity_features_test        
        out, dummy_dict, observed_vals, current_observed_vals, new_classes_to_add = get_event_activity_features_test(dummy_dict, out)
    
    """
    Case features
    """
    
    # dummify_time_features
    from dataprep.features import dummify_time_features   
    out = dummify_time_features(out, features = ["dayofweek","hourofday"]) #sysdummies, 
                
    print("categorical case features:")
    print(category_cols)
            
    from dataprep.features import categorical_case_features            
    out = categorical_case_features(out, datasub, category_cols)
    
    print("Nans in generated data:")
    print("out",np.count_nonzero(np.isnan(out.values)))
        
    print("numeric case features:")
    print(numeric_cols)
    
    from dataprep.features import numerical_case_features            
    out = numerical_case_features(out, datasub, numeric_cols)
    
        
    #Remove last event in each case
    if droplastev==True:
        print("\ndropping last event from each case")
        print("before:",len(out))
        out = out.loc[out["drop"] != 1]
        out = out.drop("drop",axis=1)
        print("after:",len(out))
        print("data in X is the",max_prefix_length,"last events, excluding the final event")
    
    #####################################
    # Separate outputs
    #####################################
    
    print("\ndropping target from X")
    
    #store the target as new series
    y = out["y_timetofinish"]
    
    #Drop everything that is not for the model to see during training
    drops = ["y_timetofinish"]
    
    #remove next activity maker to avoid peeking into the future
    X = out.drop(drops,axis=1)
    
    #Reset indexes:
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
        
    #Store possible clases for categorical columns
    dummy_dict = dummy_dict
    
    print("means")
    print("X mean:",np.mean(X))
    print("y mean:",np.mean(y))
    
    
    
    
    #####################################
    # Outputs
    
    """
    datasub #input data used, with desired prefix
    X            # X features
    y            # y remaining time at time t
    cases        # case ids for generating stats and back-linking results
    dummy_dict   # dictionary with observed classes for categorical variables
    """
        
    return X, y, cases, dummy_dict