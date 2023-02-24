# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 13:35:31 2022

@author: Mike
"""

def GenerateTrainData(df):
    
    #load experiment settings
    import pickle
    with open('settings.pickle', 'rb') as handle:
        settings = pickle.load(handle)

    """
    define variables
    """

    verbose = False

    category_cols=settings["cat_features"], #"start_day"
    numeric_cols=settings["num_features"],
    droplastev=settings["drop_last_event"],
    drop_end_target=settings["drop_last_activity_target_class"],
    get_activity_target=settings["generate_activity_target"],
    get_case_features = settings["generate_case_features"],
    dummify_time_features = settings["onehot_time_features"], 
    max_prefix_length = settings["max_prefix_length"],
    window_position=settings["trace_window_position"],
    predefined_dummies={"train":True}

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
    df = df[["id","time","event"]+category_cols+numeric_cols]
    
    #Standard dateformat to use
    dateformat = "%Y-%m-%d %H:%M:%S"
    
    # Make new case ids: ##############################
    cases = data["id"].unique().tolist()
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
    
    #make a counter to keep status
    num_cases = len(cases)
    
    # Generate features case by case
    for i in cases:
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
        
        index1 = 0
        
        """ Alternate approach:
            
        #determine whether to start in the beginning or end of trace        
        if window_position == "last_k":
        
            #if trace is smaller than desired prefix, just pick the full trace
            if max_prefix_length > len(subset):
                start = 1 #0
                stop = len(subset) - index1 # 
            
            #If the the max prefix len is smaller than the actual trace, 
            #take the K last events (sliding window approach)
            if max_prefix_length < len(subset):
                start = len(subset) - max_prefix_length
                stop = len(subset) - index1
    
            #If max prefix is identical to trace len, start from one        
            if max_prefix_length == len(subset):
                start = 1 #0
                stop = len(subset) - index1
        """
                
                
        """ if window_position == "first_k":"""
        
        #if trace is smaller than desired prefix len, just pick the full trace
        if max_prefix_length > len(subset):
            start = 1 #0
            stop = len(subset) - index1 # 
        
        #If the the max prefix len is smaller than the actual trace, 
        #take the K FIRST events (sliding window approach)
        if max_prefix_length < len(subset):
            start = 1
            stop = max_prefix_length - index1

        #If max prefix is identical to trace len, start from one        
        if max_prefix_length == len(subset):
            start = 1 #0
            stop = len(subset) - index1
        if verbose==True:
            print("start",start,"stop",stop)
        
        #Prefix capability: Subset k last events from trace
        subset = subset.loc[start:stop]
        
        #Make sure the data to be dummified also follows prefix convention
        if i == 1:
            datasub = subset
            if verbose==True:
                print("len subset:",len(subset))
                print("len datasub:",len(datasub))

        if i > 1:
            datasub = pd.concat([datasub, subset],axis=0)
            
        """
        #######################################################################
        PREFIX:
        #######################################################################
        """
        
        #Get list of events
        eventlist = subset.eventid.tolist()
        
        #store the case id
        caseid = str(i)
        
        
        #for every event:
        for event in eventlist:
            #print(event)
            
            #Generate an eventID
            event_number = event#+1
            
            #get the event for later reference
            event_activity = subset["event"].loc[event]
            
            #For the tax-approach, get next event
            if event != stop: #len(subset)-1: #if its the last event
                next_activity = subset["event"].loc[event+1]
            
            if event == stop: #len(subset)-1: #if its the last event
                next_activity = "END"
                
                    
            """
            #######################################################################
            Beginning of time features:
            """
            
            #first event
            starttime = datetime.fromtimestamp(tm.mktime(tm.strptime(subset["time"].loc[start], dateformat)))
         
            #time in secs since midnight
            t = tm.strptime(subset["time"].loc[event], dateformat) #Time now
            midnight = datetime.fromtimestamp(tm.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0) #Midnight
            timesincemidnight = (datetime.fromtimestamp(time.mktime(t)) - midnight).total_seconds()
    
            #monday = 1
            dayofweek = datetime.fromtimestamp(tm.mktime(t)).weekday()+1 
            
            #hour of day
            hour = datetime.fromtimestamp(tm.mktime(t)).hour
    
            #Time since start in seconds
            timesincestart = (datetime.fromtimestamp(time.mktime(t)) - starttime).total_seconds()
    
            #Time since last event in seconds
            if event ==start:
                t_last = tm.strptime(subset["time"].loc[event], dateformat) #Time last event: now
                timesincelastev = 0
            if event !=start:
                t_last = tm.strptime(subset["time"].loc[event-1], dateformat) #Time last event
                timesincelastev = (datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(t_last))).total_seconds()
          
            #Time until finishtime
            t_finish = tm.strptime(subset["time"].loc[stop], dateformat) #Time last event
            timetofinish = (datetime.fromtimestamp(time.mktime(t_finish)) - datetime.fromtimestamp(time.mktime(t))).total_seconds()
    
            
            #Time to next event:
            if event == stop:#len(subset)-1: #if its the last event
                t_nextev = tm.strptime(subset["time"].loc[event], dateformat) #Time last event: now
                timetonextev = 0
                
            if event != stop:#len(subset)-1: #if not last event
                t_nextev = tm.strptime(subset["time"].loc[event+1], dateformat) #Time last event
                timetonextev = (datetime.fromtimestamp(time.mktime(t_nextev))-datetime.fromtimestamp(time.mktime(t))).total_seconds()
    
            """
            #######################################################################
            End of time features for each event
            """
                    
            #Make a marker for dropping last step where remaining time (y) = 0
            drop = 0
            #Mark if it is the last activity:
            if event == stop:#len(subset)-1:
                drop = 1
                
               
            #Actual:
            results = [caseid,
                       event_number,
                       event_activity, #event
                       timesincemidnight,
                       dayofweek,
                       hour,
                       timesincestart,
                       timesincelastev,
                       timetofinish,
                       timetonextev,
                       next_activity,
                       drop]
            
            # if this is the first time, create out
            if i == 1 and event == start: #First time
                out = pd.DataFrame(results).T
            
            # else append to out (which is a dataframe)
            else:
                res_i = pd.DataFrame(results).T
                out = out.append(res_i)
            
    #Rename all static vars
    cols = ['caseid',
            'event_number',
            'event_activity',
            'timesincemidnight',
            'dayofweek',
            'hourofday',
            'timesincestart',
            'timesincelastev',
            'y_timetofinish',
            'y_timetonextev',
            'next_activity',
            'drop']
    out.columns = cols
      
    print("============================")
    print("Post-processing:")
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
        
    if get_activity_target == True:
        #dummify next event variable
        y_a = pd.get_dummies(out["next_activity"], prefix="y_a_t1")
        y_a = y_a.reset_index(drop=True)    
        #generate list of original varnames
        y_a_varnames = y_a.columns.tolist()
    
        #do it all again, but with renamed activity names
        #convert event into numerical codes
        out['next_activity'] = out['next_activity'].astype('category')
        out['next_activity'] = out['next_activity'].cat.codes
        out['next_activity'] = out['next_activity']+1
        
        #dummify next event variable
        y_a = pd.get_dummies(out["next_activity"], prefix="y_a_t1")
        y_a = y_a.reset_index(drop=True)   
        
        #generate list of new original varnames
        y_a_new_varnames = y_a.columns.tolist()
        
        #add it into the output table
        out = out.reset_index(drop=True)
        out = pd.concat([out, y_a], axis=1)
        
    """
    Activity/event features: Always included in the model
    """
    
    
    if predefined_dummies["train"] == True:
        
        #initialize dummy dict
        dummy_dict = {"event":None}
        
        #get currently observed values:
        observed_vals = np.sort(out["event_activity"].astype('str').unique()).tolist()
        
        #Convert to categorical column
        out["event_activity"] = out["event_activity"].astype('category')
        
        #generate dummies
        Dummies = pd.get_dummies(out["event_activity"])
        
        #sort the dummies
        Dummies = Dummies.sort_index(axis=1)
        print(Dummies.head())
        
        #reset index of dummies and generate a prefix
        Dummies = Dummies.reset_index(drop=True)
        Dummies = Dummies.add_prefix('ev_a_t0_')
        Dummies = Dummies.sort_index(axis=1)
        
        #reset index of other the rest of the features
        out = out.reset_index(drop=True)
        
        #add dummies to all featues, and drop original column
        dummycols = Dummies.columns.tolist()
        out = pd.concat([out,Dummies],axis=1)
        out = out.drop("event_activity",axis=1)
        
        # store observed values
        dummy_dict["event"] = observed_vals
    
    
    if predefined_dummies["train"] == False:
        
        #get previously observed values:
        observed_vals = dummy_dict["event"]
                
        
        #get currently observed values:
        current_observed_vals = np.sort(out["event_activity"].astype('str').unique()).tolist()
        print(current_observed_vals)             
        
        #Find classes not observed in current data
        new_classes_to_add = list(set(observed_vals) - set(current_observed_vals))
        print(new_classes_to_add)
        
        #Convert to categorical column
        out["event_activity"] = out["event_activity"].astype('category')
        
        #Add unobserved values
        out["event_activity"] = out["event_activity"].cat.add_categories(new_classes_to_add)
        
        #generate dummies
        Dummies = pd.get_dummies(out["event_activity"])
        
        #sort the dummies
        Dummies = Dummies.sort_index(axis=1)
        print(Dummies.head())
        
        #reset index of dummies and generate a prefix
        Dummies = Dummies.reset_index(drop=True)
        Dummies = Dummies.add_prefix('ev_a_t0_')
        Dummies = Dummies.sort_index(axis=1)
        
        #reset index of other the rest of the features
        out = out.reset_index(drop=True)
        
        #add dummies to all featues, and drop original column
        dummycols = Dummies.columns.tolist()
        out = pd.concat([out,Dummies],axis=1)
        out = out.drop("event_activity",axis=1)
        print(out.columns)
    
    """
    Case features
    """
    
    if get_case_features == True:    
        if len(category_cols) > 0:
            
            dummylist = category_cols
            print("\nDummification of",dummylist)
    
            Dummies = pd.get_dummies(datasub[dummylist])
            Dummies = Dummies.reset_index(drop=True)
            out = out.reset_index(drop=True)
            dummycols = Dummies.columns.tolist()
            out = pd.concat([out,Dummies],axis=1)
        
        if len(numeric_cols) > 0:
            #add numerical features:
            print("Adding numerical features:",numeric_cols)
            numerics = datasub[numeric_cols]
            numerics = numerics.reset_index(drop=True)
            numerics = numerics.add_prefix('num_')
            out = out.reset_index(drop=True)
            out = pd.concat([out,numerics],axis=1)
    
    
    if dummify_time_features == True:
        features = ["dayofweek","hourofday"]
        print("Dummification of time features",features)
        sysdummies = out[features]
        sysdummies = pd.get_dummies(sysdummies,prefix="t_")
        sysdummies = sysdummies.reset_index(drop=True)
        out = out.drop(features,axis=1)
        out = out.reset_index(drop=True)
        out = pd.concat([out, sysdummies],axis=1)
    
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
    
    print("\ndropping vars:")
    
    # Generate the next_activity target vector
    y_a = out[y_a_new_varnames]
    y_a["caseid"] = out["caseid"]
    y_a["event_number"] = out["event_number"]
    
    
    if drop_end_target==True:
        """ NOT SURE IF THIS SHOULD BE DONE FOR THE TAX MODEL
            as this would signal the end of a trace..
        """
        dropme = ['y_a_t1_1']#before it was: ["y_a_t1_END"]
        print("dropping last event category from y_a:",dropme)
        y_a = y_a.drop(dropme,axis=1) #drop indicator that it is the last event
    
    
    # Generate the time to next activity target vector
    y_t = out["y_timetonextev"]
    #y_t["caseid"] = out["caseid"]
    #y_t["event"] = out["event"]
    
    y = out["y_timetofinish"]
    #y["caseid"] = out["caseid"]
    #y["event"] = out["event"]
    
    #Drop everything that is not for the model to see during training
    drops = ["y_timetofinish","y_timetonextev","next_activity"] + y_a_new_varnames
    print("dropping vars from X: ",drops)
    
    #remove next activity maker to avoid peeking into the future
    X = out.drop(drops,axis=1)
    
    #Reset indexes:
    X = X.reset_index(drop=True)
    y_a = y_a.reset_index(drop=True)
    y_t = y_t.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    #Store possible clases for categorical columns
    dummy_dict = dummy_dict
    
    #####################################
    # Output stuff
    """
    datasub #input data used, with desired prefix
    X            # X features
    y_a          # y next event type
    y_t          # y time to next event
    y            # y remaining time at time t
    cases        # case ids for generating stats and back-linking results
    y_a_varnames # original varnames for y_a before renaming
    dummy_dict   # dictionary with observed classes for categorical variables
    """

    output = {"X":X, "y":y, "y_a":y_a, "y_t":y_t, "cases":cases, "y_a_varnames":y_a_varnames, "dummy_dict":dummy_dict}

    return output