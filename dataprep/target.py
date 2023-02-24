# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:10:43 2022

@author: Mike
"""

def get_next_activity_target(out):
    import pandas as pd
    
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
    
    return out, y_a_varnames, y_a_new_varnames, y_a 



def calculate_time_target(event, i, start, stop, subset, dateformat, caseid):
    import time as tm
    from datetime import datetime
    import pandas as pd
    import numpy as np
    import time
    
    #Generate an eventID
    event_number = event #+1
    
    #get the event for later reference
    event_activity = subset["event"].loc[event]
                
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
    if event == start:
        t_last = tm.strptime(subset["time"].loc[event], dateformat) #No last event
        timesincelastev = 0
        
    if event != start:
        t_last = tm.strptime(subset["time"].loc[event-1], dateformat) #Time last event
        timesincelastev = (datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(t_last))).total_seconds()
  
    #Time until finishtime
    t_finish = tm.strptime(subset["time"].loc[stop], dateformat) #Time last event
    timetofinish = (datetime.fromtimestamp(time.mktime(t_finish)) - datetime.fromtimestamp(time.mktime(t))).total_seconds()

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
               drop]
    
    out = {"caseid":caseid,
            "event_number":event_number,
            "event_activity":event_activity,
            "timesincemidnight":timesincemidnight,
            "dayofweek":dayofweek,
            "hour":hour,
            "timesincestart":timesincestart,
            "timesincelastev":timesincelastev,
            "timetofinish":timetofinish,
            "drop":drop}
        
    return out 