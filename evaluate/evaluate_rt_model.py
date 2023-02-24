
# -*- coding: utf-8 -*-

def evaluate_rt_model(data_objects, partition="test"):
    """
    Function to evaluate a RT model: model-agnostic

    Parameters
    ----------
    data_objects : dictionary
        Results of training, containing inference tables to make calculations from

    Returns
    -------
    data_objects : dictionary
    
        - prefix_performance: table
            pivottable showing absolute error per prefix
            
        - report: table
            one-row table with KPI's of interest
            
        - Inference_test: table
            updated columns with absolute error per event, and MAE over testset
        

    """
    #from sklearn.metrics import mean_squared_error, mean_absolute_error
    #from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    #from sklearn.model_selection import train_test_split as split
    
    print("Hint: evaluation metrics calculated in days.")

    import time
    import datetime
    import numpy as np
    import pandas as pd

    def AE(y, y_hat):
        AE = np.abs(y - y_hat)
        return AE


    # Aggregated metrics:
        
    def MAE(table):
        """
        Average of the Mean absolute error over each trace
        
        Input: 
        Inference table with relevant variables
       
        T = "prefixes"
    
        case = "caseid"
        y = "y"
        y_pred = "y_pred"

        """
        import numpy as np
        
        #placeholder
        MAE_i = []
        
        #get id of all cases
        ids = table.caseid.unique().tolist()
        
        #for each case, to make sure T is variable
        for idx in ids:
            sub = table.loc[table.caseid == idx]
            
            T = np.max(sub.prefixes)
            
            ae = np.abs(sub.y-sub.y_pred)
            
            mae = np.sum(ae)/T
            
            MAE_i.append(mae)
        
        #get mean of all the MAE's of each trace
        MAE = np.mean(MAE_i)/(24.0*3600) #convert to days
        
        
        #Aggregated metrics:
        table["MAE"] = MAE
        
        
        #Observation-level metrics:
        table["AE"] = AE(table.y, table.y_pred)
        
        return table
    
    def MAE_PTD(table):
        """        
        Average mean absolute error with progression-based temporal decay
        
        Input: 
        Inference table with relevant variables
       
        T = "prefixes"
    
        case = "caseid"
        y = "y"
        y_pred = "y_pred"
        
        
        
        decay = 1/t
    
        resid = np.abs(y - y_hat)
        
        EP = resid*decay

        """
        import numpy as np
        
        #placeholder
        MAEPTD_i = []
        
        #get id of all cases
        ids = table.caseid.unique().tolist()
        
        #for each case, to make sure T is variable
        for idx in ids:
            sub = table.loc[table.caseid == idx]
            
            t = np.array(range(1,len(sub)+1))
            
            decay = 1/t
            
            T = np.max(sub.prefixes)
            
            ae = np.abs(sub.y-sub.y_pred)
            
            td_resid = ae*decay
            
            mae_ptd = np.sum(td_resid)/T
            
            MAEPTD_i.append(mae_ptd)
        
        #get mean of all the MAE's of each trace
        MAEPTD = np.mean(MAEPTD_i)/(24.0*3600) #convert to days
        
        
        #Aggregated metrics:
        table["MAE_PTD"] = MAEPTD
        
        
        #Observation-level metrics:
        #table["AE_PTD"] = AE(table.y, table.y_pred)
        
        return table
        
    
    def TC(table, direction="positive"):
        """
        Calculate the TC metrics
        """
        
        import numpy as np
        
        #get id of all cases
        ids = table.caseid.unique().tolist()
        
        #placeholder
        TC_t = []
        TC_i = []
        
        #for each case
        for idx in ids:
            
            #subset on case-level
            sub = table.loc[table.caseid == idx]
            sub.index = list(range(0,len(sub)))
                        
            # TC at time t, t=0 will be zero
            tc_t_values = []

            # TC over full event-log, starting from t=1
            tc_i_values = []
            
            # for each timestep
            for t in sub.index:
                
                # starting with 0 as we use first difference, and need
                # a value such that the list has same length as event-log
                # even though we wont plot t=0
                if t == 0:
                    tc_t_values.append(0)
                
                # take first difference
                if t > 0:
                    # previous value
                    t_minus_1 = sub.loc[sub.index == t-1]["y_pred"].values[0]
                    # current value
                    t_0 = sub.loc[sub.index == t]["y_pred"].values[0]
                    
                    # Over-prediction, or under-prediction:
                    # (heaviside function)
                    if direction == "positive":
                        t_diff = ((t_0 - t_minus_1) > 0)*1
                    if direction == "negative":
                        t_diff = ((t_0 - t_minus_1) < 0)*1
                    
                    # first term: the residual
                    residual = t_0 - t_minus_1
                    
                    # second term: heaviside function
                    tc_t = (residual*t_diff)
                    
                    #append the values to trace-level list
                    tc_t_values.append(tc_t/(24.0*3600)) #convert to days

                    #append the values to event-log level list
                    tc_i_values.append(tc_t)
                    
            # append event-level residuals in pos or neg direction
            TC_t.append(tc_t_values)
            
            # append case-level diff calculations
            TC_i.append(np.mean(tc_i_values))
                    
        #Aggregated metrics:
        table["TC_i"] = np.mean(TC_i)/(24.0*3600)
        
        #event-level indicators (first value is zero)
        table["TC_t"] = [item for sublist in TC_t for item in sublist] #flatten list of lists

        return table


    def CUMSUM(table):
        """
        Accumulated error at time t of a trace (for later pivot analysis)
        """
              
        import numpy as np
        
        #get id of all cases
        ids = table.caseid.unique().tolist()
        
        cumsum = []
        
        #for each case
        for idx in ids:
            #subset on case-level
            sub = table.loc[table.caseid == idx]
            sub.index = list(range(0,len(sub)))
            
            #case level residuals        
            cumsum.append(np.cumsum(sub.AE))
        
        table["AE_CUMSUM_t"] = [item for sublist in cumsum for item in sublist] #flatten list
        
        return table
    
    """
    Evaluation
    """  
    
    # current date and time
    ts = time.time()    
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    

    time_taken = 0
    if "train_time" in data_objects:
        time_taken = data_objects["train_time"]

    ###################################################################
    # get the data
    
    if partition == "test":
        inference = data_objects["Inference_test"]
    
    if partition == "train":
        inference = data_objects["Inference_train"]
        
    model_params = data_objects["model_params"]

    ###################################################################
    # Calculate performance    
    
    inference = MAE(inference)
    inference = TC(inference, direction="positive")
    inference = CUMSUM(inference)
    inference = MAE_PTD(inference)
    
    ###################################################################
    # Get and show performance

    mae_test = np.mean(inference["MAE"])#/(24.0*3600)
    
    mae_ptd_test = np.mean(inference["MAE_PTD"])#/(24.0*3600)
    
    TC_i_avg = np.mean(inference["TC_i"]) # converted to days in function
    
    TC_t_avg = np.mean(inference["TC_t"])

    # Print status:
    print('_'*60)
    print('Test MAE:     ', mae_test, ' (days)')
    print("================================"*3)
    
    ###################################################################
    # Generate report with parameters of interest
    
    results = {"RUN":0,
               "Time_of_evaluation":timestamp,
               "Train_duration":time_taken,
               "MAE":mae_test,
               "MAE_PTD":mae_ptd_test,
               "TC_i_avg":TC_i_avg,
               "TC_t_avg":TC_t_avg}

    report = dict(results, **model_params)
    report = pd.DataFrame(report, index=[0])
    
    # store report
    data_objects["report"] = report
        
    ###################################################################
    # Prefix-level overview of performance up to 5 events
    from evaluate.prefix import prefix_pivot    
    
    n_prefixes = data_objects['PREP_SETTINGS']['max_prefix_length']+1
    
    # store prefix performance    
    data_objects["prefix_MAE_t"] = prefix_pivot(inference, metric="AE", events=n_prefixes)
    data_objects["prefix_TC_t"] = prefix_pivot(inference, metric="TC_t", events=n_prefixes)
    data_objects["prefix_AE_CUMSUM_t"] = prefix_pivot(inference, metric="AE_CUMSUM_t", events=n_prefixes)
    
    print("MAE_t:")
    print(data_objects["prefix_MAE_t"])
    
    return data_objects
