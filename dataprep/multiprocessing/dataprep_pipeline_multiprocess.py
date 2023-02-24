# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:46:33 2022

@author: Mike
"""

def prepare_train_data(log, settings=None, verbose=True):
    import pandas as pd
    import numpy as np
    
    from dataprep.dataprep_helperfunctions import GetFileInfo, Sample, drop_cases, create_new_caseids
    from dataprep.inference_tables import CaseData, GetCaseStats
    from dataprep.padding import PadInputs
    from dataprep.format import InitialFormatting
    from dataprep.partition import MakeSplitCriterion, SplitAndReshape
    from dataprep.traindata import GenerateTrainData    
    #from dataprep.helperfunctions import 
    
    #print expectations right away
    print("#"*100)
    print("Friendly reminder:")
    print("date format must be %Y-%m-%d %H:%M:%S")
    print("caseid column must be named: 'id'")
    print("event column must be names: 'event")
    print("timestamp col must be named: 'time")
    print("other input features currently not supported...")
    print("#"*100)
    
    
    # format
    df = InitialFormatting(log)#, dateformat="%Y-%m-%d %H:%M:%S"
    
    
    # subsample
    df = Sample(df, maxcases=settings["max_cases"])
    
    df.index = list(range(0,len(df)))

    if settings["max_prefix_length"] == 0:
        # get max trace length in the log
        max_length = GetFileInfo(df)
    else:
        max_length = settings["max_prefix_length"]
    
    print("mode:",settings["split_mode"])
    print("*"*250)
    
    # create new caseids after dropping/sampling cases
    df = create_new_caseids(df)
    
    # make split criteria
    split_criterion = MakeSplitCriterion(df, trainsize=settings["train_ratio"], mode=settings["split_mode"]) # "event", "case"
    
    """
    Drop cases of length less than x
    """
    
    df = drop_cases(df, min_len=settings["min_prefix_length"])
    
    # create new caseids after dropping/sampling cases
    df = create_new_caseids(df)
    
    """
    Get predefined dummies from train distribution?
    """
    dummy_dict = {"train":True}

    print("Starting multiprocessing")
    import dataprep.multiprocessing_helpers as mph
    import dataprep.traindata_multiprocess as mpt
    
    import multiprocessing
    pool = multiprocessing.Pool(16)

    dflist = mph.make_df_list(df, idvar="id")
            
    # run multiprocessing of traindata generatior
    list_of_results = pool.map(mpt.GenerateTrainData, dflist)
    
    # restructure the outputs
    X = pd.concat([d["X"] for d in list_of_results])
    y = pd.concat([d["y"] for d in list_of_results])
    y_a = [d["y_a"] for d in list_of_results]
    y_t = [d["y_t"] for d in list_of_results]
    cases = pd.concat([d["X"] for d in list_of_results])
    y_a_varnames = list_of_results[0]["y_a_varnames"]
    dummy_dict = list_of_results[0]["dummy_dict"]

    # generate the train data: target etc.
    #X, y, y_a, y_t, cases, y_a_varnames, dummy_dict = GenerateTrainData(df)

    print("Multiprocessing done")
    
    #add dummy information to dataprep settings
    settings.update(dummy_dict)
    
    # pad the data
    padded_df = PadInputs(cases, 
                      X, 
                      max_prefix_len=max_length, 
                      standardize=settings["standardize_when_padding"],
                      verbose=settings["verbose"])
    
    """
    raw training data
    """
    
    # split and reshape data for RNNs
    X_train, X_test, y_t_train, y_t_test, y_a_train, y_a_test, y_train, y_test = SplitAndReshape(padded_df, 
                                                                                                 y_a, 
                                                                                                 y_t, 
                                                                                                 y, 
                                                                                                 split_criterion, 
                                                                                                 prefixlength=max_length)
   
    
    """
    collect results
    """

    #collect all datasets
    Input_data={"x_train":X_train,
                "x_test":X_test,
                "y_train":y_train,
                "y_test":y_test,
                "y_a_train":y_a_train,
                "y_a_test":y_a_test,
                "y_t_train":y_t_train,
                "y_t_test":y_t_test,
                "dataprep_settings":settings,
                "dummy_dict":dummy_dict}

    """
    case stats for inference tables
    """
    
    if settings["inference_tables"] == True:
        
        # get case data
        Case_Data = CaseData(df)
        
        # get case stats
        CaseStats = GetCaseStats(df.rename({"activity_no":"event_number"},axis=1), 
                             padded_df, 
                             Case_Data, 
                             y_t, 
                             y_a, 
                             y, 
                             prefixwindow=max_length, 
                             dateformat="%Y-%m-%d %H:%M:%S", 
                             drop_last_ev=settings["drop_last_event"])
        
        # make inference tables
        Inference = pd.merge(left=CaseStats, right=split_criterion, on="caseid",how="left")
    
        Inference_train = Inference.loc[Inference.trainset==True].drop("trainset",axis=1)
        Inference_test = Inference.loc[Inference.trainset==False].drop("trainset",axis=1)
        print("Inference train:",len(Inference_train))
        print("Inference test: ",len(Inference_test))
        
        
        Inference_data = {"Inference_train":Inference_train,
                    "Inference_test":Inference_test}
        
        #add inference table data to output dictionary
        Input_data.update(Inference_data)
        
    """
    last-state data for non-sequential models
    """
    if settings["last_state"] == True:
        
        """
        laststate 
        """
        from dataprep.laststate_helperfunctions import Split
        
        # split data: vanilla last-state m=1 data (verenich et al., 2018)
        ls_X_train, ls_X_test = Split(X, split_criterion)
        
        last_state_data = {"ls_x_train":ls_X_train,
                            "ls_x_test":ls_X_test}
        
        #add fs data to output dictionary
        Input_data.update(last_state_data)
    
    return Input_data


def prepare_inference_data(log, settings=None, dummy_dict=None, verbose=True):
    import pandas as pd
    import numpy as np
    
    from dataprep.dataprep_helperfunctions import GetFileInfo
    from dataprep.padding import PadInputs
    from dataprep.format import InitialFormatting
    from dataprep.traindata import GenerateTrainData    
    
    
    #print expectations right away
    print("#"*100)
    print("Friendly reminder:")
    print("date format must be %Y-%m-%d %H:%M:%S")
    print("caseid column must be named: 'id'")
    print("event column must be names: 'event")
    print("timestamp col must be named: 'time")
    print("other input features currently not supported...")
    print("#"*100)

    #format
    df = InitialFormatting(log, maxcases=settings["max_cases"], train=False)#, dateformat="%Y-%m-%d %H:%M:%S"
    df.index = list(range(0,len(df)))

    if settings["max_prefix_length"] == 0:
        # get max trace length in the log
        max_length = GetFileInfo(df)
    else:
        max_length = settings["max_prefix_length"]
    
    print("mode:",settings["split_mode"])
    print("*"*250)
    
    """
    Get predefined dummies from train distribution
    """
    #dummy_dict = settings["dummy_dict"]
    dummy_dict["train"] = False
    
      
    # generate the train data: target etc.
    X, _, _, _, cases, _, _ = GenerateTrainData(df,
                                            category_cols=settings["cat_features"], #"start_day"
                                            numeric_cols=settings["num_features"],
                                            droplastev=False,
                                            drop_end_target=settings["drop_last_activity_target_class"],
                                            get_activity_target=settings["generate_activity_target"],
                                            get_case_features = settings["generate_case_features"],
                                            dummify_time_features = settings["onehot_time_features"], 
                                            max_prefix_length = settings["max_prefix_length"],
                                            window_position=settings["trace_window_position"],
                                            predefined_dummies=dummy_dict)
    
    # pad the data
    padded_df = PadInputs(cases, 
                          X, 
                          max_prefix_len=max_length, 
                          standardize=settings["standardize_when_padding"],
                          verbose=settings["verbose"])
    
    """
    raw training data
    """
    
    from dataprep.dataprep_helperfunctions import Reshape
    
    # split and reshape data for RNNs
    X_train = Reshape(padded_df,
                     prefixlength=max_length)
   
    
    """
    collect results
    """

    #collect all datasets
    Input_data={"x_online":X_train,
                "dataprep_settings":settings}
        
    """
    last-state data for non-sequential models
    """
    if settings["last_state"] == True:
        
        """
        laststate 
        """
        from dataprep.laststate_helperfunctions import PrepInference
        
        # split data: vanilla last-state m=1 data (verenich et al., 2018)
        ls_X_train = PrepInference(X)
        
        last_state_data = {"ls_x_online":ls_X_train}
        
        #add fs data to output dictionary
        Input_data.update(last_state_data)
    
    return Input_data