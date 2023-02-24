# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:46:33 2022

@author: Mike
"""

def prepare_train_data(log, settings=None, verbose=True):
    import pandas as pd
    import numpy as np
    import time
    
    from dataprep.dataprep_helperfunctions import GetFileInfo, Sample, drop_cases, create_new_caseids, convert_to_float32
    from dataprep.inference_tables import CaseData, GetCaseStats
    from dataprep.padding import pad_cases_w_merge, pad_cases
    from dataprep.format import InitialFormatting
    from dataprep.partition import MakeSplitCriterion, SplitAndReshape
    from dataprep.traindata import GenerateTrainData    
    #from dataprep.helperfunctions import 
    
    #print expectations right away
    print("#"*100)
    print("Friendly reminder:")
    print("- date format must be %Y-%m-%d %H:%M:%S")
    print("- caseid column must be named: 'id'")
    print("- event column must be names: 'event")
    print("- timestamp col must be named: 'time")
    print("#"*100)
    
    
    # format
    df = InitialFormatting(log)#, dateformat="%Y-%m-%d %H:%M:%S"
    
    print("/"*100)
    print("convert_to_float32")
    df = convert_to_float32(df)
        
    #print("/"*100)
    # subsample
    df = Sample(df, maxcases=settings["n_traces"])    
    df.index = list(range(0,len(df)))

    print("/"*100)
    if settings["max_prefix_length"] == 0:
        # get max trace length in the log
        max_length = GetFileInfo(df)
    else:
        max_length = settings["max_prefix_length"]
    
    print("mode:",settings["split_mode"])
    print("*"*100)
    
    # create new caseids after dropping/sampling cases
    print("/"*100)
    print("create_new_caseids")
    df = create_new_caseids(df)
    
    # make split criteria
    print("/"*100)
    print("MakeSplitCriterion")
    split_criterion = MakeSplitCriterion(df, trainsize=settings["train_ratio"], mode=settings["split_mode"]) # "event", "case"
    
    """
    Drop cases of length less than x
    """
    
    df = drop_cases(df, min_len=settings["min_prefix_length"])
    
    # print("/"*100)
    # print("convert_to_float32")
        
    # # create new caseids after dropping/sampling cases
    # df = convert_to_float32(df)
    
    """
    Get predefined dummies from train distribution
    """
    dummy_dict = {"train":True}
    
    print("/"*100)
    print("GenerateTrainData")
        
    # generate the train data: target etc.
    X, y, cases, dummy_dict = GenerateTrainData(df,
                                                    category_cols=settings["cat_features"], #"start_day"
                                                    numeric_cols=settings["num_features"],
                                                    droplastev=settings["drop_last_event"],
                                                    get_case_features = settings["generate_case_features"],
                                                    dummify_time_features = settings["onehot_time_features"], 
                                                    max_prefix_length = settings["max_prefix_length"],
                                                    window_position=settings["trace_window_position"],
                                                    predefined_dummies=dummy_dict)
    
    #add dummy information to dataprep settings
    settings.update(dummy_dict)
    
    # print("/"*100)
    # print("convert_to_float32: X")
    
    # # create new caseids after dropping/sampling cases
    # X = convert_to_float32(X)
    
    
    print("/"*100)
    print("PadInputs")
    # pad the data
    
    padded_df = pad_cases(cases, 
                                    X, 
                                    max_prefix_len=max_length, 
                                    padding=settings["padding"],
                                    verbose=settings["verbose"])
    
    
    # padded_df = pad_cases_w_merge(cases, 
    #                                 X, 
    #                                 max_prefix_len=max_length, 
    #                                 padding=settings["padding"],
    #                                 verbose=settings["verbose"])
    
    # print("/"*100)
    # print("convert_to_float32: padded_df")
    # padded_df = convert_to_float32(padded_df)
    
    
    # print("identical?")
    # print(padded_df.shape)
    # print(padded_df2.shape)
    # print(padded_df.columns.tolist() == padded_df2.columns.tolist())
        
    # print("Nans in generated data:")
    # print("padded_df",np.sum(padded_df.isna().sum()))
    # print("padded_df2",np.sum(padded_df2.isna().sum()))
    # padded_df.to_csv("padded_df.csv",index=False)
    # padded_df2.to_csv("padded_df2.csv",index=False)
    # print(stophere)
    
    """
    raw training data
    """
    
    print("/"*100)
    print("SplitAndReshape")
    # split and reshape data for RNNs
    X_train, X_test, y_train, y_test = SplitAndReshape(padded_df, 
                                                        X,
                                                        y, 
                                                        split_criterion, 
                                                        prefixlength=max_length)
   
    
    
    # print(stophere)
    """
    collect results
    """

    #collect all datasets
    Input_data={"x_train":X_train,
                "x_test":X_test,
                "y_train":y_train,
                "y_test":y_test,
                "dataprep_settings":settings,
                "dummy_dict":dummy_dict}
    
        
    """
    case stats for inference tables
    """
    
    if settings["inference_tables"] == True:
        
        print("/"*100)
        print("CaseData")
        # get case data
        Case_Data = CaseData(df)
        
        print("/"*100)
        print("GetCaseStats")
                
        # get case stats
        CaseStats = GetCaseStats(df.rename({"activity_no":"event_number"},axis=1), 
                          padded_df, 
                          Case_Data, 
                          y, 
                          prefixwindow=max_length, 
                          dateformat="%Y-%m-%d %H:%M:%S", 
                          drop_last_ev=settings["drop_last_event"],
                          verbose=settings["verbose"])

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
        
        #add inference table data to output dictionary
        Input_data.update(Inference_data)
        
    """
    last-state data for non-sequential models
    """
    
    print("/"*250)
    print("last_state")
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
