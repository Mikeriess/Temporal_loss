# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:23:11 2022

@author: Mike
"""



"""
dataset profiling
"""

import pickle
import pandas as pd
import numpy as np



def dataset_profiling(inf_train, inf_test, event_log, dataprep_settings):
    
    """
    Statistics: raw data
    """
    
    def calculate_case_metrics_raw(event_log):
        # Number of cases
        n_cases = len(set(event_log.id))
        
        # Max trace length of raw event log
        trace_lengths=[]
        case_durations=[]
        
        for i in set(event_log.id):
            sub = event_log.loc[event_log.id == i]
            sub.index = list(range(0,len(sub)))
            trace_lengths.append(len(sub))
            
            start_time = sub.loc[0]["time"]
            end_time = sub.loc[len(sub)-1]["time"]
            
            duration_days = (end_time - start_time).days
            case_durations.append(duration_days)
            
        max_trace_length = np.max(trace_lengths)
        avg_trace_length = np.mean(trace_lengths)
        
        # Average case duration
        avg_case_duration = np.mean(case_durations)
        return n_cases, max_trace_length, avg_trace_length, avg_case_duration
    
    # calculate for raw
    n_cases, max_trace_length, avg_trace_length, avg_case_duration = calculate_case_metrics_raw(event_log)
    
    """
    Statistics: processed data
    """
    
    def calculate_case_metrics(event_log):
        # Number of cases
        n_cases = len(set(event_log.caseid))
        
        # Max trace length of raw event log
        trace_lengths=[]
        case_durations=[]
        
        for i in set(event_log.caseid):
            sub = event_log.loc[event_log.caseid == i]
            sub.index = list(range(0,len(sub)))
            trace_lengths.append(len(sub))
                        
            duration_days = sub.loc[len(sub)-1]["caseduration_days"]
            case_durations.append(duration_days)
            
        max_trace_length = np.max(trace_lengths)
        avg_trace_length = np.mean(trace_lengths)
        
        # Average case duration
        avg_case_duration = np.mean(case_durations)
        return n_cases, max_trace_length, avg_trace_length, avg_case_duration
    
    # Truncation
    truncation_length = dataprep_settings["max_prefix_length"]
    
    train_n_cases, train_max_trace_length, train_avg_trace_length, train_avg_case_duration = calculate_case_metrics(inf_train)
    test_n_cases, test_max_trace_length, test_avg_trace_length, test_avg_case_duration = calculate_case_metrics(inf_test)
    
    dropped_due_to_censoring = n_cases - (train_n_cases + test_n_cases)
    
    """
    collecting all metrics into one table
    """
    
    # Collect metrics in table
    metrics = {"raw_n_cases":n_cases,
               "raw_max_trace_length":max_trace_length,
               "raw_avg_trace_length":np.round(avg_trace_length,decimals=2),
               "raw_avg_case_duration":np.round(avg_case_duration,decimals=2),
               "processed_truncation_length":truncation_length,
               "processed_n_cases_train":train_n_cases,
               "processed_n_cases_test":test_n_cases,
               "processed_censored":dropped_due_to_censoring,
               "processed_train_max_trace_length":train_max_trace_length,
               "processed_train_avg_trace_length":np.round(train_avg_trace_length,decimals=2),
               "processed_train_avg_case_duration":np.round(train_avg_case_duration,decimals=2),
               "processed_test_max_trace_length":test_max_trace_length,
               "processed_test_avg_trace_length":np.round(test_avg_trace_length,decimals=2),
               "processed_test_avg_case_duration":np.round(test_avg_case_duration,decimals=2)
               }
    #metrics
    
    return metrics

"""
Iterate over multiple datasets
"""

files = ["experiment_0","experiment_1","experiment_2","experiment_3","experiment_4"]
all_datasets = []
    

for file in files:
        
    with open('../results/'+file+'.pickle', 'rb') as handle:
        data_objects = pickle.load(handle)
    
    #load the raw training data
    traindata = data_objects["traindata"]
        
    ## Settings
    experiment_settings = data_objects["curr_settings"]
    dataprep_settings = traindata["dataprep_settings"]
    
    ## RAW data
    
    #load full eventlog
    event_log = traindata["log"]
    
    ## PROCESSED data
    
    #load inference tables, based on temporal split
    inf_train = traindata["Inference_train"]
    inf_test = traindata["Inference_test"]

    metrics = dataset_profiling(inf_train, inf_test, event_log, dataprep_settings)
    
    all_datasets.append(metrics)

"""
collect results
"""

metrics = pd.DataFrame(all_datasets)

"""
plots
"""







