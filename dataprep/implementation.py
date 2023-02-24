# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:42:48 2022

@author: Mike
"""

def load_data(curr_settings):
                  
    from data.mapping import mapping
    import os.path
    import pickle
    import pandas as pd
    
    #datasets = load_data(mapping, n=10000)
    file = curr_settings["dataprep_id"]
    
    #check if data exist, if yes, then load:
    if os.path.isfile('results/'+file+'.pickle'):
        print("loading dataset:",curr_settings["data"])
        with open('results/'+file+'.pickle', 'rb') as handle:
            data_objects = pickle.load(handle)
            
    #else, generate the data:
    if os.path.isfile('results/'+file+'.pickle') != True:
        print("preparing dataset:",curr_settings["data"])
        
        """
        settings for dataprep
        """
                
        PREP_SETTINGS = {"data":curr_settings["data"],
                         "dataprep_id":curr_settings["dataprep_id"],
                        
                        "inference_tables": int(curr_settings["data_inference_tables"]) == 1,
                        #"inference_tables_multiprocess": False,
                        "last_state": int(curr_settings["data_last_state"]) == 1,
                        "n_traces": int(curr_settings["data_n_traces"]),
                        
                        "max_prefix_length": int(curr_settings["data_max_prefix_length"]),
                        "min_prefix_length": int(curr_settings["data_min_prefix_length"]),
                        
                        "split_mode": curr_settings["data_split_mode"],
                        "train_ratio": curr_settings["data_train_ratio"],
                        
                        "generate_case_features": int(curr_settings["data_generate_case_features"]) == 1,
                        
                        "cat_features": int(curr_settings["data_cat_features"]) == 1, # this refer to the categorical features <<<<< need logic around this
                        "num_features": int(curr_settings["data_num_features"]) == 1, # this refer to the numerical features <<<<< need logic around this
                        
                        "standardize": int(curr_settings["data_standardize"]) == 1,
                        "padding": curr_settings["data_padding"], 
                        "trace_window_position": curr_settings["data_trace_window_position"],
                        "drop_last_event": int(curr_settings["data_drop_last_event"]) == 1,
                        
                        "onehot_time_features": int(curr_settings["data_onehot_time_features"]) == 1,
                        
                        "verbose": int(curr_settings["data_verbose"]) == 1
                        }
        
                
        """ load approach """
        print(PREP_SETTINGS)
                
        # generated prepared CSV           
        from data.evlog_standardisation import convert_to_standardised_csv, prepare_csv
        import shutil
        
        #fixed mapping for simulation data
        if curr_settings["data"] == "simulation":
            mapping = {"simulation":{'name': "simulation",
                                     'dest': 'data/simulation.csv',
                                     'type': 'csv',
                                     'sep': ',',
                                     'utc': False,
                                     'timestamp': '07/01/1970  18:59:43',
                                     'timeformat': '%d/%m/%Y %H:%M:%S',
                                     'keep_columns': ['id', 'event', 'start_datetime', 'resource', 'start_day', 'start_hour'],
                                     'new_colnames': ['caseid', 'activity','timestamp','resource', 'start_day','start_hour'],
                                     'cat_features': ['start_day', 'start_hour'],
                                     'num_features': []}}
            
            shutil.copyfile("results/simulated_log_" + str(curr_settings["RUN"]) + ".csv", 
                            'data/simulation.csv')
            file_mapping = mapping[PREP_SETTINGS["data"]]
            log = prepare_csv(file_mapping)
        
        
        #conditional mapping for benchmark data
        if curr_settings["data"] != "simulation":        
            with open('data/mapping.pickle', 'rb') as handle:
                mapping = pickle.load(handle)
            
            print("loading CSV from mapping")
            file_mapping = mapping[PREP_SETTINGS["data"]]
            log = prepare_csv(file_mapping)
        
        
        
        """ features """ 

        PREP_SETTINGS["cat_features"] = mapping[curr_settings["data"]]["cat_features"]
        
        PREP_SETTINGS["num_features"] = mapping[curr_settings["data"]]["num_features"]
                
        print("generating training data")
        from dataprep.dataprep_pipeline import prepare_train_data
        
        # prepare the data
        data_objects = prepare_train_data(log, 
                                          settings=PREP_SETTINGS, 
                                          verbose=PREP_SETTINGS["verbose"])
        
        #Store original log
        data_objects["log"] = log
        print(len(data_objects["log"]))
        
        # Store data preparation settings
        data_objects["PREP_SETTINGS"] = PREP_SETTINGS
        
        """ Implement unique filename for combination key xyz here """
        
        #Store to pickle
        with open('results/'+PREP_SETTINGS["dataprep_id"]+'.pickle', 'wb') as handle:
            pickle.dump(data_objects, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    return data_objects