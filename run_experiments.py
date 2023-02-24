# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 09:54:34 2022

@author: Mike
"""

import pandas as pd
#pd.set_option('display.max_columns', None)
import numpy as np
import pickle

import warnings
warnings.filterwarnings('ignore')

"""
Load experiments
"""

experiments = pd.read_csv("results/experiments.csv")


"""
Load simulation settings
"""

results = []
inference_tables = []

for run in experiments.index:
    print("RUN:",run)
    
    # fix random seed for reproducibility
    np.random.seed(int(run))
    
    """
    Settings from experiments
    """
    curr_settings = experiments.loc[run]
    
    if curr_settings.Done == 0:
              
        
        """
        load or create the data
        """
        if curr_settings["data"] == "simulation":
            """
            ##################################################################
            Generate simulated data
            """
                        
            from simulation.implementation import run_simulation
            log, SIM_SETTINGS = run_simulation(curr_settings)
            
            from dataprep.implementation import load_data          
            data_objects = load_data(curr_settings)
            
        if curr_settings["data"] != "simulation":
            
            """
            ##################################################################
            Load event-log data
            """           
            
            from dataprep.implementation import load_data          
            data_objects = load_data(curr_settings)
                    
        
        """
        Train a model
        """
        
        
        if curr_settings["modelname"] == "LSTM_vanilla":
            from model.train_vanilla_lstm import train_model
        
        if curr_settings["modelname"] == "LSTM":
            from model.train_custom_lstm import train_model
        
        if curr_settings["modelname"] == "RF_last_state":
            from model.train_laststate_rf import train_model
        
        if curr_settings["modelname"] == "XGB_last_state":
            from model.train_laststate_xgb import train_model
        
        
        
        """
        model settings
        """
    
        modelparams = {"BLOCK_LAYERS":int(curr_settings["num_blocks"]),
    
                        "DROPOUT_RATE":curr_settings["dropout"],
    
                        "FULLY_CONNECTED":int(curr_settings["num_units"]),
    
                        ### common:
                        "batch_size":curr_settings["batch_size"],
                        "learningrate":curr_settings["learningrate"], 
                        "optimizer":curr_settings["optimizer"],
                        "epochs":int(curr_settings["epochs"]),
                        "early_stop":int(curr_settings["early_stop"]),
                        "lr_reduce":int(curr_settings["lr_reduce"]),
                        "sgd_momentum":curr_settings["sgd_momentum"],
    
                        #misc
                        "lossfunction":curr_settings["loss_function"],
                        "y_transformation":curr_settings["y_transformation"],
    
                         # Get the loss function parameters:
                         "alpha":curr_settings["alpha"],
                         "beta":curr_settings["beta"],
                         "gamma":curr_settings["gamma"],
                         
                         #Other models
                         "n_estimators":curr_settings["n_estimators"],
                         "max_depth":curr_settings["max_depth"],
                         "eta":curr_settings["eta"],
                         "subsample":curr_settings["subsample"],
                         "colsample_bytree":curr_settings["colsample_bytree"],
    
                       }
            
            
        print(modelparams)
        data_objects["RUN"] = run
        
        data_objects = train_model(data_objects, modelparams, wandb=False)
        
        
        # if nan, mark as failed
        if np.count_nonzero(np.isnan(data_objects["Inference_test"]["y_pred"])) != 0:
            experiments.at[run,"Failed"] = 1
            
            #store status - run it again
            experiments.at[run,"Done"] = 0
        
        # if nan, dont do anything
        if np.count_nonzero(np.isnan(data_objects["Inference_test"]["y_pred"])) == 0:
            
            """
            Evaluate the model
            """
            
            
            from evaluate.evaluate_rt_model import evaluate_rt_model        
            data_objects = evaluate_rt_model(data_objects, partition=curr_settings["evaluation_partition"])
            
            """
            Store the results
            """
            
            curr_settings.Done = 1
            
            #curr_settings["RES_num_events"] = len(log)    
            curr_settings["MAE"] = data_objects["report"].loc[0]["MAE"]    
            curr_settings["Time_of_evaluation"] = data_objects["report"].loc[0]["Time_of_evaluation"]    
            curr_settings["Train_duration"] = data_objects["report"].loc[0]["Train_duration"]
            
            #get multiple cols from prefix performance
            prefix_TC_t = data_objects["prefix_TC_t"].loc[0]
            prefix_MAE_t = data_objects["prefix_MAE_t"].loc[0]
            prefix_AE_CUMSUM_t = data_objects["prefix_AE_CUMSUM_t"].loc[0]
            
            #add them to table
            curr_settings = pd.concat([curr_settings,prefix_MAE_t,prefix_TC_t,prefix_AE_CUMSUM_t])
            
            out = pd.DataFrame(curr_settings.T)
                
            results.append(out)
        
            inference_i = data_objects["Inference_test"]
            inference_i["RUN"] = run
            inference_tables.append(inference_i)
            
            """
            Store all results as pickle
            """
            
            all_data = {"curr_settings":curr_settings,
                        "PREP_SETTINGS":data_objects["PREP_SETTINGS"],
                        "modelparams":modelparams,
                        "model":data_objects["model"],
                        "TRAIN_MAE":data_objects["TRAIN_MAE"],
                        "VAL_MAE":data_objects["VAL_MAE"],
                        "training_history":data_objects["training_history"],
                        "prefix_MAE_t":prefix_MAE_t,
                        "prefix_TC_t":prefix_TC_t,
                        "prefix_AE_CUMSUM_t":prefix_AE_CUMSUM_t,
                        "report":data_objects["report"],
                        "inference_train":data_objects["Inference_train"],
                        "inference_test":data_objects["Inference_test"]}
    
            #store results
            import pickle
            with open('results/experiment_'+str(run)+'.pickle', 'wb') as handle:
                pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            #inference tables
            with open('results/inference_tables.pickle', 'wb') as handle:
                pickle.dump(inference_tables, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            #experiment inference table            
            inference_i.to_csv("results/inference_"+str(run)+".csv",index=False)
    
            
            #store results
            if run == 0:
                # Create the needed variables
                import datetime
                import time
                experiments["Finished_at"] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                
                experiments["MAE"] = curr_settings["MAE"]
                experiments["TRAIN_MAE"] = data_objects["TRAIN_MAE"]
                experiments["VAL_MAE"] = data_objects["VAL_MAE"]
                
                experiments["TC_i"] = data_objects["report"]["TC_i_avg"]
                experiments["MAE_PTD"] = data_objects["report"]["MAE_PTD"]
                experiments[prefix_MAE_t.index[0]], experiments[prefix_MAE_t.index[1]], experiments[prefix_MAE_t.index[2]], experiments[prefix_MAE_t.index[3]], experiments[prefix_MAE_t.index[4]] = [prefix_MAE_t[0], prefix_MAE_t[1], prefix_MAE_t[2],prefix_MAE_t[3],prefix_MAE_t[4]]
                experiments[prefix_TC_t.index[0]], experiments[prefix_TC_t.index[1]], experiments[prefix_TC_t.index[2]], experiments[prefix_TC_t.index[3]], experiments[prefix_TC_t.index[4]] = [prefix_TC_t[0], prefix_TC_t[1], prefix_TC_t[2],prefix_TC_t[3],prefix_TC_t[4]]
            
            experiments.at[run,"TRAIN_MAE"] = data_objects["TRAIN_MAE"]
            experiments.at[run,"VAL_MAE"] = data_objects["VAL_MAE"]
            experiments.at[run,"MAE"] = curr_settings["MAE"]
            experiments.at[run,"TC_i"] = data_objects["report"]["TC_i_avg"]
            experiments.at[run,"MAE_PTD"] = data_objects["report"]["MAE_PTD"]
            
            experiments.at[run,prefix_MAE_t.index[0]] = prefix_MAE_t[0]
            experiments.at[run,prefix_MAE_t.index[1]] = prefix_MAE_t[1]
            experiments.at[run,prefix_MAE_t.index[2]] = prefix_MAE_t[2]
            experiments.at[run,prefix_MAE_t.index[3]] = prefix_MAE_t[3]
            experiments.at[run,prefix_MAE_t.index[4]] = prefix_MAE_t[4]
            
            experiments.at[run,prefix_TC_t.index[0]] = prefix_TC_t[0]
            experiments.at[run,prefix_TC_t.index[1]] = prefix_TC_t[1]
            experiments.at[run,prefix_TC_t.index[2]] = prefix_TC_t[2]
            experiments.at[run,prefix_TC_t.index[3]] = prefix_TC_t[3]
            experiments.at[run,prefix_TC_t.index[4]] = prefix_TC_t[4]
             
            #store status
            experiments.at[run,"Done"] = 1
            
        #store status of experiments
        experiments.to_csv("results/experiments.csv", index=False)