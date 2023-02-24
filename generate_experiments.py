# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 09:53:21 2022

@author: Mike
"""


"""
Model/experiment settings
"""

from experiment.DoE import build_full_fact, fix_label_values

             
run_settings = {"modelname":["LSTM"], #["LSTM-vanilla","LSTM","XGB_last_state","RF_last_state"]                
                    # LSTM-vanilla (one block, n units)
                    # LSTM (Navarini-architecture)
                    # Last_state: Random forest
                    # Last_state: XGB
                    
                # DATA   
                "data":["sepsis", "helpdesk","traffic_fines", "hospital_billing"], #"simulation", "sepsis", "traffic_fines", "hospital_billing", "helpdesk"
                    
                # LSTM
                "num_units":[999],
                "num_blocks":[999],
                
                "epochs":[200],#,75,100, 200],
                "batch_size":[999], #
                "learningrate":[0.0], #ratio between LR and working LR on hospital billing
                "optimizer":["NONE"], #"ADAM", "NADAM", "SGD"
                "dropout":[0.2],
                
                "early_stop":[10],
                "lr_reduce":[5],
                "sgd_momentum":[0.9],
                
                # LSTM-loss
                "loss_function":["MAE","MAE_Mtd","MAE_Ptd","MAE_Etd"], #["MAE","MSE","MAE_Etd","MAE_Mtd","MAE_Ptd", "MAE_td", "huber"][0]
                
                # MAE_TD parameters
                "alpha":[0.5], 
                "beta":[0.5], #(1-beta) * mean(abs(y_pred-y_true)) + beta * mean((abs(y_pred-y_true))/alpha*t)
                "gamma":[0], # unused in current version
                
                # XGB/RF
                "n_estimators":[250],
                "max_depth":[6],
                "eta":[0.02],
                "subsample":[0.4],
                "colsample_bytree":[0.5],
                
                "y_transformation":["none"], #"log","range","standard"
                
                
                # SIMULATION SETTINGS
                "sim_save_eventlog":[1],
                "sim_statespace_size":[6],
                "sim_number_of_traces":[1050],                  
                "sim_process_entropy":["med_entropy"],#["min_entropy","med_entropy","max_entropy"],
                "sim_process_type":["memory"],#["memory","memoryless"],               
                "sim_process_memory":[4],                
                "sim_med_ent_e_steps":[5],
                "sim_med_ent_n_transitions":[5],
                "sim_med_ent_max_trials":[5],
                "sim_inter_arrival_time":[1.5],
                #lambda parameter of process noise
                "sim_process_stability_scale":[0.5],
                #probability of getting an agent
                "sim_resource_availability_p":[0.25],                          
                #waiting time in days, when no agent is available      
                "sim_resource_availability_n":[3],
                #waiting time in days, when no agent is available
                "sim_resource_availability_m":[0.041], 
                #variation between activity durations
                "sim_activity_duration_lambda_range":[0.5],
                #time-unit for a full week: days = 7, hrs = 24*7, etc.
                "sim_deterministic_offset_W":["weekdays"], # make_workweek(["weekdays","all-week"][1]),
                
                "Deterministic_offset_u":[7],
                
                
                
                # DATA PREP SETTINGS
                "data_inference_tables":[1],                           
                "data_last_state":[1],                
                "data_n_traces":[1000000000000000],
                
                "data_max_prefix_length":[20], #20 in verenich et al 2019
                "data_min_prefix_length":[3],  #as last event is dropped, and the model should predict more than one value
                
                "data_split_mode":["case"], #"case","event"
                "data_train_ratio":[0.5],
                
                "data_generate_case_features":[1], #0 = no, 1=yes
                
                "data_cat_features":[1],#"event"
                "data_num_features":[1],
                
                "data_standardize":[1],
                
                "data_padding":["leading"], #"leading","trailing"
                "data_trace_window_position":["last_k"],
                
                "data_drop_last_event":[1],
                "data_onehot_time_features":[1],
                                                
                "data_verbose":[1],
                
                # EVALUATION: where to calculate stats from
                "evaluation_partition":["test"], #test, train
                
                # Repeated runs
                "repetition":[1,2,3,4,5,6,7,8,9,10]}


"""
Simulation settings
"""



# Generate a full factorial:
df = build_full_fact(run_settings)#[0:2]

# Get string values back
df = fix_label_values(df, run_settings, variables = ["modelname",
                                                     "data",
                                                     "optimizer",
                                                     "y_transformation",
                                                     "loss_function",
                                                     "sim_process_entropy",
                                                     "sim_process_type",
                                                     "sim_deterministic_offset_W",
                                                     "data_split_mode",
                                                     "data_padding",
                                                     "data_trace_window_position",
                                                     "evaluation_partition"
                                                     ])


#change dtypes
df.n_estimators = df.n_estimators.astype(int)
df.max_depth = df.max_depth.astype(int)

df.epochs = df.epochs.astype(int)
df.batch_size = df.batch_size.astype(int)
df.num_units = df.num_units.astype(int)
df.num_blocks = df.num_blocks.astype(int)

#df.statespace_size = df.statespace_size.astype(int)



"""
Conditional hyper-parameter settings for each event-log
"""

df.loc[(df['data'] == 'sepsis'), 'learningrate'] = 0.1
df.loc[(df['data'] == 'sepsis'), 'num_blocks'] = 2
df.loc[(df['data'] == 'sepsis'), 'num_units'] = 100
df.loc[(df['data'] == 'sepsis'), 'batch_size'] = 1024
df.loc[(df['data'] == 'sepsis'), 'optimizer'] = 'NADAM'

df.loc[(df['data'] == 'helpdesk'), 'learningrate'] = 0.1
df.loc[(df['data'] == 'helpdesk'), 'num_blocks'] = 1
df.loc[(df['data'] == 'helpdesk'), 'num_units'] = 100
df.loc[(df['data'] == 'helpdesk'), 'batch_size'] = 2048
df.loc[(df['data'] == 'helpdesk'), 'optimizer'] = 'SGD'

df.loc[(df['data'] == 'traffic_fines'), 'learningrate'] = 0.1
df.loc[(df['data'] == 'traffic_fines'), 'num_blocks'] = 1
df.loc[(df['data'] == 'traffic_fines'), 'num_units'] = 100
df.loc[(df['data'] == 'traffic_fines'), 'batch_size'] = 128
df.loc[(df['data'] == 'traffic_fines'), 'optimizer'] = 'NADAM'

df.loc[(df['data'] == 'hospital_billing'), 'learningrate'] = 0.01
df.loc[(df['data'] == 'hospital_billing'), 'num_blocks'] = 2
df.loc[(df['data'] == 'hospital_billing'), 'num_units'] = 100
df.loc[(df['data'] == 'hospital_billing'), 'batch_size'] = 128
df.loc[(df['data'] == 'hospital_billing'), 'optimizer'] = 'NADAM'


"""
Generate unique string for all data settings
"""

#list all data related factors
data_factors = ["data_n_traces","data_max_prefix_length","data_min_prefix_length","data_split_mode","data_train_ratio","data_generate_case_features","data_cat_features",
                "data_num_features","data_standardize","data_padding","data_trace_window_position","data_drop_last_event","data_onehot_time_features"]

#placeholder
df["dataprep_id"] = "dataprep"

#concatenate all data related factors to a single string to get each combination as a "class"
for fact in data_factors:
    df["dataprep_id"] = df["dataprep_id"] + "_" +  df[fact].astype(str)

#change the string to a categorical variable 
df["dataprep_id"] = df.dataprep_id.astype("category")
df["dataprep_id"] = df.dataprep_id.cat.codes

#convert back to a simple ID
df["dataprep_id"] = df["data"] + "_" + df["dataprep_id"].astype(str)


"""
sort experiments by dataset
"""
sorter = ["sepsis", "helpdesk","traffic_fines","hospital_billing"]
df.sort_values(by="data", key=lambda column: column.map(lambda e: sorter.index(e)), inplace=True)
df["RUN"] = list(range(0,len(df)))

print(df)

"""
Save experiments table
"""

df.to_csv("results/experiments.csv",index=False)

