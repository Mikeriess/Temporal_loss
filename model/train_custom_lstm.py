# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:16:39 2022

@author: Mike
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 15:36:19 2022

@author: Mike
"""
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization #, Input, Embedding, Dropout, Activation, Flatten
from tensorflow.keras.models import Sequential#, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau#, ModelCheckpoint#, CSVLogger
from tensorflow.keras.optimizers import Nadam, Adam, SGD, RMSprop

from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as Kc

# TF2: Disable eager execution
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.get_logger().setLevel('INFO')

# TF2: Mixed precision
#from tensorflow.keras.mixed_precision import experimental as mixed_precision
#mixed_precision.set_policy('mixed_float16')


import time
from datetime import datetime
import numpy as np
import pandas as pd


# Callback for tracking training time per epoch:
class TimeHistory(Kc.Callback): #callbacks.
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)





def train_model(data_objects, modelparams, wandb=False):
    #### Weights and biases ####
    
    if wandb==True:
        from wandb.keras import WandbCallback
        from wandb import init
        init()
    
    #### Load the data ########
    x_train, y_train = data_objects["x_train"], data_objects["y_train"]
    x_test, y_test = data_objects["x_test"], data_objects["y_test"]
    
    
    ##########################################################
    # Transformations: Input
    ##########################################################

    #Standardize to mean 0, sd 1
    print("standardizing X")

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()

    #Transform Train:
    n = x_train.shape[0]
    t = x_train.shape[1]
    k = x_train.shape[2]

    x_train_2d = x_train.reshape(n,t*k)

    sc.fit_transform(x_train_2d)
    x_train_2d = sc.transform(x_train_2d)

    x_train = x_train_2d.reshape(n,t,k)

    #Transform Test:
    sc = StandardScaler()
    
    n = x_test.shape[0]
    t = x_test.shape[1]
    k = x_test.shape[2]

    x_test_2d = x_test.reshape(n,t*k)

    sc.fit_transform(x_test_2d)
    x_test_2d = sc.transform(x_test_2d)

    x_test = x_test_2d.reshape(n,t,k)

    ##########################################################
    # Transformations: Target
    ##########################################################
    
    y_transformation = modelparams["y_transformation"]
    print("target transformation:",y_transformation)
        
    # Standardize to mean 0, sd 1
    if y_transformation == "standard":  
        # Initialize
        sc_train = StandardScaler()
        sc_test = StandardScaler()
        
        # Fit:
        sc_train.fit_transform(y_train)
        sc_test.fit_transform(y_test)
        
        # Transform:
        y_test = sc_test.transform(y_test)
        y_train = sc_train.transform(y_train)
        
    # 0-1 Range transformations:
    if y_transformation=="range":
        
        # Calculate values
        y_train_min = np.min(y_train)
        y_train_max = np.max(y_train)
        
        y_test_min = np.min(y_test)
        y_test_max = np.max(y_test)
        
        # Transform
        y_train = (y_train -  y_train_min)/(y_train_max - y_train_min)
        y_test = (y_test -  y_test_min)/(y_test_max - y_test_min)
            
    # Log-transform       
    if y_transformation=="log":
        
        y_train = np.log(1+y_train)
        y_test = np.log(1+y_test)
        
    if y_transformation=="none":
        print("no transformation to the target variable y")
    
    ##########################################################
    # Model Type/Architecture
    ##########################################################
    # Clear the TF graph
    K.clear_session()

    print("Input data shape:",x_train.shape)

    ############################################

    # Architecture related variables
    BLOCK_LAYERS = modelparams["BLOCK_LAYERS"]
    FULLY_CONNECTED = modelparams["FULLY_CONNECTED"]
    DROPOUT_RATE = modelparams["DROPOUT_RATE"]

    #####################################
    
    # Get shape of input data
    input_dim = (x_train.shape[1], x_train.shape[2])

    # Initialize model
    model = Sequential()

    # Add first LSTM block
    if BLOCK_LAYERS == 1:
        model.add(LSTM(FULLY_CONNECTED,  #implementation=2, 
                             recurrent_dropout=DROPOUT_RATE, 
                             input_shape=input_dim,
                             return_sequences=False))
        model.add(BatchNormalization())

        
    if BLOCK_LAYERS > 1:
        for i in range(0,BLOCK_LAYERS-1):
            
            if i == 0:
                model.add(LSTM(FULLY_CONNECTED, #implementation=2, 
                               input_shape=input_dim, #input_shape=(x_train.shape[1], x_train.shape[2]), 
                               recurrent_dropout=DROPOUT_RATE, 
                               return_sequences=True))
                model.add(BatchNormalization())
            if i > 0:
                model.add(LSTM(FULLY_CONNECTED, #implementation=2, 
                               recurrent_dropout=DROPOUT_RATE, 
                               return_sequences=True))
                model.add(BatchNormalization())
                
        model.add(LSTM(FULLY_CONNECTED, #implementation=2, 
                       #input_shape=input_dim, #input_shape=(x_train.shape[1], x_train.shape[2]), 
                       recurrent_dropout=DROPOUT_RATE, 
                       return_sequences=False))
        model.add(BatchNormalization())
        
    
    model.add(Dense(1, kernel_initializer='glorot_uniform')) #, dtype='float32' #Only the softmax is adviced to be float32 

    print(model.summary())
    
    ##########################################################
    # Loss function
    ##########################################################

    if modelparams["lossfunction"] == "MAE":
        loss_func = "mae"

    if modelparams["lossfunction"] == "MSE":
        loss_func = "mse"
    
    if modelparams["lossfunction"] == "huber":
        loss_func = "huber"
        loss_func = tf.keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss")

    # Get prefix log characteristics for calculating custom losses:
    max_prefix_length = x_train.shape[1]
    n_obs = x_train.shape[0]
    cases = int(n_obs/max_prefix_length)
    
    if modelparams["lossfunction"] == "MAE_td":
        from model.losses import MAE_td, generate_prefix_weights
        time_weights = generate_prefix_weights(cases, max_prefix_length)
        loss_func = MAE_td(time_weights, 
                           alpha=modelparams["alpha"], 
                           beta=modelparams["beta"], 
                           gamma=modelparams["gamma"])
    
    if modelparams["lossfunction"] == "MAE_Etd":
        from model.losses import MAE_Etd, generate_prefix_weights
        time_weights = generate_prefix_weights(cases, max_prefix_length)
        loss_func = MAE_Etd(time_weights, alpha=modelparams["alpha"])
        
    if modelparams["lossfunction"] == "MAE_Mtd":
        from model.losses import MAE_Mtd, generate_prefix_weights
        time_weights = generate_prefix_weights(cases, max_prefix_length)
        loss_func = MAE_Mtd(time_weights, max_prefix_length, alpha=modelparams["alpha"])
    
    if modelparams["lossfunction"] == "MAE_Ptd":
        from model.losses import MAE_Ptd, generate_prefix_weights
        time_weights = generate_prefix_weights(cases, max_prefix_length)
        loss_func = MAE_Ptd(time_weights, max_prefix_length, alpha=modelparams["alpha"])
        
    
    ##########################################################
    # Train assist
    ##########################################################

    
    checkpoint_filepath = 'results/experiment_'+str(data_objects["RUN"])+"_model.h5"
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                            save_weights_only=False,
                                                            monitor='val_mae',
                                                            mode='min',
                                                            save_best_only=True)
    data_objects["model"] = checkpoint_filepath

    time_callback = TimeHistory()
    
    # Stop training early if no improvements are made
    early_stopping = EarlyStopping(patience=modelparams["early_stop"])    #42 for the navarini paper

    # Reduce learning rate (same for both intial and final)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', 
                                   factor=0.5, 
                                   patience=modelparams["lr_reduce"], 
                                   verbose=1, 
                                   mode='auto', 
                                   epsilon=0.0001, 
                                   cooldown=0, 
                                   min_lr=0)

    # Optimizer
    if modelparams["optimizer"] == "ADAM":
        optimizer = Adam(learning_rate=modelparams["learningrate"])

    if modelparams["optimizer"] == "NADAM":
        optimizer = Nadam(learning_rate=modelparams["learningrate"], beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)  #verenich values

    if modelparams["optimizer"] == "SGD":
        optimizer = SGD(learning_rate=modelparams["learningrate"], momentum=modelparams["sgd_momentum"])

    if modelparams["optimizer"] == "RMSPROP":
        optimizer = RMSprop(learning_rate=modelparams["learningrate"], rho=0.9, epsilon=1e-08, decay=0.0) #verenich values
        
    print("y_train mean:",np.mean(y_train))
    print("x_train mean:",np.mean(x_train))
    print("x_train shape:",x_train.shape)
    print("y_test mean:",np.mean(y_test))
    print("x_test mean:",np.mean(x_test))
    print("x_test shape:",x_test.shape)

    # Compile model
    model.compile(loss=loss_func, 
          optimizer=optimizer, 
          metrics=["mae"])

    # Store starttime
    start_time = time.time()
    
    # Check if W&B are to be used for storing the experiments
    if wandb==True:
        model.fit(x_train, y_train,
                  batch_size=modelparams["batch_size"],
                      callbacks=[early_stopping,
                                model_checkpoint,
                                lr_reducer,
                                WandbCallback(),
                                time_callback],
                  epochs=modelparams["epochs"],
                  verbose=1,
                  #validation_steps=128,
                  validation_split=0.2)
    else:
        history = model.fit(x_train, y_train,
                  batch_size=modelparams["batch_size"],
                      callbacks=[early_stopping,
                                model_checkpoint,
                                lr_reducer,
                                #WandbCallback(),
                                time_callback],
                  epochs=modelparams["epochs"],
                  verbose=1,
                  #validation_steps=128,
                  validation_split=0.2)
    
    #get best validation accuracy
    TRAIN_MAE = np.min(history.history["mae"])
    VAL_MAE = np.min(history.history["val_mae"])
    
    # REPLACE with best version of the model (checkpoint)
    model = load_model(checkpoint_filepath, compile=False)

    # Store endtime
    end_time = time.time()

    # Store epoch times in already existing CSV-history
    epoch_times = time_callback.times
    
    ########################################################
    # PREDICT
    ########################################################
    
    # Predict on inference tables
    y_pred_test = model.predict(x_test, verbose=1, batch_size=2048)
    y_pred_train = model.predict(x_train, verbose=1, batch_size=2048)
    
    # Load inference tables
    Inference_test = data_objects["Inference_test"] 
    Inference_train = data_objects["Inference_train"] 
    
    # Add predictions to inference table
    Inference_test["y_pred"] = y_pred_test
    Inference_train["y_pred"] = y_pred_train
    
    
    # Log transform inverse  
    if y_transformation == "log":    
        Inference_test["y_pred"] = np.exp(y_pred_test)-1
        Inference_test["y_test"] = np.exp(y_test)-1
        
        Inference_train["y_pred"] = np.exp(y_pred_train)-1
        Inference_train["y_train"] = np.exp(y_train)-1
        print("Inverse transforming predictions: log")
        
    # Range transform inverse  
    if y_transformation == "range":
        Inference_test["y_pred"] = (y_pred_test * (y_test_max - y_test_min)) + y_test_min
        Inference_test["y_test"] = (y_test * (y_test_max - y_test_min)) + y_test_min
        
        
        Inference_train["y_pred"] = (y_pred_train * (y_train_max - y_train_min)) + y_train_min
        Inference_train["y_train"] = (y_train * (y_train_max - y_train_min)) + y_train_min
        print("Inverse transforming predictions: range")
        
          
    # Standardize backtransform
    if y_transformation == "standard":  
        
        #Transform Test:
        y_pred_test = sc_test.inverse_transform(y_pred_test)
        Inference_test["y_pred"] = y_pred_test
        
        
        y_test = sc_test.inverse_transform(y_test)
        Inference_test["y_test"] = y_test        
        
        
        #Transform Test:
        y_pred_train = sc_train.inverse_transform(y_pred_train)
        Inference_train["y_pred"] = y_pred_train
        
        
        y_train = sc_train.inverse_transform(y_train)
        Inference_train["y_train"] = y_train        
        print("Inverse transforming predictions: standard")
    
    # no transformation
    if y_transformation == "none":
        print("nb: no transformation of predictions..")
    
  
   
    ########################################################
    # Store results
    ########################################################
    
    # store train history
    pd.DataFrame(history.history).to_csv("results/train_hist_"+str(data_objects["RUN"])+".csv",index=False)
    
    ## Store info in data object
    data_objects["model"] = checkpoint_filepath
    #model.save(data_objects["model"])
    
    data_objects["training_history"] = pd.DataFrame(history.history)
    
    data_objects["TRAIN_MAE"] = TRAIN_MAE/(24.0*3600)
    data_objects["VAL_MAE"] = VAL_MAE/(24.0*3600)
    
    data_objects["Inference_test"] = Inference_test
    data_objects["Inference_train"] = Inference_train
    
    data_objects["modelname"] = "custom_LSTM"
    data_objects["epoch_times"] = epoch_times

    ## Store the parameters in data object:
    data_objects["model_params"] = modelparams

    #Store number of epochs
    data_objects["epochs"] = modelparams["epochs"]

    ## Log the full time for training:
    time_sec = end_time - start_time

    ## Store train time in data object
    data_objects["train_time"] = time_sec

    return data_objects