# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 15:55:31 2022

@author: Mike
"""

def predict(data_objects, online_data):
    print("Model:",data_objects["modelname"])
    
    if data_objects["modelname"] == "vanilla_LSTM" or data_objects["modelname"] == "custom_LSTM":
        print("LSTM: predicting..")
                
        from tensorflow import keras
        import numpy as np
        model = keras.models.load_model(data_objects["model"])
        
        x_test = online_data["x_online"]
        x_test = np.asarray(x_test).astype('float32')
        
        # Predict on inference table
        y_pred = model.predict(x_test)
    
    
    if data_objects["modelname"] == "LS_RF":
        print("RF: predicting..")
        
        model = data_objects["model"]
        
        x_test = online_data["ls_x_online"]
        
        # Predict on inference table
        y_pred = model.predict(x_test)
    
    
    if data_objects["modelname"] == "LS_XGB":
        print("XGB: predicting..")
        
        model = data_objects["model"]
        
        x_test = online_data["ls_x_online"]
        
        # Predict on inference table
        y_pred = model.predict(x_test)
    
    return y_pred