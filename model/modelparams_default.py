# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 15:55:31 2022

@author: Mike
"""

# Generate the architecture:    
modelparams = {"BLOCK_LAYERS":3,
               "DROPOUT_RATE":0.2,
               "FULLY_CONNECTED":100,

               ### common:
               "batch_size":64, 
               "learningrate":0.001,
               "optimizer":"Nadam",
               "epochs":20,

               #misc
               "lossfunction":"MAE",
               "y_transformation":"range",

                # Get the loss function parameters:
                "alpha":0.5,
                "beta":0.5,
                "gamma":0.5,

               }