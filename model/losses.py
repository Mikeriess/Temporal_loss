# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 15:36:19 2022

@author: Mike
"""

def generate_prefix_weights(cases, max_prefix_length):
    import numpy as np
    ## Generate the weight vector:
    prefix_weights = []
    
    for case in range(0,cases):
        for event in range(0,max_prefix_length):
            #weight = 1/(event+1)
            #weight = (max_prefix_length-event)/max_prefix_length
            weight = event
            prefix_weights.append(weight)
    
    ## Save the penalty vector
    temporal_weight_vector = np.asarray(prefix_weights)
    
    return temporal_weight_vector

def MAPE(y_true, y_pred):
    import tensorflow.keras.backend as K
    
    return K.mean(K.abs((y_pred - y_true+1)/y_true+1), axis=-1)


"""
Losses from the paper:
"""


def MAE_Etd(time_weights, alpha=1):
    import tensorflow.keras.backend as K
    import numpy as np
    time_weights = np.exp(time_weights)
    
    def mae_w(y_true, y_pred):        
        return K.mean(K.abs(y_pred - y_true) + alpha*(K.abs(y_pred - y_true))/time_weights, axis=-1)
    
    return mae_w


def MAE_Mtd(time_weights, max_prefix_length, alpha=1):
    import tensorflow.keras.backend as K
    
    time_weights = time_weights*1
    
    def mae_w(y_true, y_pred):
        return K.mean(K.abs(y_pred - y_true) + alpha*(K.abs(y_pred - y_true))/1+time_weights, axis=-1)
    
    return mae_w


def MAE_Ptd(time_weights, max_prefix_length, alpha=1):
    import tensorflow.keras.backend as K
    
    time_weights = ((max_prefix_length-time_weights)/max_prefix_length)
    
    def mae_w(y_true, y_pred):        
        return K.mean(K.abs(y_pred - y_true) + alpha*K.abs(y_pred - y_true) ** time_weights, axis=-1) 
    
    return mae_w

"""
Custom weights:
"""

def MAE_td(time_weights, alpha=1, beta=1, gamma=1):
    import tensorflow.keras.backend as K
    
    def mae_w(y_true, y_pred):
        
        #return alpha * K.mean(K.abs(y_pred - y_true), axis=-1) + beta * K.mean((K.abs(y_pred - y_true))/gamma*time_weights, axis=-1)
        return (1-beta) * K.mean(K.abs(y_pred - y_true), axis=-1) + beta * K.mean((K.abs(y_pred - y_true))/alpha*time_weights, axis=-1)
    
    return mae_w
