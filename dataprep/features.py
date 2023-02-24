# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:18:31 2022

@author: Mike
"""

def get_event_activity_features_train(out):
    import pandas as pd
    import numpy as np
    
    #initialize dummy dict
    dummy_dict = {"event":None}
    
    #get currently observed values:
    observed_vals = np.sort(out["event_activity"].astype('str').unique()).tolist()
    
    #Convert to categorical column
    out["event_activity"] = out["event_activity"].astype('category')
    
    #generate dummies
    Dummies = pd.get_dummies(out["event_activity"])
    
    #sort the dummies
    Dummies = Dummies.sort_index(axis=1)
    print(Dummies.head())
    
    #reset index of dummies and generate a prefix
    Dummies = Dummies.reset_index(drop=True)
    Dummies = Dummies.add_prefix('activity_')
    Dummies = Dummies.sort_index(axis=1)
    
    #reset index of other the rest of the features
    out = out.reset_index(drop=True)
    
    #add dummies to all featues, and drop original column
    dummycols = Dummies.columns.tolist()
    out = pd.concat([out,Dummies],axis=1)
    out = out.drop("event_activity",axis=1)
    
    # store observed values
    dummy_dict["event"] = observed_vals    
    
    
    print("Nans after concatenation:")
    print("out",np.count_nonzero(np.isnan(out.values)))
    return out, dummy_dict, observed_vals


def get_event_activity_features_test(dummy_dict, out):
    import pandas as pd
    import numpy as np
    
    #get previously observed values:
    observed_vals = dummy_dict["event"]
    
    #get currently observed values:
    current_observed_vals = np.sort(out["event_activity"].astype('str').unique()).tolist()
    print(current_observed_vals)             
    
    #Find classes not observed in current data
    new_classes_to_add = list(set(observed_vals) - set(current_observed_vals))
    print(new_classes_to_add)
    
    #Convert to categorical column
    out["event_activity"] = out["event_activity"].astype('category')
    
    #Add unobserved values
    out["event_activity"] = out["event_activity"].cat.add_categories(new_classes_to_add)
    
    #generate dummies
    Dummies = pd.get_dummies(out["event_activity"])
    
    #sort the dummies
    Dummies = Dummies.sort_index(axis=1)
    print(Dummies.head())
    
    #reset index of dummies and generate a prefix
    Dummies = Dummies.reset_index(drop=True)
    Dummies = Dummies.add_prefix('activity_')
    Dummies = Dummies.sort_index(axis=1)
    
    #reset index of other the rest of the features
    out = out.reset_index(drop=True)
    
    #add dummies to all featues, and drop original column
    dummycols = Dummies.columns.tolist()
    out = pd.concat([out,Dummies],axis=1)
    out = out.drop("event_activity",axis=1)
    print(out.columns)
    
    print("Nans after concatenation:")
    print("out",np.count_nonzero(np.isnan(out.values)))
    return out, dummy_dict, observed_vals, current_observed_vals, new_classes_to_add


def categorical_case_features(out, datasub, category_cols):
    if len(category_cols) > 0:
        import pandas as pd
        import numpy as np
        print("\nDummification of",category_cols)    
        
        # Generate dummy df
        Dummies = pd.get_dummies(datasub[category_cols])
        dummycols = Dummies.columns.tolist()
        
        
        Dummies = Dummies.fillna(-1)
    
        # Reset the indexes
        Dummies = Dummies.reset_index(drop=True)
        out = out.reset_index(drop=True)
        
        out = pd.concat([out,Dummies], axis=1)
        
        print("Nans after concatenation:")
        print("out",np.count_nonzero(np.isnan(out.values)))
        
    return out



def numerical_case_features(out, datasub, numeric_cols):
    import pandas as pd
    import numpy as np
    
    if len(numeric_cols) > 0:
        #add numerical features:
        print("Adding numerical features:",numeric_cols)
        
        numerics = datasub[numeric_cols]
        numerics = numerics.reset_index(drop=True)
        #numerics = numerics.add_prefix('num_')
        numerics = numerics.fillna(-1)
        
        out = out.reset_index(drop=True)
        out = pd.concat([out,numerics],axis=1)
        
        print(out.shape)
        print(out.head(10))
        
        print("Nans after concatenation:")
        print("out",np.count_nonzero(np.isnan(out.values)))
                
    return out


def dummify_time_features(out, features = ["dayofweek","hourofday"]):
    import pandas as pd
    import numpy as np
    print("Dummification of time features",features)
    sysdummies = out[features]
    
    #convert to categorical first
    for feat in features:
        sysdummies[feat] = sysdummies[feat].astype("category")
    
    sysdummies = pd.get_dummies(sysdummies)
    sysdummies = sysdummies.reset_index(drop=True)
    out = out.drop(features,axis=1)
    out = out.reset_index(drop=True)
    out = pd.concat([out, sysdummies],axis=1)
    
    print("Nans after concatenation:")
    print("out",np.count_nonzero(np.isnan(out.values)))
    return out
        