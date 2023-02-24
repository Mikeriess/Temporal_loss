# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 13:30:51 2022

@author: Mike
"""

def MakeSplitCriterion(df, trainsize=0.8, mode="event"):
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import time as tm
    
    def datetime_range(start=None, end=None):
        span = end - start
        for i in range(span.days + 1):
            yield start + timedelta(days=i)
 
    #Parse date
    df["time_parsed"] = pd.to_datetime(df["time"])
    
    #Get min max dates:
    earliest_date = min(df["time_parsed"])
    lastest_date = max(df["time_parsed"])
    
    #Find the date to divide on:
    dates = list(datetime_range(start=earliest_date, end=lastest_date))
    n_dates = len(dates)
    splitpoint = n_dates*trainsize
    splitpoint = int(np.round(splitpoint,decimals=0))
    
    dividing_date = dates[splitpoint]
    dividing_date = dividing_date
    print("=======================================")
        
    print("Log starts at:",earliest_date)
    print("Last event starts at:",lastest_date)
    print("Train-test split happens at:",dividing_date)
    print("=======================================")

        
    if mode=="event":
        """
            Here we simply divide by date of the event,
            and disregard that a case could be in both train and test set
            this way
        """
        
        df["trainset"] = df["time_parsed"] < dividing_date
        df["trainset"].value_counts()
        
        split_criterion = df[["id","trainset"]]
        split_criterion = split_criterion.rename(columns={'id':'caseid',
                                'trainset':'trainset'}, inplace=False)
        
        split_criterion = split_criterion.reset_index(drop=True)
        split_criterion = split_criterion.drop_duplicates(subset="caseid",keep="first")
        
        print(len(split_criterion["caseid"].unique().tolist()))
        print(len(split_criterion))
        print(np.sum(df["trainset"]*1))
        print("=======================================")
        
    if mode=="case":
        """
            Here we remove all cases that are in both train and test set
        """
        
        # For every case, verify if it has both True & False events
        # If it has, drop that case ID
        # And remember to print it
        
        df["trainset"] = df["time_parsed"] < dividing_date
        df["trainset"].value_counts()
        
        split_criterion = df[["id","trainset"]]
        split_criterion = split_criterion.rename(columns={'id':'caseid',
                                'trainset':'trainset'}, inplace=False)
        
        split_criterion = split_criterion.reset_index(drop=True)
        
        #Groupby and get count of every unique value per case id
        validation = pd.DataFrame(split_criterion.groupby('caseid').trainset.nunique())
        validation["caseid"] = validation.index
        
        #If a caseid has both true and false within it (count == 2),
        #it should be dropped.
        print("=======================================")
        print("Dropping cases that have events in both train + testsets:")
        print("=======================================")
        print("Cases before dropping:",len(validation["trainset"]))
        validation["keep"] = validation["trainset"] == 1
        validation = validation.loc[validation["keep"]==True]
        print("Cases after dropping:",len(validation["trainset"]))
        
        #list of caseids to keep
        ids_keep = validation["caseid"]
        
        #drop those thet should not be kept
        print("Total events before:",len(split_criterion))
        split_criterion = split_criterion.loc[split_criterion["caseid"].isin(ids_keep)]
        print("Total events after:",len(split_criterion))
        split_criterion = split_criterion.drop_duplicates(subset="caseid",keep="first")
        print("=======================================")
        print(len(split_criterion))
        print(np.sum(split_criterion["trainset"]*1))
    return split_criterion


def SplitAndReshape(padded_df, X, y, split_criterion, prefixlength, verbose=False):
    import pandas as pd
    import numpy as np
        
    y = pd.DataFrame({"y":y})
    y["caseid"] = X.caseid
    y["caseid"] = y["caseid"].astype('int')
    
    print("Y:")
    print(y.shape)
    print(y.head())
        
    #prepare for join
    x = padded_df.reset_index(drop=True)
    x["caseid"] = x["caseid"].astype('int')
    
    print("X:")
    print(x.shape)
    print(x.head())
    
    # prepare the splitting criteria table
    split = split_criterion.reset_index(drop=True)
    split["caseid"] = split["caseid"].astype('int')
    
    ####################################################
    #Add the splitting cplumn (true = to trainset):
    print("merging..")

    print(x.shape)
    X = pd.merge(left=x.reset_index(drop=True),
                 right=split.reset_index(drop=True), 
                 how='left', 
                 on = 'caseid')
    print(X.shape)
    
    print(y.shape)
    y = pd.merge(left=y.reset_index(drop=True),
                 right=split.reset_index(drop=True), 
                 how='left', 
                 on = 'caseid')
    print(y.shape)
    ####################################################
    #Subset based on the date divider, made in beginning:
    X_train = X.loc[X["trainset"]==True]
    X_test = X.loc[X["trainset"]==False]
        
    y_train = y.loc[y["trainset"]==True]
    y_test = y.loc[y["trainset"]==False]
    
    #Drop system variables
    X_train = X_train.drop(["caseid","SEQID","trainset"],axis=1)
    X_test = X_test.drop(["caseid","SEQID","trainset"],axis=1)
        
    y_train = y_train.drop(["caseid","trainset"], axis=1)
    y_test = y_test.drop(["caseid","trainset"], axis=1)
    
    
    print(X_train.columns.tolist())
    
    print("final mean of X_train",np.mean(X_train))
    
    # print(nothere)
    print("Nans in generated data:")
    print("X_train",np.count_nonzero(np.isnan(X_train.values)))
    
        
    ############################# 
    X_train = X_train.values
    X_test = X_test.values

    y_train = y_train.values
    y_test = y_test.values

    #Reshape:
    print("reshaping..")
    
    #time, n, k
    timesteps = prefixlength
    observations = y_train.shape[0] #int(X.shape[0]/prefixlength)
    k = X_train.shape[1]
    
    print("timesteps, observations, k",timesteps,observations,k)
    print("X_train shape:",X_train.shape)
    print("y_train shape:",y_train.shape)
    
    #Reshape the data
    X_train = X_train.reshape(observations, timesteps, k)
    X_test = X_test.reshape(y_test.shape[0], timesteps, X_test.shape[1])
    
    print("Trainset size (with prefixes of ",prefixlength,"):",y_train.shape[0])
    print("Testset size (with prefixes of ",prefixlength,"):",y_test.shape[0])
    print("==========================================")
    
    #Check the shapes
    print("X: observations, timesteps, vars")
    print(X_train.shape)
    print("X mean:",np.mean(X_train))
    
    print("y_train: observations, labels")
    print(y_train.shape)
    print("y mean:",np.mean(y_train))
    
        
    return X_train, X_test, y_train, y_test