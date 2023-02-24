# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:43:03 2022

@author: Mike
"""

mapping = {"sepsis":
                                {"name":"sepsis",
                                 "dest":"data/sepsis/Sepsis Cases - Event Log.xes",
                                 "type":"xes",
                                 "utc":True,
                                 "timestamp":"2014-10-22 11:27:00+02:00",
                                             "keep_columns":["case:concept:name","concept:name","time:timestamp","org:group"],
                                             "new_colnames":["caseid","activity","timestamp","resource"],
                                 "cat_features":[],
                                 "num_features":[]},

            "traffic_fines": 
                                {"name":"traffic_fines",
                                 "dest":"data/traffic_fines/Road_Traffic_Fine_Management_Process.xes",
                                 "type":"xes",
                                 "utc":True,
                                 "timestamp":"2006-07-24 00:00:00+02:00",
                                             "keep_columns":["case:concept:name","concept:name","time:timestamp","org:resource"],
                                             "new_colnames":["caseid","activity","timestamp","resource"],
                                 "cat_features":[],
                                 "num_features":[]},
            "hospital_billing":
                                {"name":"hospital_billing",
                                 "dest":"data/hospital_billing/Hospital Billing - Event Log.xes",
                                 "type":"xes",
                                 "utc":True,
                                 "timestamp":"2012-12-16 19:33:10+01:00",
                                             "keep_columns":["case:concept:name","concept:name","time:timestamp","org:resource"],
                                             "new_colnames":["caseid","activity","timestamp","resource"],
                                 "cat_features":[],
                                 "num_features":[]},
            "service_desk":
                                {"name":"service_desk",
                                 "dest":"data/BPI2014/Service desk//Detail_Incident_Activity.csv",
                                 "type":"csv",
                                 "sep":";",
                                 "utc":False,
                                 "timestamp":"07-01-2013 08:17:17",
                                 "timeformat":"%d-%m-%Y %H:%M:%S", #"%Y-%m-%d %H:%M:%S",
                                             "keep_columns":["Incident ID","IncidentActivity_Type","DateStamp","Assignment Group"],
                                             "new_colnames":["caseid","activity","timestamp","resource"],
                                 "cat_features":[],
                                 "num_features":[]},
            "helpdesk":
                                {"name":"helpdesk",
                                 "dest":"data/helpdesk/finale_helpdesk.csv",
                                 "type":"csv",
                                 "sep":",",
                                 "utc":False,
                                 "timestamp":"07/01/2013 08:17:17",
                                 "timeformat":"%Y/%m/%d %H:%M:%S",
                                 "keep_columns":["Case ID","Activity","Complete Timestamp","Resource",],
                                 "new_colnames":["caseid","activity","timestamp","resource",],
                                 "cat_features":[],
                                 "num_features":[]}}
