import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
import sys
sys.path.append('../space/')
from space import utils as su


def make_ml_datasets(datasets,x_params,y_params):
    data  = pd.concat(su.listify(datasets),axis=1)
    xdataset = data[x_params]
    ydataset = data[y_params]
    return xdataset,ydataset


def make_multixing_list(index,time,sat,theta,phi,dt = 45  , dth = 7.5, dph = 7.5):
    time_pos = pd.DataFrame({'time' : time, 'sat' : sat,'theta' : theta, 'phi' : phi},index= index)
    time_pos = time_pos.sort_values(by='time')

    list_multiXings= [ pd.DataFrame({'time' : [time_pos.iloc[0].time], 'sat' : [time_pos.iloc[0].sat], 'theta' : [time_pos.iloc[0].theta], 'phi' : [time_pos.iloc[0].phi], 'idx' : [time_pos.index[0]] }) ]
    
    for i  in range(len(time_pos)-1):
        if any((time_pos.iloc[i+1].time-list_multiXings[-1].time)/timedelta(minutes=1)<=dt) and any(abs(list_multiXings[-1].theta - time_pos.iloc[i+1].theta)<=dth*np.pi/180) and any(abs(list_multiXings[-1].phi - time_pos.iloc[i+1].phi)<=dph*np.pi/180):  
            list_multiXings[-1] = pd.concat([list_multiXings[-1],pd.DataFrame({'time' : [time_pos.iloc[i+1].time],'sat' : [time_pos.iloc[i+1].sat],'theta' : [time_pos.iloc[i+1].theta],'phi' : [time_pos.iloc[i+1].phi], 'idx' : [time_pos.index[i+1]]} ) ],axis=0,ignore_index=True)
        
        elif (len(list_multiXings)>=2) and any((time_pos.iloc[i+1].time-list_multiXings[-2].time)/timedelta(minutes=1)<=dt) and any(abs(list_multiXings[-2].theta - time_pos.iloc[i+1].theta)<=dth*np.pi/180) and any(abs(list_multiXings[-2].phi - time_pos.iloc[i+1].phi)<=dph*np.pi/180):
    
            list_multiXings[-2] = pd.concat([list_multiXings[-2],pd.DataFrame({'time' : [time_pos.iloc[i+1].time],'sat' : [time_pos.iloc[i+1].sat],'theta' : [time_pos.iloc[i+1].theta],'phi' : [time_pos.iloc[i+1].phi], 'idx' : [time_pos.index[i+1]]} ) ],axis=0,ignore_index=True)
        elif (len(list_multiXings)>=3) and any((time_pos.iloc[i+1].time-list_multiXings[-3].time)/timedelta(minutes=1)<=dt) and any(abs(list_multiXings[-3].theta - time_pos.iloc[i+1].theta)<=dth*np.pi/180) and any(abs(list_multiXings[-3].phi - time_pos.iloc[i+1].phi)<=dph*np.pi/180):
    
            list_multiXings[-3] = pd.concat([list_multiXings[-3],pd.DataFrame({'time' : [time_pos.iloc[i+1].time],'sat' : [time_pos.iloc[i+1].sat],'theta' : [time_pos.iloc[i+1].theta],'phi' : [time_pos.iloc[i+1].phi], 'idx' : [time_pos.index[i+1]]} ) ],axis=0,ignore_index=True)
        else:
            list_multiXings.append(pd.DataFrame({'time' : [time_pos.iloc[i+1].time],'sat' : [time_pos.iloc[i+1].sat],'theta' : [time_pos.iloc[i+1].theta],'phi' : [time_pos.iloc[i+1].phi], 'idx' : [time_pos.index[i+1]]} ))
            
    return list_multiXings


def make_trainset_testset_from_multixing_list(list_multiXings,xdataset,ydataset, test_size=0.1,r_state = None ):
    if r_state is None:
        ttrain, ttest , _ , _ = train_test_split(list_multiXings, list_multiXings, test_size=test_size)
    else :
        ttrain, ttest , _ , _ = train_test_split(list_multiXings, list_multiXings, test_size=test_size, random_state=r_state)
    ttrain = pd.concat(ttrain,ignore_index=True)
    xtrain = xdataset[xdataset.index.isin(ttrain.idx.values)]
    ytrain = ydataset[ydataset.index.isin(ttrain.idx.values)]
    ttest = pd.concat(ttest,ignore_index=True)
    xtest = xdataset[xdataset.index.isin(ttest.idx.values)]
    ytest = ydataset[ydataset.index.isin(ttest.idx.values)]
    xtrain,ytrain = shuffle(xtrain,ytrain, random_state=42)
    xtest,ytest = shuffle(xtest,ytest, random_state=42)
    return xtrain,ytrain,xtest,ytest

