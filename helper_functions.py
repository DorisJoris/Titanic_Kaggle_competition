# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 17:13:30 2022

@author: bejob
"""

#%% Import 

import pandas as pd

from sklearn.model_selection import KFold

#%% Get list of train_functions
def get_train_functions_list(modul):
    train_functions = list()
    for function in dir(modul):
        if function[0:6] == 'train_':
            train_functions.append(getattr(modul, function))
    return train_functions

#%% Kfold model tester

def kfold_model_tester(train_functions, n_splits, x, y):
    kf = KFold(n_splits=10, shuffle = True)

    folds = list()

    for train_index, val_index in kf.split(x,y):
        for function in train_functions:
            folds.append(function(x[train_index], y[train_index],
                                  x[val_index], y[val_index]))
            

    df = pd.DataFrame(folds, columns = ['Type', 'Validation accuracy'])
    stats = df.groupby('Type').mean()
    
    return stats