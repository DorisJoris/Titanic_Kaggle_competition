# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:44:40 2022

@author: bejob
"""

#%% Import 

from dataprovider import Dataprovider 

import model_functions
from helper_functions import get_train_functions_list
from helper_functions import kfold_model_tester

#%% Train functions

train_functions = get_train_functions_list(model_functions)


#%% Data

dp = Dataprovider()

x = dp.training_data.x_onehotencoded.knn_imputed
y = dp.training_data.y


model_stats = kfold_model_tester(train_functions, n_splits=10, x=x, y=y)

model_stats.plot.bar(rot=15, ylim=(0, 1))
