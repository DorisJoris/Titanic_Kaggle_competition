# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:30:22 2022

@author: bejob
"""

#%% Import libraries 

from dataprovider import Dataprovider 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

#%% train/validation split

dp = Dataprovider()

x = dp.training_data.x_onehotencoded.knn_imputed
y = dp.training_data.y

x_train, x_validation, y_train, y_validation = train_test_split(x, 
                                                                y, 
                                                                test_size = 0.2)

#%% Train logistic regrssion

classifier = LogisticRegression(random_state = 0) 
classifier = classifier.fit(x_train, y_train)

#%% Validation
classifier.score(x_validation,y_validation)