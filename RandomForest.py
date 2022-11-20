# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 16:25:54 2022

@author: bejob
"""

#%% Import libraries 

from dataprovider import Dataprovider 

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier


#%% train/validation split

dp = Dataprovider()

x = dp.training_data.x_onehotencoded.knn_imputed
y = dp.training_data.y

x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size = 0.2)


#%% Train decisiontree

classifier = RandomForestClassifier(criterion = 'entropy') 
classifier = classifier.fit(x_train, y_train)


#%% Validation
classifier.score(x_validation,y_validation)
# 0.8100558659217877 KNNImputer (criterion = 'gini')
# 0.7541899441340782 IterativeImputer (criterion = 'gini')

# 0.7821229050279329 KNNImputer (criterion = 'entropy') 
# 0.770949720670391 IterativeImputer (criterion = 'entropy') 