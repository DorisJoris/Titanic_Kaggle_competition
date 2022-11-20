# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 16:25:54 2022

@author: bejob
"""

#%% Import libraries 

from dataprovider import Dataprovider 

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier


#%% train/validation split

dp = Dataprovider()

x = dp.training_data.x_onehotencoded.knn_imputed
y = dp.training_data.y

x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size = 0.2)


#%% Train decisiontree

classifier = DecisionTreeClassifier() 
classifier = classifier.fit(x_train, y_train)


#%% Validation
classifier.score(x_validation,y_validation)
# 0.7653631284916201 KNNImputer
# 0.7541899441340782 IterativeImputer