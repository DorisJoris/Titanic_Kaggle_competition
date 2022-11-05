# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 16:25:54 2022

@author: bejob
"""

#%% Import libraries 

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.impute import KNNImputer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.ensemble import RandomForestClassifier

#%% Data import

train_df = pd.read_csv('Data/train.csv')
test_df = pd.read_csv('Data/test.csv')


#%% one-hot-encoding

x_categorical = train_df[['Pclass', 'Sex',  'SibSp',
       'Parch', 'Embarked']].to_numpy()

x_numerical = train_df[['Age', 'Fare']].to_numpy()

encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(x_categorical)
x_categorical = encoder.transform(x_categorical).toarray()

x = np.concatenate((x_categorical, x_numerical), axis =1)

#%% impute Nan-values KNNImputer
imputer = KNNImputer(n_neighbors=2)

x = imputer.fit_transform(x)

#%% impute Nan-values IterativeImputer
imputer = IterativeImputer()

imputer.fit(x)

x = imputer.transform(x)

#%% train/validation split

y = train_df['Survived'].to_numpy()

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