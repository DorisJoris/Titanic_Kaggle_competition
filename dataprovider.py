# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 16:25:54 2022

@author: bejob
"""

#%% Import libraries 

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize
from sklearn.impute import KNNImputer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#%% X class

class X:
    def __init__(self, x):
        self.unimputed = x
        self.knn_imputed = normalize(self.knn_impute(x), axis=1)
        self.iterative_imputed = normalize(self.iterative_impute(x), axis=1)
    def knn_impute(self, x):
        imputer = KNNImputer(n_neighbors=2)

        return imputer.fit_transform(x)

    def iterative_impute(self, x):
        imputer = IterativeImputer()

        imputer.fit(x)

        return imputer.transform(x)        

#%% Data class
    
class TrainingData:
    def __init__(self, dataframe):
        self.df = dataframe
        self.x_raw = self.get_x()
        self.x_onehotencoded = X(self.get_x_onehotencoded())
        self.y = self.get_y()
    
    def get_x_onehotencoded(self):

        x_categorical = self.df[['Pclass', 'Sex',  
               'Embarked']].to_numpy()
        
        x_numerical = self.df[['SibSp', 'Parch', 'Age', 'Fare']].to_numpy()
        
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(x_categorical)
        x_categorical = encoder.transform(x_categorical).toarray()
        
        return np.concatenate((x_categorical, x_numerical), axis =1)
    
    def get_x(self):
        return self.df[['Pclass', 'Sex',  'SibSp',
               'Parch', 'Embarked', 'Age', 'Fare']].to_numpy()
    
    def get_y(self):
        return self.df['Survived'].to_numpy()

class TestData:
    def __init__(self, dataframe):
        self.df = dataframe
        self.x_raw = self.get_x()
        self.x_onehotencoded = X(self.get_x_onehotencoded())
    
    def get_x_onehotencoded(self):

        x_categorical = self.df[['Pclass', 'Sex',  
               'Embarked']].to_numpy()
        
        x_numerical = self.df[['SibSp', 'Parch', 'Age', 'Fare']].to_numpy()
        
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(x_categorical)
        x_categorical = encoder.transform(x_categorical).toarray()
        
        return np.concatenate((x_categorical, x_numerical), axis =1)
    
    def get_x(self):
        return self.df[['Pclass', 'Sex',  'SibSp',
               'Parch', 'Embarked', 'Age', 'Fare']].to_numpy()
    

#%% Dataprovider class
class Dataprovider:
    def __init__(self):
        self.training_data = TrainingData(pd.read_csv('Data/train.csv'))
        self.test_data = TestData(pd.read_csv('Data/test.csv'))


#%%
if __name__ == "__main__":
    dp = Dataprovider()
    dpdf = dp.training_data.df
    
    for column in list(dpdf.columns):
        print(column)
        print(dpdf[column].isna().sum())
        print("-----")