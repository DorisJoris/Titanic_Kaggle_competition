# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 16:25:54 2022

@author: bejob
"""

#%% Import libraries 

import pandas as pd
import seaborn as sns

#%% Data import

train_df = pd.read_csv('Data/train.csv')
test_df = pd.read_csv('Data/test.csv')

#%% Exploration

sns.set_theme(style='ticks')

#visu_df = train_df
sns.pairplot(train_df, hue='Survived')

#%% Data cleaning

train_df.drop("Name")