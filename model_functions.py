# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:03:15 2022

@author: bejob
"""

#%% Import libraries 

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from tensorflow import keras


#%% Logistic Regression

def train_logistic_regression(x_train, y_train, x_val, y_val):
    name = 'Logistic regression'
    classifier = LogisticRegression(random_state = 0) 
    classifier = classifier.fit(x_train, y_train)
    score = classifier.score(x_val, y_val)
    return (name, score)

#%% Decision Tree

def train_decision_tree(x_train, y_train, x_val, y_val):
    name = 'Decision tree'
    classifier = DecisionTreeClassifier() 
    classifier = classifier.fit(x_train, y_train)
    score = classifier.score(x_val, y_val)
    return (name, score)

#%% Random forest

def train_random_forest(x_train, y_train, x_val, y_val):
    name = 'Random forest'
    classifier = RandomForestClassifier(criterion = 'entropy') 
    classifier = classifier.fit(x_train, y_train)
    score = classifier.score(x_val, y_val)
    return (name, score)

#%% Gradientboosting

def train_gradientboosting(x_train, y_train, x_val, y_val):
    name = 'Gradientboosting'
    classifier = GradientBoostingClassifier(learning_rate = 0.01,
                                            n_estimators = 700) 
    classifier = classifier.fit(x_train, y_train)
    score = classifier.score(x_val, y_val)
    return (name, score)

#%% Histgradientboosting

def train_histgradientboosting(x_train, y_train, x_val, y_val):
    name = 'HistGradientboosting'
    classifier = HistGradientBoostingClassifier() 
    classifier = classifier.fit(x_train, y_train)
    score = classifier.score(x_val, y_val)
    return (name, score)

#%% Neural Network

def train_neural_network(x_train, y_train, x_val, y_val):
    name = 'Neural network'
    inputs = keras.Input(shape=(len(x_train[0]),))
    dense = keras.layers.Dense(8, activation = 'relu')(inputs)
    outputs = keras.layers.Dense(1, activation = 'sigmoid')(dense)

    classifier = keras.Model(inputs = inputs,
                        outputs = outputs,
                        name = 'titanic_nn')
    
    classifier.compile(
        loss = 'binary_crossentropy',
        optimizer= 'adam',
        metrics = ['accuracy']
        )
    
    callback = keras.callbacks.EarlyStopping(patience = 3)

    classifier = classifier.fit(x_train, 
                                y_train,
                                batch_size = 64, 
                                epochs = 1000,
                                validation_data = (x_val, y_val),
                                callbacks = [callback]
                                )
    
    score = classifier.history['val_accuracy'][-1]
    return (name, score)