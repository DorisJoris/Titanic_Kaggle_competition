# -*- coding: utf-8 -*-

#%% Import libraries 

from dataprovider import Dataprovider 

from sklearn.model_selection import train_test_split

from tensorflow import keras

#%% network

inputs = keras.Input(shape=(13,))

dense = keras.layers.Dense(8, activation = 'relu')(inputs)

outputs = keras.layers.Dense(1, activation = 'sigmoid')(dense)

model = keras.Model(inputs = inputs,
                    outputs = outputs,
                    name = 'titanic_nn')

model.summary()


#%% train/validation split

dp = Dataprovider()

x = dp.training_data.x_onehotencoded.iterative_imputed
y = dp.training_data.y

x_train, x_validation, y_train, y_validation = train_test_split(x, 
                                                                y, 
                                                                test_size = 0.2)

#%% Compilying
model.compile(
    loss = 'binary_crossentropy',
    optimizer= 'adam',
    metrics = ['accuracy']
    )

#%%
callback = keras.callbacks.EarlyStopping(patience = 3)

history = model.fit(x_train, 
                    y_train,
                    batch_size = 64, 
                    epochs = 1000,
                    validation_data = (x_validation, y_validation),
                    callbacks = [callback]
                    )

history.history['val_accuracy'][-1]
