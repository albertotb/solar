#!/usr/bin/env python
# coding: utf-8

# # Build train and test matrices
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, Reshape, Add, Multiply, Subtract, Dropout
from keras.layers import Conv2D, MaxPooling2D, LocallyConnected1D, Conv1D, UpSampling1D, MaxPooling1D, Dot, Concatenate

import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import TimeSeriesSplit

from utils.build_matrix import df_shift
from utils.clr import CyclicLR


def to_array(X_train, y_train, X_test, y_test, id_sensor='AP5', val=0.1):
    ''' Converts dataframe to numpy array for predicting any given sensor. val specifies the fraction
    of training samples to be used as validation. '''
    X_tr1_1_np = X_train.values
    y_tr1_1_np = y_train[id_sensor].values

    #val_idx = int((1 - val)*len(y_tr1_1_np))

    X_te1_1_np = X_test.values
    y_te1_1_np = y_test[id_sensor].values

    #return X_tr1_1_np[:val_idx], y_tr1_1_np[:val_idx], X_tr1_1_np[val_idx:], y_tr1_1_np[val_idx:], X_te1_1_np, y_te1_1_np
    return X_tr1_1_np, y_tr1_1_np, X_te1_1_np, y_te1_1_np


def make_model_sensor(idx_sensor, n_sensors=16):
    ''' Returns a model using all the sensors to predict index_sensor '''
    xin = Input(shape=(n_sensors, 1), name='main_input')
    x = LocallyConnected1D(8, 7, data_format = 'channels_last', padding='valid')(xin)
    x = Activation('relu')(x)
    x = LocallyConnected1D(16, 5, data_format = 'channels_last', padding='valid')(x)
    x = Activation('relu')(x)
    x = Conv1D(32, 3, data_format = 'channels_last', padding='causal')(x)
    xl = Flatten()(x)
    xl = Dropout(0.2)(xl)
    xo = Dense(1)(xl)

    # use date info here?
    xinf = Flatten()(xin)
    s  = Dense(5)(xinf)
    s = Activation('tanh')(s)
    s = Dense(2)(s)
    s = Activation('softmax')(s)

    # sort of residual connection
    xin_0 = Activation('relu')(xin)
    xin_1 = Lambda(lambda x : x[:, idx_sensor, :])(xin_0)
    xo_m = Dot(axes=1)([Concatenate()([xo, xin_1]), s])
    xo_m = Activation('relu')(xo_m)

    model = Model(inputs=[xin], outputs=[xo_m])
    return model


def make_model_sensor_2D(idx_sensor, n_sensors=16):
    ''' Returns a model using all the sensors to predict index_sensor '''
    xin = Input(shape=(n_sensors, 1), name='lon_input')
    x = LocallyConnected1D(8, 7, data_format = 'channels_last', padding='valid')(xin)
    x = Activation('relu')(x)
    x = LocallyConnected1D(16, 5, data_format = 'channels_last', padding='valid')(x)
    x = Activation('relu')(x)
    x = Conv1D(32, 3, data_format = 'channels_last', padding='causal')(x)
    xl = Flatten()(x) 

    yin = Input(shape=(n_sensors, 1), name='lat_input')
    y = LocallyConnected1D(8, 7, data_format = 'channels_last', padding='valid')(xin)
    y = Activation('relu')(x)
    y = LocallyConnected1D(16, 5, data_format = 'channels_last', padding='valid')(x)
    y = Activation('relu')(x)
    y = Conv1D(32, 3, data_format = 'channels_last', padding='causal')(x)
    yl = Flatten()(y)

    xc = Concatenate()([xl, yl])
    xc = Dropout(0.2)(xc)
    xo = Dense(1)(xc)

    # use date info here?
    xinf = Flatten()(xin)
    s  = Dense(5)(xinf)
    s = Activation('tanh')(s)
    s = Dense(2)(s)
    s = Activation('softmax')(s)

    # sort of residual connection
    xin_0 = Activation('relu')(xin)
    xin_1 = Lambda(lambda x : x[:, idx_sensor, :])(xin_0)
    xo_m = Dot(axes=1)([Concatenate()([xo, xin_1]), s])
    xo_m = Activation('relu')(xo_m)

    model = Model(inputs=[xin, yin], outputs=[xo_m])
    return model


df = pd.read_pickle('/home/SHARED/SOLAR/data/oahu_min_final.pkl')  
df_roll = df_shift(df, periods=3)

# Split target (time t) and variables (times t-1 to t-width+1)
y = df_roll['t']
X = df_roll.drop(columns='t', level='time')

# Split train-test, approximately 12 and 4 months respectively
X_train, X_test = X[:'2011-07-31'], X['2011-08-01':]
y_train, y_test = y[:'2011-07-31'], y['2011-08-01':]

# We only use the previous timestep as features
X_tr1 = X_train['t-1']
y_tr1 = y_train

X_te1 = X_test['t-1']
y_te1 = y_test

# We load the info of the sensors to extract the longitude information
info = pd.read_pickle('/home/SHARED/SOLAR/data/info.pkl')

# Sorted longitudes
lon = info['Longitude'].sort_values(ascending=False).drop('AP3')
lat = info['Latitude'].sort_values(ascending=False).drop('AP3')

# Finally, we sort the data according to sensor's longitude
X_tr_lon = X_tr1[lon.index]
y_tr_lon = y_tr1[lon.index]
X_te_lon = X_te1[lon.index]
y_te_lon = y_te1[lon.index]

X_tr_lat = X_tr1[lat.index]
y_tr_lat = y_tr1[lat.index]
X_te_lat = X_te1[lat.index]
y_te_lat = y_te1[lat.index]

lr = 0.0001
opt = keras.optimizers.Adam(lr=lr)

# We add a callback to log metrics and another one to schedule the learning rate
c1 = keras.callbacks.BaseLogger(stateful_metrics=None)
c2 = CyclicLR(step_size=250, base_lr=lr)
c3 = keras.callbacks.History()

batch_size = 1 << 11   # as big as possible so we can explore many models
epochs = 1 << 5


def train_and_test_sensor(idx_sensor, id_sensor, n_sensors):
    X_tr1, y_tr1, X_te1, y_te1 = to_array(X_tr_lon, y_tr_lon, X_te_lon, y_te_lon, id_sensor=id_sensor)

    # Validation using TS split (just to obtain different MAE estimations, no hyperoptimization for the moment)
    cv_loss = []
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=5).split(X_tr1):
        model = make_model_sensor(idx_sensor, n_sensors=n_sensors)
        model.compile(opt, loss='mean_absolute_error')
        model.fit(np.atleast_3d(X_tr1[tr_idx]), y_tr1[tr_idx], 
                  batch_size=batch_size, 
                  epochs=epochs, 
                  validation_data=(np.atleast_3d(X_tr1[va_idx]), y_tr1[va_idx]), 
                  callbacks=[c2, c3], 
                  verbose=0)
        cv_loss.append(c3.history['val_loss'][-1])

    # Testing
    model = make_model_sensor(idx_sensor, n_sensors=n_sensors)
    model.compile(opt, loss='mean_absolute_error')
    model.fit(np.atleast_3d(X_tr1), y_tr1, 
              batch_size=batch_size, 
              epochs=epochs, 
              validation_data=(np.atleast_3d(X_te1), y_te1), 
              callbacks=[c2, c3], 
              verbose=0)
    test_loss = c3.history['val_loss'][-1]

    print('MAE_val ', cv_loss)
    print('MAE_test ', test_loss)

    return test_loss, cv_loss


def train_and_test_sensor_2D(idx_sensor, id_sensor, n_sensors):
    X_tr1, y_tr1, X_te1, y_te1 = to_array(X_tr_lon, y_tr_lon, X_te_lon, y_te_lon, id_sensor=id_sensor)
    X_tr2, y_tr2, X_te2, y_te2 = to_array(X_tr_lat, y_tr_lat, X_te_lat, y_te_lat, id_sensor=id_sensor)

    # Validation using TS split (just to obtain different MAE estimations, no hyperoptimization for the moment)
    cv_loss = []
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=5).split(X_tr1):
        model = make_model_sensor_2D(idx_sensor, n_sensors=n_sensors)
        model.compile(opt, loss='mean_absolute_error')
        model.fit([np.atleast_3d(X_tr1[tr_idx]), np.atleast_3d(X_tr2[tr_idx])],
                  y_tr1[tr_idx], 
                  batch_size=batch_size, 
                  epochs=epochs, 
                  validation_data=([np.atleast_3d(X_tr1[va_idx]), np.atleast_3d(X_tr2[va_idx])], 
                                   y_tr1[va_idx]), 
                  callbacks=[c2, c3], 
                  verbose=0)
        cv_loss.append(c3.history['val_loss'][-1])

    # Testing
    model = make_model_sensor_2D(idx_sensor, n_sensors=n_sensors)
    model.compile(opt, loss='mean_absolute_error')
    model.fit([np.atleast_3d(X_tr1), np.atleast_3d(X_tr2)], 
              y_tr1, 
              batch_size=batch_size, 
              epochs=epochs, 
              validation_data=([np.atleast_3d(X_te1), np.atleast_3d(X_te2)], 
                                y_te1), 
              callbacks=[c2, c3], 
              verbose=0)
    test_loss = c3.history['val_loss'][-1]

    print('MAE_val ', cv_loss)
    print('MAE_test ', test_loss)

    return test_loss, cv_loss


maes1 = {}
maes2 = {}
for idx_sensor, id_sensor in enumerate(lon.index.values):
    print(idx_sensor, id_sensor)
    maes1[id_sensor], _ = train_and_test_sensor(idx_sensor, id_sensor, n_sensors=16)
    maes2[id_sensor], _ = train_and_test_sensor_2D(idx_sensor, id_sensor, n_sensors=16)

maes1 = pd.Series(maes1, name='MAE').sort_values()
maes2 = pd.Series(maes2, name='MAE').sort_values()

print(maes1)
print(maes2)

