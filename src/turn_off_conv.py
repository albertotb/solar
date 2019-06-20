#!/usr/bin/env python
# coding: utf-8

# # Build train and test matrices
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import keras

from utils.build_matrix import df_shift, to_array
from utils.clr import CyclicLR
from utils.models import conv1D_lon, conv1D_lon_lat

'''Code to study influence of sensors in predictive error in a target sensor'''
target_sensor = "DH10"
predictors = ["DH1","DH7","DH2","DH4","DH3","DH5","DH10","AP1","AP5","AP4","AP6","AP7"]
#predictors = ["DH10", "DH9", "DH6", "DH8", "DH11"]

path = 'results/' + str(target_sensor) + "_SW.csv" ## to save results

df = pd.read_pickle('/home/SHARED/SOLAR/data/oahu_min_final.pkl')
df_roll = df_shift(df, periods=1)

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
info = info.drop("AP3")
info = info[["Longitude", "Latitude"]]
info["isPredictor"] = 0
info.isPredictor[info.index.isin(predictors)] = 1 ## to know which are the predictors
info["MAE"] = 0 ## to store MAEs

info.to_csv(path)
# Sorted longitudes
lon = info['Longitude'].sort_values(ascending=False)
lat = info['Latitude'].sort_values(ascending=False)

# Keep just target sensors
lon = lon[lon.index.isin(predictors)]
lat = lat[lat.index.isin(predictors)]

# Drop all sensors except predictors
X_tr1 = X_tr1[predictors]
y_tr1 = y_tr1[predictors]
X_te1 = X_te1[predictors]
y_te1 = y_te1[predictors]

# Finally, we sort the data according to sensor's longitude
X_tr_lon = X_tr1[lon.index]
y_tr_lon = y_tr1[lon.index]
X_te_lon = X_te1[lon.index]
y_te_lon = y_te1[lon.index]

X_tr_lat = X_tr1[lat.index]
y_tr_lat = y_tr1[lat.index]
X_te_lat = X_te1[lat.index]
y_te_lat = y_te1[lat.index]

##
lr = 0.0001
opt = keras.optimizers.Adam(lr=lr)

# We add a callback to log metrics and another one to schedule the learning rate
c1 = keras.callbacks.BaseLogger(stateful_metrics=None)
c2 = CyclicLR(step_size=250, base_lr=lr)
c3 = keras.callbacks.History()

batch_size = 1 << 11   # as big as possible so we can explore many models
epochs = 1 << 5


def train_and_test_sensor(idx_sensor, id_sensor, n_sensors, use_lat=False):
    X_tr1, y_tr1, X_te1, y_te1 = to_array(X_tr_lon, y_tr_lon, X_te_lon, y_te_lon, id_sensor=id_sensor)

    if use_lat:
        X_tr2, y_tr2, X_te2, y_te2 = to_array(X_tr_lat, y_tr_lat, X_te_lat, y_te_lat, id_sensor=id_sensor)


    # Validation using TS split (just to obtain different MAE estimations, no hyperoptimization for the moment)
    cv_loss = []
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=5).split(X_tr1):

        if not use_lat:
            train_data = np.atleast_3d(X_tr1[tr_idx])
            validation_data = np.atleast_3d(X_tr1[va_idx])
            model = conv1D_lon(idx_sensor, n_sensors=n_sensors)

        else:
            train_data = [np.atleast_3d(X_tr1[tr_idx]), np.atleast_3d(X_tr2[tr_idx])]
            validation_data = [np.atleast_3d(X_tr1[va_idx]), np.atleast_3d(X_tr2[va_idx])]
            model = conv1D_lon_lat(idx_sensor, n_sensors=n_sensors)

        model.compile(opt, loss='mean_absolute_error')
        model.fit(train_data, y_tr1[tr_idx],
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(validation_data, y_tr1[va_idx]),
                  callbacks=[c2, c3],
                  verbose=0)

        cv_loss.append(c3.history['val_loss'][-1])

    # Testing
    if not use_lat:
        train_data = np.atleast_3d(X_tr1)
        validation_data = np.atleast_3d(X_te1)
        model = conv1D_lon(idx_sensor, n_sensors=n_sensors)

    else:
        train_data = [np.atleast_3d(X_tr1), np.atleast_3d(X_tr2)]
        validation_data = [np.atleast_3d(X_te1), np.atleast_3d(X_te2)]
        model = conv1D_lon_lat(idx_sensor, n_sensors=n_sensors)

    model.compile(opt, loss='mean_absolute_error')
    model.fit(train_data, y_tr1,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(validation_data, y_te1),
              callbacks=[c2, c3],
              verbose=0)

    test_loss = c3.history['val_loss'][-1]

    #model.save('../models/conv1D_{}_{:1d}.h5'.format(id_sensor, use_lat))

    print('MAE_val ', cv_loss)
    print('MAE_test ', test_loss)

    return test_loss, cv_loss

idx_sensor = np.where(lon.index.values == target_sensor)[0][0]

##
test_loss, _ = train_and_test_sensor(idx_sensor, target_sensor, n_sensors=len(predictors))

info.MAE[info.index == target_sensor] = test_loss

#path = 'results/' str(target_sensor) + "_NE.csv"
info.to_csv(path)

#print(df_res)

