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
info["MAE"] = 0 ## to store MAEs
##
u = np.array([1,1]) # Direction to order sensors. (1,0) = Longitude, (0,1) = Latitude
norm_u = np.sqrt(np.sum(u**2))
u_n = u/norm_u
##
info['Order'] = np.dot(info[["Longitude", "Latitude"]].values, u_n)
order = info['Order'].sort_values(ascending=False)
##
path = 'results/conv1D_Long' + str(u[0]) + "_Lat" + str(u[1]) + ".csv" ## to save results

# Finally, we sort the data according to the defined order
X_tr_ord = X_tr1[order.index]
y_tr_ord = y_tr1[order.index]
X_te_ord = X_te1[order.index]
y_te_ord = y_te1[order.index]


lr = 0.0001
opt = keras.optimizers.Adam(lr=lr)

# We add a callback to log metrics and another one to schedule the learning rate
c1 = keras.callbacks.BaseLogger(stateful_metrics=None)
c2 = CyclicLR(step_size=250, base_lr=lr)
c3 = keras.callbacks.History()

batch_size = 1 << 11   # as big as possible so we can explore many models
epochs = 1 << 5


def train_and_test_sensor(idx_sensor, id_sensor, n_sensors, use_lat=False):
    X_tr1, y_tr1, X_te1, y_te1 = to_array(X_tr_ord, y_tr_ord, X_te_ord, y_te_ord, id_sensor=id_sensor)

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

    model.save('../models/conv1D_{}_{:1d}.h5'.format(id_sensor, use_lat))

    print('MAE_val ', cv_loss)
    print('MAE_test ', test_loss)

    return test_loss, cv_loss


maes1 = {}
maes2 = {}
for idx_sensor, id_sensor in enumerate(order.index.values):
    print(idx_sensor, id_sensor)
    test_loss, _ = train_and_test_sensor(idx_sensor, id_sensor, n_sensors=16)
    info.MAE[info.index == id_sensor] = test_loss

info.to_csv(path)
