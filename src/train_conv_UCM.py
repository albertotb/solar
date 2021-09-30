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

if len(sys.argv) < 3:
    print('argerror')
    sys.exit(1)

epochs=int(sys.argv[1])
lag=int(sys.argv[2])

PATH = '/home/SHARED/SOLAR/matrices/'

X_train = pd.read_csv('{}/train_validation/X_tr_val.csv'.format(PATH))
y_train = pd.read_csv('{}/train_validation/Y_tr_val.csv'.format(PATH))

X_test = pd.read_csv('{}/test/X_test.csv'.format(PATH))
y_test = pd.read_csv('{}/test/Y_test.csv'.format(PATH))

sensors = (X_train.columns[X_train.columns.str.endswith('_ns0')]
                  .str.extract(r'(.*?)_', expand=False)
                  .str.upper())

# We load the info of the sensors to extract the longitude information
info = pd.read_pickle('/home/SHARED/SOLAR/data/info.pkl')

# Sorted longitudes
lon = info['Longitude'].sort_values(ascending=False).drop('AP3')
lat = info['Latitude'].sort_values(ascending=False).drop('AP3')

# Finally, we sort the data according to sensor's longitude
lon_idx = lon.index.map(pd.Series(range(len(sensors)), index=sensors)).values
lat_idx = lat.index.map(pd.Series(range(len(sensors)), index=sensors)).values

X_tr1 = (X_train.drop(columns=['elevation', 'azimut'])
                .to_numpy()
                .reshape(-1, 16, 3)[:, lon_idx, lag])

X_te1 = (X_test.drop(columns=['elevation', 'azimut'])
               .to_numpy()
               .reshape(-1, 16, 3)[:, lon_idx, lag])

X_tr2 = (X_train.drop(columns=['elevation', 'azimut'])
                .to_numpy()
                .reshape(-1, 16, 3)[:, lat_idx, lag])

X_te2 = (X_test.drop(columns=['elevation', 'azimut'])
               .to_numpy()
               .reshape(-1, 16, 3)[:, lat_idx, lag])

print(X_tr1.shape)
print(X_te1.shape)
print(X_tr2.shape)
print(X_te2.shape)

y_tr1 = y_train.to_numpy()
y_te1 = y_test.to_numpy()


lr = 0.0001
opt = keras.optimizers.Adam(lr=lr)

# We add a callback to log metrics and another one to schedule the learning rate
c1 = keras.callbacks.BaseLogger(stateful_metrics=None)
c2 = CyclicLR(step_size=250, base_lr=lr)
c3 = keras.callbacks.History()

batch_size = 1 << 11   # as big as possible so we can explore many models


def train_and_test_sensor(idx_sensor, id_sensor, n_sensors, use_lat=False):
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

    model.save('../models/conv1D_{}_{:1d}_UCM{:1d}.h5'.format(id_sensor, use_lat, lag))

    print('MAE_val ', cv_loss)
    print('MAE_test ', test_loss)

    return test_loss, cv_loss


maes1, _ = train_and_test_sensor(idx_sensor, id_sensor, n_sensors=16)
maes2, _ = train_and_test_sensor(idx_sensor, id_sensor, n_sensors=16, use_lat=True)

maes1 = pd.Series(maes1, name='conv1D_lon')
maes2 = pd.Series(maes2, name='conv1D_lon_lat')

df_res = pd.concat([maes1, maes2], axis=1, sort=True)
df_res.to_pickle('conv1D_UCM{1d}.pkl'.format(lag))
print(df_res)

