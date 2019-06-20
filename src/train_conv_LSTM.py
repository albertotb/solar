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
from utils.models import conv1D_lon, conv1D_lon_lat, conv1D_lon_LSTM


df = pd.read_pickle('/home/SHARED/SOLAR/data/oahu_min_final.pkl')
df_roll = df_shift(df, periods=3)

# Split target (time t) and variables (times t-1 to t-width+1)
y = df_roll['t']
X = df_roll.drop(columns='t', level='time')

# Split train-test, approximately 12 and 4 months respectively
X_train, X_test = X[:'2011-07-31'], X['2011-08-01':]
y_train, y_test = y[:'2011-07-31'], y['2011-08-01':]

# We load the info of the sensors to extract the longitude information
info = pd.read_pickle('/home/SHARED/SOLAR/data/info.pkl')

# Sorted longitudes
lon = info['Longitude'].sort_values(ascending=False).drop('AP3')
lat = info['Latitude'].sort_values(ascending=False).drop('AP3')

# Finally, we sort the data according to sensor's longitude
lon_idx = lon.index.map(pd.Series(range(df.shape[1]), index=df.columns)).values

X_tr1 = X_train.to_numpy().reshape(-1, 3, df.shape[1], 1)[:, :, lon_idx, :]
y_tr1 = y_train.to_numpy()[:, lon_idx]

X_te1 = X_test.to_numpy().reshape(-1, 3, df.shape[1], 1)[:, :, lon_idx, :]
y_te1 = y_test.to_numpy()[:, lon_idx]


def train_and_test(batch_size, epochs, n_steps, n_sensors):

    lr = 0.0001
    opt = keras.optimizers.Adam(lr=lr)

    c1 = CyclicLR(step_size=250, base_lr=lr)

    # Validation using TS split (just to obtain different MAE estimations, no hyperoptimization for the moment)
    cv_loss = []
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=5).split(X_tr1):
        model = conv1D_lon_LSTM(n_steps=n_steps, n_sensors=n_sensors)
        model.compile(opt, loss='mean_absolute_error')
        hist = model.fit(X_tr1[tr_idx], y_tr1[tr_idx],
                         batch_size=batch_size,
                         epochs=epochs,
                         validation_data=(X_tr1[va_idx], y_tr1[va_idx]),
                         callbacks=[c1],
                         verbose=0)
        cv_loss.append(hist.history['val_loss'][-1])

    # Testing
    model = conv1D_lon_LSTM(n_steps=n_steps, n_sensors=n_sensors)
    model.compile(opt, loss='mean_absolute_error')
    hist = model.fit(X_tr1, y_tr1,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_te1, y_te1),
              callbacks=[c1],
              verbose=0)

    test_loss = hist.history['val_loss'][-1]

    print('MAE_val ', cv_loss)
    print('MAE_test ', test_loss)

    return test_loss, cv_loss, model


batch_size = 100   # as big as possible so we can explore many models
epochs = 50

_, _, model = train_and_test(batch_size, epochs, n_steps=3, n_sensors=16)

model.save('../models/conv1D_LSTM.h5')
maes1 = pd.Series(np.mean(np.abs(model.predict(X_te1) - y_te1), axis=0), index=lon.index)

maes1.to_pickle('../results/conv1D_LSTM.pkl')
print(maes1)
