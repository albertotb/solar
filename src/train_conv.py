#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

import sys
import feather
import keras
from keras import backend as K
from keras.datasets import cifar10
from keras.layers import (Activation, Add, Concatenate, Conv1D, Conv2D, Dense,
                          Dot, Dropout, Flatten, Input, Lambda,
                          LocallyConnected1D, MaxPooling1D, MaxPooling2D,
                          Multiply, Reshape, Subtract, UpSampling1D)
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from utils.clr import CyclicLR

from sklearn.model_selection import TimeSeriesSplit


# In pandas 0.24, use df.to_numpy() instead of df.values. Also care with non-numeric columns
def window_stack(df, width=3):
    """Converts a pandas.DataFrame with shape (n, d) to another one with shape
       (n-width, d*width) by stacking shifted versions of itself
    """
    n = df.shape[0]
    a = np.hstack(list(df.values[(width-1-i):(n-i)] for i in range(0, width)))

    times = [('t' if not idx else 't-{:d}'.format(idx))
             for idx in range(width)]
    columns = pd.MultiIndex.from_product(
        (times, df.columns), names=('time', 'location'))

    return pd.DataFrame(a, index=df.index[width-1:], columns=columns)


def make_features(fname, width):
    """ Given the data fnmae, creates the features and returns the train-test split. 
    Last value to be returned is the ordenation of the features. """
    df = (feather.read_dataframe(fname)
                 .set_index('Datetime'))

    df_roll = window_stack(df, width=width)

    # Split target (time t) and variables (times t-1 to t-width+1)
    y = df_roll['t']
    X = df_roll.drop(columns='t', level='time')

    # Split train-test, approximately 12 and 4 months respectively
    X_train, X_test = X[:'2011-07-31'], X['2011-08-01':]
    y_train, y_test = y[:'2011-07-31'], y['2011-08-01':]

    # For the moment, we only use the previous timestep as features
    X_tr1 = X_train['t-1']
    y_tr1 = y_train

    X_te1 = X_test['t-1']
    y_te1 = y_test

    # We load the info of the sensors to extract the longitude information
    info = pd.read_csv('/home/SHARED/SOLAR/data/info.csv')

    info.Location = info.Location.apply(
        lambda x: (x[:2] + x[-2:]).replace('_', ''))
    info.index = info.Location
    # Sorted longitudes
    longs = info['       Longitude'].sort_values(ascending=False)

    # We drop two sensors (they are different compared to the other 17, since they are "tilted")
    X_tr1.drop('GT_AP6', inplace=True, axis=1)
    y_tr1.drop('GT_AP6', inplace=True, axis=1)
    X_tr1.drop('GT_DH1', inplace=True, axis=1)
    y_tr1.drop('GT_DH1', inplace=True, axis=1)
    X_te1.drop('GT_AP6', inplace=True, axis=1)
    y_te1.drop('GT_AP6', inplace=True, axis=1)
    X_te1.drop('GT_DH1', inplace=True, axis=1)
    y_te1.drop('GT_DH1', inplace=True, axis=1)

    # Just some auxiliar code to homogeneize name of sensors across different tables
    def homogen_name(x): return x[-4:].replace('_', '')
    X_tr1.columns = [homogen_name(x) for x in X_tr1.columns.values.tolist()]
    y_tr1.columns = [homogen_name(x) for x in y_tr1.columns.values.tolist()]
    X_te1.columns = [homogen_name(x) for x in X_te1.columns.values.tolist()]
    y_te1.columns = [homogen_name(x) for x in y_te1.columns.values.tolist()]

    # Finally, we sort the data according to sensor's longitude
    X_tr1_1 = X_tr1[longs.index]
    y_tr1_1 = y_tr1[longs.index]
    X_te1_1 = X_te1[longs.index]
    y_te1_1 = y_te1[longs.index]

    return X_tr1_1, y_tr1_1, X_te1_1, y_te1_1, longs


def to_array(X_tr, y_tr, X_te, y_te, sensor='AP5'):
    ''' Converts dataframe to numpy array for predicting any given sensor. '''
    X_tr1_1_np = X_tr.values
    y_tr1_1_np = y_tr[sensor].values

    X_te1_1_np = X_te.values
    y_te1_1_np = y_te[sensor].values

    return X_tr1_1_np, y_tr1_1_np, X_te1_1_np, y_te1_1_np


def make_convolutional_model(index_sensor, n_sensors=17):
    ''' Returns a model using all the sensors to predict index_sensor '''
    xin = Input(shape=(n_sensors, 1), name='main_input')
    x = LocallyConnected1D(
        8, 7, data_format='channels_last', padding='valid')(xin)
    x = Activation('relu')(x)
    x = LocallyConnected1D(
        16, 5, data_format='channels_last', padding='valid')(x)
    x = Activation('relu')(x)
    x = Conv1D(32, 3, data_format='channels_last', padding='causal')(x)
    xl = Flatten()(x)
    xl = Dropout(0.2)(xl)
    xo = Dense(1)(xl)

    # use date info here?
    xinf = Flatten()(xin)
    s = Dense(5)(xinf)
    s = Activation('tanh')(s)
    s = Dense(2)(s)
    s = Activation('softmax')(s)

    # sort of residual connection
    xin_0 = Activation('relu')(xin)
    xin_1 = Lambda(lambda x: x[:, index_sensor, :])(xin_0)
    xo_m = Dot(axes=1)([Concatenate()([xo, xin_1]), s])
    xo_m = Activation('relu')(xo_m)

    model = Model(inputs=[xin], outputs=[xo_m])
    return model


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('ArgError')
        sys.exit()

    fname = sys.argv[1]
    width = int(sys.argv[2])

    X_train, y_train, X_test, y_test, longs = make_features(fname, width)
    longs_np = longs.index.values

    lr = 0.0001
    batch_size = 1 << 11   # as big as possible so we can explore many models
    epochs = 1 << 5

    opt = keras.optimizers.Adam(lr=lr)

    # We add a callback to log metrics and another one to schedule the learning rate
    c1 = keras.callbacks.BaseLogger(stateful_metrics=None)
    c2 = CyclicLR(step_size=250, base_lr=lr)
    c3 = keras.callbacks.History()

    def train_and_test_sensor(id_sensor=4):
        X_tr, y_tr, X_te, y_te = to_array(
            X_train, y_train, X_test, y_test, sensor=longs_np[id_sensor])

        # Validation using TS split (just to obtain different MAE estimations, no hyperoptimization for the moment)
        for tr_idx, va_idx in TimeSeriesSplit(n_splits=5).split(X_tr):
            model = make_convolutional_model(id_sensor, n_sensors=17)
            model.compile(opt, loss='mean_absolute_error')
            model.fit(np.atleast_3d(X_tr[tr_idx]), y_tr[tr_idx], batch_size=batch_size, epochs=epochs, validation_data=(
                np.atleast_3d(X_tr[va_idx]), y_tr[va_idx]), callbacks=[c2, c3], verbose=0)
            print('MAE_val ', c3.history['val_loss'][-1])

        # Testing
        model = make_convolutional_model(id_sensor, n_sensors=17)
        model.compile(opt, loss='mean_absolute_error')
        model.fit(np.atleast_3d(X_tr), y_tr, batch_size=batch_size, epochs=epochs,
                  validation_data=(np.atleast_3d(X_te), y_te), callbacks=[c2, c3], verbose=0)

        print('MAE_test ', c3.history['val_loss'][-1])
        return longs_np[id_sensor], c3.history['val_loss'][-1]

    maes = {}
    for i in range(len(longs_np)):
        print(i, longs_np[i])
        sensor, mae = train_and_test_sensor(i)
        maes[sensor] = mae

    maes = pd.Series(maes, name='MAE').sort_values()
    print(maes)
