#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import math
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from joblib import Parallel, delayed

DATA_PATH = '/home/SHARED/SOLAR/data/'


def get_grid(lonlat, decimals=3):

    prec = 10**decimals
    max_min = lonlat.agg([np.max, np.min])

    lat = np.arange(int(math.floor(max_min.loc['amin',  'Latitude']*prec)),
                    int(math.ceil( max_min.loc['amax',  'Latitude']*prec))+1)/prec

    lon = np.arange(int(math.floor(max_min.loc['amin', 'Longitude']*prec)),
                    int(math.ceil( max_min.loc['amax', 'Longitude']*prec))+1)/prec

    xx, yy = np.meshgrid(lon, lat)
    return pd.DataFrame({'lon': xx.ravel(), 'lat': yy.ravel()})


def train_gpr(df, datetime):
    X_train = df[['Longitude', 'Latitude']]
    y_train = df['GHI']

    X_test = get_grid(df[['Longitude', 'Latitude']])

    gpr = GaussianProcessRegressor(kernel=RBF(10, (1e-3, 1e-3)), n_restarts_optimizer=10)
    gpr.fit(X_train, y_train)

    X_test['GHI'] = gpr.predict(X_test)
    return datetime, X_test.set_index(['lon', 'lat'])


# read minute data and location info
df   = pd.read_pickle(DATA_PATH + 'oahu_min_final.pkl')
info = pd.read_pickle(DATA_PATH + 'info.pkl')

df_long = (df.stack()
             .reset_index('Datetime')
             .join(info[['Longitude', 'Latitude']])
             .rename(columns={0: 'GHI'})
             .dropna())

res = Parallel(n_jobs=8)(delayed(train_gpr)(df, datetime)
                         for datetime, df in df_long.groupby('Datetime'))

df_wide = pd.concat(dict(res)).unstack(level=['lon', 'lat'])

df_wide.to_pickle('ghi_map.pkl')
