#!/usr/bin/env python
# coding: utf-8

# # Build train and test matrices

import pandas as pd
import numpy as np
import feather
from utils.download_data import timer

with timer():
    df = (feather.read_dataframe('/home/SHARED/SOLAR/data/oahu_min.feather')
                 .set_index('Datetime'))


# I feel this function can also be done for pd.DataFrame
def window_stack(a, width=3):
    n = a.shape[0]
    return np.hstack(list(a[(width-1-i):(n-i)] for i in range(0, width)))


# In pandas 0.24, use df.to_numpy() instead of df.values. Also care with non-numeric columns
width = 61
a = window_stack(df.values, width=width)


times   = [ ('t' if not idx else 't-{:d}'.format(idx)) for idx in range(width) ]
columns = pd.MultiIndex.from_product((times, df.columns), names=('time', 'location'))


# Convert back to DataFrame, just for convenience of having indexes
df_roll = pd.DataFrame(a, index=df.index[width-1:], columns=columns)

# Split target (time t) and variables (times t-1 to t-width+1)
y = df_roll['t']
X = df_roll.drop(columns='t', level='time')

# Split train-test, approximately 12 and 4 months respectively
X_train, X_test = X[:'2011-07-31'], X['2011-08-01':]
y_train, y_test = y[:'2011-07-31'], y['2011-08-01':]


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


#from contextlib import contextmanager
#from timeit import default_timer
#
#@contextmanager
#def timer(tag=None):
#    start = default_timer()
#    try:
#        yield
#    finally:
#        end = default_timer()
#        print(('[{}] '.format(tag) if tag else '') +
#               'Elapsed time (s): {:.6f}'.format(end - start))
#
#
#from sklearn.linear_model import MultiTaskElasticNetCV
#from sklearn.multioutput import MultiOutputRegressor
#
#enet = MultiOutputRegressor(ElasticNetCV(cv=TimeSeriesSplit(n_splits=5), l1_ratio=0.5), n_jobs=10)
#with timer():
#    enet.fit(X_train, y_train)
#
#
#y_test_pred = pd.DataFrame(enet.predict(X_test), index=y_test.index, columns=y_test.columns)
#
#res = pd.concat((y_test, y_test_pred), axis=1, keys=['Actual', 'Pred'])
#
#res.stack(level='location').groupby('location').apply(lambda s: mean_absolute_error(s['Actual'], s['Pred'])).sort_values()


# Conclusiones:
# * Ajustar los 19 modelos ENet tarda poco más de 1h con `n_jobs=10`
# * Parece que hay bastantes diferencias entre estaciones
# 
# **TODO**:
# * Hacer un mapa de las estaciones coloreado por MAE, a ver si tienen alguna distribución espacial
# * Probar otros modelos
# * Normalizar las series temporales entre [0, 1]?
# * Probar con distintos valores de `width`, como si fuera un hiperparámetro
# * Ver sparsity
# * Ajustar otros modelos (RandomForest, GradientBoosting, Neural Networks)
