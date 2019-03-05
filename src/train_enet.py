#!/usr/bin/env python
# coding: utf-8

# # Build train and test matrices
import sys
import pandas as pd
import numpy as np
import feather
import pickle
from sklearn.linear_model import ElasticNetCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from utils.download_data import timer


# In pandas 0.24, use df.to_numpy() instead of df.values. Also care with non-numeric columns
def window_stack(df, width=3):
    """Converts a pandas.DataFrame with shape (n, d) to another one with shape
       (n-width, d*width) by stacking shifted versions of itself
    """
    n = df.shape[0]
    a = np.hstack(list(df.values[(width-1-i):(n-i)] for i in range(0, width)))

    times   = [ ('t' if not idx else 't-{:d}'.format(idx)) for idx in range(width) ]
    columns = pd.MultiIndex.from_product((times, df.columns), names=('time', 'location'))

    return pd.DataFrame(a, index=df.index[width-1:], columns=columns)


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('ArgError')
        sys.exit()

    fname = sys.argv[1]
    width = int(sys.argv[2])

    df = (feather.read_dataframe(fname)
                 .set_index('Datetime'))

    df_roll = window_stack(df, width=width)

    mem = df_roll.memory_usage(index=True, deep=True)
    print(mem)
    print(mem.sum()*1e-9)

    # Split target (time t) and variables (times t-1 to t-width+1)
    y = df_roll['t']
    X = df_roll.drop(columns='t', level='time')

    # Split train-test, approximately 12 and 4 months respectively
    X_train, X_test = X[:'2011-07-31'], X['2011-08-01':]
    y_train, y_test = y[:'2011-07-31'], y['2011-08-01':]

    enet = MultiOutputRegressor(ElasticNetCV(cv=TimeSeriesSplit(n_splits=5), l1_ratio=0.5), n_jobs=10)
    with timer():
        enet.fit(X_train, y_train)

    y_test_pred = pd.DataFrame(enet.predict(X_test), index=y_test.index, columns=y_test.columns)
    res = pd.concat((y_test, y_test_pred), axis=1, keys=['Actual', 'Pred'])

    with open('model_{}.pkl'.format(width), 'wb') as f:
        pickle.dump({'model': enet, 'pred': res}, f)
