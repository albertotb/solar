#!/usr/bin/env python
# coding: utf-8

# # Build train and test matrices
import sys
import pandas as pd
import numpy as np
import feather
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
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

    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print('ArgError')
        sys.exit()

    fname = sys.argv[1]
    width = int(sys.argv[2])
    location = int(sys.argv[3])
    if len(sys.argv) == 5:
        seed = int(sys.argv[4])

    df = (feather.read_dataframe(fname)
                 .set_index('Datetime'))

    df_roll = window_stack(df, width=width)

    mem = df_roll.memory_usage(index=True, deep=True)
    print('Memory: {:.2f} Gb'.format(mem.sum()*1e-9))

    # Split target (time t) and variables (times t-1 to t-width+1)
    y = df_roll['t']
    X = df_roll.drop(columns='t', level='time')

    # Split train-test, approximately 12 and 4 months respectively
    X_train, X_test = X[:'2011-07-31'], X['2011-08-01':]
    y_train, y_test = y[:'2011-07-31'].iloc[:, location], y['2011-08-01':].iloc[:, location]

    params = {'n_estimators'      : [100, 500],
              'max_features'      : [0.2, 0.4, 0.6],
              'min_samples_split' : [5, 20, 40],
              'min_samples_leaf'  : [1, 2, 5],
              'max_depth'         : [None]}

    np.random.seed(seed)

    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    rfr = GridSearchCV(RandomForestRegressor(), param_grid=params, n_jobs=10,
            cv=TimeSeriesSplit(n_splits=5), scoring=mae_scorer)

    with timer():
        rfr.fit(X_train, y_train)

    with open('rfr_{}_{:02d}.pkl'.format(width, location), 'wb') as f:
        pickle.dump(rfr, f)
