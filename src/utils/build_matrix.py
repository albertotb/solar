'''
Functions to build df_roll (Each time step in a different index)
and to transform it into numpy arrays
'''

import numpy as np
import pandas as pd

# https://stackoverflow.com/questions/15722324/sliding-window-in-numpy
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


def df_shift(df, periods=1):
    return (pd.concat([df] + [ df.tshift(t+1, freq='1min') for t in range(periods) ], axis=1, 
                      keys=['t'] + [ 't-{:d}'.format(t+1) for t in range(periods) ],
                      names=['time', 'location'])
              .dropna())


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
