'''
Functions to build df_roll (Each time step in a different index)
and to transform it into numpy arrays
'''

import numpy as np
import pandas as pd


def window_stack_forward(a, stepsize=1, width=3):
    """https://stackoverflow.com/questions/15722324/sliding-window-in-numpy"""
    return np.hstack(a[i:1 + i - width or None:stepsize] for i in range(0, width))

# 


def window_stack(a, width=3, step=1):
    """
    I feel this function can also be done for pd.DataFrame
    """
    n = a.shape[0]
    return np.hstack(list(a[(width - 1 - i):(n - i)] for i in range(0, width, step)))


def df_to_roll(df, width, step):
    '''
    Transform the df into df_roll, with an multi index columns.
    One column for each time slice
    This is: ['t','t-step','t-2*step',...]
    '''
    np_array = window_stack(df.to_numpy(), width, step)
    times = [('t' if not idx else 't-{:d}'.format(idx))
             for idx in range(0, width, step)]
    columns = pd.MultiIndex.from_product(
        (times, df.columns), names=('time', 'location'))
    # Convert back to DataFrame, just for convenience of having indexes
    df_roll = pd.DataFrame(
        np_array, index=df.index[width - 1:], columns=columns)
    return df_roll
