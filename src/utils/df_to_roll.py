'''
Functions to build df_roll (Each time step in a different index)
and to transform it into numpy arrays
'''

import numpy as np
import pandas as pd


def window_stack_forward(a, stepsize=1, width=3):
    """https://stackoverflow.com/questions/15722324/sliding-window-in-numpy"""
    return np.hstack(a[i:1 + i - width or None:stepsize] for i in range(0, width))


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


def to_array_cov1d(df_roll, chosen_times=None, scale=1., reverse_times=True):
    '''
    Convert the rolled dataframe into the X and y numpy arrays.
    The X array has several time slices, given in chosen_times and
    sorts them in ascending order if reverse_times=True.
    Output shape X_np : (item, sensor, time_slice), 'time last'
    Output shape y_np : (item, sensor)
    '''
    y = df_roll['t']
    n_sensors = y.shape[1]
    X = df_roll.drop(columns='t', level='time')

    if chosen_times is None:
        chosen_times = X.columns.levels[0].tolist()
        chosen_times.remove('t')
    assert 't' not in chosen_times, 'Time slice t should not be in chosen_times'

    if reverse_times:
        chosen_times = chosen_times[::-1]

    #  This is the correct way of getting the correct order of the array
    X_np = df_roll[chosen_times].to_numpy().reshape(
        (-1, len(chosen_times), n_sensors))
    # Swap time_slice and sensor axes
    X_np = np.swapaxes(X_np, 1, 2)
    y_np = y.to_numpy()

    if scale != 1.:
        X_np = X_np / scale
        y_np = y_np / scale

    assert X_np.shape[2] == len(
        chosen_times), 'X_np does not have correct number of time slices'
    assert X_np.shape[0] == y_np.shape[0], 'X_np and y_np do not have the same number of items'
    assert X_np.shape[1] == y_np.shape[1], 'X_np and y_np do not have the same number of sensors'

    return X_np, y_np


def to_array_cov2d(df_roll, chosen_times=None, scale=1., reverse_times=True):
    '''
    Convert the rolled dataframe into the X and y numpy arrays.
    The X array has several time slices, given in chosen_times and
    sorts them in ascending order if reverse_times=True.
    Output shape X_np : (item, sensor, time_slice, 1), 'channels_last'
    Output shape y_np : (item, sensor)
    '''
    X_np, y_np = to_array_cov1d(df_roll, chosen_times, scale, reverse_times)
    # Add last axis: channels
    X_np = np.expand_dims(X_np, axis=3)
    return X_np, y_np
