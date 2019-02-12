#!/usr/bin/env python
# coding: utf-8

# # Build train and test matrices
import sys
import pandas as pd
import numpy as np
import feather
from utils.download_data import timer

with timer():
    df = (feather.read_dataframe('/home/SHARED/SOLAR/data/oahu_min.feather')
                 .set_index('Datetime'))


df = df.iloc[0:10, 0:3]

# I feel this function can also be done for pd.DataFrame
def window_stack(a, width=3):
    n = a.shape[0]
    return np.hstack(list(a[(width-1-i):(n-i)] for i in range(0, width)))

# In pandas 0.24, use df.to_numpy() instead of df.values. Also care with non-numeric columns
width = 3
a = window_stack(df.values, width=width)

n = df.shape[0]

times   = [ ('t' if not idx else 't-{:d}'.format(idx)) for idx in range(width) ]
columns = pd.MultiIndex.from_product((times, df.columns), names=('time', 'location'))

# Convert back to DataFrame, just for convenience of having indexes
df_roll = pd.DataFrame(a, index=df.index[width-1:], columns=columns)
print(df_roll.shape)
print(df_roll)

df1 = pd.concat((df[(width-1-i):(n-i)].reset_index().drop(columns='Datetime') for i in range(0, width)),
                axis=1, keys=times, names=['time', 'location']).set_index(df.index[width-1:])

df2 = pd.concat((df[(width-1-i):(n-i)] for i in range(0, width)),
                axis=1, keys=times, names=['time', 'location'], join_axes=[df.index[width-1:]])
print(df1.shape)
print(df1)

print(df2.shape)
print(df2)

print(df1.equals(df_roll))
print(np.isclose(df1, df_roll).all())
pd.util.testing.assert_frame_equal(df_roll, df1)
pd.util.testing.assert_frame_equal(df1, df2)
