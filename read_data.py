#!/usr/bin/env python

import pandas as pd
import numpy as np

df = pd.read_feather('/home/atorres/SHARED/SOLAR/data/oahu.feather')

df1 = (df.replace(to_replace=-99999.0, value=np.nan)
         .drop(columns=['Date', 'Seconds', 'Year', 'DOY', 'HST']))

print(df1.shape)
print(df1.isnull().mean())

# impute nans and remove leading
df2 = (df1.set_index('Datetime')
          .ffill()
          .dropna())

print(df2.shape)
print(df2.isnull().mean())

# some days have data from 4:30 to 19:30, others from
# 5:00 to 20:00 and others from 5:30 to 20:30
# resample is adding new rows for every day so they all
# have same range from 4:30 to 20:30
df_min = df2.resample('min').first().dropna()

print(df_min.shape)
print(df_min.isnull().mean())
df_min.reset_index().to_feather('/home/atorres/SHARED/SOLAR/data/oahu_min.feather')
