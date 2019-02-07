#!/usr/bin/env python

import pandas as pd
from download_data import timer

df = pd.read_feather('./data/oahu.feather')

with timer(tag='pandas'):
    dt = (df['Date'] + '_' +
          df['HST'].astype(str).str.pad(4, fillchar='0') +
          df['Seconds'].astype(str).str.pad(2, fillchar='0'))

    df['Datetime'] = pd.to_datetime(dt, format='%Y%m%d_%H%M%S')

import numpy as np
from multiprocessing import Pool

num_partitions = 8
num_cores = 8

def parallel_df(df, func):
    df_split = np.array_split(df, num_partitions)
    with Pool(num_cores) as pool:
        df = pd.concat(pool.map(func, df_split))
    return df

def create_dt(df):
    dt = (df['Date'] + '_' +
          df['HST'].astype(str).str.pad(4, fillchar='0') +
          df['Seconds'].astype(str).str.pad(2, fillchar='0'))

    df['Datetime'] = pd.to_datetime(dt, format='%Y%m%d_%H%M%S')
    return df

with timer(tag='pool'):
    df = parallel_df(df, create_dt)

import dask.dataframe as dd

with timer(tag='dask_map'):
    ddf = dd.from_pandas(df, npartitions=num_partitions)

    ddf['Datetime'] = (ddf['Date'] + '_' +
                       ddf['HST'].astype(str).str.pad(4, fillchar='0') +
                       ddf['Seconds'].astype(str).str.pad(2, fillchar='0'))

    ddf['Datetime'] = ddf['Datetime'].map_partitions(pd.to_datetime, format='%Y%m%d_%H%M%S', meta=('datetime64[ns]'))
    ddf.compute()

with timer(tag='dask'):
    ddf = dd.from_pandas(df, npartitions=num_partitions)

    ddf['Datetime'] = (ddf['Date'] + '_' +
                       ddf['HST'].astype(str).str.pad(4, fillchar='0') +
                       ddf['Seconds'].astype(str).str.pad(2, fillchar='0'))

    df = ddf.compute()
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y%m%d_%H%M%S')
