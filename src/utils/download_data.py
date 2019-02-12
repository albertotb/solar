#!/usr/bin/env python
# coding: utf-8

import requests
import zipfile
import io
from contextlib import contextmanager
from timeit import default_timer
import pandas as pd

@contextmanager
def timer(tag=None):
    start = default_timer()
    try:
        yield
    finally:
        end = default_timer()
        print(('[{}] '.format(tag) if tag else '') +
               'Elapsed time (s): {:.6f}'.format(end - start))


if __name__ == '__main__':

    # Download data from https://data.nrel.gov/submissions/11
    url = 'https://midcdmz.nrel.gov/oahu_archive/rawdata/Oahu_GHI/{}.zip'

    mrange = pd.date_range('2010-03', '2011-11', freq='M').strftime('%Y%m')

    with timer(tag='download'):
        for year_month in mrange:
            r = requests.get(url.format(year_month))
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path='./data')

    # Info from https://midcdmz.nrel.gov/oahu_archive/
    # Stations: location https://midcdmz.nrel.gov/oahu_archive/instruments.html
    # Map: https://midcdmz.nrel.gov/oahu_archive/map.jpg

    fields = [ 'Seconds', 'Year', 'DOY', 'HST', 'GH_DH3', 'GH_DH4', 'GH_DH5', 'GH_DH10',
               'GH_DH11', 'GH_DH9', 'GH_DH2', 'GH_DH1', 'GT_DH1', 'GH_AP6', 'GT_AP6',
               'GH_AP1', 'GH_AP3',  'GH_AP5', 'GH_AP4', 'GH_AP7', 'GH_DH6', 'GH_DH7',
               'GH_DH8']

    drange = pd.date_range('2010-03-18', '2011-10-31', freq='D').strftime('%Y%m%d')

    with timer(tag='read_csv'):
        df_dict = { date: pd.read_csv('./data/{}.txt'.format(date), header=None, names=fields)
                    for date in drange }

    with timer(tag='concat'):
        df = (pd.concat(df_dict)
                .reset_index()
                .rename(columns={'level_0': 'Date'})
                .drop(columns='level_1'))

    # Much slower, not vectorized
    #with timer():
    #    dt = df[['Date', 'HST', 'Seconds']].apply(
    #             lambda s: '{Date}_{HST:04d}{Seconds:02d}'.format(**s.to_dict()),
    #             axis=1)

    with timer(tag='to_datetime'):
        dt = (df['Date'] + '_' +
              df['HST'].astype(str).str.pad(4, fillchar='0') +
              df['Seconds'].astype(str).str.pad(2, fillchar='0'))

        df['Datetime'] = pd.to_datetime(dt, format='%Y%m%d_%H%M%S')

    with timer(tag='to_feather'):
        df.to_feather('./data/oahu.feather')
