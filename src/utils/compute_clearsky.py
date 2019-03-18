#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import feather
import pysolar
import pvlib
from download_data import timer

DATA_PATH = '/home/SHARED/SOLAR/data/'

def clearsky_pysolar(s, info=None):
    ''' Compute the total direct radiation under a clear sky model.
        `get_altitude()` returns negative values when night time so we filter 
        those and assign a radiation of 0
    '''
    lat, lon = info.loc[s['Location'], ['Latitude', 'Longitude']]
    deg = pysolar.solar.get_altitude(lat, lon, s['Datetime'])
    if deg <= 0:
        radiation = 0.
    else:
        radiation = pysolar.radiation.get_radiation_direct(s['Datetime'], deg)
    return(pd.Series([radiation, deg]))


def clearsky_pvlib(df, info=None, model='ineichen'):
    ''' Compute the total direct radiation under a clear sky model using the pvlib library.
        Ref: https://pvlib-python.readthedocs.io/en/latest/clearsky.html
    '''
    lat, lon = info.loc[df.name, ['Latitude', 'Longitude']]
    location = pvlib.location.Location(lat, lon, tz='Pacific/Honolulu')
    return(location.get_clearsky(df.index.get_level_values('Datetime'), model=model))


if __name__ == '__main__':

    # read minute data and location info
    df = (feather.read_dataframe(DATA_PATH + 'oahu_min.feather')
                 .set_index('Datetime'))

    info = pd.read_csv(DATA_PATH + 'info.csv')

    # normalize location names by removing "HL_"
    info['Location'] = info['Location'].str.replace('(HL)?_', '')
    info.set_index('Location', inplace=True)
    info.to_pickle(DATA_PATH + 'info.pkl')

    # convert from wide form (one column per station-radiation_type) to long form
    df1 = df.stack().reset_index().rename(columns={'level_1': 'Radiation_Location', 0: 'Values'})

    # split the location column into the sensor type (GH, GT) and the location name
    df1[['Radiation', 'Location']] = df1['Radiation_Location'].str.split('_', n=1, expand=True)

    # convert back to wide form but now with only 2 columns, one for GH and other for GT
    df2 = (df1.pivot_table(index=['Datetime', 'Location'], columns='Radiation', values='Values')
              .reset_index()
              .rename(columns={'GH': 'GHI', 'GT': 'GTI'}))

    # add timezone info (Hawaii standard time)
    df2['Datetime'] = df2['Datetime'].dt.tz_localize('HST')

    # apply function to every row
    with timer(tag='pysolar'):
        df2[['Pysolar', 'Altitude']] = df2[['Datetime', 'Location']].apply(clearsky_pysolar, axis=1, info=info)

    # we set the index because pvlib functions need a Datetimeindex
    df2.set_index(['Location', 'Datetime'], inplace=True)

    with timer(tag='ineichen'):
        df2['Ineichen'] = df2.groupby('Location').apply(clearsky_pvlib, info=info, model='ineichen')['ghi']

    with timer(tag='haurwitz'):
        df2['Haurwitz'] = df2.groupby('Location').apply(clearsky_pvlib, info=info, model='haurwitz')['ghi']

    with timer(tag='solis'):
        df2['Solis'] = df2.groupby('Location').apply(clearsky_pvlib, info=info, model='simplified_solis')['ghi']

    df2.to_pickle(DATA_PATH + 'oahu_min_cs.pkl')
