#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import feather
import pysolar
from download_data import timer

def clear_sky(s, info=None):
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
    return(radiation)


if __name__ == '__main__':

    # read minute data and location info
    df = (feather.read_dataframe('/home/SHARED/SOLAR/data/oahu_min.feather')
                 .set_index('Datetime'))

    info = pd.read_csv('/home/SHARED/SOLAR/data/info.csv')

    # normalize location names by removing "HL_"
    info['Location'] = info['Location'].str.replace('(HL)?_', '')
    info.set_index('Location', inplace=True)

    # convert from wide form (one column per station-radiation_type) to long form
    df1 = df.stack().reset_index().rename(columns={'level_1': 'Radiation_Location', 0: 'Values'})

    # split the location column into the sensor type (GH, GT) and the location name
    df1[['Radiation', 'Location']] = df1['Radiation_Location'].str.split('_', n=1, expand=True)

    # convert back to wide form but now with only 2 columns, one for GH and other for GT
    df2 = df1.pivot_table(index=['Datetime', 'Location'], columns='Radiation', values='Values').reset_index()

    # add timezone info (Hawaii standard time)
    df2['Datetime'] = df2['Datetime'].dt.tz_localize('HST')

    # apply function to every row
    with timer():
        df2['ClearSky'] = df2[['Datetime', 'Location']].apply(clear_sky, axis=1, info=info)

    # TODO: set negative GH to 0 and normalize with ClearSky
    df2.to_pickle('/home/SHARED/SOLAR/data/oahu_min_norm.pkl')
