#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:49:53 2024

@author: ngourgue
"""
import pandas as pd 


def readLatLon(site, path = '/home/ngourgue/climavion/params.csv'):
    '''
    Extract latitude and longitude about site.

    Parameters
    ----------
    site : string
        Site name, SIRTA, Orsay...
    path : string, optional
        The file where we extract lat and lon. The default is '/home/ngourgue/climavion/params.csv'.

    Returns
    -------
    lat : float
        Site's latitude.
    lon : float
        Site's longitude.

    '''
    #open file
    data = pd.read_csv(path, index_col=0)
    #filter site
    line = data.loc[site]
    #extract data
    lat = line['lat']
    lon = line['lon']
    return lat, lon