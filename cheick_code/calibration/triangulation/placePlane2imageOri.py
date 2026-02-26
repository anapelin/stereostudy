#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:12:26 2024

@author: ngourgue

convention x = -lat, y = lon, z = lat 
"""

#%% import 
import os#, time
# import copy as cp
import numpy as np
import pandas as pd
import datetime as dt

from matplotlib import pyplot as plt

#%% set path
path_here = os.getcwd()
if path_here != os.path.join("/home", os.environ['USER'], "climavion", "detection_contrail"):
    os.chdir(os.path.join("/home", os.environ['USER'], "climavion", "detection_contrail"))
path_here = os.getcwd()

#%% import custom
from AVION import findplanes, datespan, extract_plane, convAltitude
from setup_variable import pathBuilder, readLatLon
from image import imagetime, readimage2

#%% main

if __name__ == '__main__':
    #init time
    year = 2024
    month = 3
    day = 4
    hour = 8
    minute = 56
    
    durationMinute = 18
    minuteStep = 2
    
    beg = dt.datetime(year, month, day, hour, minute)
    end = beg + dt.timedelta(minutes = durationMinute)
    
    #init path
    site = 'SIRTA'
    path = pathBuilder()
    path.setAuto()
    path.setSite(site)
    
    outputFolder = '/home/ngourgue/Documents/these/colab/eurocontrole/202403040_contrail2'
    
    #other params
    process = {'open' : {'imtype' : 1, 'Site' : path.getSite()}}
    last = 3
    verbose = ['debug']
    method = 'hour'
    radius=6370.0
    lat_site, lon_site = readLatLon(site)
    
    #create csv with lat lon plane
    path.setDateDay(end)
    # flines = readRadar(path, last, method = method, verbose = verbose)
    flines = pd.read_csv(path.getPathRadarHour(last), header = None)
    flines.columns = ['date', 'time', '2', '3', 'flightname', 'country', '6', 'alt', '8', 'lat', 'lon', '11', '12', 'speed', 'bearing', '15', '16']
       
    flines = flines.loc[flines['alt'] > 25000].copy() #filter alt
    flines['dateTime'] = pd.to_datetime(flines['date']+flines['time'], format='%Y/%m/%d%H:%M:%S.%f') #datetime conversion
    flines = flines[flines['dateTime']<=end]
    flines = flines[flines['dateTime']>=beg]
    
    #convAlt
    flines['altkm'] = convAltitude(flines['alt'].values, 
                                     dt.datetime.strptime(flines['date'].iloc[0]+flines['time'].iloc[0], '%Y/%m/%d%H:%M:%S.%f'),
                                     unit = 'km', path=path)
    
    #calcul X, Y, Z normalise on camera sphere.
    flines['X'] = -radius*(flines['lat']-lat_site)/180.*np.pi
    flines['Y'] = radius*np.cos((lat_site+flines['lat'])/2./180.*np.pi)*(flines['lon']-lon_site)/180.*np.pi
    flines['Z'] = flines['altkm']
    
    flines['xNorm'] = flines['X']/np.sqrt(flines['X']**2 + flines['Y']**2 + flines['Z']**2)
    flines['yNorm'] = flines['Y']/np.sqrt(flines['X']**2 + flines['Y']**2 + flines['Z']**2)
    flines['zNorm'] = flines['Z']/np.sqrt(flines['X']**2 + flines['Y']**2 + flines['Z']**2)
    
    #only data
    planeData = pd.DataFrame(columns=['altkm', 'lon', 'lat', 'datetime', 'xImage', 'yImage', 'xNorm', 'yNorm', 'zNorm', 'flightname'])
    
    for timestamp in datespan(beg, end, delta= dt.timedelta(minutes=minuteStep)):
        
        #set datetime
        path.setDateDay(timestamp)
        #open image
        realsec, warn = imagetime(path = path, imtype = process['open']['imtype'], 
                                  site = path.getSite(), cle = False, verbose = verbose)
        
        found, imRGBori = readimage2(path, imtype = process['open']['imtype'], cle = False, verbose = verbose)
        
        if warn == 'Fail':
            ValueError('image not found')
        hourim=timestamp.hour+timestamp.minute/60.+(realsec+5.)/3600.
        
        #calcul plane position
        xflight, yflight, tflight, flight_details, age_plane =\
            findplanes(path, Datetime = timestamp + dt.timedelta(minutes=1), 
                       last = last, hourim = hourim, imageShape = imRGBori.shape,
                       site = site, dpixel=2, method = method, verbose = verbose)     
        
            
        #plot
        #transpose imRGBOri
        imRGBTrans = imRGBori.transpose(1, 0, 2)
        figi = plt.figure()
        axesi = figi.add_subplot(1, 1, 1)
        axesi.imshow(imRGBTrans[::-1,:, :])
        for flightName in  flight_details.keys():
            if flightName in flines['flightname'].unique():
    
                flight = flight_details[flightName]
                flightname, code, xx, yy, tt, flightfeet, flightbearing, flightspeed, xflightexact, \
                    yflightexact = extract_plane(flight)
                #add plane
                tmpPlaneData = pd.DataFrame(columns=['altkm', 'lon', 'lat', 'datetime', 'xImage', 'yImage', 'xNorm', 'yNorm', 'zNorm', 'flightname'])
                flinesPlane = flines[flines['flightname'] == flightName]
                tmpPlaneData[['altkm', 'lat', 'lon',  'datetime', 'xNorm', 'yNorm', 'zNorm', 'flightname']] = \
                    flinesPlane[['altkm', 'lat', 'lon', 'dateTime', 'xNorm', 'yNorm', 'zNorm', 'flightname']]
                #search best way
                for i, tt1 in enumerate(tt):
                    ind = (tmpPlaneData['datetime'].dt.hour + tmpPlaneData['datetime'].dt.minute/60 + tmpPlaneData['datetime'].dt.second/3600).where(
                        tmpPlaneData['datetime'].dt.hour + tmpPlaneData['datetime'].dt.minute/60 + tmpPlaneData['datetime'].dt.second/3600 == tt1).dropna()
                    if ind.shape[0] > 0:
                        indi = ind.index.values[0]
                        tmpPlaneData.loc[indi, 'xImage'] = xx[i]
                        tmpPlaneData.loc[indi, 'yImage'] = yy[i]
                    else:
                        #search nearest point
                        indi = ((tmpPlaneData['datetime'].dt.hour + tmpPlaneData['datetime'].dt.minute/60 + tmpPlaneData['datetime'].dt.second/3600 - tt1)**2).argmin()
                        tmpPlaneData.loc[tmpPlaneData.index[indi], 'xImage'] = xx[i]
                        tmpPlaneData.loc[tmpPlaneData.index[indi], 'yImage'] = yy[i]
                        
                tmpPlaneData.dropna(inplace=True)
                planeData = pd.concat([planeData, tmpPlaneData])
                
                axesi.scatter(xx, yy, s = 2, label = flightname)
            figi.legend()
        axesi.set_title('%s'%(timestamp.strftime('%Y%m%d_%H%M%S')))
            
        figi.savefig(os.path.join(outputFolder, '%s_3minAvion.png'%(timestamp.strftime('%Y%m%d_%H%M%S'))))
        
    planeData.to_csv(os.path.join(outputFolder, 'dataPlane.csv'))
        