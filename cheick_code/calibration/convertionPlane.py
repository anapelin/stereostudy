#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 16:37:30 2023

@author: ngourgue
"""

import os, sys
import copy as cp
import numpy as np
import pandas as pd
import datetime as dt

#%%
if not '/home/ngourgue/climavion/detection_contrail' in sys.path:
    sys.path.append('/home/ngourgue/climavion/detection_contrail')
    
from AVION import readLatLon, convAltitude2

#%% open file
path = '/homedata/ngourgue/Images/SIRTA/2019'
name = '2019contrails_mix.csv'
filenameOri = os.path.join(path, name)
fileOri = pd.read_csv(filenameOri)

#%% function

def changeCoordonate(dateTime, flightname):
    
    if type(dateTime) == str:
        dateTime = dt.datetime.strptime(dateTime, '%Y-%m-%d %H:%M:%S')
        
    #readRadarfile 
    pathRadar = '/homedata/ngourgue/ADSB/2019/%02d/%04d%02d%02d/%04d%02d%02d_hr%02d_min%02d_last3min.bst'%(
        dateTime.month, dateTime.year, dateTime.month, dateTime.day, dateTime.year, dateTime.month, 
        dateTime.day, dateTime.hour, dateTime.minute+1)
    
    radarData = pd.read_csv(pathRadar, header=None)
    
    flightData = radarData[radarData[4] == flightname]
    
    flightData.loc[:, 'dateTime'] = pd.to_datetime(flightData[0]+' '+flightData[1], 
                                                   format ='%Y/%m/%d %H:%M:%S.%f').copy()
    
    flightData.at[:, 'diffTime'] = np.abs(flightData['dateTime'] - dateTime)
    line = flightData.iloc[np.argmin(flightData['diffTime'])]

    return float(line[8]), float(line[9]), float(line[10])

        
        
#%%
alts=[]
lats=[]
lons=[]

lat_site, lon_site = readLatLon('SIRTA')

for i in range(fileOri.shape[0]):
    alt, lat, lon = changeCoordonate(fileOri['datetime'].iloc[i], fileOri['flightname'].iloc[i])   
    alts.append(alt)
    lats.append(lat)
    lons.append(lon)
    
fileOri['datetime'] = pd.to_datetime(fileOri['datetime'], format = '%Y-%m-%d %H:%M:%S')
newFile = cp.deepcopy(fileOri)
radius=6370.0
newFile['Y'] = radius*(lats-lat_site)/180.*np.pi
newFile['X'] = radius*np.cos((lat_site+lats)/2./180.*np.pi)*(lons-lon_site)/180.*np.pi
for i in range(newFile.shape[0]):
    newFile.at[i, 'Z'] = convAltitude2(np.array(alts)[i], newFile['datetime'].iloc[i], unit = 'km')

# Normalise
matTmp = [-newFile['Y']/np.sqrt(newFile['X']**2+newFile['Y']**2+newFile['Z']**2), 
           newFile['X']/np.sqrt(newFile['X']**2+newFile['Y']**2+newFile['Z']**2), 
           newFile['Z']/np.sqrt(newFile['X']**2+newFile['Y']**2+newFile['Z']**2)]

matTmp = np.array(matTmp)
newFile['Z'] = matTmp[2]
newFile['X'] = matTmp[0]
newFile['Y'] = matTmp[1]


#%%
namenew = '2019contrails_newAlt.csv'
filenamenew = os.path.join(path, namenew)
newFile.to_csv(filenamenew)
