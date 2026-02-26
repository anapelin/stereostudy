#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:07:29 2024

@author: ngourgue


Conversion 
Init lat en Y et lon en X
during normalisation
after X = -Y
and   Y = X
"""

#%% import 
import os, time
import copy as cp
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
from setup_variable.position import readLatLon
from setup_variable.path_file import pathBuilder
from AVION import readRadar, convAltitude

#%% def droite
def calPoint(p, t, d):
    return p + t*d   

###############################################################################

def calDroite(p, txyz, d, i):
    return calPoint(p, txyz[:, i], d)

###############################################################################

def calDroiteFull(p, d, mini, maxi, length = 101):
    t = np.linspace(mini, maxi, length)
    tx = t
    ty = t
    tz = t
    txyz = np.array([tx, ty, tz])
    v = []
    for i in range(txyz.shape[1]):
        vi = calDroite(p, txyz, d, i)
        v.append(vi)
        
    va = np.array(v)
        
    return va

###############################################################################

def normVect(vector):
    return vector/np.sqrt((vector*vector).sum())

###############################################################################

def calcNearest(p1, p2, d1, d2, verbose = ['']):
    n = np.cross(d1, d2) 

    n2 = np.cross(d2, n)
    n1 = np.cross(d1, n)
    
    # vn2 = calDroiteFull(p2, n2, 0, maxi)
    # vn1 = calDroiteFull(p1, n1, 0, maxi)
    
    #nearest
    c1 = p1 + np.dot((p2-p1), n2)/np.dot(d1, n2) * d1
    #p1 is the nearest
    
    c2 = p2 + np.dot((p1-p2), n1)/np.dot(d2, n1) * d2
    
    #distance d
    d = np.linalg.norm(c1-c2)
    if 'all' in verbose or 'debug' in verbose or 'dist' in verbose: 
        print('distance entre c1 et c2 :', d)
    
    return c1, c2, d

###############################################################################

def calcLinePoint(p1, p2, lineAvion1, dAvion2, verbose = [''] ):
    d = []
    for i in range(lineAvion1.shape[1]):
        pAvion1 = lineAvion1[:, i]
        d1 = normVect(pAvion1)
        cSirtai, cBretyi, di = calcNearest(p1, p2, d1, dAvion2)
        d.append(di)  
        
    indMin = np.where(d == min(d))[0]
    
    return indMin

###############################################################################

def calcNearestLP(p1, p2, lineAvion1, dAvion2, verbose = ['']):
    indMin = calcLinePoint(p1, p2, lineAvion1, dAvion2)
    if 'all' in verbose or 'debug' in verbose or 'ind' in verbose :
        print('minimum index : %d'%(indMin))
    pAvion1Min = lineAvion1[:, indMin].reshape([3,])
    d1Min = normVect(pAvion1Min)
    c1Min, c2Min, dMin = calcNearest(p1, p2, d1Min, dAvion2, verbose)
    
    return c1Min, c2Min, d1Min, pAvion1Min, dMin
    
###############################################################################

def calcNearestLL(p1, p2, lineAvion1, lineAvion2, verbose = ['']):
    dist2 = []
    for i in range(lineAvion2.shape[1]):
        d2i = normVect(lineAvion2[:, i])
        c1Min, c2Min, d1Min, pAvion1Min, diMin = calcNearestLP(p1, p2, lineAvion1, d2i, verbose = verbose)
        dist2.append(diMin)
    
    indMin2 = np.where(dist2 == min(dist2))[0]
    # dBretyMin = lineBrety[:, indMin2]
    pAvion2Min = lineAvion2[:, indMin2].reshape([3,])
    d2Min = normVect(pAvion2Min)
    c1Min, c2Min, d1Min, pAvion1Min, dMin1 = calcNearestLP(p1, p2, lineAvion1, d2Min, verbose = ['ind', 'dist'])
    return c1Min, c2Min, d1Min, d2Min, pAvion1Min, dMin1

###############################################################################

def genDroite(data, ind1, ind2, nb = 10):
    p1 = data.loc[ind1, ['X', 'Y', 'Z']].values
    p2 = data.loc[ind2, ['X', 'Y', 'Z']].values
    
    lineX = np.linspace(p1[0], p2[0], nb)
    lineY = np.linspace(p1[1], p2[1], nb)
    lineZ = np.linspace(p1[2], p2[2], nb)
    
    line  = np.array([lineX, lineY, lineZ])
    
    return line

###############################################################################

def calcLink09(p1, p2, lineA1, lineA2, verbose = ['']):
    
    #cacl for each point lineA2 the nearest point in lineA1
    ind2 = np.ones([lineA2.shape[0]])*-1
    
    for i in range(lineA2.shape[0]):
        indi = calcLinePoint(p1, p2, lineA1.transpose(), lineA2[i, :])
        ind2[i] = indi
    
    ind1 = np.ones([lineA1.shape[0]])*-1
    for i in range(lineA1.shape[0]):
        indi = calcLinePoint(p2, p1, lineA2.transpose(), lineA1[i, :])
        ind1[i] = indi
    
    
    return ind1, ind2

###############################################################################

def calcCMat(p1, p2, d1s, n2):
    p1s = np.zeros_like(d1s)
    p1s = p1s + p1
    
    numerator = np.dot((p2-p1), n2.transpose())
    
    denominator = np.dot(d1s, n2.transpose())
    if type(denominator) == np.ndarray:
        denominator = np.diagonal(denominator)
    
    numDem = numerator/denominator
    
    product = np.zeros_like(d1s.transpose())
    product = product + numDem
    
    c1 = p1s + product.transpose() * d1s
    
    return c1

def calcNearests(p1, p2, d1s, d2s, verbose = ['']):
    n = np.cross(d1s, d2s) 

    n2 = np.cross(d2s, n)
    n1 = np.cross(d1s, n)
    
    # vn2 = calDroiteFull(p2, n2, 0, maxi)
    # vn1 = calDroiteFull(p1, n1, 0, maxi)
    
    #nearest
    # c1 = p1 + np.dot((p2-p1), n2)/np.dot(d1s, n2) * d1s
    #décompose c1
    c1 = calcCMat(p1, p2, d1s, n2)
    #p1 is the nearest
    
    # c2 = p2 + np.dot((p1-p2), n1)/np.dot(d2, n1) * d2
    c2 = calcCMat(p2, p1, d2s, n1)
    
    #distance d
    if len(c1.shape) > 1:
        d = np.linalg.norm(c1-c2,  axis = 1)
    elif len(c1.shape) == 1:
        d = np.linalg.norm(c1-c2)
    if 'all' in verbose or 'debug' in verbose or 'dist' in verbose: 
        print('distance entre c1 et c2 :', d)
    
    return c1, c2, d

###############################################################################

def calcInd(p1, p2, d1s, d2s, loop = True, verbose = ['']):
    ind = []
    # c1si, c2si, dsi = calcNearests(p1, p2, d1s[0, :], d2s[0, :], verbose = verbose)
    for i in range(d2s.shape[0]):
        d2si = np.zeros_like(d2s)
        d2si = d2si + d2s[i, :]
        c1si, c2si, dsi = calcNearests(p1, p2, d1s, d2si, verbose = verbose)
        indi = np.where(dsi == np.min(dsi))[0]
        ind.append(indi)
        
    return ind

###############################################################################

def calcNearestMat(p1, p2, d1sM, d2sN, verbose = ['']):
    n = np.cross(d1sM[:, 0, :], d2sN[0, :, :])
    
    n2 = np.cross(d2sN, n)
    n1 = np.cross(d1sM, n)
    
    # vn2 = calDroiteFull(p2, n2, 0, maxi)
    # vn1 = calDroiteFull(p1, n1, 0, maxi)
    
    #nearest
    # c1 = p1 + np.dot((p2-p1), n2)/np.dot(d1s, n2) * d1s
    #décompose c1
    c1 = calcCMat(p1, p2, d1sM, n2)
    #p1 is the nearest
    
    # c2 = p2 + np.dot((p1-p2), n1)/np.dot(d2, n1) * d2
    c2 = calcCMat(p2, p1, d2sN, n1)
    
    #distance d
    if len(c1.shape) > 1:
        d = np.linalg.norm(c1-c2,  axis = 1)
    elif len(c1.shape) == 1:
        d = np.linalg.norm(c1-c2)
    if 'all' in verbose or 'debug' in verbose or 'dist' in verbose: 
        print('distance entre c1 et c2 :', d)
    
    return c1, c2, d
    return None

###############################################################################

def calcNearestsLP(p1, p2, d1s, d2s, loop = True, verbose = ['']):
    if loop == True:
        ind2 = calcInd(p1, p2, d1s, d2s, loop = loop, verbose = verbose)
            
        ind1 = calcInd(p1, p2, d2s, d1s, loop = loop, verbose = [''])
        
    else:
        #shape of repeat
        d1sM = np.repeat(d1s, d2s.shape[0], axis = 0)
        d2sN = np.repeat(d2s, d1s.shape[0], axis = 0)
        
        #case d1s.shape = 2.3 d2s.sjape = 4.3
        # d2sN = d2sN[[0,2,4,6, 1,3,5,7], :]
        
        #cases general
        resort = np.arange(0, d2sN.shape[0])
        resort = (resort%d2s.shape[0])*d1s.shape[0]+resort//d2s.shape[0]
        d2sN = d2sN[resort, :]
        
        
        c1s, c2s, ds =  calcNearests(p1, p2, d1sM, d2sN, verbose = [''])
        c1s = c1s.reshape([d1s.shape[0], d2s.shape[0], c1s.shape[1]])
        c2s = c2s.reshape([d1s.shape[0], d2s.shape[0], c2s.shape[1]])
        ds  = ds.reshape([d1s.shape[0], d2s.shape[0]])
        ind1 = ds.argmin(axis = 1)
        ind2 = ds.argmin(axis = 0)
    return ind1, ind2
    
    

#%% def points
def createSite(site = 'SIRTA', conv = 'conv1'):
    radius=6370.0
    
    if site == 'SIRTA':
        #p1 = sirta
        p = np.array([0., 0., 0.])
    
    elif site == 'BRETIGNY':
        lat_sirta, lon_sirta = readLatLon('SIRTA')
        # lat_sirta = 48.713
        # lon_sirta = 2.208
        alt_sirta  = 0 #valeur à corriger
        
        #p2 = bretigny #valeur à corriger à partir de eurocontrole
        lat_brety = 48.610
        lon_brety = 2.308
        alt_brety = 0.150
        
        #projection brety on sirta in km
        Y_bretyAC = radius*np.deg2rad(lat_brety - lat_sirta)
        X_bretyAC = radius*np.cos(np.deg2rad((lat_sirta+lat_brety)/2.))*np.deg2rad(lon_brety-lon_sirta)
        Z_bretyAC = alt_brety - alt_sirta
        
        if conv == 'conv2':
            X_brety = -Y_bretyAC
            Y_brety =  X_bretyAC
            Z_brety =  Z_bretyAC
        elif conv == 'conv1':
            X_brety = X_bretyAC
            Y_brety = Y_bretyAC
            Z_brety = Z_bretyAC
        #distance brety sirta ~16.5km  
        p = np.array([X_brety, Y_brety, Z_brety])
    
    return p

###############################################################################

def createAvion(flightname = 'EZY14DT', conv = 'conv1', ref = 'SIRTA'):
    if flightname == 'EZY14DT':
        lat_avion = 48.73073
        lon_avion = 2.17311
        alt_avion = 10.86147
    
    elif flightname == 'EZY93JB':
        lat_avion = 48.72
        lon_avion = 2.17
        alt_avion = 11.43
        
    elif flightname == 'RYR8603':
        lat_avion = 48.73
        lon_avion = 2.17
        alt_avion = 11.76
    else:
        print("flightname unknow")
        p = None
        
    radius=6370.0
    if ref == 'SIRTA':
        # lat_sirta = 48.713
        # lon_sirta = 2.208
        lat_site, lon_site = readLatLon('SIRTA')
        alt_site  = 0 #valeur à corriger
        
    elif ref == 'BRETIGNY':
        lat_site = 48.610
        lon_site = 2.308
        alt_site = 0.150
    else:
        print("ref unknow")
        p = None
        
    #avion
    Y_avionAC = radius*np.deg2rad(lat_avion - lat_site)
    X_avionAC = radius*np.cos(np.deg2rad((lat_avion+lat_site)/2.))*np.deg2rad(lon_avion-lon_site)
    Z_avionAC = alt_avion - alt_site
        
    if conv == 'conv2':
        X_avion = -Y_avionAC
        Y_avion =  X_avionAC
        Z_avion =  Z_avionAC
    elif conv == 'conv1':
        X_avion = X_avionAC
        Y_avion = Y_avionAC
        Z_avion = Z_avionAC
    else:
        print("conv unknow")
        p = None
    
    p = np.array([X_avion, Y_avion, Z_avion])
    
    return p

################################################################################
    
def initSite(conv  = 'conv1', verbose  = ['']):
    p1 = createSite(site = 'SIRTA', conv = conv)
    p2 = createSite(site = 'BRETIGNY', conv = conv)
    
    dist = np.sqrt((p2*p2).sum())
    if 'all' in verbose or 'debug' in verbose or 'dist' in verbose :
        print('distance beetwen sirta et bretigny : %f'%dist)
    
    return p1, p2

###############################################################################

def extractAvionTraj(flightname = 'EZY14DT', path = None, verbose = ['']):
    last = 3
    method = 'hour'
    
    if path == None:
        path = pathBuilder()
        path.setSite('SIRTA')
        path.setAuto()
        path.setDateDay(dt.datetime(2019, 6, 1, 5, 42))
    #open radar file
    flines = readRadar(path, last, method = method, verbose = verbose)
    if type(flines) == bool:
        return False
    elif flines.shape[0] == 0:
        return False
    #filter plane with altitude
    conversionWithPressure = True
        
    # pathRadar = path.getPathRadarTime(last)
    # flines = pd.read_csv(pathRadar, header = None)
    flines.columns = ['date', 'time', '2', '3', 'flightname', 'country', '6', 'alt', '8', 'lat', 'lon', '11', '12', 'speed', 'bearing', '15', '16']
       
    #filter alt and lat, lon
    flines = flines.loc[flines['alt'] > 25000].copy()
    flines['dateTime'] = pd.to_datetime(flines['date']+flines['time'], format='%Y/%m/%d%H:%M:%S.%f')
    flines3 = flines.copy()
    
    #conversion alt
    if conversionWithPressure == True:
        flines = None
        del(flines)
        if flines3.shape[0] > 100000:
            lat_site, lon_site = readLatLon('SIRTA')
            #delete more than one degree to preserve rame
            flines3 = flines3.loc[flines3['lat'] < lat_site+1].copy()
            flines3 = flines3.loc[flines3['lat'] > lat_site-1].copy()
            flines3 = flines3.loc[flines3['lon'] < lon_site+1].copy()
            flines3 = flines3.loc[flines3['lon'] > lon_site-1].copy()
        flines3['altkm'] = convAltitude(flines3['alt'].values, 
                                         dt.datetime.strptime(flines3['date'].iloc[0]+flines3['time'].iloc[0], '%Y/%m/%d%H:%M:%S.%f'),
                                         unit = 'km', path = path)
        
    else:
        feet2km=0.3048e-3
        flines3['altkm'] = flines3['alt']*feet2km
    flines =flines3[flines3['flightname'] == flightname]
    
    return flines

###############################################################################

def avionProjSite(avionDF, site = 'SIRTA', conv = 'conv1'):
    if site == 'SIRTA':
        lat_site, lon_site = readLatLon('SIRTA')
        alt_site = 0.
        
    elif site == 'BRETIGNY':
        lat_site, lon_site = 48.610, 2.308
        alt_site = 0.150
        
    radius=6370.0
    avionDFSite = avionDF.copy()
    avionDFSite['Y'] = radius*np.deg2rad(avionDFSite['lat']-lat_site)
    avionDFSite['X'] = radius*np.cos(np.deg2rad((lat_site+avionDFSite['lat'])/2.))*np.deg2rad(avionDFSite['lon']-lon_site)
    avionDFSite['Z'] = avionDFSite['altkm'] - alt_site
    
    if conv  ==  'conv2':
        yTMP = cp.deepcopy(avionDFSite['Y'])
        avionDFSite['Y'] = avionDFSite['X']
        avionDFSite['X'] = -yTMP
        matPos = [ avionDFSite['X']    /np.sqrt(avionDFSite['X']**2+avionDFSite['Y']**2+avionDFSite['Z']**2), 
                   avionDFSite['Y']    /np.sqrt(avionDFSite['X']**2+avionDFSite['Y']**2+avionDFSite['Z']**2), 
                   avionDFSite['Z']    /np.sqrt(avionDFSite['X']**2+avionDFSite['Y']**2+avionDFSite['Z']**2)]
        

    elif conv == 'conv1':
        matPos = [ avionDFSite['X']    /np.sqrt(avionDFSite['X']**2+avionDFSite['Y']**2+avionDFSite['Z']**2), 
                   avionDFSite['Y']    /np.sqrt(avionDFSite['X']**2+avionDFSite['Y']**2+avionDFSite['Z']**2), 
                   avionDFSite['Z']    /np.sqrt(avionDFSite['X']**2+avionDFSite['Y']**2+avionDFSite['Z']**2)]
    flines2 = avionDFSite[['date', 'time', 'flightname', 'altkm', 'lat', 'lon', 'X', 'Y', 'Z']].copy()
    
    matPos1 = np.array(matPos)
    flines2['Xnorm'] = matPos1[0, :]
    flines2['Ynorm'] = matPos1[1, :]
    flines2['Znorm'] = matPos1[2, :]
    
    return flines2

###############################################################################

#%% def plot

def plotAvion(p1, p2, d1, d2, pAvion, c1, c2):
    
    maxi = 20
    mini = 0
    # calcul
    v1 = calDroiteFull(p1, d1, mini, maxi)
    v2 = calDroiteFull(p2, d2, mini, maxi)
    
    #plot 
    fig1 = plt.figure()
    ax = fig1.add_subplot(projection='3d')
    ax.plot(v1[:, 0], v1[:, 1], zs=v1[:, 2], zdir='z', label='camSirta', c= 'red',)
    
    ax.plot(v2[:, 0], v2[:, 1], zs=v2[:, 2], zdir='z', label='camBretigny', c= 'orange')
    
    ax.scatter(p2[0], p2[1], zs = p2[2], zdir = 'z', c = 'pink',  label = 'bretigny')
    ax.scatter(p1[0], p1[1], zs = p1[2], zdir = 'z', c = 'aqua', label = 'sirta')
    
    #ax.plot(vn2[:, 0], vn2[:, 1], zs = vn2[:,2], zdir = 'z', c = 'magenta', label = 'n2')
    #ax.plot(vn1[:, 0], vn1[:, 1], zs = vn1[:,2], zdir = 'z', c = 'deepskyblue', label = 'n1')
    
    ax.scatter(c2[0], c2[1], zs = c2[2], zdir = 'z', c = 'darkviolet',  label = 'c2')
    ax.scatter(c1[0], c1[1], zs = c1[2], zdir = 'z', c = 'royalblue', label = 'c1')
    
    ax.scatter(pAvion[0], pAvion[1], zs = pAvion[2], zdir = 'z', c = 'black', label = 'avion')
    
    ax.legend()

###############################################################################

def plotTraj(p1, p2, pAvion1, indSirta, indBrety):
    
    #norm
    
    #plot 
    fig1 = plt.figure()
    ax = fig1.add_subplot(projection='3d')
    # ax.plot(v1[:, 0], v1[:, 1], zs=v1[:, 2], zdir='z', label='camSirta', c= 'red',)
    
    # ax.plot(v2[:, 0], v2[:, 1], zs=v2[:, 2], zdir='z', label='camBretigny', c= 'orange')
    
    ax.scatter(p2[0], p2[1], zs = p2[2], zdir = 'z', c = 'pink',  label = 'bretigny')
    ax.scatter(p1[0], p1[1], zs = p1[2], zdir = 'z', c = 'aqua', label = 'sirta')
    
    #ax.plot(vn2[:, 0], vn2[:, 1], zs = vn2[:,2], zdir = 'z', c = 'magenta', label = 'n2')
    #ax.plot(vn1[:, 0], vn1[:, 1], zs = vn1[:,2], zdir = 'z', c = 'deepskyblue', label = 'n1')
    
    # ax.scatter(c2[0], c2[1], zs = c2[2], zdir = 'z', c = 'darkviolet',  label = 'c2')
    # ax.scatter(c1[0], c1[1], zs = c1[2], zdir = 'z', c = 'royalblue', label = 'c1')
    
    ax.scatter(pAvion1[:,0], pAvion1[:, 1], zs = pAvion1[:, 2], zdir = 'z', c = indSirta, label = 'avionSirta',
               cmap = 'jet', s = indBrety/5)
    # ax.scatter(pAvion1[:,0], pAvion1[:, 1], zs = pAvion1[:, 2], zdir = 'z', c = indBrety, label = 'avionBrety',
    #             cmap = 'Reds', alpha = 0.2)
    
    ax.legend()

###############################################################################

#%% params
steps = ['test_équations', 'inversXY', 'functions', 'conv1', 'conv2', 'traj1', 'traj2', 'noLoop',
         'compTime', 'ajout error']
step = steps[6]
#%%
if __name__ == '__main__' and step == steps[0]:
    radius=6370.0
    maxi = 20
    mini = 0
    
    #p1 = sirta
    # lat_sirta, lon_sirta = readLatLon('SIRTA')
    lat_sirta = 48.713
    lon_sirta = 2.208
    alt_sirta  = 0 #valeur à corriger
    
    p1 = np.array([0., 0., 0.])
    
    #p2 = bretigny #valeur à corriger à partir de eurocontrole
    lat_brety = 48.610
    lon_brety = 2.308
    alt_brety = 0.150
    
    #projection brety on sirta in km
    X_brety = radius*np.deg2rad(lat_brety - lat_sirta)
    Y_brety = radius*np.cos(np.deg2rad((lat_sirta+lat_brety)/2.))*np.deg2rad(lon_brety-lon_sirta)
    Z_brety = alt_brety - alt_sirta
    
    #distance brety sirta ~16.5km  
    p2 = np.array([X_brety, Y_brety, Z_brety])
    dist = np.sqrt((p2*p2).sum())
    
    #d1
    #point avion EZY14DT 01/06/2019 05:42:06.99
    #XYZ norm = np.array([-0.174, -0.226, 0.959])
    lat_avion = 48.73
    lon_avion = 2.17
    alt_avion = 10.86
    
    #avion sirta
    X_avionS = radius*np.deg2rad(lat_avion - lat_sirta)
    Y_avionS = radius*np.cos(np.deg2rad((lat_avion+lat_sirta)/2.))*np.deg2rad(lon_avion-lon_sirta)
    Z_avionS = alt_avion - alt_sirta
    
    d_avionS = [X_avionS, Y_avionS, Z_avionS]
    norm1 = d_avionS/np.sqrt(X_avionS**2+Y_avionS**2+Z_avionS**2)
    d1 = np.array(norm1)
    
    v1 = calDroiteFull(p1, d1, mini, maxi)
    
    #d2 theorique

    #projection avion on brety in km
    X_avionB = radius*np.deg2rad(lat_avion - lat_brety)
    Y_avionB = radius*np.cos(np.deg2rad((lat_avion+lat_brety)/2.))*np.deg2rad(lon_avion-lon_brety)
    Z_avionB = alt_avion - alt_brety
    
    d_avionB = [X_avionB, Y_avionB, Z_avionB]
    norm2 = d_avionB/np.sqrt(X_avionB**2+Y_avionB**2+Z_avionB**2)
    # norm2 = np.array([X_avion, Y_avion, Z_avion])
    # norm2 = norm2/np.sqrt((norm2*norm2).sum())
    d2 = np.array(norm2)
    
    v2 = calDroiteFull(p2, d2, mini, maxi)
    
    #%% cal triang
    #perpendicular
    n = np.cross(d1, d2) 

    n2 = np.cross(d2, n)
    n1 = np.cross(d1, n)
    
    vn2 = calDroiteFull(p2, n2, 0, maxi)
    vn1 = calDroiteFull(p1, n1, 0, maxi)
    
    #nearest
    c1 = p1 + np.dot((p2-p1), n2)/np.dot(d1, n2) * d1
    #p1 is the nearest
    
    c2 = p2 + np.dot((p1-p2), n1)/np.dot(d2, n1) * d2
    
    #distance d
    d = np.linalg.norm(c1-c2)
    print('distance entre c1 et c2 :', d)
    
    #%% plot
    fig1 = plt.figure()
    ax = fig1.add_subplot(projection='3d')
    ax.plot(v1[:, 0], v1[:, 1], zs=v1[:, 2], zdir='z', label='camSirta', c= 'red',)
    
    ax.plot(v2[:, 0], v2[:, 1], zs=v2[:, 2], zdir='z', label='camBretigny', c= 'orange')
    
    ax.scatter(p2[0], p2[1], zs = p2[2], zdir = 'z', c = 'pink',  label = 'bretigny')
    ax.scatter(p1[0], p1[1], zs = p1[2], zdir = 'z', c = 'aqua', label = 'sirta')
    
    #ax.plot(vn2[:, 0], vn2[:, 1], zs = vn2[:,2], zdir = 'z', c = 'magenta', label = 'n2')
    #ax.plot(vn1[:, 0], vn1[:, 1], zs = vn1[:,2], zdir = 'z', c = 'deepskyblue', label = 'n1')
    
    ax.scatter(c2[0], c2[1], zs = c2[2], zdir = 'z', c = 'darkviolet',  label = 'c2')
    ax.scatter(c1[0], c1[1], zs = c1[2], zdir = 'z', c = 'royalblue', label = 'c1')
    
    ax.scatter(X_avionS, Y_avionS, zs = Z_avionS, zdir = 'z', c = 'black', label = 'avion')
    
    ax.legend()
    
#%% plot calcul droite
elif __name__ == '__main__' and step == steps[1]:
    radius=6370.0
    maxi = 20
    mini = 0
    
    #%% points
    #p1 = sirta
    # lat_sirta, lon_sirta = readLatLon('SIRTA')
    lat_sirta = 48.713
    lon_sirta = 2.208
    alt_sirta  = 0 #valeur à corriger
    
    p1 = np.array([0., 0., 0.])
    
    #p2 = bretigny #valeur à corriger à partir de eurocontrole
    lat_brety = 48.610
    lon_brety = 2.308
    alt_brety = 0.150
    
    #projection brety on sirta in km
    X_bretyAC = radius*np.deg2rad(lat_brety - lat_sirta)
    Y_bretyAC = radius*np.cos(np.deg2rad((lat_sirta+lat_brety)/2.))*np.deg2rad(lon_brety-lon_sirta)
    Z_bretyAC = alt_brety - alt_sirta
    
    X_brety = -Y_bretyAC
    Y_brety =  X_bretyAC
    Z_brety =  Z_bretyAC
    #distance brety sirta ~16.5km  
    p2 = np.array([X_brety, Y_brety, Z_brety])
    dist = np.sqrt((p2*p2).sum())
    
    #d1
    #point avion EZY14DT 01/06/2019 05:42:06.99
    #XYZ norm = np.array([-0.174, -0.226, 0.959])
    lat_avion = 48.73
    lon_avion = 2.17
    alt_avion = 10.86
    #avion sirta
    X_avionSAC = radius*np.deg2rad(lat_avion - lat_sirta)
    Y_avionSAC = radius*np.cos(np.deg2rad((lat_avion+lat_sirta)/2.))*np.deg2rad(lon_avion-lon_sirta)
    Z_avionSAC = alt_avion - alt_sirta
    
    X_avionS = -Y_avionSAC
    Y_avionS =  X_avionSAC
    Z_avionS =  Z_avionSAC
    
    #%% droite
    d_avionS = [X_avionS, Y_avionS, Z_avionS]
    norm1 = d_avionS/np.sqrt(X_avionS**2+Y_avionS**2+Z_avionS**2)
    d1 = np.array(norm1)
    
    v1 = calDroiteFull(p1, d1, mini, maxi)
    
    #d2 theorique

    #projection avion on brety in km
    X_avionBAC = radius*np.deg2rad(lat_avion - lat_brety)
    Y_avionBAC = radius*np.cos(np.deg2rad((lat_avion+lat_brety)/2.))*np.deg2rad(lon_avion-lon_brety)
    Z_avionBAC = alt_avion - alt_brety
    
    X_avionB = -Y_avionBAC
    Y_avionB = X_avionBAC
    Z_avionB = Z_avionBAC
    
    d_avionB = [X_avionB, Y_avionB, Z_avionB]
    norm2 = d_avionB/np.sqrt(X_avionB**2+Y_avionB**2+Z_avionB**2)
    # norm2 = np.array([X_avion, Y_avion, Z_avion])
    # norm2 = norm2/np.sqrt((norm2*norm2).sum())
    d2 = np.array(norm2)
    
    v2 = calDroiteFull(p2, d2, mini, maxi)
    
    #%% cal triang
    #perpendicular
    n = np.cross(d1, d2) 

    n2 = np.cross(d2, n)
    n1 = np.cross(d1, n)
    
    vn2 = calDroiteFull(p2, n2, 0, maxi)
    vn1 = calDroiteFull(p1, n1, 0, maxi)
    
    #nearest
    c1 = p1 + np.dot((p2-p1), n2)/np.dot(d1, n2) * d1
    #p1 is the nearest
    
    c2 = p2 + np.dot((p1-p2), n1)/np.dot(d2, n1) * d2
    
    #distance d
    d = np.linalg.norm(c1-c2)
    print('distance entre c1 et c2 :', d)
    
    #%% plot
    fig1 = plt.figure()
    ax = fig1.add_subplot(projection='3d')
    ax.plot(v1[:, 0], v1[:, 1], zs=v1[:, 2], zdir='z', label='camSirta', c= 'red',)
    
    ax.plot(v2[:, 0], v2[:, 1], zs=v2[:, 2], zdir='z', label='camBretigny', c= 'orange')
    
    ax.scatter(p2[0], p2[1], zs = p2[2], zdir = 'z', c = 'pink',  label = 'bretigny')
    ax.scatter(p1[0], p1[1], zs = p1[2], zdir = 'z', c = 'aqua', label = 'sirta')
    
    #ax.plot(vn2[:, 0], vn2[:, 1], zs = vn2[:,2], zdir = 'z', c = 'magenta', label = 'n2')
    #ax.plot(vn1[:, 0], vn1[:, 1], zs = vn1[:,2], zdir = 'z', c = 'deepskyblue', label = 'n1')
    
    ax.scatter(c2[0], c2[1], zs = c2[2], zdir = 'z', c = 'darkviolet',  label = 'c2')
    ax.scatter(c1[0], c1[1], zs = c1[2], zdir = 'z', c = 'royalblue', label = 'c1')
    
    ax.scatter(X_avionS, Y_avionS, zs = Z_avionS, zdir = 'z', c = 'black', label = 'avion')
    
    ax.legend()


#%% functions
elif __name__ == '__main__' and step == steps[2]:
    radius=6370.0
    maxi = 20
    mini = 0
    
    # points
    #p1 = sirta
    pSirta = createSite(site = 'SIRTA')
    
    #p2 = bretigny #valeur à corriger à partir de eurocontrole
    pBrety = createSite(site = 'BRETIGNY')
    
    #distance brety sirta ~16.5km  
    dist = np.sqrt((pBrety*pBrety).sum())
    
    #avions
    #point avion EZY14DT 01/06/2019 05:42:06.99  
    #avions = ['EZY14DT', 'EZY93JB', 'RYR8603']
    pAvionSirta = createAvion(flightname='EZY14DT', ref = 'SIRTA')
    
    #projection avion on brety in km
    pAvionBrety = createAvion(flightname='EZY14DT', ref = 'BRETIGNY')
    
    # droites
    dSirta = normVect(pAvionSirta)
    vSirta = calDroiteFull(pSirta, dSirta, mini, maxi)
    
    dBrety = normVect(pAvionBrety)
    vBrety = calDroiteFull(pBrety, dBrety, mini, maxi)
    
    # cal triang    
    cSirta, cBrety, d = calcNearest(pSirta, pBrety, dSirta, dBrety, verbose=['dist'])
    
    #%% plot
    fig1 = plt.figure()
    ax = fig1.add_subplot(projection='3d')
    ax.plot(vSirta[:, 0], vSirta[:, 1], zs=vSirta[:, 2], zdir='z', label='camSirta', c= 'red',)
    
    ax.plot(vBrety[:, 0], vBrety[:, 1], zs=vBrety[:, 2], zdir='z', label='camBretigny', c= 'orange')
    
    ax.scatter(pBrety[0], pBrety[1], zs = pBrety[2], zdir = 'z', c = 'pink',  label = 'bretigny')
    ax.scatter(pSirta[0], pSirta[1], zs = pSirta[2], zdir = 'z', c = 'aqua', label = 'sirta')
    
    #ax.plot(vn2[:, 0], vn2[:, 1], zs = vn2[:,2], zdir = 'z', c = 'magenta', label = 'n2')
    #ax.plot(vn1[:, 0], vn1[:, 1], zs = vn1[:,2], zdir = 'z', c = 'deepskyblue', label = 'n1')
    
    ax.scatter(cBrety[0], cBrety[1], zs = cBrety[2], zdir = 'z', c = 'darkviolet',  label = 'nearest to Sirta')
    ax.scatter(cSirta[0], cSirta[1], zs = cSirta[2], zdir = 'z', c = 'royalblue', label = 'nearest to Bretigny')
    
    ax.scatter(pAvionSirta[0], pAvionSirta[1], zs = pAvionSirta[2], zdir = 'z', c = 'black', label = 'avion')
    
    ax.legend()
    
#%% conv1
elif __name__ == '__main__'  and step == steps[3]:
    #creation site
    pSirta, pBrety = initSite()
    
    path = pathBuilder()
    path.setSite('SIRTA')
    path.setAuto()
    path.setDateDay(dt.datetime(2019, 6, 1, 5, 42))
    #creation avion traj
    avionDF = extractAvionTraj(flightname='EZY14DT', path=path)
    avionSirta = avionProjSite(avionDF=avionDF, site='SIRTA')
    avionBrety = avionProjSite(avionDF=avionDF, site='BRETIGNY')
    
    #calcul point 1
    indice = 37465
    # dSirta = avionSirta.loc[avionSirta.index.max(), ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta = avionSirta.loc[indice, ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta = np.array(dSirta, dtype = np.float64)
    # dBrety = avionBrety.loc[avionBrety.index.max(), ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety = avionBrety.loc[indice, ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety = np.array(dBrety, dtype = np.float64)
    cSirta, cBrety, d1 = calcNearest(pSirta, pBrety, dSirta, dBrety, verbose=['dist'])
    
    #%% plot
    pAvion = avionSirta.loc[indice, ['X', 'Y', 'altkm']].values
    plotAvion(pSirta, pBrety, dSirta, dBrety, pAvion, cSirta, cBrety)
    
    #%% calcul point 2
    indice2 = 37512
    # dSirta = avionSirta.loc[avionSirta.index.max(), ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta2 = avionSirta.loc[indice2, ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta2 = np.array(dSirta2, dtype = np.float64)
    # dBrety = avionBrety.loc[avionBrety.index.max(), ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety2 = avionBrety.loc[indice2, ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety2 = np.array(dBrety2, dtype = np.float64)
    cSirta2, cBrety2, d2 = calcNearest(pSirta, pBrety, dSirta2, dBrety2)
    
    #%% plot 2
    pAvion2 = avionSirta.loc[indice2, ['X', 'Y', 'altkm']].values
    plotAvion(pSirta, pBrety, dSirta2, dBrety2, pAvion2, cSirta2, cBrety2)
 
#%% conv2
elif __name__ == '__main__' and step == steps[4]:
    #creation site
    pSirta, pBrety = initSite(conv = 'conv2')
    
    path = pathBuilder()
    path.setSite('SIRTA')
    path.setAuto()
    path.setDateDay(dt.datetime(2019, 6, 1, 5, 42))
    #creation avion traj
    avionDF = extractAvionTraj(flightname='EZY14DT', path=path)
    avionSirta = avionProjSite(avionDF=avionDF, site='SIRTA', conv='conv2')
    avionBrety = avionProjSite(avionDF=avionDF, site='BRETIGNY', conv='conv2')
    
    #calcul point 1
    indice = 37465
    # dSirta = avionSirta.loc[avionSirta.index.max(), ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta = avionSirta.loc[indice, ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta = np.array(dSirta, dtype = np.float64)
    # dBrety = avionBrety.loc[avionBrety.index.max(), ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety = avionBrety.loc[indice, ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety = np.array(dBrety, dtype = np.float64)
    cSirta, cBrety, d1 = calcNearest(pSirta, pBrety, dSirta, dBrety, verbose=['dist'])
    
    #%% plot
    pAvion = avionSirta.loc[indice, ['X', 'Y', 'altkm']].values
    plotAvion(pSirta, pBrety, dSirta, dBrety, pAvion, cSirta, cBrety)
    
    #%% calcul point 2
    indice2 = 37512
    # dSirta = avionSirta.loc[avionSirta.index.max(), ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta2 = avionSirta.loc[indice2, ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta2 = np.array(dSirta2, dtype = np.float64)
    # dBrety = avionBrety.loc[avionBrety.index.max(), ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety2 = avionBrety.loc[indice2, ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety2 = np.array(dBrety2, dtype = np.float64)
    cSirta2, cBrety2, d2 = calcNearest(pSirta, pBrety, dSirta2, dBrety2, verbose=['dist'])
    
    #%% plot 2
    pAvion2 = avionSirta.loc[indice2, ['X', 'Y', 'altkm']].values
    plotAvion(pSirta, pBrety, dSirta2, dBrety2, pAvion2, cSirta2, cBrety2)
    
    
#%% trajectory sirta point brety
elif __name__ == '__main__' and step == steps[5]:
    #creation site
    pSirta, pBrety = initSite(conv = 'conv2')
    
    path = pathBuilder()
    path.setSite('SIRTA')
    path.setAuto()
    path.setDateDay(dt.datetime(2019, 6, 1, 5, 42))
    #creation avion traj
    avionDF = extractAvionTraj(flightname='EZY14DT', path=path)
    avionSirta = avionProjSite(avionDF=avionDF, site='SIRTA', conv='conv2')
    avionBrety = avionProjSite(avionDF=avionDF, site='BRETIGNY', conv='conv2')
    
    #calcul point 1
    indice = 37465
    # dSirta = avionSirta.loc[avionSirta.index.max(), ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta = avionSirta.loc[indice, ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta = np.array(dSirta, dtype = np.float64)
    # dBrety = avionBrety.loc[avionBrety.index.max(), ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety = avionBrety.loc[indice, ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety = np.array(dBrety, dtype = np.float64)
    cSirta, cBrety, d1 = calcNearest(pSirta, pBrety, dSirta, dBrety, verbose=['dist'])
    
    #%% plot
    pAvion = avionSirta.loc[indice, ['X', 'Y', 'altkm']].values
    plotAvion(pSirta, pBrety, dSirta, dBrety, pAvion, cSirta, cBrety)
    
    #%% calcul point 2
    indice2 = 37512
    # dSirta = avionSirta.loc[avionSirta.index.max(), ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta2 = avionSirta.loc[indice2, ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta2 = np.array(dSirta2, dtype = np.float64)
    # dBrety = avionBrety.loc[avionBrety.index.max(), ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety2 = avionBrety.loc[indice2, ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety2 = np.array(dBrety2, dtype = np.float64)
    cSirta2, cBrety2, d2 = calcNearest(dSirta2, dBrety2, pSirta, pBrety, verbose=['dist'])
    
    #%% plot 2
    pAvion2 = avionSirta.loc[indice2, ['X', 'Y', 'altkm']].values
    plotAvion(pSirta, pBrety, dSirta2, dBrety2, pAvion2, cSirta2, cBrety2)
    
    #%% calcul droite 1
    pAvion1 = pAvion
    pAvion2 = pAvion2
    dBrety2 = dBrety
    
    lineAvionX = np.linspace(pAvion1[0], pAvion2[0], 10)
    lineAvionY = np.linspace(pAvion1[1], pAvion2[1], 10)
    lineAvionZ = np.linspace(pAvion1[2], pAvion2[2], 10)
    
    lineSirta  = np.array([lineAvionX, lineAvionY, lineAvionZ])
    
    #comparaison des distances d
    # d = []
    # for i in range(lineSirta.shape[1]):
    #     pAvionSirta = lineSirta[:, i]
    #     dSirtai = normVect(pAvionSirta)
    #     cSirtai, cBretyi, di = calcNearest(dSirtai, dBrety2, pSirta, pBrety)
    #     d.append(di)       
    # indMin = np.where(d == min(d))[0]
    # indMin = calcLinePoint(pSirta, pBrety, lineSirta, dBrety2)
    # pAvionSirtaMin = lineSirta[:, indMin].reshape([3,])
    # dSirtaMin = normVect(pAvionSirtaMin)
    # cSirtaMin, cBretyMin, dMin = calcNearest(dSirtaMin, dBrety2, pSirta, pBrety, verbose = ['dist'])
    cSirtaMin, cBretyMin, dSirtaMin, pAvionSirtaMin, dMin = calcNearestLP(pSirta, pBrety, lineSirta, dBrety2, verbose = ['ind', 'dist'])
    
    #%% plot droite 1
    plotAvion(pSirta, pBrety, dSirtaMin, dBrety2, pAvionSirtaMin, cSirtaMin, cBretyMin)
    
#%% trajectory sirta trajectory brety
elif __name__ == '__main__' and step == steps[6]:
    pSirta, pBrety = initSite(conv = 'conv2')
    
    path = pathBuilder()
    path.setSite('SIRTA')
    path.setAuto()
    path.setDateDay(dt.datetime(2019, 6, 1, 5, 42))
    #creation avion traj
    avionDF = extractAvionTraj(flightname='EZY14DT', path=path)
    avionSirta = avionProjSite(avionDF=avionDF, site='SIRTA', conv='conv2')
    avionBrety = avionProjSite(avionDF=avionDF, site='BRETIGNY', conv='conv2')
    
    #%%calcul point 1
    indice1 = 37465
    # dSirta = avionSirta.loc[avionSirta.index.max(), ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta = avionSirta.loc[indice1, ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta = np.array(dSirta, dtype = np.float64)
    # dBrety = avionBrety.loc[avionBrety.index.max(), ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety = avionBrety.loc[indice1, ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety = np.array(dBrety, dtype = np.float64)
    cSirta, cBrety, d1 = calcNearest(pSirta, pBrety, dSirta, dBrety, verbose=['dist'])
    
    # calcul point 2
    indice2 = 37512
    # dSirta = avionSirta.loc[avionSirta.index.max(), ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta2 = avionSirta.loc[indice2, ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta2 = np.array(dSirta2, dtype = np.float64)
    # dBrety = avionBrety.loc[avionBrety.index.max(), ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety2 = avionBrety.loc[indice2, ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety2 = np.array(dBrety2, dtype = np.float64)
    cSirta2, cBrety2, d2 = calcNearest(dSirta2, dBrety2, pSirta, pBrety, verbose=['dist'])
    
    #%% Generation line
    lineSirta = genDroite(avionSirta, indice1, indice2)
    
    lineBrety = genDroite(avionBrety, indice1, indice2)
    
    #%% calcul
    # distBrety = []
    # for i in range(lineBrety.shape[1]):
    #     dBretyi = normVect(lineBrety[:, i])
    #     cSirtaMin, cBretyMin, dSirtaMin, pAvionSirtaMin, dMin = calcNearestLP(pSirta, pBrety, lineSirta, dBretyi, verbose = ['ind', 'dist'])
    #     distBrety.append(dMin)
    
    # indMin2 = np.where(distBrety == np.min(distBrety))[0]
    # # dBretyMin = lineBrety[:, indMin2]
    # pAvion2Min = lineBrety[:, indMin2].reshape([3,])
    # d2Min = normVect(pAvion2Min)
    # cSirtaMin2, cBretyMin2, dSirtaMin2, pAvionSirtaMin2, dMin12 = calcNearestLP(pSirta, pBrety, lineSirta, d2Min, verbose = ['ind', 'dist'])
    
    cSirtaMin, cBretyMin, dSirtaMin, dBretyMin, pAvionSirtaMin, dMin1 = calcNearestLL(pSirta, pBrety, lineSirta, lineBrety, verbose = ['ind', 'dist'])
    
    #%% plot
    plotAvion(pSirta, pBrety, dSirtaMin, dBretyMin, pAvionSirtaMin, cSirtaMin, cBretyMin)
    
#%% same without loop
elif __name__ == '__main__' and step == steps[7]:
    pSirta, pBrety = initSite(conv = 'conv2')
    
    path = pathBuilder()
    path.setSite('SIRTA')
    path.setAuto()
    path.setDateDay(dt.datetime(2019, 6, 1, 5, 42))
    #creation avion traj
    avionDF = extractAvionTraj(flightname='EZY14DT', path=path)
    avionSirta = avionProjSite(avionDF=avionDF, site='SIRTA', conv='conv2')
    avionBrety = avionProjSite(avionDF=avionDF, site='BRETIGNY', conv='conv2')
    
    #%% point by point
    #calcul point 1
    indice1 = 37465
    # dSirta = avionSirta.loc[avionSirta.index.max(), ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta = avionSirta.loc[indice1, ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta = np.array(dSirta, dtype = np.float64)
    # dBrety = avionBrety.loc[avionBrety.index.max(), ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety = avionBrety.loc[indice1, ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety = np.array(dBrety, dtype = np.float64)
    cSirta, cBrety, d1 = calcNearest(pSirta, pBrety, dSirta, dBrety, verbose=['dist'])
    
    # calcul point 2
    indice2 = 37512
    # dSirta = avionSirta.loc[avionSirta.index.max(), ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta2 = avionSirta.loc[indice2, ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta2 = np.array(dSirta2, dtype = np.float64)
    # dBrety = avionBrety.loc[avionBrety.index.max(), ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety2 = avionBrety.loc[indice2, ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety2 = np.array(dBrety2, dtype = np.float64)
    cSirta2, cBrety2, d2 = calcNearest(dSirta2, dBrety2, pSirta, pBrety, verbose=['dist'])    
    
    #%% vector point to point
    dSirtas = np.array([dSirta, dSirta2])
    dBretys = np.array([dBrety, dBrety2])  
    pAvionSirta = np.array([avionSirta.loc[indice1, ['X', 'Y', 'Z']].values,
                            avionSirta.loc[indice2, ['X', 'Y', 'Z']].values])     
    
    cSirtas, cBretys, ds = calcNearests(pSirta, pBrety, dSirtas, dBretys)
    
    #%% plot
    for i in range(dSirtas.shape[0]):
        plotAvion(pSirta, pBrety, dSirtas[i,:], dBretys[i,:], pAvionSirta[i,:], cSirtas[i,:], cBretys[i,:])
        
    #%% calcul line à line avec output list d'indice.
    indSirta, indBrety = calcLink09(pSirta, pBrety, dSirtas, dBretys)
    #%% vector + loop point to line
    indSirta, indBrety = calcNearestsLP(pSirta, pBrety, dSirtas, dBretys, loop = True)
    
    #%% matrice point to line.
    dBretys2 = np.zeros([4, 3])
    dBretys2[:2, :] = dBretys
    dBretys2[2:, :] = dBretys
    dSirtas2 = np.zeros([5, 3])
    dSirtas2[:2, :] = dSirtas
    dSirtas2[2:4, :] = dSirtas
    dSirtas2[4, :] = dSirtas[0, :]
    indSirta2, indBrety2 = calcNearestsLP(pSirta, pBrety, dSirtas2, dBretys2, loop = False)
        
#%% comparaison temps 
elif __name__ == '__main__' and step == steps[8]:
    pSirta, pBrety = initSite(conv = 'conv2')
    
    path = pathBuilder()
    path.setSite('SIRTA')
    path.setAuto()
    path.setDateDay(dt.datetime(2019, 6, 1, 5, 42))
    #creation avion traj
    avionDF = extractAvionTraj(flightname='EZY14DT', path=path)
    avionSirta = avionProjSite(avionDF=avionDF, site='SIRTA', conv='conv2')
    avionBrety = avionProjSite(avionDF=avionDF, site='BRETIGNY', conv='conv2')
    
    #%% all plane
    dSirta = avionSirta.loc[:, ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta = np.array(dSirta, dtype = np.float64)
    dBrety = avionBrety.loc[:, ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety = np.array(dBrety, dtype = np.float64)
    # cSirta, cBrety, d1 = calcNearest(pSirta, pBrety, dSirta, dBrety, verbose=['dist'])
    
    #%% ref calcLink09 on full plane
    indSirta, indBrety = calcLink09(pSirta, pBrety, dSirta, dBrety)
    #%%
    debutLink09 = time.time()
    indSirta, indBrety = calcLink09(pSirta, pBrety, dSirta, dBrety)
    print('fin calcLink09in : %.4f'%(time.time()-debutLink09))
    
    debutsLP = time.time()
    indSirta2, indBrety2 = calcNearestsLP(pSirta, pBrety, dSirta, dBrety, loop = True)
    print('fin Nearests LP Loop in : %.4f'%(time.time()-debutsLP))
    
    debutNoLoop = time.time()
    indSirta3, indBrety3 = calcNearestsLP(pSirta, pBrety, dSirta, dBrety, loop = False)
    print('fin Nearests LP noLoop in : %.4f'%(time.time()-debutNoLoop))
    
#%% ajout d'erreur pour estimé la précision.
elif __name__ == '__main__' and step == steps[9]:
    #%%
    #site creation
    pSirta, pBrety = initSite(conv = 'conv2')
    
    path = pathBuilder()
    path.setSite('SIRTA')
    path.setAuto()
    path.setDateDay(dt.datetime(2019, 6, 1, 5, 42))
    #creation avion traj
    avionDF = extractAvionTraj(flightname='EZY14DT', path=path)
    avionSirta = avionProjSite(avionDF=avionDF, site='SIRTA', conv='conv2')
    avionBrety = avionProjSite(avionDF=avionDF, site='BRETIGNY', conv='conv2')
    # all plane
    dSirta = avionSirta.loc[:, ['Xnorm', 'Ynorm', 'Znorm']].values
    dSirta = np.array(dSirta, dtype = np.float64)
    dBrety = avionBrety.loc[:, ['Xnorm', 'Ynorm', 'Znorm']].values
    dBrety = np.array(dBrety, dtype = np.float64)
    #ref
    indSirtaRef, indBretyRef = calcNearestsLP(pSirta, pBrety, dSirta, dBrety, loop = True)
    #conv array
    indSirtaRef = np.array(indSirtaRef)
    indBretyRef = np.array(indBretyRef)
    
    #%% deformation systématique coord 0 = 0.5
    dSirta2 = cp.deepcopy(dSirta)
    dSirta2[:,2] = dSirta2[:,2] + 0.1
    indSirtaRef2, indBretyRef2 = calcNearestsLP(pSirta, pBrety, dSirta2, dBrety, loop = True)
    indSirtaRef2 = np.array(indSirtaRef2)
    indBretyRef2 = np.array(indBretyRef2)
    
    #%% plot
    pAvion1 = avionSirta.loc[:, ['X', 'Y', 'Z']].values
    plotTraj(pSirta, pBrety, pAvion1, indSirtaRef2, indBretyRef2)
    
    
    