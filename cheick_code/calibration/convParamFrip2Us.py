#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:04:47 2022

@author: ngourgue
"""

import os, sys

import pandas as pd
import numpy as np
import copy as cp

from scipy.optimize import curve_fit

from matplotlib import pyplot as plt

if not '/home/ngourgue/climavion/detection_contrail' in sys.path:
    sys.path.append('/home/ngourgue/climavion/detection_contrail')
from calibration import initR

#%% extract site function
def convInputToInt(inputValue, indexSite):
    try :
        indexFileSite = int(inputValue)
    except:
        indexFileSite = input('Select index of line to extrat (enter int number into'+str(indexSite.values)+') :')
        indexFileSite = convInputToInt(indexFileSite, indexSite)
        if indexFileSite in indexSite.values:
            pass
        else:
            indexFileSite = convInputToInt(inputValue, indexSite)
    return indexFileSite

def inputIndexSite(fileSite):
    print(fileSite[['Cityx', 'Date startx', 'Date endx']])
    indexFileSite = input('Select index of line to extrat :')
    indexFileSite = convInputToInt(indexFileSite, fileSite.index)
    
    fileSite = fileSite.loc[indexFileSite]
    return fileSite

def deletePixel(string):
    value = float(string[:string.find('pixel')])
    return value

def deleteDegre(string):
    value = float(string[:string.find('°')])
    return value

def poly9bis(x, a, b, c, d, e):
    yCal =  a*x + b*(x**3) + c*(x**5) + d*(x**7) + e*(x**9)
    return yCal

#%% execution
if __name__ == '__main':
    #%% open file
    fileParamFripon = '/home/ngourgue/climavion/detection_contrail/calibration/Astrometry results.csv'
    
    if os.path.isfile(fileParamFripon):
        fileFripon = pd.read_csv(fileParamFripon)
    else:
        raise ValueError('FileParamFripon not exist or not file.')
        
    invRot = False
    
    #%% select Site
    list_sites = fileFripon['Cityx']
    site = input('Select site into list :'+str(list_sites.values))
    
    
    #%% extract site
    fileSite = fileFripon.loc[fileFripon['Cityx']==site]
    
    if type(fileSite) == pd.core.frame.DataFrame:
        fileSite = inputIndexSite(fileSite)
    
    elif type(fileSite) == pd.core.series.Series:
        pass
    else :
        raise ValueError('type fileSite is unknow. Type :'+str(type(fileSite)))
        
    #%% mise en forme
    #extract information
    fileSiteForUs = fileSite[['Cityx', 'x0x', 'y0x', 'Vx', 'Sx', 'Dx', 'Px', 'Qx', 'Theta Xx', 'Theta Yx', 'Theta Zx', 'Anix', 'Phix']]
    
    #create dataFrame
    columns = fileSiteForUs.index.values
    dfSite = pd.DataFrame(columns= columns)
    #create row
    dfSite.loc[0] = 0
    
    #cityx
    dfSite.loc[0, 'Cityx'] = fileSiteForUs['Cityx']
    dfSite.set_index('Cityx', inplace=True)
    
    #Inversion x and y because python matrix has x in vectical and y in horizontal
    
    #y0x
    xo = fileSiteForUs['x0x']
    xo = deletePixel(xo)
    dfSite.loc[site, 'x0x'] = xo
    dfSite.rename(columns = {'x0x' : 'xo'}, inplace= True)
    #x0x
    yo = fileSiteForUs['y0x']
    yo = deletePixel(yo)
    dfSite.loc[site, 'y0x'] = yo
    dfSite.rename(columns = {'y0x' : 'yo'}, inplace= True)
    
    #Anix
    K1 = fileSiteForUs['Anix']
    dfSite.loc[site, 'Anix'] = K1
    dfSite.rename(columns = {'Anix' : 'K1'}, inplace= True)
    
    #Angle
    
    #extract value
    wx = fileSiteForUs['Theta Xx']
    wy = fileSiteForUs['Theta Yx']
    wz = fileSiteForUs['Theta Zx']
    phi = fileSiteForUs['Phix']
    
    #conv to float
    wx = deleteDegre(wx)
    wy = deleteDegre(wy)
    wz = deleteDegre(wz)
    phi = deleteDegre(phi)
    
    # #conv to radian
    # wx = np.deg2rad(wx)
    # wy = np.deg2rad(wy)
    # wz = np.deg2rad(wz)
    # phi = np.deg2rad(phi)
    
    if invRot == True:
        w = np.array([wx, wy, wz])
        matRot = initR(w)
        matRotInv = np.linalg.inv(matRot)
        wx = np.arctan2(matRotInv[2,1], matRotInv[2,2])
        wy = np.arctan2(-matRotInv[2,0], np.sqrt(matRotInv[2, 1]**2+matRotInv[2,2]**2))
        wz = np.arctan2(matRotInv[2,0], matRotInv[0, 0])
    
    #put in df
    dfSite.loc[site, 'Theta Xx'] = wx
    dfSite.loc[site, 'Theta Yx'] = wy
    dfSite.loc[site, 'Theta Zx'] = wz
    dfSite.loc[site, 'Phix'] = phi
    
    dfSite.rename(columns = {'Theta Xx' : 'wx', 'Theta Yx' : 'wy', 'Theta Zx' : 'wz', 'Phix': 'phi'}, inplace= True)
    
    #%% inversion polynome
    #Inversion V, S, D, P, Q because we want to projet real world in image
    a = fileSiteForUs['Vx']
    b = fileSiteForUs['Sx']
    c = fileSiteForUs['Dx']
    d = fileSiteForUs['Px']
    e = fileSiteForUs['Qx']
    
    x = np.linspace(0, 0.45, 1000)
    y = poly9bis(x, a, b, c, d, e)
    invP1, pcov = curve_fit(poly9bis, y, x)
    
    #verification graphique
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Inversion polynome')
    ax1.set_xlabel('elevation')
    ax1.set_ylabel('Rayon')
    ax1.plot(y, x, 'b-', label= 'original curve')
    ax1.plot(y, poly9bis(y, *invP1), 'g-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, e=%5.3f' %(
        invP1[0], invP1[1], invP1[2], invP1[3], invP1[4]))
    ax1.legend()
    fig.show()
    
    dfSite.loc[site, 'Vx'] = invP1[0]
    dfSite.loc[site, 'Sx'] = invP1[1]
    dfSite.loc[site, 'Dx'] = invP1[2]
    dfSite.loc[site, 'Px'] = invP1[3]
    dfSite.loc[site, 'Qx'] = invP1[4]
    
    dfSite.rename(columns = {'Vx' : 'a1', 'Sx' : 'a2', 'Dx' : 'a3', 'Px' : 'a4', 'Qx' : 'a5'}, inplace=True)
    
    #%% lat lon
    if site == 'Orsay':
        lat = 48.706433
        lon =  2.179331
    else:
        print("site unknow. Site know : ['Orsay']. site :", site)
        
    dfSite.loc[site, 'lat'] = lat
    dfSite.loc[site, 'lon'] = lon
    
    #%% open the old value
    paramsFileName = '/home/ngourgue/climavion/params.csv'
    paramsFile = pd.read_csv(paramsFileName, index_col=0)
    
    if site in paramsFile.index.values:
        print('site existe we ecrase value')
        paramsFile.loc[site] = dfSite.loc[site]
    else:
        print('site not exist in params. concat new site')
        paramsFile = pd.concat([paramsFile, dfSite])
        
    #%% save new Params
    paramsFile.to_csv(paramsFileName)
    print(site, ' params save to ', paramsFileName)
    
#%% inv model
if __name__ == '__main__':
    fileParamModel = '/home/ngourgue/climavion/params.csv'
    site = 'SIRTA'
    
    if os.path.isfile(fileParamModel):
        fileModel = pd.read_csv(fileParamModel, index_col='site')
    else:
        raise ValueError('FileParamFripon not exist or not file.')
        
    fileSite = fileModel.loc[site]
    fileSite2 = cp.deepcopy(fileSite)
    #cityx
    # fileSite.set_index('site', inplace=True)
    #%% inversion polynome
    #Inversion V, S, D, P, Q because we want to projet real world in image
    a = fileSite['a1']
    b = fileSite['a2']
    c = fileSite['a3']
    d = fileSite['a4']
    e = fileSite['a5']
    
    x = np.linspace(0, 0.45, 1000)
    y = poly9bis(x, a, b, c, d, e)
    invP1, pcov = curve_fit(poly9bis, y, x)
    
    #verification graphique
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Inversion polynome')
    ax1.set_xlabel('elevation')
    ax1.set_ylabel('Rayon')
    ax1.plot(y, x, 'b-', label= 'original curve')
    ax1.plot(y, poly9bis(y, *invP1), 'g-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, e=%5.3f' %(
        invP1[0], invP1[1], invP1[2], invP1[3], invP1[4]))
    ax1.legend()
    fig.show()
    
    fileSite2.loc['a1'] = invP1[0]
    fileSite2.loc['a2'] = invP1[1]
    fileSite2.loc['a3'] = invP1[2]
    fileSite2.loc['a4'] = invP1[3]
    fileSite2.loc['a5'] = invP1[4]
    
    #%%
    # dfSite.rename(columns = {'Vx' : 'a1', 'Sx' : 'a2', 'Dx' : 'a3', 'Px' : 'a4', 'Qx' : 'a5'}, inplace=True)*
    fileSite2.name = 'SIRTA_W'
    fileModel.loc['SIRTA_W', ] = fileSite2
        
    fileModel.to_csv(fileParamModel)
    print(site, ' params save to ', fileParamModel)
        
    
    
