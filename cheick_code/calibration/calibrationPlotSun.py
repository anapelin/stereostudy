#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 20:18:23 2023

@author: ngourgue
"""

#%% importation
import numpy  as np

import os, sys, random, copy
import datetime as dt

from scipy.optimize import minimize, basinhopping
from matplotlib import pyplot as plt

if not "/home/ngourgue/climavion/detection_contrail/calibration" in sys.path:
    sys.path.append("/home/ngourgue/climavion/detection_contrail/calibration")
from calibrationM1     import (epsilonfO, saveCalParams, epsilonfNorm, droiteError, 
                               mixteSunPlane, plotParam, convBeta2Param, convPara2Beta)#, epsilonfC123, f)
from calibrationFripon import (epsiFrip, epsiFripNorm, droiteErrorFrip, mixteFripon, 
                               saveFripParams, plotFrip, convBeta2Frip, convFrip2Beta, invModel)
from baseCalibration   import (readCalParams, normaliseBeta, 
                               UnnormaliseBeta, loadContrailData, loadSunData, separateData)#, psiSun

from useCalibration    import (printError, worldToImage) 

if not "/home/ngourgue/climavion/detection_contrail" in sys.path:
    sys.path.append("/home/ngourgue/climavion/detection_contrail")
from image             import imagetime, pre_processing, readimage

from setup_variable    import pathBuilder



#%% load data
random.seed(5)
XvSoleil, xSun, indic   = loadSunData(2019, "sun_zoonFull", part = 'random')
XvSoleilAuto, xSunAuto  = loadSunData(2019, 'sun')

XvSoleilFit, XvSoleilTest, xSunFit, xSunTest = separateData(XvSoleil, xSun, indic)

XvSoleilMean, xSunMean, indice = loadSunData(2019, "sun_zoonMean", part='random')
XvSoleilMFit, XvSoleilMTest, xSunMFit, xSunMTest = separateData(XvSoleilMean, xSunMean, indice)

XvAvion, xPlane, indices = loadContrailData(2019, 'contrails_newAlt', part ='random')
XvAvionFit, XvAvionTest, xPlaneFit, xPlaneTest = separateData (XvAvion, xPlane, indices)

#%%
## x:435.654 y:326.377 dt: 20/05/2019 15:30 theta: 0.917 phi: -1.351
#2019-06-28 10:40, 2019-04-20 05:32:00, 2019-06-01 12:32, 2019-06-01 15:04, 2019-04-20 11:12,
xSunPlot = xSunMTest[:, :5]
xSoleilPlot = XvSoleilMTest[:, :5]
xDatePlot = [dt.datetime(2019,  1, 28,  8, 50),
             dt.datetime(2019,  1, 28, 10, 26),
             dt.datetime(2019,  2, 10, 12, 56),
             dt.datetime(2019,  2, 10, 13, 28),
             dt.datetime(2019,  2, 20, 10, 16)]

site = 'SIRTA'

#before : 
beta = np.array([222,0,0,0,0, 383,513, 0.055,0,0, 0,0 ], dtype = np.float128)
b, x0, y0, theta, K1, phi = convBeta2Frip(beta)
#after
# b, x0, y0, theta, K1, phi = readCalParams(method = 'csv', site = site)

x, y =    invModel(xSoleilPlot,b, x0, y0, theta, K1, phi, site)




#%% read image 
path = pathBuilder()
path.setAuto()
path.setSite(site)
path.setImageFolder('/homedata/ngourgue/Images/')
path.setCompFolder('/homedata/ngourgue/COMP')
images = []
for timestamp in xDatePlot:
    DirectoryImage = DirectoryImage =  path.getPathFolder(timestamp, 'day')
    imageDateInput =  "%04d%02d%02d%02d%02d%02d"%(timestamp.year, timestamp.month,  timestamp.day,
                                              timestamp.hour, timestamp.minute, timestamp.second)
    found, imRGBori = readimage(DirectoryImage, imageDateInput, imtype = '03', 
                                site = site, cle = False, verbose = [''])
    images.append(imRGBori)

#%% plot
for i in range(len(images)):
    figi = plt.figure(num = i)
    
    axesi = figi.add_subplot(1, 1, 1)
    axesi.imshow(images[i])
    axesi.scatter(1024-int(xSunPlot[1, i]), int(xSunPlot[0, i]), c = 'red',   s= 50, label= 'annotation')
    axesi.scatter(1024-int(y[i]),    int(x[i]),    c = 'green', s= 50, label= 'projection')
    figi.legend()


