#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:15:12 2024

@author: ngourgue

Projection brétigny.
"""

#%% importation 
import os
import copy as cp
import numpy as np
import datetime as dt

from matplotlib import pyplot as plt

#%% custom importation
path_here = os.getcwd()
path = "/data/common/STEREOSTUDYIPSL/Datasets/gQg5IUvV/srf02_0a_skyimgLz2_v01_20250406_044600_853"
image_name = "20250406050000_01.jpg"
path_here = os.path.join(path, image_name)

from image import readimage, pre_processing
from setup_variable import pathBuilder
from setup_variable.position import readLatLon
from calibration import worldToImage, convGeoCarto, reversZoom, projCoord, projCoord2

#%% function
def noName():
    return None

#%% image SIRTA

year = 2023#2023
month = 3#3
day = 20#20

hour = 10#10
minute = 46#46

xmax = 901
szamax = 60

path = pathBuilder()
path.setAuto()
path.setSite('SIRTA')
path.setDateDay(dt.datetime(year, month, day, hour, minute))

imageName = path.getPathImage().split('/')[-1][:14]
DirectoryImage = '/'.join(path.getPathImage().split('/')[:-1])
found, image = readimage(DirectoryImage = DirectoryImage, imageDate = imageName, 
                      imtype = '01', site = 'SIRTA', cle = False, verbose = [''])

image1 = cp.deepcopy(image)
image1[300, 400, :] = [0, 255, 0]

imZoom     = pre_processing(image = image1, 
                            processing = [{'name' : 'zoom', 'xmax' : xmax, 'szamax' : szamax}], 
                            path = path, cle = False, verbose = [''])

#%% test zoom and reverse zoom
ix = 600
iy = 400
ixz, iyz = projCoord(ix, iy, image, szamax = 60, xmax = 901, zoom = True, imShow = True)
ixNew, iyNew = projCoord(ixz, iyz, image, szamax = 60, xmax= 901, zoom = False, imShow = True)

ix2 = 400
iy2 = 600
ixz2, iyz2 = projCoord(ix2, iy2, image, szamax = 60, xmax = 901, zoom = True, imShow = False)
ixNew2, iyNew2 = projCoord(ixz2, iyz2, image, szamax = 60, xmax = 901, zoom = False, imShow = False)

ix3 = 600
iy3 = 400
ixz3, iyz3 = projCoord2(ix3, iy3, szamax = 60, xmax = 901, zoom = True, imShow = True)
ixNew3, iyNew3 = projCoord2(ixz3, iyz3, szamax = 60, xmax = 901, zoom = False, imShow = True)

ix4 = 400
iy4 = 600
ixz4, iyz4 = projCoord2(ix4, iy4, szamax = 60, xmax = 901, zoom = True, imShow = False)
ixNew4, iyNew4 = projCoord2(ixz4, iyz4, szamax = 60, xmax = 901, zoom = False, imShow = False)
        
#%% point bretigny  
lat_brety = 48.6 #°
lon_brety = 2.3  #°
alt_avion = 10   #km
radius=6370.0    #radius earth km

lat_site, lon_site = readLatLon('SIRTA')
y_brety = radius*(lat_brety-lat_site)/180.*np.pi
x_brety = radius*np.cos((lat_site+lat_brety)/2./180.*np.pi)*(lon_brety-lon_site)/180.*np.pi

matPos = [-y_brety  /np.sqrt(x_brety**2+y_brety**2+alt_avion**2), 
           x_brety  /np.sqrt(x_brety**2+y_brety**2+alt_avion**2), 
           alt_avion/np.sqrt(x_brety**2+y_brety**2+alt_avion**2)]

matPos1 = np.array(matPos)

ixs, iys = worldToImage(XPosition = matPos1, imageShape = image.shape, methodRead = "csv", site = 'SIRTA', zoom = False)
ixs = ixs-68
iys = iys-182
xR, yR= reversZoom(ixs, iys, szamax= 60, xmax =901, xRGBc = 339, yRGBc= 339)

ixsZoom, iysZoom = worldToImage(XPosition = matPos1, imageShape = imZoom.shape, methodRead = "csv", site = 'SIRTA', zoom = True)

#%% plot
fig1 = plt.figure()
axes1 = fig1.add_subplot(1, 1, 1)
axes1.imshow(image1[:,:, :])
ixs, iys = projCoord(ixsZoom, iysZoom, image, szamax = 60, xmax= 901, zoom = False, imShow = True)
axes1.scatter(ixs , iys , c= 'red', s =2)

fig2 = plt.figure()
axes2 = fig2.add_subplot(1, 1, 1)
axes2.imshow(imZoom)
axes2.scatter(ixsZoom, iysZoom, c= 'red', s =2)
