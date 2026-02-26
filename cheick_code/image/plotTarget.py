#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:18:52 2024

@author: ngourgue
"""

import numpy as np
import os 

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
#%% path

pathFolder = '/home/ngourgue/Documents/roboflow/'

listFiles = os.listdir(pathFolder)

dateStr = '20190703_0930'

fileDate = False
for file in listFiles:
    if dateStr in file:
        fileDate = file
        
if not fileDate == False:
    fileFull = os.path.join(pathFolder, fileDate)
        
#%% open mat

image = np.load(fileFull)

#%% config plot
classes=['Sky', 'contrail_old', 'contrail_very_old', 'contrail_young', 'parasite', 'sun', 'unkown', 'black']
colors = ['blue', 'red', 'orange', 'darkviolet', 'aqua', 'magenta', 'turquoise', 'black']
cmap = LinearSegmentedColormap.from_list('roboflow', colors, N = len(colors))


#%% plot semantic
fig1 = plt.figure()

axes1 = fig1.add_subplot(1, 1, 1)
im = axes1.imshow(image, cmap = cmap)

cbar = fig1.colorbar(im, ax = axes1,  drawedges = True,
              boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])

cbar.set_ticks(ticks = np.arange(0, len(colors)), labels = classes)

axes1.set_title('Target with semantic segmentation')

#%% plot instance
fig2 = plt.figure()

axes2 = fig2.add_subplot(1, 1, 1)
im = axes.im