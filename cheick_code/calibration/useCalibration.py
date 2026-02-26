#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:03:19 2022

@author: ngourgue

file to use calibration
"""
import numpy as np

from .baseCalibration import readCalParams
from .calibrationM1 import world2image as W2IM1
from .calibrationFripon import world2image as W2IM2
from .calibrationFripon import image2world
from .FriponModel import Fripon
from .OriginalModel import OrigineModel

#%% world2image
def worldToImage(XPosition, imageShape= np.array([768,1024, 3]), zoom = True, methodRead = "csv", site = "SIRTA"):
    """
    Function to use to calculate image coordonate of plane with function.

    Parameters
    ----------
    XPosition : array
        X, Y, Z, 1 plane position. Vector dimension 4.
    imageShape : array, optional
        Image size.. The default is np.array([768,1024, 3]).
    zoom : bool, optional
        If images is zoomed or not. The default is True.
    methodRead : string, optional
        Type of csv file where is save params. The default is "csv".

    Returns
    -------
    x : float
        X coordonate.
    y : float
        y coordonate.

    """
    params = readCalParams(site = site)
    if len(params) == 6:
        x, y = W2IM2(XPosition, imageShape= imageShape, zoom = zoom, methodRead = methodRead, site = site)
    elif len(params) == 8:
        x, y = W2IM1(XPosition, imageShape= imageShape, zoom = zoom, methodRead = methodRead, site = site)
        
    return x, y

###############################################################################

def useCalibration(XPosition, imageShape= np.array([768,1024, 3]), zoom = True, methodRead = "csv", model="SIRTA"):
    """
    Function to use calculate image coordonate of plane with object model.

    Parameters
    ----------
    XPosition : array
        X, Y, Z, 1 plane position. Vector dimension 4.
    imageShape : array, optional
        Image size.. The default is np.array([768,1024, 3]).
    zoom : bool, optional
        If images is zoomed or not. The default is True.
    methodRead : string, optional
        Type of csv file where is save params. The default is "csv".
    model : string, optional
        Type of model. The default is "Fripon".

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    x : float
        X coordonate.
    y : float
        y coordonate.

    """
    
    if model in ['Orsay', 'SIRTA']:
        model = Fripon(model)
    elif model == 'Original':
        model = OrigineModel()
    else :
        raise ValueError('model unknow')
        
    x, y = model.useModel(XPosition= XPosition, imageShape= imageShape, zoom= zoom)
    return x, y


###############################################################################

def printError (beta,  XvSoleilFit, XvSoleilTest, xSunFit,   xSunTest,   funcSun, 
                beta2, XvAvionFit,  XvAvionTest,  xPlaneFit, xPlaneTest, funcCon,
                model = None):
    """
    Print different error.

    Parameters
    ----------
    beta : array
        Parameters before optimisation.
    XvSoleilFit : array
        Matrix to optimise beta with sun data. Data in real world.
    XvSoleilTest : array
        Matrix to test beta with sun data. Data in real world.
    xSunFit : array
        Matrix to optimise beta with sun data. Data in image space.
    xSunTest : array
        Matrix to test beta with sun data. Data in image space.
    funcSun : function
        Function to estimate error in sun dataset.
    beta2 : array
        Parameters after optimisation.
    XvAvionFit : array
        Matrix to optimise beta with contrail data. Data in real world.
    XvAvionTest : array
        Matrix to test beta with contrail data. Data in real world.
    xPlaneFit : array
        Matrix to optimise beta with contrail data. Data in image space.
    xPlaneTest : array
        Matrix to test beta with contrail data. Data in image space..
    funcCon : function
        Function to estimate error in contrail dataset.

    Returns
    -------
    None.
    
    Plot
    ----
    print 8 differents error.

    """
    if model == None:
        errorFSB = np.sqrt(funcSun(beta, XvSoleilFit,  xSunFit)/XvSoleilFit.shape[1])
        errorTSB = np.sqrt(funcSun(beta, XvSoleilTest, xSunTest)/XvSoleilTest.shape[1])
        errorFAB = np.sqrt(funcCon(beta, XvAvionFit,   xPlaneFit, False)/XvAvionFit.shape[1])
        errorTAB = np.sqrt(funcCon(beta, XvAvionTest,  xPlaneTest, False)/XvAvionTest.shape[1])
        
        errorFSA = np.sqrt(funcSun(beta2, XvSoleilFit,  xSunFit)/XvSoleilFit.shape[1])
        errorTSA = np.sqrt(funcSun(beta2, XvSoleilTest, xSunTest)/XvSoleilTest.shape[1])
        errorFAA = np.sqrt(funcCon(beta2, XvAvionFit,   xPlaneFit, False)/XvAvionFit.shape[1])
        errorTAA = np.sqrt(funcCon(beta2, XvAvionTest,  xPlaneTest, False)/XvAvionTest.shape[1])
    else:
        errorFSB = np.sqrt(funcSun(beta, XvSoleilFit,  xSunFit, model)/XvSoleilFit.shape[1])
        errorTSB = np.sqrt(funcSun(beta, XvSoleilTest, xSunTest, model)/XvSoleilTest.shape[1])
        errorFAB = np.sqrt(funcCon(beta, XvAvionFit,   xPlaneFit, model, False)/XvAvionFit.shape[1])
        errorTAB = np.sqrt(funcCon(beta, XvAvionTest,  xPlaneTest, model, False)/XvAvionTest.shape[1])
        
        errorFSA = np.sqrt(funcSun(beta2, XvSoleilFit,  xSunFit, model)/XvSoleilFit.shape[1])
        errorTSA = np.sqrt(funcSun(beta2, XvSoleilTest, xSunTest, model)/XvSoleilTest.shape[1])
        errorFAA = np.sqrt(funcCon(beta2, XvAvionFit,   xPlaneFit, model, False)/XvAvionFit.shape[1])
        errorTAA = np.sqrt(funcCon(beta2, XvAvionTest,  xPlaneTest, model, False)/XvAvionTest.shape[1])

    print("\n Before \n error Test Avion :%.2f, Test Sun :%.2f \n Fit Avion :%.2f, Fit Sun :%.2f "%(
                              errorTAB, errorTSB, errorFAB, errorFSB)+
          "\n After \n error Test Avion :%.2f, Test Sun :%.2f \n Fit Avion :%.2f, Fit Sun :%.2f"%(
                              errorTAA, errorTSA, errorFAA, errorFSA))
    
###############################################################################

#%% determination coordate

def imageToWorld(XPosition, imageShape= np.array([768,1024, 3]), zoom = True, methodRead = "csv", model="SIRTA_W", output = 'dico'):
    params = readCalParams(site = model)
    if len(params) == 6:
        x, y, z = image2world(XPosition, imageShape= imageShape, zoom = zoom, methodRead = methodRead, site = model)
    elif len(params) == 8:
        print("not implemented")
        
    if output == 'dico':
        return {'name': 'carteNorm', 'x':x, 'y':y, 'z':z}
    
    else:
        return False
    
    
    