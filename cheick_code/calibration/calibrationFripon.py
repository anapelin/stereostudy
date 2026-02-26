#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:23:30 2022

@author: ngourgue

all function you need to use fripon model 

rayon : calcul radius with cartesian coordonate.
angle : calcul angle with cartesian coordonate.
cart2polar : use rayon and angle to convert cartesian coordonate to polar.
distSolid : integrate an lens distorsion.
poly0 : integrate raidale distorsion.
distAsym : integrate a phasique distorsion.
model : integrate all preceent function to convert image coordonate to real world
coordonate.

But our application we want to convert real coordonate to image coordonate so
we inverse model.
polR2cart : conversion radius and angle coordonate to x and y.
invDistAsym : inversion the phasique distorsion.
invModel : all precedent function to convert real world to image coordonate.
invModelNorm : use to optimisation with normalise input. unnormalise and call
invModel.

cost function 
espriFrip : calcul distance between projection and reference.
espriFropNorm : Calcul distance to optmise parameters.
droiteErrorFrip : calcul distance between projection plane and reference 
contrail.
mixteFripon : combine espritFrip and droiteErrorFrip.

saveFripPraram : save parameters in file.

world2image : convert real world position to image position.

plotFrip : print value parameters.
convFrip2Beta : conversion fripon parameters so Beta vector.
convBeta2Frip : conversion Beta vector to Frip parameters.

"""

#%% importation
import copy, os, sys
import numpy as np
import pandas as pd

if not '/home/ngourgue/climavion/detection_contrail/calibration' in sys.path:
    sys.path.append('/home/ngourgue/climavion/detection_contrail/calibration')
from .baseCalibration import (Cartesian2Spherical, initR, NormalisationCal, reversZoom, 
                              readCalParams, Spherical2Cartesian, UnnormaliseBeta,
                              projCoord2, applyZoom)
#%% calibration fripon

def rayon(x, y, x0, y0):
    """
    Calcul radius with x and y coordonate and the center of image.

    Parameters
    ----------
    x : int
        Coordonate X.
    y : int
        Coordonate Y.
    x0 : int
        Center coordonate x.
    y0 : int
        Center coordonate y.

    Returns
    -------
    r : float
        Radius in pixel.

    """
    r = np.sqrt((x-x0)**2+(y-y0)**2)
    return r

###############################################################################

def angle(x, y, x0, y0):
    """
    Calcul angle with x and y coorondate and the center of image.

    Parameters
    ----------
    x : int
        Coordonate X.
    y : int
        Coordonate Y.
    x0 : int
        Center coordonate x.
    y0 : int
        Center coordonate y.

    Returns
    -------
    alpha : float
        Angle in radian.

    """
    # alpha = np.arctan2((y-y0), (x-x0))
    if y-y0 > 0:
        alpha = np.arccos((x-x0)/np.sqrt((x-x0)**2+(y-y0)**2))
    elif y-y0 == 0 and x-x0 == 0:
        alpha = 0
    else:
        alpha = - np.arccos((x-x0)/np.sqrt((x-x0)**2+(y-y0)**2))
        
    return alpha

###############################################################################
    
def cart2polR(x, y, x0, y0):
    """
    Calcul Radius and Angle with x and y coordonate and the center of image.

    Parameters
    ----------
    x : int
        Coordonate X.
    y : int
        Coordonate Y.
    x0 : int
        Center coordonate x.
    y0 : int
        Center coordonate y.

    Returns
    -------
    r : float
        Radius in pixel.
    alpha : float
        Angle in radian.

    """
    r = rayon(x, y, x0, y0)
    alpha = angle(x, y, x0, y0)
    return r, alpha

###############################################################################

def distSolid(z, k1, k2):
    """
    Calcul the solid distorsion (lens deformation) on radius parameters with angle. 

    Parameters
    ----------
    z : float
        Phase angle in radian.
    k1 : float
        Distorsion ponderation .
    k2 : float
        Angle deformation.

    Returns
    -------
    r : float
        Radius with deformation in pixel.

    """
    r = k1*np.sin(k2*z)
    return r

###############################################################################

def poly9(r, a):
    """
    Calcul the radiale correction on radius parameters with angle.

    Parameters
    ----------
    r : float
        Radius in pixel.
    a : float
        Radiale angle.

    Returns
    -------
    z : float
        Radius with radiale deformation.

    """
    z = a[0]*r+a[1]*r**3+a[2]*r**5+a[3]*r**7+a[4]*r**9
    return z

###############################################################################

def distAsym(Rold, K1, phi, A, site = 'SIRTA_W'):
    """
    Calcul distosion on phase.

    Parameters
    ----------
    Rold : float
        Input radius.
    K1 : float
        Deformartion ponderation.
    phi : float
        Phase in radian.
    A : float
        Angle in radian.

    Returns
    -------
    r : float
        Output radius.

    """
    r = Rold*(1+K1*np.sin(A+phi))
    if site == 'Orsay':
        r = r/1000
    return r

###############################################################################

def equationSimon(X1, Y1, Z1, theta):
    X2 = X1
    Y2 = Y1*np.cos(theta[0]) - Z1*np.sin(theta[0])
    Z2 = Y1*np.sin(theta[0]) + Z1*np.cos(theta[0])
    
    X3 = X2*np.cos(theta[1]) + Z2*np.sin(theta[1])
    Y3 = Y2
    Z3 = -X2*np.sin(theta[1]) + Z2*np.cos(theta[1])
    
    X4 = X3*np.cos(theta[2]) - Y3*np.sin(theta[2])
    Y4 = X3*np.sin(theta[2]) + Y3*np.cos(theta[2])
    Z4 = Z3
    
    return np.array([X4, Y4, Z4])
###############################################################################

def model(x, y, a, x0, y0, theta, K1, phi, site = 'SIRTA_W'):
    """
    Model function. Project in 3D world the coordonate pixel.

    Parameters
    ----------
    x : float
        X coordonate in image.
    y : float
        Y coordonate in image.
    a : array
        Polynome to ponderate radius.
    x0 : int
        X center coordonate in image.
    y0 : int
        Y center coorondate in image.
    theta : array
        Angle to matrix rotation.
    K1 : float
        Ponderation distorsion phase.
    phi : float
        Phase distortion in radian.

    Returns
    -------
    X : array
        X, Y, Z coordonate.

    """
    mat = True
    Rold, alpha = cart2polR(x, y, x0, y0)
    rNew = distAsym(Rold, K1, phi, alpha, site = site)
    z = poly9(rNew, a)
    #conversion polaire vers cartésien
    X1, Y1, Z1 = Spherical2Cartesian(theta=z, phi=alpha, method='Simon')
    # X1 = -np.cos(alpha)*np.sin(z)  #- -> np.pi - alpha; + -> alpha
    # Y1 = np.sin(alpha)*np.sin(z)
    # Z1 = np.cos(z)
    if mat == True:
        if site == 'Orsay':
            Rx, Ry, Rz = initR(theta, method = 'Simon')
            XYZ2 = np.matmul(Rx, np.array([X1, Y1, Z1]))
            XYZ3 = np.matmul(Ry, XYZ2)
            XYZ4 = np.matmul(Rz, XYZ3)
            XYZ = XYZ4
        elif site  == 'SIRTA_W':
            Rot = initR(theta, method = 'Cardan')
            Rotm1 = np.linalg.inv(Rot)
            XYZ = np.matmul(Rotm1, np.array([X1, Y1, Z1]))
            XYZ[0] = -XYZ[0]
    else:
        XYZ = equationSimon(X1, Y1, Z1, theta)
   
    return XYZ

###############################################################################

#%% inv model

def polR2cart(R, theta, x0, y0):
    """
    Conversion polar coordonate to cartesian coordonate.

    Parameters
    ----------
    R : float
        Radius in pixel.
    theta : float
        Phase angle in radian.
    x0 : int
        X coordonate of center.
    y0 : int
        Y coordonate of center.

    Returns
    -------
    x : float
        X coordonate.
    y : float
        Y coordonate.

    """
    x = R*np.cos(theta)+x0
    y = R*np.sin(theta)+y0
    return x, y

###############################################################################
    
def invDistAsym(r, K1, phi, A, site = 'SIRTA'):
    """
    Correction of radius with phase deformation

    Parameters
    ----------
    r : float
        Radius in pixel.
    K1 : float
        Pondération of déformation.
    phi : float
        Phase in radian.
    A : float
        Phase angle in radian.

    Returns
    -------
    Rold : float
        Output angle radian.

    """
    Rold = r/(1+K1*np.sin(A+phi))
    if site == 'Orsay':
        Rold = Rold*1000
    return Rold

###############################################################################

def rotation(Px, theta, invEqu = True, method = 'old'):
        if method == 'old':
            # self.method = 'Cardan'
            Rot = initR()
            if invEqu == True:     
                RotInv = np.linalg.inv(Rot)
                Px1 = np.dot(RotInv, Px)
            else:
                Px1 = np.dot(Rot, Px)
        elif method =='Simon':
            # method = 'Simon'
            Rx, Ry, Rz = initR(theta, method = 'Simon')
            if invEqu == True:
                Rz1 = np.linalg.inv(Rz)
                Vz = np.dot(Rz1, Px)
                Ry1 = np.linalg.inv(Ry)
                Vy = np.dot(Ry1, Vz)
                Rx1 = np.linalg.inv(Rx)
                Vx = np.dot(Rx1, Vy)
                Px1 = Vx
            else:
                Vx = np.dot(Rx, Px)
                Vy = np.dot(Ry, Vx)
                Vz = np.dot(Rz, Vy)
                Px1 = Vz
        
        elif method == 'Nico':
            # method = 'Nico'
            Rot = initR(theta)
            if invEqu == True:
                RotInv = np.linalg.inv(Rot)
                Px1 = np.dot(RotInv, Px)
            else:
                Px1 = np.dot(Rot, Px)
            
        else :
            raise ValueError('method is unknow :'+method)
        return Px1
    
###############################################################################

def invModel(Px, b, x0, y0, theta, K1, phi, site = 'SIRTA'):
    """
    Using model to calculate coordonate image with real position.

    Parameters
    ----------
    Px : array
        Object position longitude latitude et altitude normalise.
    b : array
        Polynome parameters for radial déformation.
    x0 : int
        X coordonate center.
    y0 : int
        Y coordonate center.
    theta : array
        Vector 3 angles in radian.
    K1 : float
        Ponderation to phase distorsion.
    phi : float
        Phase in radian to shit angle.
    
    raise
    -----
    error : Px problem with size.

    Returns
    -------
    x : float
        X coordonate in image.
    y : float
        Y coordonate in image.

    """
    #check size Px
    if Px.shape[0] == 4:
        Px = Px[:3,:]
    elif Px.shape[0] == 3:
        pass
    else:
        raise ValueError('size on Px in not correct', Px.shape)
    # invEqu = True
    if site == 'SIRTA':
        Px1 = rotation(Px, theta, True, method = 'Nico')
        z, alpha = Cartesian2Spherical(Px1[0], Px1[1], Px1[2], method='tan')
    elif site == 'Orsay':
        Px1 = rotation(Px, theta, True, method = 'Simon')
        z, alpha = Cartesian2Spherical(Px1[0], Px1[1], Px1[2], method='Simon')

    rNew = poly9(z, b)
    R = invDistAsym(rNew, K1, phi, alpha, site)
    x, y = polR2cart(R, alpha, x0, y0)
    return x, y

###############################################################################
    
def invModelNorm(params, Px):
    """
    Calculate coordonate in image with normalisation.

    Parameters
    ----------
    params : array
        Vector contain all parameters.
    Px : array
        Vector to position.

    Returns
    -------
    x : float
        X coordonate in image.
    y : float
        Y coordonate in image.

    """
    b = NormalisationCal(params[:5], 'k', normalise = False)
    x0  = NormalisationCal(params[5], 'xo', normalise = False)
    y0  = NormalisationCal(params[6], 'yo', normalise = False)
    theta = NormalisationCal(params[7:10], 'w', normalise = False)
    K1   = NormalisationCal(params[10], 'p1', normalise = False)
    phi   = NormalisationCal(params[11], 'w', normalise = False)
    x, y = invModel(Px, b, x0, y0, theta, K1, phi)
    return x, y

###############################################################################

#%% cost function
def epsiFrip(params, Px, xSun, site = 'SIRTA'):
    """
    Calcul euclidian distance error in image space.

    Parameters
    ----------
    params : array
        All parameters, vector dimension 12.
    Px : array
        Position in real world, latitude longitude altitude normalize, matrix 
        dimenson 3 *pour number of object.
    xSun : array
        Position in image, x and y coordonate, matrix dimension 2 * number of object.

    Returns
    -------
    epsilon : float
        Raw error with distance euclidienne.

    """
    b = params[:5]
    x0 = params[5]
    y0 = params[6]
    theta = params[7:10]
    K1 = params[10]
    phi = params[11]
    x, y = invModel(Px, b, x0, y0, theta, K1, phi, site)
    epsilon = np.sum((x-xSun[0, :])**2+(y-xSun[1,:])**2)
    return epsilon

###############################################################################
    
def epsiFripNorm(params, Px, xSun, site = 'SIRTA'):
    """
    Calcul euclidian distance error in image space with an normalise paramters.

    Parameters
    ----------
    params : array
        All parameters, vector dimension 12 with normalisation.
    Px : array
        Position in real world, latitude longitude altitude normalize, matrix 
        dimenson 3 * number of object.
    xSun : array
        Position in image, x and y coordonate, matrix dimension 2 * number of object.

    Returns
    -------
    epsilon : float
        Raw error with distance euclidienne.

    """
    params1 = copy.deepcopy(params)
    # params1[:5]   = NormalisationCal(params[:5],   'k', normalise = False)
    # params1[5]    = NormalisationCal(params[5],    'xo', normalise = False)
    # params1[6]    = NormalisationCal(params[6],    'yo', normalise = False)
    # params1[7:10] = NormalisationCal(params[7:10], 'w', normalise = False)
    # params1[10]   = NormalisationCal(params[10],   'p1', normalise = False)
    # params1[11]   = NormalisationCal(params[11],   'w', normalise = False)
    params1 = UnnormaliseBeta(params1)
    
    epsilon = epsiFrip(params1, Px, xSun, site)
    return epsilon

###############################################################################
    
def droiteErrorFrip(params, XvAvion, xPlane, norm= True):
    """
    Calcul euclidian distance error in image space into contrail line and
    plane position. The error is based on hypothesys in moment the contrail line 
    is in prolongement of plane.

    Parameters
    ----------
    params : array
        All parameters, vector dimension 12.
    XvAvion : array
        Position in real world, latitude longitude altitude normalize matrix 
        dimension 3 * number of object.
    xPlane : array
        Value save about contrail line. Matrix dimension 2 * number of object.
        first value dist and angle.
    norm : bool, optional
        If parameters is normalise. The default is True.

    Returns
    -------
    epsilon : float
        Raw error with distance euclidienne into contrail line and plane position.

    """

    #call function
    if norm:
        x, y  = invModelNorm(params, XvAvion)
    else:
        b = params[:5]
        x0 = params[5]
        y0 = params[6]
        theta = params[7:10]
        K1 = params[10]
        phi = params[11]
        x, y = invModel(XvAvion, b, x0, y0, theta, K1, phi)

    x = x -68
    y = y -182
    # if np.min(xvCal)<1:
    #     return 9999999
    x, y = reversZoom(x, y, szamax= 60, xmax =901, xRGBc = 339, yRGBc= 339)

    # find distance
    X0 = xPlane[:,1] * np.array([np.cos(xPlane[:,0]), np.sin(xPlane[:,0])])
    a = np.tan(xPlane[:,0]+np.pi/2) #coeff directeur
    deltaY = a/(X0[1,:]- a*X0[0,:]) #A
    deltaX = -1/(X0[1,:]- a*X0[0,:]) #B
    c = 1 #C

    dist = np.abs(deltaY*y+deltaX*x+c)/np.sqrt(deltaX**2+deltaY**2)
    epsilon = np.sum(dist)

    return epsilon    

###############################################################################

def mixteFripon(params, XvSoleil, xSun, XvAvion, xPlane, ponderation = 1, norm = True):
    """
    Calcul an error base on mix into euclidean distance in sun image and 
    euclidean distance in contrail line. 

    Parameters
    ----------
    params : array
        All parameters, vector dimension 12All parameters, vector dimension 12.
    XvSoleil : array
        Position in real world, latitude longitude altitude normalize, matrix 
        dimenson 3 * number of object.
    xSun : array
        Position in image, x and y coordonate, matrix dimension 2 * number of object.
    XvAvion : array
        Position in real world, latitude longitude altitude normalize matrix 
        dimension 3 * number of object.
    xPlane : array
        Value save about contrail line. Matrix dimension 2 * number of object.
        first value dist and angle.
    ponderation : float, optional
        Coefficient to multiplicate contrail error. The default is 1.
    norm : bool, optional
        If params is normalise or not. The default is True.

    Returns
    -------
    errorTot : float
        Sum on different error.

    """
    params1 = copy.deepcopy(params)
    errorPlane = droiteErrorFrip(params1, XvAvion, xPlane, norm)
    if norm == True:
        errorSun = epsiFripNorm(params1, XvSoleil, xSun)
    else :
        errorSun = epsiFrip(params1, XvSoleil, xSun)
    errorTot =  errorSun + errorPlane*ponderation
    return errorTot

###############################################################################    
#%% function use calibration
def saveFripParams (b, x0, y0, theta, K1, phi, method = "csv", path = "/home/ngourgue/climavion/params.csv", site = 'SIRTA'):
    """
    Save parameters in a file. Csv by default.

    Parameters
    ----------
    b : array
        Polynome to module radiale deformation.
    x0 : float
        X coordonate of center on image.
    y0 : float
        Y coordonate of center on image
    theta : array
        Vector about angle to matrix rotation.
    K1 : float
        Ponderation on phase distorsion.
    phi : float
        Phase shift.
    method : string, optional
        Type of file to save. The default is "csv".
    path :string, optional
        Path where to save file with parameters. The default is "/home/ngourgue/climavion/params.csv".

    Returns
    -------
    bool
        Succes or not to save.

    """
    if method == "csv":
        df0 = pd.read_csv(path)
        df1 = df0[df0.site == site]
        beta  = np.zeros(12)
        beta[:5]   = b
        beta[5]    = x0
        beta[6]    = y0 
        beta[7:10] = theta 
        beta[10]   = K1
        beta[11]   = phi
        colonne = df1.columns
        df1[colonne[1:13]] = beta
        df0[df0.site == site] = df1
        df0.to_csv(path, index=False)
        return True
    else:
        print("paramters can't be save. method not function.")
        return False

###############################################################################

#%% world2image

def world2image(XPosition, imageShape= np.array([768,1024, 3]), zoom = True, methodRead = "csv", site = "SIRTA"):
    """
    calcul with position the coordonate in image.

    Parameters
    ----------
    XPosition : array
        Latitude, Longitude and Altitude noramilaze.
    imageShape : array, optional
        Shape of image. The default is np.array([768,1024, 3]).
    zoom : Bool, optional
        If it's original image or zoom image. The default is True.
    methodRead : string, optional
        Method to read parameters. The default is "csv".

    Raises
    ------
    ValueError
        Image shape unknow.

    Returns
    -------
    array
        Coordonate.
        
    """
    #read params
    b, x0, y0, theta, K1, phi = readCalParams(method = methodRead, site = site)
    #convert XPosition
    # if len(XPosition) == 3:
    #     if type(XPosition[0]) is int or type(XPosition[0]) is float or type(XPosition[0]) is np.float64:
    #         XPosition = np.array([XPosition[0], XPosition[1], XPosition[2], 1])
    #     else:
    #         XPosition = np.array([XPosition[0], XPosition[1], XPosition[2], np.ones(len(XPosition[0]))])
    #change if len == 4 change to 3
    #image 3D
    if len(imageShape) == 3:
        #test size image
        #original image
        if np.all(imageShape == np.array([768, 1024, 3])) and site == 'SIRTA':
            x, y = invModel(XPosition,b, x0, y0, theta, K1, phi, site) #projection coordonné transposer de l'image puis inversion grand axe.
            return(x, y)
        #transpose image
        elif np.all(imageShape == np.array([1024, 768, 3])) and site == 'SIRTA':
            x, y = invModel(XPosition,b, x0, y0, theta, K1, phi, site)#projection coordonné transposer de l'image puis inversion grand axe.
            return(y, x)
        #zoom or transforme image
        elif np.all(imageShape == np.array([901, 901, 3])) and site == 'SIRTA':
            x, y = invModel(XPosition,b, x0, y0, theta, K1, phi, site)#projection coordonné transposer de l'image puis inversion grand axe.
            x = x-68
            y = y-182 
            xR, yR= reversZoom(x, y, szamax= 60, xmax =901, xRGBc = 339, yRGBc= 339)
            return(xR, yR)
        #cropped image old cal
        elif  np.all(imageShape == np.array([674, 674, 3])) and site == 'SIRTA':
            x, y = invModel(XPosition,b, x0, y0, theta, K1, phi, site)
            x = x-66
            y = y-161
            return(x, y)
        #cropped image new cal
        elif  np.all(imageShape == np.array([678, 678, 3])) and site == 'SIRTA':
            x, y = invModel(XPosition,b, x0, y0, theta, K1, phi, site)
            x = x-68
            y = y-182
            return(x, y)
        #original image Orsay
        elif np.all(imageShape == np.array([1280, 960, 3])) and site == 'Orsay':
            x, y = invModel(XPosition,b, x0, y0, theta, K1, phi, site)
            return (x, y)
        #zoom or transforme image orsay
        elif np.all(imageShape == np.array([901, 901, 3])) and site == 'Orsay':
            x, y =invModel(XPosition,b, x0, y0, theta, K1, phi, site)
            x = x-160
            y = y-33
            if zoom:
                xR, yR= reversZoom(x, y, szamax= 60, xmax =901, xRGBc = 450.5, yRGBc= 450.5)
            else:
                xR, yR = x, y

            return(xR, yR)
        else:
            raise ValueError("image.shape not corresponded to correct value.\n image.shape : %d, %d, %d\n"%(imageShape[0], 
                             imageShape[1], imageShape[2])+"correct values : \n768, 1024, 3 : original"+\
                             " image\n1024, 768, 3 : transpose image\n901, 901, 3 : zoom or transforme"+\
                             " image\n678, 678, 3 : cropped image")
    #image 2D    
    elif len(imageShape) == 2:
        #original image Orsay
        if np.all(imageShape == np.array([1280, 960])) and site == 'Orsay':
            x, y = invModel(XPosition,b, x0, y0, theta, K1, phi, site)
            return (x, y)
        #zoom or transforme image orsay
        elif np.all(imageShape == np.array([901, 901])) and site == 'Orsay':
            x, y =invModel(XPosition,b, x0, y0, theta, K1, phi, site)
            x = x-160
            y = y-33
            # x = x - 33
            # y = y - 160
            if zoom:
                xR, yR= reversZoom(x, y, szamax= 60, xmax =901, xRGBc = 450.5, yRGBc= 450.5)
            else:
                xR, yR = x, y

            return(xR, yR)
        else:
            raise ValueError("image.shape not corresponded to correct value.\n image.shape : %d, %d, %d\n"%(imageShape[0], 
                             imageShape[1], imageShape[2])+"correct values : \n768, 1024, 3 : original"+\
                             " image\n1024, 768, 3 : transpose image\n901, 901, 3 : zoom or transforme"+\
                             " image\n678, 678, 3 : cropped image")
    else:
        raise ValueError("image is not correct. image 2D or 3D is need to argument image.")

###############################################################################

#%% image to world
def image2world(XPosition, imageShape= [901, 901, 3], zoom = True, methodRead = 'csv', site = 'SIRTA_W'):
    
    #read params
    a, x0, y0, theta, K1, phi = readCalParams(method = methodRead, site = site)
    
    if len(imageShape) == 3:
        #original image
        if np.all(imageShape == np.array([768, 1024, 3])) and site == 'SIRTA_W':
            print('not implemented')
        #transpose image
        elif np.all(imageShape == np.array([1024, 768, 3])) and site == 'SIRTA_W':
            print('not implemented')
        #zoom or transforme image
        elif np.all(imageShape == np.array([901, 901, 3])) and site == 'SIRTA_W':
            ix, iy, altkm = XPosition
            #coordonner réel mais on veut le coordonné biaisées
            # xOri, yOri = projCoord2(ix, iy, image = None, szamax=60, xmax=901, zoom = True, imShow = False)
            xUz, yUz = applyZoom(ix, iy, 60, 901, 339, 339)
            xUc = xUz + 68
            yUc = yUz + 182
            XYZ = model(xUc, yUc, a, x0, y0, theta, K1, phi)
            
            return XYZ
    else:
        print('not implemented')
    
    
    return None

###############################################################################

#%%
def plotFrip(b , x0, y0, theta, K1, phi):
    """
    Plot name and value on parameters.

    Parameters
    ----------
    b : array
        Vector with 5, value of radiale distorsion.
    x0 : float
        X coordonate of center.
    y0 : float
        Y coordonate of center.
    theta : array
        Vector with 3 values, angle of rotation matrix.
    K1 : float
        Ponderation of phase distorsion.
    phi : float
        Phase.

    Returns
    -------
    None.

    """
    print("b = %.2f %.2f %.2f %.2f %.2f"%(b[0], b[1], b[2], b[3], b[4]))
    print("center x0 : %.2f, y0 %.2f"%(x0, y0))
    print("w angle  alpha : %.5f, beta : %.5f, gamma : %.5f"%(theta[0], theta[1], theta[2]))
    print("K1 : %.5f, phi : %.5f"%(K1, phi))

###############################################################################

def convFrip2Beta(b, x0, y0, theta, K1, phi):
    """
    Transform multi parameters to a vector.

    Parameters
    ----------
    b : array
        Vector with 5, value of radiale distorsion.
    x0 : float
        X coordonate of center.
    y0 : float
        Y coordonate of center.
    theta : array
        Vector with 3 values, angle of rotation matrix.
    K1 : float
        Ponderation of phase distorsion.
    phi : float
        Phase.

    Returns
    -------
    beta : array
        Vector Output.

    """
    beta = np.zeros((12), dtype=np.float128)
    beta[:5] = b
    beta[5]  = x0
    beta[6]  = y0
    beta[7:10] = theta
    beta[10] = K1
    beta[11] = phi
    return beta

###############################################################################

def convBeta2Frip(beta):
    """
    Transform vector to multi parameters

    Parameters
    ----------
    beta : array
        Vector input.

    Returns
    -------
   b : array
        Vector with 5, value of radiale distorsion.
    x0 : float
        X coordonate of center.
    y0 : float
        Y coordonate of center.
    theta : array
        Vector with 3 values, angle of rotation matrix.
    K1 : float
        Ponderation of phase distorsion.
    phi : float
        Phase.

    """
    b = beta[:5]
    x0 = beta[5]
    y0 = beta[6]
    theta = beta[7:10]
    K1= beta[10]
    phi = beta[11]
    return b, x0, y0, theta, K1, phi

###############################################################################

def convGeoCarto(lat, lon, alt, method = ''):
    #theta angle d'élèvation (latitude)
    #phi angle azimutal (longitude)
    #h altitude en m
    
    #conv degree to radian
    theta = np.deg2rad(lat, dtype=np.float128)
    phi   = np.deg2rad(lon, dtype=np.float128)
    
    if method == 'Clarke 1880 IGN':
        a = 6378249,2
        b = 6356515
    elif method == 'IAG GRS 80':
        a = 6378137
        f = 1./298.257222101
        b = a*(1-f)
    elif method == 'WGS 84':
        a = 6378137
        f = 1./298.257223563
        b = a*(1-f)
    elif method == 'HAYFORD 1909':
        a = 6378388
        f = 1/297
        b = a*(1-f)
    else:
        print('unknow method')
        return None, None, None
    
    e2 = np.array((a**2-b**2)/(a**2), dtype=np.float128)
    W  = np.sqrt(1-e2*np.sin(theta, dtype=np.float128)**2, dtype=np.float128)
    # rho = a*(1-e2)/(W**3)
    N   = np.array(a/W , dtype=np.float128)
    # r   = N* np.cos(theta)
    
    X = np.array((N+alt)*np.cos(theta, dtype=np.float128)*np.cos(phi, dtype=np.float128), dtype=np.float128)
    Y = np.array((N+alt)*np.cos(theta, dtype=np.float128)*np.sin(phi, dtype=np.float128), dtype=np.float128)
    Z = np.array((N*(1-e2)+alt)*np.sin(theta, dtype=np.float128), dtype=np.float128)
    
    return X, Y, Z
    
def convCartoGeo(X, Y, Z, method= ''):
    
    if method == 'Clarke 1880 IGN':
        a = 6378249,2
        b = 6356515
    elif method == 'IAG GRS 80':
        a = 6378137
        f = 1./298.257222101
        b = a*(1-f)
    elif method == 'WGS 84':
        a = 6378137
        f = 1./298.257223563
        b = a*(1-f)
    elif method == 'HAYFORD 1909':
        a = 6378388
        f = 1/297
        b = a*(1-f)
    else:
        print('unknow method')
        return None, None, None

    e2 = np.array((a**2-b**2)/(a**2), dtype=np.float128)
    R = np.sqrt(X**2 + Y**2 + Z**2, dtype=np.float128)
    
    phi = np.arctan2(Y, X, dtype=np.float128) #longitude est
    
    µ = np.arctan(Z/np.sqrt(X**2+Y**2, dtype=np.float128)*((1-f)+(e2*a/R)), dtype=np.float128)
    theta = np.arctan2(Z*(1-f)+e2*a*(np.sin(µ, dtype=np.float128)**3), (1-f)*(np.sqrt(X**2+Y**2, dtype=np.float128)-e2*a*np.cos(µ, dtype=np.float)**3), dtype=np.float128) #latitude
    
    h = np.sqrt(X**2+Y**2, dtype=np.float128)*np.cos(theta, dtype=np.float128) + Z*np.sin(theta, dtype=np.float128) - a*np.sqrt(1-e2*np.sin(theta, dtype=np.float128)**2, dtype=np.float128) #altitude
    
    #conv rad to degree
    theta = np.rad2deg(theta, dtype=np.float128)
    phi   = np.rad2deg(phi, dtype=np.float128)
    return theta, phi, h



#%% paramters
if __name__ == '__main__':
    theta = np.array([0, 0, 0])
    
    x0  = 383
    y0  = 513
    
    a = np.array([1, 0, 0, 0, 0])
    
    K1 = 2.5*10**(-3)
    phi = 0

#%% model


