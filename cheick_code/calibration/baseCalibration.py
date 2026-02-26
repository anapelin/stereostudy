#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:07:07 2022

@author: ngourgue

file contain different functon which you need with all model.

Cartesoan2Spherical and Spherical2Cartesian can convert coordonate to spherical
to cartesian and the opposite.

initR to create rotation matrix.

NormalisationCal to normalise a parameter with a function to optimisation.
reducto and amplificatum apply the normalisation or apply unnormalisation.
normalisationBeta and UnnormaliseBeta can normalise or unnormalise the vector
which contain all parameters.

reversZoom can to calculate coordonate in old image.

readCalParam can read parameters in file.

psiSun convert sun ephem to be coherent with orther mathematique convention.

loadSunData and loadContrailData can extract data to csv file to do an 
optimisation. 
Separate data can create 4 daatset with 2 original. Projection data, reference 
data and after projection data to optimisation and test, reference data to 
optimisation and test.

"""

import os, random
import copy as cp
import numpy  as np
import pandas as pd

# from deco import timeDuration  # Commented out - library not available
###############################################################################

#%% Cpnversion cartesian and spherical
def Cartesian2Spherical(XCam, YCam, ZCam, method = 'tan'):
    """
    Conversion X, Y, Z coordonate to spherical theta phi. 

    Parameters
    ----------
    XCam : float
        X coordonate.
    YCam : float
        Y coordonate.
    ZCam : float
        Z coordonate.

    Returns
    -------
    psi : array
        Vector 2 dimensions. theta phi.

    """
    if method == 'tan':
        theta = np.arctan2(np.sqrt(XCam**2+YCam**2),ZCam)
        phi   = np.arctan2(YCam, XCam)
        
    elif method == 'cos':
        r = np.sqrt(XCam**2+YCam**2+ZCam**2)
        theta = np.arccos(ZCam/r)
        phi   = (-np.arctan2(YCam, XCam)) % (2.*np.pi) - np.pi #--because of the minus sign in S2C
    
    elif method == 'Simon':
        theta = np.arctan2(np.sqrt(XCam**2+YCam**2),ZCam)
        phi   = np.arctan2(YCam, -1.*XCam)  #--minus sign to match Simon's convention
    psi = [theta, phi]
    return psi

###############################################################################

def Spherical2Cartesian(R = None, t = None, theta = 0, phi = 0, lamda = 1, inversion = False, method = 'old'):
    """
    Conversion theta phi coordonate to cartesian X, Y, Z, T

    Parameters
    ----------
    R : array
        Matrix 3*3. Rotation matrix.
    t : array
        Vector 3. Translation vector.
    theta : float
        Angle radiale.
    phi : float
        Angle of phase.
    lamda : float, optional
        DESCRIPTION. The default is 1.
    inversion : bool, optional
        Inversion of x and y axis. The default is False.

    Returns
    -------
    Xv : array
        Output vector. X, Y, Z, T.

    """
    if method == 'old':
        if type(theta) is int or type(theta) is float:
            mat = np.zeros([4,4])
            mat[:3,:3] = lamda * np.transpose(R)
            mat[:3,3] = np.dot(-1*np.transpose(R), t)
            mat[ 3, 3] = 1
            vect = np.zeros([4,1])
            #inversion
            if inversion:
                vect[0] = np.sin(phi)*np.sin(theta)
                vect[1] = np.cos(phi)*np.sin(theta)
            else:
                vect[0] = np.cos(phi)*np.sin(theta)
                vect[1] = np.sin(phi)*np.sin(theta)
            vect[2] = np.cos(theta)
            vect[3] = 1
            Xv = np.dot(mat, vect)
        else:
            mat = np.zeros([4,4])
            mat[:3,:3] = lamda * np.transpose(R)
            mat[:3,3] = np.dot(-1*np.transpose(R), t)
            mat[ 3, 3] = 1
            vect = np.zeros([4,(len(theta))])
            #inversion
            if inversion:
                vect[0] = np.sin(phi)*np.sin(theta)
                vect[1] = np.cos(phi)*np.sin(theta)
            else:
                vect[0] = np.cos(phi)*np.sin(theta)
                vect[1] = np.sin(phi)*np.sin(theta)
            vect[2] = np.cos(theta)
            vect[3] = 1
            Xv = np.dot(mat, vect)
        return Xv
    elif method == 'Simon':
        X = -np.cos(phi)*np.sin(theta)  # note the - sign
        Y = np.sin(phi)*np.sin(theta)
        Z = np.cos(theta)
        return X,Y,Z

    else:
        print("Method unknow")
        return None
    

#%% Normalisation and transformation
def NormalisationCal(value, name, norm = "Normal", normalise = True):
    """
    function to normalise parameters.

    Parameters
    ----------
    value : float
        Value to normalise.
    name : string
        Name of value.
    norm : string, optional
        Type of noramlisation. The default is "Normal".
    normalise : Bool, optional
        If you want to normalise or Unnormalise. The default is True.

    Raises
    ------
    ValueError
        Normalisation non implented.

    Returns
    -------
    output : float
        Value after normalisation or unnormalisation.

    """
    
    if name == "alphaX":
        if norm == "Normal":
            meanaX = 222.8
            stdaX  = 1
            if normalise is True:
                output = reducto(value, meanaX, stdaX)
            else :
                output = amplificatum(value, meanaX, stdaX)
        else :
            raise ValueError("norm unknow")
    
    elif name == "alphaY":
        if norm == "Normal":
            meanaY = 222.8
            stdaY  = 1
            if normalise is True:
                output = reducto(value, meanaY, stdaY)
            else:
                output = amplificatum(value, meanaY, stdaY)
        else :
            raise ValueError("norm unknow")
    
    elif name == "s":
        if norm == "Normal":
            means = 0.0
            stds  = 0.01
            if normalise is True:
                output = reducto(value, means, stds)
            else :
                output = amplificatum(value, means, stds)
        else:
            raise ValueError("norm unknow")
    
    elif name == "xo":
        if norm == "Normal":
            meanxo = 383
            stdxo  = 1
            if normalise is True:
                output = reducto(value, meanxo, stdxo)
            else :
                output = amplificatum(value, meanxo, stdxo)
        else:
            raise ValueError("norm unknow")
            
    elif name == "yo":
        if norm == "Normal":
            meanyo = 513
            stdyo  = 1
            if normalise is True:
                output = reducto(value, meanyo, stdyo)
            else :
                output = amplificatum(value, meanyo, stdyo)
        else:
            raise ValueError("norm unknow")
    
    elif name in ["w1", "w2", "w3", "w"]:
        if norm == "Normal":
            meanw = 0
            stdw  = 0.1
            if normalise is True:
                output = reducto(value, meanw, stdw)
            else :
                output = amplificatum(value, meanw, stdw)
        else:
            raise ValueError("norm unknow")
            
    elif name in ["tx", "ty", "tz", "t"]:
        if norm == "Normal":
            meant = 0
            stdt  = 0.1
            if normalise is True:
                output = reducto(value, meant, stdt)
            else :
                output = amplificatum(value, meant, stdt)
        else:
            raise ValueError("norm unknow")
    
    elif name in ["k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k"]:
        if norm == "Normal":
            meank = 0
            stdk  = 1
            if normalise is True:
                output = reducto(value, meank, stdk)
            else :
                output = amplificatum(value, meank, stdk)
        else:
            raise ValueError("norm unknow")
    
    elif name in ["p1", "p2", "p12"]:
        if norm == "Normal":
            meanp12 = 0
            stdp12  = 0.001
            if normalise is True:
                output = reducto(value, meanp12, stdp12)
            else :
                output = amplificatum(value, meanp12, stdp12)
        else:
            raise ValueError("norm unknow")
    
    elif name in ["p3", "p4", "p34"]:
        if norm == "Normal":
            meanp34 = 0
            stdp34   = 0.1
            if normalise is True:
                output = reducto(value, meanp34, stdp34)
            else : 
                output = amplificatum(value, meanp34, stdp34)
        else:
            raise ValueError("norm unknow")
    elif name in ["p", "p1234"]:
        if norm == "Normal":
            meanp = 0
            stdp   = 0.1
            if normalise is True:
                output = reducto(value, meanp, stdp)
            else : 
                output = amplificatum(value, meanp, stdp)
        else:
            raise ValueError("norm unknow")
    else:
            raise ValueError("name unknow")
            
    return output

###############################################################################
        
def reducto(value, mean, std):
    """
    Normalisation with normal equation.

    Parameters
    ----------
    value : float
        Value to normalise.
    mean : float
        Reference.
    std : float
        Stantard deviation about value.

    Returns
    -------
    Float
        Value after normalisation.

    """
    return (value-mean)/std

###############################################################################
    
def amplificatum (value, mean, std):
    """
    Unnormalize value.
    Parameters
    ----------
    value : float
        Value to unnormalize.
    mean : float
        Reference value.
    std : float
        Stantard deviation.

    Returns
    -------
    float
        Value unnormalized.

    """
    return value*std+mean

###############################################################################



#%% sun transform
def psiSun(sun):
    """
    Calculate sun angle with our orientation

    Parameters
    ----------
    sun : array
        Vector ephem theta and phi angle to sun observation.

    Returns
    -------
    Vector output theta and phi.

    """
    if type(sun) is list:
        psis = []
        for su in sun:
            theta = np.pi/2.-su.alt
            phi = np.pi - su.az*1.
            psis.append([theta, phi])
        return(psis)
    else:
        theta = np.pi/2.-sun.alt
        phi = np.pi - sun.az*1.
        return(theta, phi)
    
###############################################################################
    
#%%load data
def loadSunData(year, name, month = None, part= 'full', path = "/homedata/ngourgue/Images"):
    """
    Load sun data.

    Parameters
    ----------
    year : int
        Year to exact data.
    name : string
        Name of file.
    month : int, optional
        Month to extract data. The default is None.
    part : string, optional
        Possible : beg, end, full, random. Value for separate data. 
        The default is 'full'.
    path : string, optional
        Path where are save file. The default is "/home/ngourgue/Images".

    Returns
    -------
    XvSoleil : array
        Matrix 4*nb data. Position of sun in real world. Ephem projet in X, Y, Z, T.
    
    xSun : array
        Matrix 3*nb data. Position of sun in image. X, Y.
        
    indic : array, optional
        Vector nb data. Indice number for fit dataset. Default no define.

    """
    
    if month is None:
        sunData = pd.read_csv(os.path.join(path, 'SIRTA', "%04d"%year,"%04d%s.csv"%(year,name)))
    else:
        sunData = pd.read_csv(os.path.join(path, 'SIRTA', "%04d"%year,"%02d"%month, 
                                           "%04d%02dsun.csv"%(year, month)))
    #solar params
    # lamda = 1
    t = np.zeros(3)
    R = np.eye(3) 
    thetaSoleil = sunData["theta"]
    phiSoleil   = sunData["phi"]
    XvSoleil = Spherical2Cartesian(R, t, thetaSoleil, phiSoleil)
    xs = sunData["x"]
    ys = sunData["y"]
    xSun = np.array([xs, ys, np.ones([len(xs)])])
    
    Parts = ['beg', 'end', 'full', 'random']
    if not part in Parts:
        return None, None
    
    if part == "beg":
        XvSoleil = XvSoleil[:,:int(XvSoleil.shape[1]/2)]
        xSun    = xSun[:,:XvSoleil.shape[1]]
    elif part == 'end':
        XvSoleil = XvSoleil[:,int(XvSoleil.shape[1]/2):]
        xSun    = xSun[:,-XvSoleil.shape[1]:]
    elif part == 'full':
        pass
    elif part == 'random':
        random.seed(10)
        indic = random.sample(list(np.arange(0, XvSoleil.shape[1], 1)), int(XvSoleil.shape[1]/2))
        return XvSoleil, xSun, indic

    return XvSoleil, xSun

###############################################################################

def loadContrailData(year, name, month = None, part = None, path = "/homedata/ngourgue/Images/"):
    """
    Load contrail data.

    Parameters
    ----------
    year : int
        Year to exact data.
    name : string
        Name of file.
    month : int, optional
        Month to extract data. The default is None.
    part : string, optional
        Possible : random. Value for separate data. The default is 'None'.
    path : string, optional
        Path where are save file. The default is "/home/ngourgue/Images".

    Returns
    -------
    XvAvion : array
        Matrix 4*nb data. Position of plane in real world. Ephem projet in X, Y, Z, T.
    
    xPlane : array
        Matrix 2*nb data. Equation on contrail line in image. X, Y.
        
    indice : array, optional
        Vector nb data. Indice number for fit dataset. Default no define.

    """
    #extract csv file
    planeData = pd.read_csv(os.path.join(path, 'SIRTA', "%04d"%year,"%04d%s.csv"%(year,name)))
    
    #extract plane value
    XvAvion = planeData[['X', 'Y', 'Z']].values
    XvAvion = np.append(XvAvion, np.ones([XvAvion.shape[0],1]), axis = 1)
    XvAvion = np.transpose(XvAvion)
    
    #extract contrail line
    xPlane = planeData[['theta', 'dist']].values
    
    if part == 'random':
        random.seed(5)
        indice = random.sample(list(np.arange(0, XvAvion.shape[1], 1)), int(XvAvion.shape[1]/2))
        return XvAvion, xPlane, indice
        
    return XvAvion, xPlane    

###############################################################################

def separateData(XvSoleil, xSun, indic):
    """
    To split data in fit and test.

    Parameters
    ----------
    XvSoleil : array
        Input data.
    xSun : array
        input data.
    indic : array
        indice to split.

    Returns
    -------
    XvSoleilFit : array
        Data output to fit for XvSoleil.
    XvSoleilTest : array
        Data output to test for XvSoleil.
    xSunFit : array
        Data output to fit for xSun.
    xSunTest : array
        Daat output to test for xSun.

    """
    XvSoleilFit = XvSoleil[:, indic]
    if xSun.shape[0] == 2 or xSun.shape[0] == 3:
        xSunFit = xSun[:,indic]
    elif xSun.shape[1] == 2 or xSun.shape[1] == 3:
        xSunFit = xSun[indic, :]
    else:
        print('no')
    
    indicTest = set(np.arange(XvSoleil.shape[1])).difference(set(indic))
    indicTest = list(indicTest)
    XvSoleilTest = XvSoleil[:, indicTest]
    
    if xSun.shape[0] == 2 or xSun.shape[0] == 3:
        xSunTest = xSun[:, indicTest]
    elif xSun.shape[1] == 2 or xSun.shape[1] == 3:
        xSunTest = xSun[indicTest, :]
    else:
        print('no')
    
    
    return  XvSoleilFit, XvSoleilTest, xSunFit, xSunTest

###############################################################################

#%% évaluation to create an object to create model.

class modelCal():
    
    def __init__(self, modelName):
        self.name = modelName
        self.method = 'Cardan'
        ValueError("you need to implement this method")
    
    def initR(self):
        """
        Create rotation Matrix with 3 angles.
    
        Parameters
        ----------
        w : array
            Vector size 3. 3 angles to rotation matrix.
    
        Returns
        -------
        R : array
            Matrix size 3*3 rotation matrix.
    
        """
        R = np.zeros([3,3])
        
        if self.method == 'Cardan':
            R[0,0] = np.cos(self.w[0])*np.cos(self.w[1])
            R[1,0] = np.sin(self.w[0])*np.cos(self.w[1])
            R[2,0] =-np.sin(self.w[1])
            R[0,1] = np.cos(self.w[0])*np.sin(self.w[1])*np.sin(self.w[2]) - np.sin(self.w[0])*np.cos(self.w[2])
            R[1,1] = np.sin(self.w[0])*np.sin(self.w[1])*np.sin(self.w[2]) + np.cos(self.w[0])*np.cos(self.w[2])
            R[2,1] = np.cos(self.w[1])*np.sin(self.w[2])
            R[0,2] = np.cos(self.w[0])*np.sin(self.w[1])*np.cos(self.w[2]) + np.sin(self.w[0])*np.sin(self.w[2])
            R[1,2] = np.sin(self.w[0])*np.sin(self.w[1])*np.cos(self.w[2]) - np.cos(self.w[0])*np.sin(self.w[2])
            R[2,2] = np.cos(self.w[1])*np.cos(self.w[2])
            
        elif self.method == 'Euler':
            R[0, 0] = np.cos(self.w[2])*np.cos(self.w[0]) - np.sin(self.w[2])*np.cos(self.w[1])*np.sin(self.w[0])
            R[1, 0] = np.sin(self.w[2])*np.cos(self.w[0]) + np.cos(self.w[2])*np.cos(self.w[1])*np.sin(self.w[0])
            R[2, 0] = np.sin(self.w[1])*np.sin(self.w[0])
            
            R[0, 1] =-np.cos(self.w[2])*np.sin(self.w[1]) - np.sin(self.w[2])*np.cos(self.w[1])*np.cos(self.w[0])
            R[1, 1] =-np.sin(self.w[2])*np.sin(self.w[0]) + np.cos(self.w[2])*np.cos(self.w[1])*np.cos(self.w[0])
            R[2, 1] = np.sin(self.w[1])*np.cos(self.w[0])
            
            R[0, 2] = np.sin(self.w[2])*np.sin(self.w[1])
            R[1, 2] =-np.cos(self.w[2])*np.sin(self.w[1])
            R[2, 2] = np.cos(self.w[1])
            
        elif self.method == 'Simon':
            Rx = initRi('x', self.w[0])
            Ry = initRi('y', self.w[1])
            Rz = initRi('z', self.w[2])
            return Rx, Ry, Rz
        
        elif self.method == 'Nico':
            Rx = initRi('x', self.w[0])
            Ry = initRi('y', self.w[1])
            Rz = initRi('z', self.w[2])
            Rzy = np.matmul(Rz, Ry)
            Rzyx = np.matmul(Rzy, Rx)
            return Rzyx
        
        return R
    
    def model(self):
        ValueError("you need to implement this method")
        
    def normaliseBeta(self):
        ValueError("you need to implement this method")
    
    def unnormaliseBeta(self):
        ValueError("you need to implement this method")
        
    def readParams(self):
        ValueError("you need to implement this method")
        
    def saveParams(self):
        ValueError("you need to implement this method")
        
    def setParams(self, w):
        self.w = w
        
    def getParams(self):
        return self.w
    
    def useModel(self):
         ValueError("you need to implement this method")
         
###############################################################################

#%% initR
def initR (w, method = 'Cardan'):
    """
    Create rotation Matrix with 3 angles.

    Parameters
    ----------
    w : array
        Vector size 3. 3 angles to rotation matrix.
    axes : array
        Vector size 3. str to axes rotation.

    Returns
    -------
    R : array
        Matrix size 3*3 rotation matrix.

    """
    # OLD = Rz(w0)Ry(w1)Rx(w2)
    # R = np.ones([3,3])
    # R[0,0] = np.cos(w[0])*np.cos(w[1])
    # R[1,0] = np.sin(w[0])*np.cos(w[1])
    # R[2,0] =-np.sin(w[1])
    
    # R[0,1] = np.cos(w[0])*np.sin(w[1])*np.sin(w[2]) - np.sin(w[0])*np.cos(w[2])
    # R[1,1] = np.sin(w[0])*np.sin(w[1])*np.sin(w[2]) + np.cos(w[0])*np.cos(w[2])
    # R[2,1] = np.cos(w[1])*np.sin(w[2])
    
    # R[0,2] = np.cos(w[0])*np.sin(w[1])*np.cos(w[2]) + np.sin(w[0])*np.sin(w[2])
    # R[1,2] = np.sin(w[0])*np.sin(w[1])*np.cos(w[2]) - np.cos(w[0])*np.sin(w[2])
    # R[2,2] = np.cos(w[1])*np.cos(w[2])
    
    #Rx(w[0])*Ry(w[1])*Rz(w[2]) ->
    # R = np.ones([3,3])
    # R[0, 0] = np.cos(w[1])*np.cos(w[2])
    # R[1, 0] =-np.sin(w[0])*np.sin(w[1])*np.cos(w[2]) + np.cos(w[0])*np.sin(w[2])
    # R[2, 0] = np.cos(w[0])*np.sin(w[1])*np.cos(w[2]) + np.sin(w[0])*np.sin(w[2])
    
    # R[0, 1] =-np.cos(w[0])*np.sin(w[2])
    # R[1, 1] = np.sin(w[0])*np.sin(w[1])*np.sin(w[2]) + np.cos(w[0])*np.cos(w[2])
    # R[2, 1] =-np.cos(w[0])*np.sin(w[1])*np.sin(w[2]) + np.sin(w[0])*np.cos(w[2])
    
    # R[0, 2] =-np.sin(w[1])
    # R[1, 2] = np.sin(w[0])*np.cos(w[1])
    # R[2, 2] = np.cos(w[0])*np.cos(w[1])
    
    #Rz(w[0])*Ry(w[1])*Rx(w[2]) -> celui la avant
    if method == 'Cardan':
        R = np.ones([3,3])
        R[0, 0] = np.cos(w[0])*np.cos(w[1])
        R[1, 0] = np.sin(w[0])*np.cos(w[1])
        R[2, 0] =-np.sin(w[1])
        
        R[0, 1] =-np.sin(w[0])*np.cos(w[2]) + np.cos(w[0])*np.sin(w[1])*np.sin(w[2])
        R[1, 1] = np.cos(w[0])*np.cos(w[2]) + np.sin(w[0])*np.sin(w[1])*np.sin(w[2])
        R[2, 1] = np.cos(w[1])*np.sin(w[2])
        
        R[0, 2] = np.sin(w[0])*np.sin(w[1]) + np.cos(w[0])*np.sin(w[1])*np.cos(w[2])
        R[1, 2] =-np.cos(w[0])*np.sin(w[2]) + np.sin(w[0])*np.sin(w[1])*np.cos(w[2])
        R[2, 2] = np.cos(w[1])*np.cos(w[2])
    
    #Rz(w[0])*Rx(w[1])*Rz(w[2]) ->
    elif method == 'Euler': 
        R = np.ones([3,3])
        R[0, 0] = np.cos(w[2])*np.cos(w[0]) - np.sin(w[2])*np.cos(w[1])*np.sin(w[0])
        R[1, 0] = np.sin(w[2])*np.cos(w[0]) + np.cos(w[2])*np.cos(w[1])*np.sin(w[0])
        R[2, 0] = np.sin(w[1])*np.sin(w[0])
        
        R[0, 1] =-np.cos(w[2])*np.sin(w[1]) - np.sin(w[2])*np.cos(w[1])*np.cos(w[0])
        R[1, 1] =-np.sin(w[2])*np.sin(w[0]) + np.cos(w[2])*np.cos(w[1])*np.cos(w[0])
        R[2, 1] = np.sin(w[1])*np.cos(w[0])
        
        R[0, 2] = np.sin(w[2])*np.sin(w[1])
        R[1, 2] =-np.cos(w[2])*np.sin(w[1])
        R[2, 2] = np.cos(w[1])
        
    elif method == 'Simon':
         # (u',v',w') = ( cos(theta3)  -sin(theta3)   0   ) *( cos(theta2)  0 sin(theta2) ) * (  1      0              0          ) * (u) 
         #              ( sin(theta3)   cos(theta3)   0   )  (      0       1     0       )   (  0  cos(theta3)  -sin(theta3)     )   (v) 
         #              (   0              0          1   )  ( -sin(theta2) 0 cos(theta2) )   (  0  sin(theta3)   cos(theta3)   0 )   (w)    
        
        R1 = initRi('x', w[0])
        R2 = initRi('y', w[1])
        R3 = initRi('z', w[2])
        return R1, R2, R3
    
        # R12 = applyRi(R1, R2)
        # R123 = applyRi(R12, R3)
    
    return R

###############################################################################

def initRi(ax, theta):
    """
    Create rotation Matrix with 3 angles.

    Parameters
    ----------
    ax : string
        Axes to apply rotation.
    theta : float
        Angle in radian to apply rotation.

    Returns
    -------
    R : array
        Matrix size 3*3 rotation matrix.

    """
    R = np.zeros([3,3])
    if ax == 'x':
        R[0, 0] = 1
        R[1, 1] = np.cos(theta)
        R[1, 2] = -np.sin(theta)
        R[2, 1] = np.sin(theta)
        R[2, 2] = np.cos(theta)
    elif ax =='y':
        R[0, 0] = np.cos(theta)
        R[0, 2] = np.sin(theta)
        R[1, 1] = 1
        R[2, 0] = -np.sin(theta)
        R[2, 2] = np.cos(theta)
    elif ax == 'z':
        R[0, 0] = np.cos(theta)
        R[0, 1] = -np.sin(theta)
        R[1, 0] = np.sin(theta)
        R[1, 1] = np.cos(theta)
        R[2, 2] = 1
    else:
        raise ValueError('ax is not recocnize. possible value : x, y, z.\n ax :, '+ax)
    
    return R
    
###############################################################################
        
#%% readCalParams
def readCalParams(site = "SIRTA", method = "csv", path = None):
    """Default path is auto-detected relative to this file."""
    if path is None:
        # Auto-detect path relative to this file
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(os.path.dirname(current_dir), 'params.csv')
    
    return _readCalParams_impl(site, method, path)

def _readCalParams_impl(site = "SIRTA", method = "csv", path = "/home/ngourgue/climavion/params.csv"):
    """
    Read Parameters to project object in image.

    Parameters
    ----------
    method : string, optional
        Type of file. The default is "csv".
    path : string, optional
        Path of file. The default is "/home/ngourgue/climavion/params.csv".

    Returns
    -------
    array
        Tuple of array to export parameters.

    """
    if method == "csv":
        df = pd.read_csv(path)
        line = df[df['site'] == site]
        beta = line.values[0][1:]
        if len(beta) == 25:
            alphaX = beta[0][0]
            alphaY = beta[1][0]
            s      = beta[2][0]
            xo     = beta[3:5].reshape(2,)
            w      = beta[5:8].reshape(3,)
            t      = beta[8:11].reshape(3,)
            k      = beta[11:19].reshape(8,)
            p      = beta[19:23].reshape(4,)
            return alphaX, alphaY, s, xo, w, t, k, p
        elif len(beta) == 14:
            b     = beta[:5]
            x0    = beta[5]
            y0    = beta[6]
            theta = beta[7:10]
            K1    = beta[10]
            phi   = beta[11]
            return b, x0, y0, theta, K1, phi
    return(False)

###############################################################################

def normaliseBeta(beta):
    """
    Normalise vector.

    Parameters
    ----------
    beta : array
        Vector input.
    model : string, optional
        Type of model. The default is 'M1'.

    Returns
    -------
    beta1 : array
        Output vector with normalisation.

    """
    if len(beta) == 23:
        beta1 = np.zeros((23), dtype=np.float128)
        beta1[0]     = NormalisationCal(beta[0],     'alphaX', normalise= True)
        beta1[1]     = NormalisationCal(beta[1],     'alphaY', normalise= True)
        beta1[2]     = NormalisationCal(beta[2],     's',      normalise= True)
        beta1[3]     = NormalisationCal(beta[3],     'xo',     normalise= True)
        beta1[4]     = NormalisationCal(beta[4],     'yo',     normalise= True)
        beta1[5:8]   = NormalisationCal(beta[5:8],   'w',      normalise= True)
        beta1[8:11]  = NormalisationCal(beta[8:11],  't',      normalise= True)
        beta1[11:19] = NormalisationCal(beta[11:19], 'k',      normalise= True)
        beta1[19]    = NormalisationCal(beta[19],   'p1',      normalise= True)
        beta1[20]    = NormalisationCal(beta[20],   'p2',      normalise= True)
        beta1[21]    = NormalisationCal(beta[21],   'p3',      normalise= True)
        beta1[22]    = NormalisationCal(beta[22],   'p4',      normalise= True)
    elif len(beta) == 12:
        beta1 = np.zeros((12), dtype=np.float128)
        beta1[:5]   = NormalisationCal(beta[:5],   'k',  normalise= True)
        beta1[5]    = NormalisationCal(beta[5],    'xo', normalise= True)
        beta1[6]    = NormalisationCal(beta[6],    'yo', normalise= True)
        beta1[7:10] = NormalisationCal(beta[7:10], 'w',  normalise= True)
        beta1[10]   = NormalisationCal(beta[10],   'p1', normalise= True)
        beta1[11]   = NormalisationCal(beta[11],   'w',  normalise= True)
    return beta1

###############################################################################

def UnnormaliseBeta(beta):
    """
    Unnormalise vector

    Parameters
    ----------
    beta : array
        Apply an unnormalisation.
    model : string, optional
        Type of model. The default is 'M1'.

    Returns
    -------
    beta1 : array
        Output vector with unnormalisation.

    """
    if len(beta) == 23:
        beta1=np.zeros((23), dtype = np.float128)
        beta1[0]     = NormalisationCal(beta[0],     'alphaX', normalise= False)
        beta1[1]     = NormalisationCal(beta[1],     'alphaY', normalise= False)
        beta1[2]     = NormalisationCal(beta[2],     's',      normalise= False)
        beta1[3]     = NormalisationCal(beta[3],     'xo',     normalise= False)
        beta1[4]     = NormalisationCal(beta[4],     'yo',     normalise= False)
        beta1[5:8]   = NormalisationCal(beta[5:8],   'w',      normalise= False)
        beta1[8:11]  = NormalisationCal(beta[8:11],  't',      normalise= False)
        beta1[11:19] = NormalisationCal(beta[11:19], 'k',      normalise= False)
        beta1[19]   = NormalisationCal(beta[19],   'p1',      normalise= False)
        beta1[20]   = NormalisationCal(beta[20],   'p2',      normalise= False)
        beta1[21]   = NormalisationCal(beta[21],   'p3',      normalise= False)
        beta1[22]   = NormalisationCal(beta[22],   'p4',      normalise= False)
    elif len(beta) == 12:
        beta1 = np.zeros((12), dtype = np.float128)
        beta1[:5]   = NormalisationCal(beta[:5],   'k',  normalise= False)
        beta1[5]    = NormalisationCal(beta[5],    'xo', normalise= False)
        beta1[6]    = NormalisationCal(beta[6],    'yo', normalise= False)
        beta1[7:10] = NormalisationCal(beta[7:10], 'w',  normalise= False)
        beta1[10]   = NormalisationCal(beta[10],   'p1', normalise= False)
        beta1[11]   = NormalisationCal(beta[11],   'w',  normalise= False)
    return beta1

###############################################################################

#%% reversZoom
def reversZoom (iix, iiy, szamax= 60, xmax =901, xRGBc = 350, yRGBc= 350):
    """
    Calcul pixel coordonate in original image.

    Parameters
    ----------
    iix : int
        X coordonate in zoom image.
    iiy : int
        Y coordonate in zoom image.
    szamax : float, optional
        Angle in degree to cut the image. The default is 60.
    xmax : int, optional
        Size of zoom image. The default is 901.
    xRGBc : int, optional
        X center coordonate in original image. The default is 350.
    yRGBc : int, optional
        Y center coordonate in original image. The default is 350.

    Returns
    -------
    None.

    """
    szamaxrad = np.radians(szamax)
    xRGB = xRGBc*2
    yRGB = yRGBc*2
    #reverse calculate
    ddx = iix - xRGBc
    ddy = iiy - yRGBc
    theta  = (np.pi/xRGB)*(ddx**2+ddy**2)**0.5
    cosalpha = (np.pi/theta)*(ddx/xRGB)
    sinalpha = (np.pi/theta)*(ddy/yRGB)
    dr = np.tan(theta)/np.tan(szamaxrad)*(xmax/2)
    dx = cosalpha*dr
    dy = sinalpha*dr
    ix = dx + xmax/2
    iy = dy + xmax/2
    if type(ix) == float:
        return(int(ix), int(iy))
    else:
        return(np.array([ix, iy], dtype = np.int16))
    
###############################################################################

def applyZoom(ix, iy, szamax= 60, xmax =901, xRGBc = 350, yRGBc= 350):
    """
    Calcul pixel coordonate in zoom image.

    Parameters
    ----------
    iix : int
        X coordonate in zoom image.
    iiy : int
        Y coordonate in zoom image.
    szamax : float, optional
        Angle in degree to cut the image. The default is 60.
    xmax : int, optional
        Size of zoom image. The default is 901.
    xRGBc : int, optional
        X center coordonate in original image. The default is 350.
    yRGBc : int, optional
        Y center coordonate in original image. The default is 350.

    Returns
    -------
    None.

    """
    szamaxrad = np.radians(szamax)
    dx=ix-xmax/2
    dy=iy-xmax/2
    dr=(dx**2.0+dy**2.0)**0.5
    cosalpha=dx/dr
    sinalpha=dy/dr
    #regridding to a flat map
    theta=np.arctan(dr*np.tan(szamaxrad)*2./xmax)
    #--find pixels
    iix=xRGBc+theta/np.pi*xRGBc*2*cosalpha
    iiy=yRGBc+theta/np.pi*yRGBc*2*sinalpha
            
    if type(iix) == float:
        return(int(iix), int(iiy))
    else:
        return(np.array([iix, iiy], dtype = np.float16))
    
###############################################################################

def projCoord(ix, iy, image, szamax = 60, xmax = 901, zoom = True, imShow = True):
    """
    Coord image Ori vers image zoom.

    Parameters
    ----------
    ix : int
        Coord big axe. Horizontal with imshow
    iy : int
        Coord small axe. Vertical with imshow
    image : TYPE
        DESCRIPTION.
    szamax : TYPE, optional
        DESCRIPTION. The default is 60.
    xmax : TYPE, optional
        DESCRIPTION. The default is 901.
    xRGBc : TYPE, optional
        DESCRIPTION. The default is 350.
    yRGBc : TYPE, optional
        DESCRIPTION. The default is 350.

    Returns
    -------
    int
        DESCRIPTION.
    int
        DESCRIPTION.

    """
    imageInd = np.zeros_like(image, dtype=np.uint16)
    szamaxrad = np.deg2rad(szamax)
    for i in range(image.shape[0]):
        imageInd[i, :, 0] = i
    for j in range(image.shape[1]):
        imageInd[:, j, 1] = j
    imageInd[:, :, 2] = 1
    #crop 
    # ixc = ix - 164
    # iyc = iy - 68
    output = zoom_wihout_loop(imageInd, xmax = xmax, szamaxrad= szamaxrad, xIm = 678, yIm = 678,
                              xImc=407, yImc=503, verbose = [''])
    
    if imShow == False:
        if zoom == True:
            ixz = np.where(output[:, int(output.shape[1]/2), 0] == ix)[0].mean()
            iyz = np.where(output[int(output.shape[0]/2), :, 1] == iy)[0].mean()    
            #inv big axes
            iyz = 900-iyz
            return ixz, iyz
        
        else:
            #inv big axes
            iyInv = 900 - iy
            ixOri, iyOri, _ = output[int(ix), int(iyInv), :]
            return ixOri, iyOri

    else:
        if zoom == True:
            iyz = np.where(output[:, int(output.shape[1]/2), 0] == iy)[0].mean()
            ixz = np.where(output[int(output.shape[0]/2), :, 1] == ix)[0].mean()    
            #inv big axes
            ixz = 900-ixz
            return ixz, iyz
        
        else:
            #inv big axes
            ixInv = 900 - ix
            iyOri, ixOri, _ = output[int(iy), int(ixInv), :]
            return ixOri, iyOri

###############################################################################

def projCoord2(ix, iy, szamax = 60, xmax = 901, zoom = True, imShow = True):
    """
    

    Parameters
    ----------
    ix : int or float
        DESCRIPTION.
    iy : int or float
        DESCRIPTION.
    szamax : int, optional
        DESCRIPTION. The default is 60.
    xmax : int, optional
        DESCRIPTION. The default is 901.
    zoom : bool, optional
        True if input image is zoom image. False if input image is ori image. 
        The default is True.
    imShow : bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    if imShow == True:
        if zoom == True:
            #crop
            ixc = ix - 164
            iyc = iy - 68
            #zoom
            ixZoom, iyZoom = reversZoom(ixc, iyc, 60, 901, 339, 339)
            #inv
            ixInv = 900-ixZoom
            return ixInv, iyZoom
        
        else:
            #inv
            ixInv = 900 - ix
            #unZoom
            ixZoom, iyZoom = applyZoom(ixInv, iy, 60, 901, 339, 339)
            #uncrop
            ixUc = ixZoom + 164
            iyUc = iyZoom + 68
            return ixUc, iyUc
    else:
       if zoom == True:
           #crop
           iyc = iy - 164
           ixc = ix - 68
           #zoom
           iyZoom, ixZoom = reversZoom(iyc, ixc, 60, 901, 339, 339)
           #inv
           iyInv = 900-iyZoom
           return ixZoom, iyInv
       
       else:
           #inv
           iyInv = 900 - iy
           #unZoom
           iyZoom, ixZoom = applyZoom(iyInv, ix, 60, 901, 339, 339)
           #uncrop
           iyUc = iyZoom + 164
           ixUc = ixZoom + 68
           return iyUc, ixUc     
            

#%% zoom_image
def zoom_with_loop(image, xmax, szamaxrad, xIm, yIm, xImc, yImc, lin = False, verbose= ['']):
    """
    make zoom with double loop for.
    it's costly in time but it's the original transformation.

    Parameters
    ----------
    image : Array
        Input image.
    xmax : Int
        Size of output image.
    szamaxrad : Float
        Angle in radian to zoom.
    xIm : Int
        Width util of input image.
    yIm : Int
        Height util of input image.
    xImc : Int
        X center of input image.
    yImc : Int
        Y center of input image.
    lin : Bool, optional 
        The default is False.
    verbose : List of string, optional 
        The default is [''].

    Returns
    -------
    imz : Array
        Image with zoom.

    """
    
    # start_time = time.time()
    
    imz = np.zeros([xmax, xmax, 3], dtype=np.uint8)
    for ix in range(0,xmax):
        for iy in range(0,xmax):
            dx=ix-int(xmax/2)
            dy=iy-int(xmax/2)
            dr=(dx**2.0+dy**2.0)**0.5
            if dr < xmax/2.: 
                if dr != 0: 
                    cosalpha=dx/dr
                    sinalpha=dy/dr
                else: 
                    cosalpha=1.0
                    sinalpha=0.0
                if lin:
                    #regridding to a hemispheric map
                    theta=dr*szamaxrad*2./float(xmax)
                else:
                    #regridding to a flat map
                    theta=np.arctan(dr*np.tan(szamaxrad)*2./float(xmax))
                #--find pixels
                iix=int(round(xImc+theta/np.pi*float(xIm)*cosalpha))
                iiy=int(round(yImc+theta/np.pi*float(yIm)*sinalpha))
                #--regrid
                imz[ix,iy,:] = np.uint8(image[iix,iiy,:])
                
    # dtime = time.time() - start_time
    # if 'all' in verbose or 'time' in verbose:
    #     print("time for loop zoom : %.2f"%dtime)
        
    return imz


###############################################################################

def zoom_wihout_loop(image, xmax, szamaxrad, xIm, yIm, xImc, yImc, verbose = ['']):
    """
    zoom on image without loop. it's near to instantane and the result is same
    than zoom with loop.

    Parameters
    ----------
    image : Array
        Input image.
    xmax : Int
        Size width and height of output image.
    szamaxrad : Float
        Angle in radian about zoom.
    xIm : Int
        Width until to input image.
    yIm : Int
        Height until to input image.
    xImc : Int
        Width center to input image.
    yImc : Int
        Height center to input image.
    verbose : , optional
        The default is [''].

    Returns
    -------
    imageZoom : Array
        Image with zoom.

    """
    
    # start_time = time.time()
    
    image1 = cp.deepcopy(image)
    #create matrix
    matImage = np.zeros([xmax, xmax, 8])
    
    #create coordoante x,y
    #discusion if int is necessary
    for i in range(xmax):
        matImage[i,:, 0] = i-int(xmax/2)
        matImage[:,i, 1] = i-int(xmax/2)
        
    #create radius
    matImage[:,:, 2] = (matImage[:,:, 0]**2.0+matImage[:,:,1]**2.0)**0.5
    
    #create alpha
    #cas radius >0
    indicOmega = np.where(matImage[:,:,2] > 0)
    #cas radius < xmax/2
    indicAlpha = np.where(matImage[indicOmega[0], indicOmega[1],2] < xmax/2)[0]
    matImage[indicOmega[0][indicAlpha], indicOmega[1][indicAlpha], 3] = \
    matImage[indicOmega[0][indicAlpha], indicOmega[1][indicAlpha], 0]/matImage[indicOmega[0][indicAlpha], 
                                                                        indicOmega[1][indicAlpha], 2]
    matImage[indicOmega[0][indicAlpha], indicOmega[1][indicAlpha], 4] = \
    matImage[indicOmega[0][indicAlpha], indicOmega[1][indicAlpha], 1]/matImage[indicOmega[0][indicAlpha], 
                                                                        indicOmega[1][indicAlpha], 2]
    #cas radius = 0
    indicOmicron = np.where(matImage[:,:,2] == 0)
    matImage[indicOmicron[0], indicOmicron[1], 3] = 1
    matImage[indicOmicron[0], indicOmicron[1], 4] = 0
    
    #delete radius > xmax/2
    indicBeta = np.where(matImage[:,:,2] > xmax/2)
    matImage[indicBeta[0], indicBeta[1], 3:] = 0
    
    #create theta, theta max = np.pi/3
    matImage[:,:,5] = np.arctan(matImage[:,:,2]*np.tan(szamaxrad)*2./float(xmax))
    matImage[indicBeta[0], indicBeta[1], 3:] = 0
    
    matImage[:,:,6] = np.round(xImc+matImage[:,:, 5]*float(xIm)*matImage[:,:, 3]/np.pi)
    matImage[:,:,7] = np.round(yImc+matImage[:,:, 5]*float(yIm)*matImage[:,:, 4]/np.pi)
    matImage[indicBeta[0], indicBeta[1], 3:] = 0
    matCoord = np.array(matImage[:,:,6:], dtype = np.int16)
    # #set zeros for indice >xmax/2
    # indicBeta = np.where(mat1[:,:,2] > xmax/2)
    # mat2[indicBeta[0], indicBeta[1]] = 0
    
    if len(image1.shape) == 3:
        if image1.dtype == 'uint8':
            imageZoom = np.uint8(image1[matCoord[:,:,0], matCoord[:,:,1],:])
        elif image1.dtype == 'uint16':
            imageZoom = np.uint16(image1[matCoord[:,:,0], matCoord[:,:,1],:])

    elif len(image1.shape) == 2:
        if image1.dtype == 'uint8':
            imageZoom = np.uint8(image1[matCoord[:,:,0], matCoord[:,:,1]])
        elif image1.dtype == 'uint16':
            imageZoom = np.uint16(image1[matCoord[:,:,0], matCoord[:,:,1]])
        
    # dtime = time.time() - start_time
    # if 'all' in verbose or 'time' in verbose:
    #     print('time for zoom witout loop : %.2f'%dtime)
        
    return imageZoom


###############################################################################
# @timeDuration  # Commented out - library not available
def zoom_image (image, xmax = 901, szamax = 60, verbose = [''], with_loop = False):
    """
    zoom on image, resize with xmax and delete perimeter after szamax.
    Zoom restaure straight line to have plane trajectories as straight line.

    Parameters
    ----------
    image : array,
        Input image.
    xmax : int, optinal
        Size new image. The default is 901
    szamax : int, optinal
        Angle to filter image. The default is 60.
    verbose : list of stirng, optional
        List of element to print . The default is [''].
    with_loop : bool, optional
        Calcul resize with loop or with matrix. The default is False.

    Raises
    ------
    ValueError
        Image size is not correct.

    Returns
    -------
    imageZoom : array
        Output image.

    """
    
    # start_time=time.time()
    
    szamaxrad = np.radians(szamax)
    imageZoom = np.zeros((xmax, xmax, 3), np.uint8)
    
    #extract size and dimension input
    if len(image.shape)==3:
        xImo, yImo, nbcolor = image.shape
    elif len(image.shape)==2:
        xImo, yImo = image.shape

    #determine  if image is cropped
    if xImo == 768 and yImo == 1024:
        #we have original image
        croppedParam = croppedSize()
        xImmax, xImmin = croppedParam.xmax, croppedParam.xmin
        xIm = xImmax-xImmin
        yImmax, yImmin = croppedParam.ymax, croppedParam.ymin
        yIm = yImmax-yImmin
    elif xImo == 678 and yImo == 678:
        #we have cropped image
        xImmax, xImmin = 678, 0
        xIm = xImmax-xImmin
        yImmax, yImmin = 678, 0
        yIm = yImmax-yImmin
    elif yImo == 1280 and xImo == 960:
        croppedParam = croppedSize('Orsay')
        xImmax, xImmin = croppedParam.xmax, croppedParam.xmin
        xIm = xImmax-xImmin
        yImmax, yImmin = croppedParam.ymax, croppedParam.ymin
        yIm = yImmax-yImmin
    elif xImo == 1280 and yImo == 960:
        croppedParam = croppedSize('Orsay')
        yImmax, yImmin = croppedParam.xmax, croppedParam.xmin
        yIm = yImmax-yImmin
        xImmax, xImmin = croppedParam.ymax, croppedParam.ymin
        xIm = xImmax-xImmin
    else :
        #we have an unknow cas
        # sys.exit('Image size does not exist')
        raise ValueError("image's shape is not correct shpae :"+','.join("%d"%i for i in image.shape))

    #--finding center
    #case same size X and Y
    if xIm == yIm:
        xImc = int(xImmin+xIm/2)
        yImc = int(yImmin+yIm/2)
    #case different size Y and Y
    else:
        sizemax = max(xIm, yIm)
        center = int(sizemax/2)
        # case Saturate min
        if xImmin == 0 or yImmin == 0:
            xImc = xImmax - center
            yImc = yImmax - center
        #case Saturate max
        elif xImmax == xImo or yImmax == yImo:
            xImc = xImmin + center
            yImc = yImmin + center
            
    #zoom without for
    if with_loop is True:
        imageZoom = zoom_with_loop(image, xmax, szamaxrad, xIm, yIm, xImc, yImc, verbose = verbose)
    else :
        imageZoom = zoom_wihout_loop(image, xmax, szamaxrad, xIm, yIm, xImc, yImc, verbose = verbose)

    #--reverse image to get orientation right
    if xImo in [768, 678] and yImo in [1024, 678]:
        if len(image.shape) == 3:
            imageZoom=imageZoom[:,::-1,:]
        elif len(image.shape) == 2:
            imageZoom=imageZoom[:,::-1]
    

    # dtime=time.time()-start_time
    # if 'all' in verbose or 'time' in verbose:
    #     print("duration of zoom image : %.2f s" %(dtime))
    return imageZoom

###############################################################################

#%% croppedSize
class croppedSize:   
    def __init__(self, size = 'SIRTA'):
        if size == 'SIRTA':
            self.xmin = 68
            self.xmax = 746
            self.ymin = 164
            self.ymax = 842
        elif size in ['FRIPON', 'Orsay']:
            self.xmin = 33 # - 15
            self.xmax = 934
            self.ymin = 160
            self.ymax = 1061
            
#%% cropped
def cropped(image):
    """
    cropped image without black strip

    Parameters
    ----------
    image : TYPE array
        DESCRIPTION image to cropped.

    Raises
    ------
    ValueError
        DESCRIPTION shape problem.

    Returns
    -------
    imageCropped : TYPE array
        DESCRIPTION image cropped.

    """
    image1 = cp.deepcopy(image)
    if np.all(image1.shape == (768, 1024, 3)):
        croppedParam = croppedSize('SIRTA')
    elif np.all(image1.shape == (1280, 960, 3)):
        croppedParam = croppedSize('Orsay')
    xsize = croppedParam.xmax - croppedParam.xmin
    ysize = croppedParam.ymax - croppedParam.ymin
    if len(image1.shape) == 3:
        imageCropped = np.zeros([xsize, ysize, image1.shape[2]])
        imageCropped = image1[croppedParam.ymin:croppedParam.ymax, croppedParam.xmin:croppedParam.xmax,:]
        if np.all(image1.shape == [768, 1024, 3]):
            imageCropped = imageCropped[:, ::-1, :]
    elif len(image1.shape) == 2:
        imageCropped = np.zeros([xsize, ysize])
        imageCropped = image1[croppedParam.ymin:croppedParam.ymax, croppedParam.xmin:croppedParam.xmax]
        imageCropped = imageCropped[:, ::-1]
    else :
        raise ValueError('shape of image is not correct')

    return imageCropped        
    
###############################################################################
