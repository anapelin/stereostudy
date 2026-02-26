#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:23:30 2022

@author: ngourgue

Class  Fripon : 
    Model to make calibration. 
    Model has parameters in attribut and method to calculate position.

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
from .baseCalibration import (modelCal, NormalisationCal, reversZoom, UnnormaliseBeta, 
                             Cartesian2Spherical)

from scipy.optimize import minimize
# %% Friponmodel

class Fripon(modelCal):
    """
    Class to use Fripon model.
    
    Attribues
    ---------
    name : string, Fix
        Name of model. value = 'Fripon'.
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
        
    Methods
    -------
    poly 9 : use a 9 degree polynome to radiale distorsion.
    
    polR2cart : convert polar coordonate to cartesiane.
    
    invDistAsym : calcul radiale distorsion.
    
    model : method to project coordonate.
    
    modelNorm : method to project coordonate with normalise parameters.
    
    saveParams : save params in csv file.
    
    plotParams : print each params with his name.
    
    useModel : make projection on image and adapte zoom and resize.
    
    convParam2Beta : get and vector with all parameters.
    
    setBeta : set each parameters with one vector.
    
    setParams : set each paramters.
    
    readParams : read csv file with parameters values.
    
    getParams : get parameters.
    
    convBeta2Params : convert an vector to multiparamters.
    """
    
    
    def __init__(self, site):
        super().__init__('Fripon')
        self.site = site
        self.readParams()

    # %%% model (5 functions)
    
    def poly9(self, r):
        """
        Calcul the radiale correction on radius parameters with angle.
    
        Parameters
        ----------
        r : float
            Radius in pixel.
        b : float
            Radiale angle.
    
        Returns
        -------
        z : float
            Radius with radiale deformation.
    
        """
        z = self.b[0]*r+self.b[1]*r**3+self.b[2]*r**5+self.b[3]*r**7+self.b[4]*r**9
        return z
    
    ###############################################################################
    
    # def poly9Inv(self, r):
    #     """
    #     Calcul the radiale correction on radius parameters with angle.
    
    #     Parameters
    #     ----------
    #     r : float
    #         Radius in pixel.
    #     b : float
    #         Radiale angle.
    
    #     Returns
    #     -------
    #     z : float
    #         Radius with radiale deformation.
    
    #     """
    #     z = self.b[0]*r+\
    #         self.b[1]*r**(1/3)+\
    #         self.b[2]*r**(1/5)+\
    #         self.b[3]*r**(1/7)+\
    #         self.b[4]*r**(1/9)
    #     return z
    
    ###########################################################################
    
    def polR2cart(self, R, theta):
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
        x = R*np.cos(theta)+self.x0
        y = R*np.sin(theta)+self.y0
        return x, y
    
    ###############################################################################
        
    def invDistAsym(self, r, A):
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
        Rold = r/(1+self.K1*np.sin(A+self.phi))
        if self.site == 'Orsay':
            Rold = Rold * 1000
        return Rold
    
    ###############################################################################
    
    def rotation(self, Px, invEqu = True, method = 'old'):
        if method == 'old':
            # self.method = 'Cardan'
            Rot = self.initR()
            if invEqu == True:     
                RotInv = np.linalg.inv(Rot)
                Px1 = np.dot(RotInv, Px)
            else:
                Px1 = np.dot(Rot, Px)
        elif method =='Simon':
            self.method = 'Simon'
            Rx, Ry, Rz = self.initR()
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
            self.method = 'Nico'
            Rot = self.initR()
            if invEqu == True:
                RotInv = np.linalg.inv(Rot)
                Px1 = np.dot(RotInv, Px)
            else:
                Px1 = np.dot(Rot, Px)
            
        else :
            raise ValueError('method is unknow :'+method)
        return Px1
    
    ###############################################################################
    
    def model(self, Px):
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
        # invEqu = True
        if Px.shape[0] == 4:
            Px = Px[:3,:]
        elif Px.shape[0] == 3:
            pass
        else:
            raise ValueError('size on Px in not correct', Px.shape)
        
        if self.site == 'SIRTA':
            Px1 = self.rotation(Px, False, method = 'Nico')
            z, alpha = Cartesian2Spherical(Px1[0], Px1[1], Px1[2], method = 'tan')
        elif self.site == 'Orsay':
            Px1 = self.rotation(Px, True, method = 'Simon')
            z, alpha = Cartesian2Spherical(Px1[0], Px1[1], Px1[2], method = 'Simon')
        rNew = self.poly9(z)
        R = self.invDistAsym(rNew, alpha)
        x, y = self.polR2cart(R, alpha)

        return x, y
    
    ###############################################################################
    
    def modelNorm(self, params, Px):
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
        self.setParams(b, x0, y0, theta, K1, phi)
        x, y = self.model(Px)
        return x, y
    
    ###############################################################################
    
    # %%% function use calibration (9 parameters)
            
    def saveParams (self, method = "csv", path = "/home/ngourgue/climavion/"):
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
        w : array
            Vector about angle to matrix rotation.
        K1 : float
            Ponderation on phase distorsion.
        phi : float
            Phase shift.
        method : string, optional
            Type of file to save. The default is "csv".
        path :string, optional
            Path where to save file with parameters. The default is "/home/ngourgue/climavion/".
    
        Returns
        -------
        bool
            Succes or not to save.
    
        """
        if method == "csv":
            beta  = np.zeros([1,14])
            beta[0,:5]   = self.b
            beta[0,5]    = self.x0
            beta[0,6]    = self.y0 
            beta[0,7:10] = self.w
            beta[0,10]   = self.K1
            beta[0,11]   = self.phi
            df_input = pd.read_csv(os.path.join(path, "params.csv"), index_col=0)
            beta[0,12] = df_input['lat'].loc[self.site]
            beta[0,13] = df_input['lon'].loc[self.site]
            df_input.loc[self.site] = beta
            df_input.to_csv(os.path.join(path, "params.csv"), index=True)
            return True
        else:
            print("paramters can't be save. method not function.")
            return False
    
    ###############################################################################
            
    def plotParams(self):
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
        print("b = %.2f %.2f %.2f %.2f %.2f"%(self.b[0], self.b[1], self.b[2], self.b[3], self.b[4]))
        print("center x0 : %.2f, y0 %.2f"%(self.x0, self.y0))
        print("w angle  alpha : %.5f, beta : %.5f, gamma : %.5f"%(self.w[0], self.w[1], self.w[2]))
        print("K1 : %.5f, phi : %.5f"%(self.K1, self.phi))
    
    ###############################################################################
    
    def useModel(self, XPosition, imageShape= np.array([768,1024, 3]), zoom = True, methodRead = "csv"):
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
        #convert XPosition
        if len(XPosition) == 3:
            if type(XPosition[0]) is int or type(XPosition[0]) is float or type(XPosition[0]) is np.float64:
                XPosition = np.array([XPosition[0], XPosition[1], XPosition[2], 1])
            else:
                XPosition = np.array([XPosition[0], XPosition[1], XPosition[2], np.ones(len(XPosition[0]))])
        #image 3D
        if len(imageShape) == 3:
            #test size image
            #original image
            if np.all(imageShape == np.array([768, 1024, 3])):
                x, y =    self.model(XPosition)
                return(x, y)
            #transpose image
            elif np.all(imageShape == np.array([1024, 768, 3])):
                x, y = self.model(XPosition)
                return(y, x)
            #zoom or transforme image
            elif np.all(imageShape == np.array([901, 901, 3])):
                x, y = self.model(XPosition)
                if self.site == 'SIRTA':
                    x = x-68
                    y = y-182
                    xR, yR= reversZoom(x, y, szamax= 60, xmax =901, xRGBc = 339, yRGBc= 339)
                elif self.site == 'Orsay':
                    x = x-30
                    y = y-109
                    xR, yR= reversZoom(x, y, szamax= 60, xmax =901, xRGBc = 451, yRGBc= 451)
                return(xR, yR)
            #cropped image old cal
            elif  np.all(imageShape == np.array([674, 674, 3])):
                x, y = self.model(XPosition)
                x = x-66
                y = y-161
                return(x, y)
            #cropped image new cal
            elif  np.all(imageShape == np.array([678, 678, 3])):
                x, y = self.model(XPosition)
                x = x-68
                y = y-182
                return(x, y)
            else:
                raise ValueError("image.shape not corresponded to correct value.\n image.shape : %d, %d, %d\n"%(imageShape[0], 
                                 imageShape[1], imageShape[2])+"correct values : \n768, 1024, 3 : original"+\
                                 " image\n1024, 768, 3 : transpose image\n901, 901, 3 : zoom or transforme"+\
                                 " image\n678, 678, 3 : cropped image")
        #image 2D    
        elif len(imageShape) == 2:
            if np.all(imageShape == np.array([768, 1024])):
                x, y =    self.model(XPosition)
                return(x, y)
            #transpose image
            elif np.all(imageShape == np.array([1024, 768])):
                x, y = self.model(XPosition)
                return(y, x)
            #zoom or transforme image
            elif np.all(imageShape == np.array([901, 901])):
                x, y = self.model(XPosition)
                if self.site == 'SIRTA':
                    x = x-68
                    y = y-182
                    xR, yR= reversZoom(x, y, szamax= 60, xmax =901, xRGBc = 339, yRGBc= 339)
                elif self.site == 'Orsay':
                    x = x-30
                    y = y-109
                    xR, yR= reversZoom(x, y, szamax= 60, xmax =901, xRGBc = 451, yRGBc= 451)
                return(xR, yR)
            #cropped image old cal
            elif  np.all(imageShape == np.array([674, 674])):
                x, y = self.model(XPosition)
                x = x-66
                y = y-161
                return(x, y)
            #cropped image new cal
            elif  np.all(imageShape == np.array([678, 678])):
                x, y = self.model(XPosition)
                x = x-68
                y = y-182
                return(x, y)
            elif np.all(imageShape == np.array([960, 1280])):
                x, y = self.model(XPosition)
                return (x, y)
            elif np.all(imageShape == np.array([1280, 960])):
                x, y = self.model(XPosition)
                return (x, y)
            else:
                raise ValueError("image.shape not corresponded to correct value.\n image.shape : %d, %d, %d\n"%(imageShape[0], 
                                 imageShape[1], imageShape[2])+"correct values : \n768, 1024, 3 : original"+\
                                 " image\n1024, 768, 3 : transpose image\n901, 901, 3 : zoom or transforme"+\
                                 " image\n678, 678, 3 : cropped image")
        else:
            raise ValueError("image is not correct. image 2D or 3D is need to argument image.")

    ###############################################################################

    def convParam2Beta(self):
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
        beta[:5] = self.b
        beta[5]  = self.x0
        beta[6]  = self.y0
        beta[7:10] = self.w
        beta[10] = self.K1
        beta[11] = self.phi
        return beta
    
    ###############################################################################
    
    def setBeta (self, beta):
        """
        set params with a vector input.

        Parameters
        ----------
        beta : array
            Input parameters.

        Returns
        -------
        None.

        """
        self.b   = beta[:5]
        self.x0  = beta[5]
        self.y0  = beta[6]
        self.w   = beta[7:10]
        self.K1  = beta[10]
        self.phi = beta[11]
        
    ###############################################################################

    def setParams (self, b, x0, y0, theta, K1, phi):
        """
        To set params.

        Parameters
        ----------
        b : array
            Polynome to module radiale deformation.
        x0 : float
            X coordonate of center on image.
        y0 : float
            Y coordonate of center on image
        w : array
            Vector about angle to matrix rotation.
        K1 : float
            Ponderation on phase distorsion.
        phi : float
            Phase shift.

        Returns
        -------
        None.

        """
        self.b = b
        self.x0 = x0
        self.y0 = y0
        self.w = theta
        self.K1 = K1
        self.phi = phi
        
    ###############################################################################
    
    def readParams (self, method = "csv", path = "/home/ngourgue/climavion/params.csv"):
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
            df = pd.read_csv(path, index_col=0)
            beta = df.loc[self.site].values
            self.b     = beta[:5]
            self.x0    = beta[5]
            self.y0    = beta[6]
            self.w     = beta[7:10]
            self.K1    = beta[10]
            self.phi   = beta[11]
        else:
            raise ValueError('method is not implemented')
    
    ###############################################################################
    
    def getParams (self):
        """
        return Params

        Returns
        -------
        b : array
            Polynome to module radiale deformation.
        x0 : float
            X coordonate of center on image.
        y0 : float
            Y coordonate of center on image
        w : array
            Vector about angle to matrix rotation.
        K1 : float
            Ponderation on phase distorsion.
        phi : float
            Phase shift.

        """
        return self.b, self.x0, self.y0, self.w, self.K1, self.phi
        
    ###############################################################################

    def convBeta2Params(self, beta):
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
              
def Error(a, x, y):
    yCal =  a[0]*x + a[1]*(x**3) + a[2]*(x**5) + a[3]*(x**7) + a[4]*(x**9)
    epsilon = np.sum((yCal-y)**2)
    return epsilon
        
def poly9(a, x):
    yCal =  a[0]*x + a[1]*(x**3) + a[2]*(x**5) + a[3]*(x**7) + a[4]*(x**9)
    return yCal

def poly9bis(x, a, b, c, d, e):
    yCal =  a*x + b*(x**3) + c*(x**5) + d*(x**7) + e*(x**9)
    return yCal

def poly10bis(x, a, b, c, d, e):
    # yCal =  a*x + b*(x**3) + c*(x**5) + d*(x**7) + e*(x**9) + f*(x**11)
    # yCal = a*(1-np.exp(-(b + c*x + d*x**3)))
    yCal = a*x + b*(x**(1/3)) + c*(x**(1/5)) + d*(x**(1/7)) + e*(x**(1/9))
    return yCal

def convTheta (data):
    for i in range(3):
        theta = data['theta'+str(i+1)].loc['ORSAY']
        theta = float(theta[:-1])
        data['theta'+str(i+1)].loc['ORSAY'] = theta
        
def convPhi(data):
    phi = data['phi'].loc['ORSAY']
    phi = float(phi[:-1])
    data['phi'].loc['ORSAY'] = phi

def convLatLon(data):
    lat = data['lat'].loc['ORSAY']
    lat = float(lat)
    data['lat'].loc['ORSAY'] = lat
    
    lon = data['lon'].loc['ORSAY']
    lon = float(lon)
    data['lon'].loc['ORSAY'] = lon
# %%% cost function
def epsilonModel(params, Px, xSun, model):
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
    
    model.setBeta(params)
    x, y = model.model(Px)
    epsilon = np.sum((x-xSun[0, :])**2+(y-xSun[1,:])**2)
    return epsilon

###############################################################################

def epsilonNorm(params, Px, xSun, model):
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
    params1 = UnnormaliseBeta(params1)
    
    epsilon = epsilonModel(params1, Px, xSun, model)
    return epsilon

###############################################################################

def droiteError(params, XvAvion, xPlane, model, norm= True):
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
        x, y  = model.modelNorm(params, XvAvion)
    else:
        model.setBeta(params)
        x, y = model.model(XvAvion)

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

def mixSunPlane(params, XvSoleil, xSun, XvAvion, xPlane, model, ponderation = 1, norm = True, c1 = False):
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
    errorPlane = droiteError(params1, XvAvion, xPlane, model, norm)
    if norm == True:
        errorSun = epsilonNorm(params1, XvSoleil, xSun, model)
    else :
        errorSun = epsilonModel(params1, XvSoleil, xSun, model)
    errorTot =  errorSun + errorPlane*ponderation
    return errorTot

###############################################################################    

# %%%Move file
def mouveFripon(name, year, month, day):
    #create folder
    if not os.path.isdir('/homedata/'+os.environ['USER']+'/Images/%s/%04d/%02d/%04d%02d%02d'%(
         name, year, month, year, month, day)):
         os.system('mkdir /homedata/'+os.environ['USER']+'/Images/%s/%04d/%02d/%04d%02d%02d'%(
             name, year, month, year, month, day))
    os.system('mv /homedata/'+os.environ['USER']+'/Images/%s/%04d/%02d/*_%04d%02d%02dT* '%(
         name, year, month, year, month, day) +\
         '/homedata/'+os.environ['USER']+'/Images/%s/%04d/%02d/%04d%02d%02d'%(
         name, year, month, year, month, day))
        
###############################################################################
#%% paramters
if __name__ == '__main__':
    theta = np.array([0, 0, 0])
    
    x0  = 383
    y0  = 513
    
    a = np.array([1, 0, 0, 0, 0])
    
    K1 = 2.5*10**(-3)
    phi = 0

    Model = Fripon('ORSAY')
    Model.setParams(a, x0, y0, theta, K1, phi)

    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    
    fileData = "/home/ngourgue/climavion/detection_contrail/calibration/Astrometry results.csv"
    dataFripon = pd.read_csv(fileData)
    length = 1000
    a = dataFripon[['Vx', 'Sx', 'Dx', 'Px', 'Qx']].iloc[2].values
    x = np.linspace(0, 450, length)
    
    y = poly9(a, x/1000)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x/1000, y, 'b-')
    ax.set_title('Lens correction')
    ax.set_ylabel('elevation')
    ax.set_xlabel('Rayon')
    
    
    from scipy.optimize import curve_fit
    #vérif
    popt, pcov = curve_fit(poly9bis, x/1000, y)
    
    diff = popt - a
    
    
    x2 = np.linspace(0, np.max(x)/1000, length)
    invP, pcov = curve_fit(poly10bis, y, x/1000)
    
    invP1, pcov = curve_fit(poly9bis, y, x/1000)

    
    print('original :', a, "\ncalcul :", popt)
    print('inversion :', invP)

    ax.plot(x/1000, poly9bis(x/1000, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, e=%5.3f'%(
        popt[0], popt[1], popt[2], popt[3], popt[4]))
    ax.legend()
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Inversion ')
    ax.set_xlabel('elevation')
    ax.set_ylabel('Rayon')
    ax1.plot(y, x/1000, 'b-')
    #ax1.plot(y, poly10bis(y, *invP), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, e=%5.3f' %(
    #    invP[0], invP[1], invP[2], invP[3], invP[4]))
    ax1.plot(y, poly9bis(y, *invP1), 'g-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, e=%5.3f' %(
        invP1[0], invP1[1], invP1[2], invP1[3], invP1[4]))
    ax1.legend()
    
