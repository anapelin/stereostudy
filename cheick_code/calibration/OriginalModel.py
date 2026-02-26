#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 17:07:33 2021

@author: ngourgue

Class  OrigineModel : 
    Model to make calibration. 
    Model has parameters in attribut and method to calculate position.
    
part 1 : 
    initP : create matrix to transformation.
    convxChapeautox : adapt value to image size. 
    initK : create matrix affine.
    
part 2 :
    go : function to lens distorsion
    rChapeauForward : radiale distorsion
    deltacForward : phasique distorsion
    
Part 3 :
    EuclideanTransformation : Matrix rotation and translation
    Distorsion : combination radiale and phasic distorsion
    AffineTransformation : Transforme x and y value to adapte to image size.

Backward projection :
    invK : inversion of K matrix.
    convxvToxvChapeauTilde : conversion image coordonate to normalise coordonate.
    Cartesian2Polar : Convert catesiane coordonate to polar coordonate.
    
    Need to complet but not use.
    deltacBackward : 
    rChapeauBackward : 
    invGo : 
    p2b: 
    psiBackward :
        
function :
    model : model function 
    fNorm : model function with normalise parameters. use to optimise.
    
        
cost function :
    epsilonModel : cost function base on distance into projection and annotation.
    epsilonNorm : same cost function as epsilonModel but params is normalise.
    droiteError : cost function base on distance into plane pojection and contrail  line.
    mixSunPlane : cost function mixte droiteError and epsilonModel.s
        
        
    
"""

#%% importation
import ephem, datetime, os, copy, random, sys#, time

import numpy  as np
import pandas as pd

from skimage.io        import imread
from skimage.draw      import circle_perimeter
from skimage.feature   import canny
from skimage.transform import hough_circle, hough_circle_peaks

from matplotlib import pyplot as plt

if not '/home/ngourgue/climavion/detection_contrail/calibration' in sys.path:
    sys.path.append('/home/ngourgue/climavion/detection_contrail/calibration')
from .baseCalibration import (Cartesian2Spherical, NormalisationCal, reversZoom, 
                             Spherical2Cartesian, modelCal, normaliseBeta, 
                             UnnormaliseBeta)
# from sympy import symbols, Eq, solve

# %%Forward camera model overview
#2.1.1 Projective transformation camera model

class OrigineModel(modelCal):
    """
    Attribues
    ---------
    name : string
        Name of model to do difference into Fripon and Origine model.
    alphaX : float.
        Horizontale coefficent about image size.
    alphaY : float
        Verticale coefficient about image size.
    s : float
        Link into Horizontale et Verticale axes.
    xo : array
        X and Y coordonate of center.
    w : array
        Vector with 3 values, angle of rotation matrix.
    t : array
        Vector with 3 values, value of translation vector.
    k : array
        Vector with 8 values, value of radiale distorsion .
    p : array
        Vector with 4 values, value of phasique distosion.

    methods
    -------
    initP : create different matrix to transformation.
    
    convxChapeautox : projection normalise value to image size.
    
    initK :create matrix to adapte value to size image.
    
    initR : create rotation matrix.
    
    go : method where it choice lens deformation model.
    
    rChapeauForward : calcul r with radial distorsion.
    
    deltacForward : calcul phase distorsion.
    
    EuclideanTransformation : Integrate rotation and translation transformation.
    
    Distorsion : integrate radiale and  phase distorsion.
    
    AffineTransformation : adapte value to image size.
    
    model : method to calcul coordonate in image.
    
    modelNorm : method to calcul coordonate in image with normalise parameters.
    
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
    
    def __init__(self):
        super().__init__('Origine')        

    # %%%part use (4 functions)
    
    def initP (self, method ="", R=None):
        """
        create matrice to transformation in functon of method.
        affine : need alphaX, alphaY, s, xo, yo. create a 3*3 matrix.
        perspective : create a pespective matrix. 3*4 matrix.
        euclidiean : create an euclidiant project. Rotation matric + Translation vector
    
        Parameters
        ----------
        method : string, optional
            Name of matrix yoi want, value possible affine, perspective, euclidean. The default is "".
        alphaX : float, optional
            Porportion about x axis. The default is None.
        alphaY : float, optional
            Proportion about y axis. The default is None.
        s : float, optional
            Deformation between alphaX, alphaY. The default is None.
        xo : float, optional
            First center coordonate. The default is None.
        yo : float, optional
            Second center coordonate. The default is None.
        R : array, optional
            Matrix 3*3 of rotation. The default is None.
        t : array, optional
            Vector 3 of translation. The default is None.
    
        Returns
        -------
        P : matrix,
            Output matrix.
    
        """
        if method =="affine":
            P = np.zeros([3,3])
            P[2,2] = 1
            if not self.alphaX is None:
                P[0,0] = self.alphaX
            else :
                print("error method affine need alphaX")
                return None
            if not self.alphaY is None:
                P[1,1] = self.alphaY
            else :
                print("error method affine need alphaY")
                return None
            if not self.s is None:
                P[0,1] = self.s
            else:
                print("error method affine need s")
                return None
            if not self.xo is None:
                P[0,2] = self.xo[0]
                P[1,2] = self.xo[1]
            else:
                print("error method affine need xo")
                return None
            
        elif method =="perspective" or method =="":
            P = np.zeros([3,4])
            P[1,1] = 1
            P[0,0] = 1
            P[2,2] = 1
            
        elif method =="euclidean":
            P = np.zeros([4,4])
            P[3,3] = 1
            if not R is None:
                P[:3,:3]= R
            else :
                print("error method Euclidean need R")
                return None
            if not self.t is None:
                P[:3,3] = self.t
            else :
                print("error method Euclidean need t")
                return None
        else :
            print("method is unknow : %s. method possible : affine, perspective, euclidean"%method)
            return None
        
        return P 
    
    ###############################################################################
    
    def convxChapeautox(self, xvChapeau): 
        """
        Ajust size to image. alphaX and alphaY to x and y dimension. s to the defomration
        into x and y axis. xo and yo to the coordonate x and y.
    
        Parameters
        ----------
        xvChapeau : array
            Vector 3 dimension.
        alphaX : float
            Value to adapt x to image size.
        alphaY : float
            Value to adapt y to image size.
        s : float
            Value to link to x and y axis.
        xo : float
            Value x center.
        yo : float
            Value ycenter.
    
        Returns
        -------
        xv : array
            Vector 3 dimensions. The 2 first value is x and y coordonate.
    
        """
        xv = np.dot(self.initP("affine"), xvChapeau)
        return xv
    
    ###############################################################################
    
    def initK (self):
        """
        create K matrix.
    
        Parameters
        ----------
        alphaX : float
            Value to adapt x to image size.
        alphaY : float
            Value to adapt y to image size.
        s : float
            Value to link to x and y axis.
        xo : float
            Value x center.
        yo : float
            Value ycenter.
    
        Returns
        -------
        K : array
            Matrix 3*3.
    
        """
        K = self.initP("affine")
        return K
    
    ###############################################################################
    
    #initR
    
    ###############################################################################
    
    def Cartesian2Polar(self, xvChapeauTilde, xo = [383, 504]):
        """
        Conversion xvChapeauTilde global position to polar annotation. 
    
        Parameters
        ----------
        xvChapeauTilde : array
            Vector 3 dimensions. Global position
        xo : array, optional
            Coordonat fo image center. The default is [383, 504].
    
        Returns
        -------
        r : float
            Radius coordonate.
        phi : float
            Angle coordonate.
    
        """
        xChapeauTilde = xvChapeauTilde[0]
        yChapeauTilde = xvChapeauTilde[1]
        r = np.sqrt((xChapeauTilde-xo[0])**2 + (yChapeauTilde-xo[1])**2)
        if r == 0:
            phi = 0
        else :
            phi = np.arctan2((yChapeauTilde-xo[1])/r, (xChapeauTilde - xo[0])/r)
        return r, phi
    
    ##############################################################################

    # %%%2.1.2 Distortion model (3 functions)
    
    def go(self, theta, method = ""):
        """
        fonction g0
    
        Parameters
        ----------
        theta : float
            Angle in randion. Correspond to angle into center and point on camera.
            like a rayon about image.
        method : string, optional
            Method choice to g0. The default is "".
    
        Returns
        -------
        float
            Output value.
    
        """
        #perspective projection (not fisheye)
        if method == "perspective":
            return np.tan(theta)
        #equidistant projection
        elif method == "equidistant" or method == "":
            return theta
        #equisolid angle projection
        elif method == "equisolid" :
            return 2*np.sin(theta/2)
        # stereographic projection
        elif method == "stereographic":
            return 2*np.tan(theta/2)
        #orthographic projection
        elif method == "orthographic":
            return np.sin(theta)
        else:
            print("arg method is unknow : %s. arg method possible : perspective,"+\
                  "equidistant, equisolid, stereographic, orthographic." %method)
            return None
       
    ###############################################################################
    
    def rChapeauForward (self, theta, method = ""):
        """
        function r chapeau theta, distorition about theta.
    
        Parameters
        ----------
        theta : float
            Angle in radian.
        k : array
            Vector 8 dimensions. About theta distotion.
        method : string, optional
            Gà method. The default is "".
    
        Returns
        -------
        output : float
            Distortion value.
    
        """
        k2 =self.k[0]
        k3 =self.k[1]
        k4 =self.k[2]
        k5 =self.k[3]
        k6 =self.k[4]
        k7 =self.k[5]
        k8 =self.k[6]
        k9 =self.k[7]
        output = self.go(theta, method = method) + k2*theta**2 + k3*theta**3 + k4*theta**4 +\
                  k5*theta**5 + k6*theta**6 + k7*theta**7 + k8*theta**8 + k9*theta**9
        return output
    
    ##############################################################################
    
    def deltacForward (self, theta, phi, method = ""):
        """
        function about distorition with phi.
    
        Parameters
        ----------
        theta : float
            Angle in radian.
        phi : float
            Angle in radian.
        p : array
            Vector 4 dimension. About phi distortion.
        k : array
            Vector 8 dimensions. About theta distotion.
        method : string, optional
            Method choice to g0. The default is "".
    
        Returns
        -------
        deltacx : TYPE
            DESCRIPTION.
        deltacy : TYPE
            DESCRIPTION.
    
        """
        p1 = self.p[0]
        p2 = self.p[1]
        p3 = self.p[2]
        p4 = self.p[3]
        # print(np.max(rChapeauForward(theta, k)))
        deltacx = (p1*(1+2*np.cos(phi)**2)+2*p2*np.sin(phi)*np.cos(phi))*\
              (self.rChapeauForward(theta, method)**2 + p3*self.rChapeauForward(theta, method)**4 +\
               p4*self.rChapeauForward(theta, method)**6)
        deltacy = (2*p1*np.sin(phi)*np.cos(phi)+p2*(1+2*np.sin(phi)**2))*\
              (self.rChapeauForward(theta, method)**2 + p3*self.rChapeauForward(theta, method)**4 +\
               p4*self.rChapeauForward(theta, method)**6)

        return  deltacx, deltacy
    
    ###############################################################################

    #%%2.1.3 Forward camera overview (3 functions)
    
    def EuclideanTransformation(self, Xv, R):
        """
        Convertion Xv coordonate to image coordonate.
    
        Parameters
        ----------
        Xv : array
            Vector 4 dimensions.
        R : array
            Matrix 3*3 rotation.
        t : array
            Vector 3 translation.
    
        Returns
        -------
        XCam : array
            Vecor output.
    
        """
        XCam = np.dot(self.initP("Euclidean", R = R), Xv)
        return XCam

    ###############################################################################
    
    def Distortion(self, psi, method):
        """
        Correction about image défomration.
        Parameters
        ----------
        psi : array
            Vector contain theta et phi.
        method : string
            Method choice to g0.
        k : array
            Vector 8 dimensions. About theta distotion.
        p : array
            Vector 4 dimension. About phi distortion.
    
        Returns
        -------
        xTildeChapeau : float
            First coordonate x after distorition correction 
        yTildeChapeau : float
            Second coordonate y after distorition correction 
    
        """
        theta, phi = psi
        xTildeChapeau = self.rChapeauForward(theta, method)*np.cos(phi)+self.deltacForward(theta, phi, method)[0]
        yTildeChapeau = self.rChapeauForward(theta, method)*np.sin(phi)+self.deltacForward(theta, phi, method)[1]
        return xTildeChapeau, yTildeChapeau
    
    ###############################################################################

    def AffineTransformation(self, R, xvTildeChapeau):
        """
        Adapt trasnformation to image size.
    
        Parameters
        ----------
        R : array
            Matrix 3*3. Affine Matrix.
        xvTildeChapeau : array
            Input vector. After distortion correction.
    
        Returns
        -------
        xv : array
            final coordonate.
    
        """
        xv = np.dot(R, xvTildeChapeau)
        return xv
    
    ###############################################################################

    #%% function (3 functions)
    
    def model (self, XvSoleil, inversion = False):
        """
        Global function to convert XvSoleil (sun position world(X, Y, Z, 1)) to
        xv (sun position image (x, y, 1)).
    
        Parameters
        ----------
        XvSoleil : array
            Input vector.
        alphaX : float
            Porportion about x axis. The default is None.
        alphaY : float
            Proportion about y axis. The default is None.
        s : float
            Deformation between alphaX, alphaY. The default is None.
        xo : array
            Vector 2 dimensions. Center coordonate.
        w : array
            Vector 3 dimensions. 3 angles to rotation matrix.
        t : array
            Vector 3 dimensions. 3 distances to translation matrix.
        k : array
            Vector 8 dimensions. About theta distotion.
        p : array
            Vector 4 dimension. About phi distortion.
        inversion : Bool, optional
            Inversion X and Y axis . The default is False.
    
        Returns
        -------
        xv : array
            Output vector. x image, y image, 1.
    
        """
        R = self.initR()
        matEucli = self.initP("euclidean", R = R)
        XvCam = np.dot(matEucli, XvSoleil)
        #2
        psi = Cartesian2Spherical(XvCam[0], XvCam[1], XvCam[2])
        #3
        xTildeChapeau, yTildeChapeau = self.Distortion(psi, "") 
        #4
        K = self.initP("affine")
        if type(xTildeChapeau) is np.float64:
            xvTildeChapeau = np.array([xTildeChapeau, yTildeChapeau, 1])
        else:
            xvTildeChapeau = np.array([xTildeChapeau, yTildeChapeau, np.ones([len(xTildeChapeau)])])
        xv = np.dot(K, xvTildeChapeau)
        #inverse x and y
        if inversion:
            xvtmp = copy.deepcopy(xv[0,:])
            xv[0,:] = xv[1,:]
            xv[1,:] = xvtmp
        return xv
    
    ###############################################################################

    def modelNorm(self, Params, XvSoleil):
        """
        f with normalisation. 
    
        Parameters
        ----------
        XvSoleil : array
            Input vector.
        alphaX : float
            Porportion about x axis. The default is None.
        alphaY : float
            Proportion about y axis. The default is None.
        s : float
            Deformation between alphaX, alphaY. The default is None.
        xo : array
            Vector 2 dimensions. Center coordonate.
        w : array
            Vector 3 dimensions. 3 angles to rotation matrix.
        t : array
            Vector 3 dimensions. 3 distances to translation matrix.
        k : array
            Vector 8 dimensions. About theta distotion.
        p : array
            Vector 4 dimension. About phi distortion.
    
        Returns
        -------
        xv : array
            Output vector. x image, y image, 1.
    
        """
        # p0 = copy.deepcopy(p)
        #Un normalise
        alphaX = NormalisationCal(Params[0],     "alphaX", normalise = False)
        alphaY = NormalisationCal(Params[1],     "alphaY", normalise = False)
        s      = NormalisationCal(Params[2],     "s",      normalise = False)
        xo     = NormalisationCal(Params[3],     "xo",     normalise = False)
        yo     = NormalisationCal(Params[4],     "yo",     normalise = False)
        w      = NormalisationCal(Params[5:8],   "w",      normalise = False)
        t      = NormalisationCal(Params[8:11],  "t",      normalise = False)
        k      = NormalisationCal(Params[11:19], "k",      normalise = False) 
        p1     = NormalisationCal(Params[19], "p1",      normalise = False)
        p2     = NormalisationCal(Params[20], "p2",      normalise = False)
        p3     = NormalisationCal(Params[21], "p3",      normalise = False)
        p4     = NormalisationCal(Params[22], "p4",      normalise = False)
        pQ      = np.array([p1, p2, p3, p4]) 
        self.setParams(alphaX, alphaY, s, np.array([xo, yo]), w, t, k, pQ)
        xv = self.model(XvSoleil)
        
        return xv
    
    ###############################################################################

    #%% function use calibration (9 functions)
    
    def saveParams (self, method = "csv", path = "/home/ngourgue/climavion/"):
        """
        Function to save parameters. 
    
        Parameters
        ----------
        alphaX : float.
            Horizontale coefficent about image size.
        alphaY : float
            Verticale coefficient about image size.
        s : float
            Link into Horizontale et Verticale axes.
        xo : array
            X and Y coordonate of center.
        w : array
            Vector with 3 values, angle of rotation matrix.
        t : array
            Vector with 3 values, value of translation vector.
        k : array
            Vector with 8 values, value of radiale distorsion .
        p : array
            Vector with 4 values, value of phasique distosion.
        method : string, optional
            Type of file to save csv. The default is "csv".
        path : string, optional
            Path where save. The default is "/home/ngourgue/climavion/".
    
        Returns
        -------
        bool
            Succes to save or not.
    
        """
        if method == "csv":
            beta = np.zeros(23)
            beta[0]     = self.alphaX
            beta[1]     = self.alphaY
            beta[2]     = self.s
            beta[3:5]   = self.xo
            beta[5:8]   = self.w
            beta[8:11]  = self.t
            beta[11:19] = self.k
            beta[19:23] = self.p
            df = pd.DataFrame(columns= ["params"], data = beta)
            df.to_csv(os.path.join(path, "calibration_parameters.csv"), index=False)
            return True
        else:
            print("paramters can't be save. method not function.")
            return False
        
    ###############################################################################
    
    def plotParams (self):
        """
        Plot name and value on parameters.
    
        Parameters
        ----------
        alphaX : float.
            Horizontale coefficent about image size.
        alphaY : float
            Verticale coefficient about image size.
        s : float
            Link into Horizontale et Verticale axes.
        xo : array
            X and Y coordonate of center.
        w : array
            Vector with 3 values, angle of rotation matrix.
        t : array
            Vector with 3 values, value of translation vector.
        k : array
            Vector with 8 values, value of radiale distorsion .
        p : array
            Vector with 4 values, value of phasique distosion.
    
        Returns
        -------
        None.
    
        """
        print("alpha X : %.2f, Y : %.2f, s : %.3f"%(self.alphaX, self.alphaY, self.s))
        print("center xo : %.2f, yo %.2f"%(self.xo[0], self.xo[1]))
        print("w angle  alpha : %.5f, beta : %.5f, gamme : %.5f"%(self.w[0], self.w[1], self.w[2]))
        print("translation tx : %.6f, ty : %.6f, tz : %.8f"%(self.t[0], self.t[1], self.t[2]))
        print("k vector :", self.k)
        print("p vector :", self.p)
        
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
            Coordonate
    
        """
        #read params
        self.readParams(method = methodRead)
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
                x, y, _= self.model(XPosition)
                return(x, y)
            #transpose image
            elif np.all(imageShape == np.array([1024, 768, 3])):
                x, y, _= self.model(XPosition)
                return(y, x)
            #zoom or transforme image
            elif np.all(imageShape == np.array([901, 901, 3])):
                x, y, _= self.model(XPosition)
                x = x-68
                y = y-182
                xR, yR= reversZoom(x, y, szamax= 60, xmax =901, xRGBc = 339, yRGBc= 339)
                return(xR, yR)
            #cropped image old cal
            elif  np.all(imageShape == np.array([674, 674, 3])):
                x, y, _= self.model(XPosition)
                x = x-66
                y = y-161
                return(x, y)
            #cropped image new cal
            elif  np.all(imageShape == np.array([678, 678, 3])):
                x, y, _= self.model(XPosition)
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
            print("not implement part")
            pass
            return(False)
        else:
            raise ValueError("image is not correct. image 2D or 3D is need to argument image.")
    
    ###############################################################################

    def convParam2Beta (self):
        """
        Transform multi parameters to a vector.
    
        Parameters
        ----------
        alphaX : float.
            Horizontale coefficent about image size.
        alphaY : float
            Verticale coefficient about image size.
        s : float
            Link into Horizontale et Verticale axes.
        xo : array
            X and Y coordonate of center.
        w : array
            Vector with 3 values, angle of rotation matrix.
        t : array
            Vector with 3 values, value of translation vector.
        k : array
            Vector with 8 values, value of radiale distorsion .
        p : array
            Vector with 4 values, value of phasique distosion.
    
        Returns
        -------
        beta : array
            Output vector.
    
        """
        beta=np.zeros((23), dtype=np.float128)
        beta[0] = self.alphaX
        beta[1] = self.alphaY
        beta[2] = self.s
        beta[3] = self.xo[0]
        beta[4] = self.xo[1]
        beta[5:8]   = self.w
        beta[8:11]  = self.t
        beta[11:19] = self.k
        beta[19:]   = self.p
        return beta
    
    ##############################################################################
    
    def setParams(self, alphaX, alphaY, s, xo, w, t, k, p):
        """
        Set input parameters. 
    
        Parameters
        ----------
        alphaX : float.
            Horizontale coefficent about image size.
        alphaY : float
            Verticale coefficient about image size.
        s : float
            Link into Horizontale et Verticale axes.
        xo : array
            X and Y coordonate of center.
        w : array
            Vector with 3 values, angle of rotation matrix.
        t : array
            Vector with 3 values, value of translation vector.
        k : array
            Vector with 8 values, value of radiale distorsion .
        p : array
            Vector with 4 values, value of phasique distosion.

        Returns
        -------
        None.

        """
        self.alphaX = alphaX
        self.alphaY = alphaY
        self.s = s
        self.xo = xo
        self.w = w
        self.t = t
        self.k = k
        self.p = p
    
    ###############################################################################
    
    def setBeta(self, beta):
        """
        

        Parameters
        ----------
        beta : array
            Vector 23 parameters.

        Returns
        -------
        None.

        """
        self.alphaX = beta[0]
        self.alphaY = beta[1]
        self.s      = beta[2]
        self.xo     = beta[3:5]
        self.w      = beta[5:8]
        self.t      = beta[8:11]
        self.k      = beta[11:19]
        self.p      = beta[19:]
        
    ###############################################################################
    
    def readParams (self, method = "csv", path = "/home/ngourgue/climavion/params.csv"):
        """
        Read Parameters to project object in image.
    
        Parameters
        ----------
        method : string, optional
            Type of file. The default is "csv".
        path : string, optional
            Path of file. The default is "/home/ngourgue/climavion/calibration_parameters.csv".
    
        Returns
        -------
        array
            Tuple of array to export parameters.
    
        """
        if method == "csv":
            df = pd.read_csv(path)
            beta = df['SIRTA'].values
            self.alphaX = beta[0]
            self.alphaY = beta[1]
            self.s      = beta[2]
            self.xo     = beta[3:5][:,0]
            self.w      = beta[5:8][:,0]
            self.t      = beta[8:11][:,0]
            self.k      = beta[11:19][:,0]
            self.p      = beta[19:][:,0]
        else:
            raise ValueError('method is not implemented')
        
    ###############################################################################
        
    def getParams (self):
        """
        return Params

        Returns
        -------
        alphaX : float.
            Horizontale coefficent about image size.
        alphaY : float
            Verticale coefficient about image size.
        s : float
            Link into Horizontale et Verticale axes.
        xo : array
            X and Y coordonate of center.
        w : array
            Vector with 3 values, angle of rotation matrix.
        t : array
            Vector with 3 values, value of translation vector.
        k : array
            Vector with 8 values, value of radiale distorsion .
        p : array
            Vector with 4 values, value of phasique distosion.

        """
        return self.alphaX, self.alphaY, self.s, self.xo, self.w, self.t, self.k , self.p
        
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
        alphaX : float.
            Horizontale coefficent about image size.
        alphaY : float
            Verticale coefficient about image size.
        s : float
            Link into Horizontale et Verticale axes.
        xo : array
            X and Y coordonate of center.
        w : array
            Vector with 3 values, angle of rotation matrix.
        t : array
            Vector with 3 values, value of translation vector.
        k : array
            Vector with 8 values, value of radiale distorsion .
        p : array
            Vector with 4 values, value of phasique distosion.

    
        """
        self.alphaX = beta[0]
        self.alphaY = beta[1]
        self.s      = beta[2]
        self.xo     = beta[3:5][:, 0]
        self.w      = beta[5:8][:, 0]
        self.t      = beta[8:11][:, 0]
        self.k      = beta[11:19][:, 0]
        self.p      = beta[19:][:, 0]
        return  self.alphaX, self.alphaY, self.s, self.xo, self.w, self.t, self.k , self.p
    
    ###############################################################################
# %% other function (1 functions)
def convBeta2Param (beta):
    """
    Conversion vector parameters to multi parameters

    Parameters
    ----------
    beta : array
        Vector 23 parameters.

    Returns
    -------
    alphaX : float.
        Horizontale coefficent about image size.
    alphaY : float
        Verticale coefficient about image size.
    s : float
        Link into Horizontale et Verticale axes.
    xo : array
        X and Y coordonate of center.
    w : array
        Vector with 3 values, angle of rotation matrix.
    t : array
        Vector with 3 values, value of translation vector.
    k : array
        Vector with 8 values, value of radiale distorsion .
    p : array
        Vector with 4 values, value of phasique distosion.

    """
    alphaX = beta[0]
    alphaY = beta[1]
    s      = beta[2]
    xo     = beta[3:5]
    w      = beta[5:8]
    t      = beta[8:11]
    k      = beta[11:19]
    p      = beta[19:]
    return alphaX, alphaY, s, xo, w, t, k, p

###############################################################################

# %% cost function (6 functions)
    
def epsilonModel(params, XvSoleil, xSun, model):
    """
    Cost function original old and unuse.

    Parameters
    ----------
    params : array
        Parameters vector. Vector with 23 dimensions.
    XvSoleil : array
        Input vector. Vector with 4 dimensions.
    xSun : array
        Reference vector. Vector with 3 dimensions.
    c1 : bool, optional
        Apply constrain c1. The default is True.

    Returns
    -------
    epsilon : float
        Error between projection (xSoleil after transformation) and reference (xSun)

    """
    params1 = copy.deepcopy(params)
    model.setBeta(params1)
    xvCal  = model.model(XvSoleil)
    if type(xSun[0]) is int:
        epsilonv =(xvCal[0]-xSun[0])**2+(xvCal[1]-xSun[1])**2
    else :
        epsilonv =(xvCal[0,:]-xSun[0,:])**2+(xvCal[1,:]-xSun[1,:])**2
    epsilon = np.sum(epsilonv)
    return epsilon

###############################################################################

def c1f(thetamax, k, method = ""):
    if method == "orthographic":
        # thetamax = np.pi/2
        c1 = 0
        for n in range(2, 10):
            c1n = 0
            for i in range(1, n+2):
                prodN = 0
                for j in range(1, i+1):
                    prodN *= (n+2-j)
                    
                c1n+= np.power(2, i)*\
                      (np.cos(thetamax/2)*(i%2)+np.sin(thetamax/2)*((i+1)%2))*\
                       np.power(thetamax, n-i+1)*\
                      (np.cos((i+2)*np.pi/2)+np.sin((i+2)*np.pi/2))*\
                       prodN
                       
            c1 += k[n-2]*c1n
    elif method == "":
        # thetamax = np.pi/2
        c1 = 0
        for n in range(2,10):
            c1 += k[n-2]*(1/n+2)*np.power(thetamax, n+2)
    return(c1)
    
###############################################################################

def epsilonC1(params, XvSoleil, xSun, c1 = True):
    """
    Cost function with c1 possibility constrain. use to evaluate error.

    Parameters
    ----------
    params : array
        Parameters vector. Vector with 23 dimensions.
    XvSoleil : array
        Input vector. Vector with 4 dimensions.
    xSun : array
        Reference vector. Vector with 3 dimensions.
    c1 : bool, optional
        Apply constrain c1. The default is True.

    Returns
    -------
    epsilon : float
        Error between projection (xSoleil after transformation) and reference (xSun)

    """

    epsilonv = epsilonModel(params, XvSoleil, xSun)
    _, _, _, _, _, _, k, _ = convBeta2Param(params)
    if c1 == True:
        c1v = 0
        c1f(1.5, k, method = "")
        epsilon = epsilonv + c1v * xSun.shape[0]
    else:
        epsilon = epsilonv
        
    return epsilon

###############################################################################

def epsilonNorm(params, XvSoleil, xSun, model, tz = None, c1 = True, rand = None):
    """
    Cost function to optimisation with normalise parameters. 
    
    Possibility to separate tz and add c1. 
    add possibilité to add a random error to comparate with droiteError.

    Parameters
    ----------
    params : array
        Parameters vector. Vector with 23 dimensions.
    XvSoleil : array
        Input vector. Vector with 4 dimensions.
    xSun : array
        Reference vector. Vector with 3 dimensions.
    tz : array, optional
        Value of tz component. The default is None.
    c1 : bool, optional
        Apply constrain c1. The default is True.
    rand : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    epsilon : float
        Error.

    """
    params1 = copy.deepcopy(params)
    if tz is None:
        params1 = copy.deepcopy(params)
    else :
        params1 = copy.deepcopy(params)
        params1 = np.inset(params1, 12, tz, axis = 0)
    
    xvCal = model.modelNorm(params1, XvSoleil)
    if type(xSun[0]) is int:
        epsilonv =(xvCal[0]-xSun[0])**2+(xvCal[1]-xSun[1])**2
    else :
        epsilonv =(xvCal[0,:]-xSun[0,:])**2+(xvCal[1,:]-xSun[1,:])**2
    epsilonv = np.sum(epsilonv)
    
    
    c1v = 0
    _, _, _, _, _, _, k, _ = convBeta2Param(params)
    c1v = c1f(np.pi/2, k, method = "")**2
    if c1 == True:
        epsilon = epsilonv + c1v*len(epsilonv)
    else :
        epsilon = epsilonv
    if not rand == None:
        epsilon += random.random()*rand
    # print("espsilon : %.2f c1 : %.2f"%(epsilon3, c1v))
    return epsilon

###############################################################################

def droiteError(params, XvAvion, xPlane, model, norm= True):
    """
    Cost function to calculate distance between projection plane and contrail
    line.
    
    The cost function suppose the contrail is in the prolongement of plane.
    but we are not sure about time precision and the precison about the line
    equation.

    Parameters
    ----------
    params : array
        Vector with all parameters for the model.
    XvAvion : array
        Matrix with plane position in real world.
    xPlane : array
        Matrix equation about line contrail.
    norm : bool, optional
        If params is normalise or not. The default is True.

    Returns
    -------
    epsilon : float
        Raw error.

    """
    params1 = copy.deepcopy(params)

    #call function
    if norm:
        xvCal = model.modelNorm(params1, XvAvion)
    else:
        model.setBeta(params1)
        xvCal = model.model(XvAvion)

    xvCal[0,:] = xvCal[0,:] -68
    xvCal[1,:] = xvCal[1,:] -182
    # if np.min(xvCal)<1:
    #     return 9999999
    xvCal[0,:], xvCal[1,:] = reversZoom(xvCal[0,:], xvCal[1,:], szamax= 60, 
                                        xmax =901, xRGBc = 339, yRGBc= 339)

    # find distance
    X0 = xPlane[:,1] * np.array([np.cos(xPlane[:,0]), np.sin(xPlane[:,0])])
    a = np.tan(xPlane[:,0]+np.pi/2) #coeff directeur
    deltaY = a/(X0[1,:]- a*X0[0,:]) #A
    deltaX = -1/(X0[1,:]- a*X0[0,:]) #B
    c = 1 #C

    dist = np.abs(deltaY*xvCal[1,:]+deltaX*xvCal[0,:]+c)/np.sqrt(deltaX**2+deltaY**2)
    epsilon = np.sum(dist)
    
    return epsilon

###############################################################################

def mixSunPlane(params, XvSoleil, xSun, XvAvion, xPlane, model, ponderation = 1, c1 = True, norm = True):
    """
    Cost function which add Sun error and plane error. You can ponderate error.

    Parameters
    ----------
    params : array
        Vector with all parameters about model.
    XvSoleil : array
        Matrix : data sun in real world.
    xSun : array
        Matrix : data sun with annotation.
    XvAvion : array
        Matrix : data plane in real world.
    xPlane : array
        Matrix : data equation contrail line.
    ponderation : float, optional
        Coefficient with multiplication plane error. The default is 1.
    c1 : bool, optional
        If we want to use . The default is True.
    norm : bool, optional
        IF params is normalise or not. The default is True.

    Returns
    -------
    errorTot : float
        Raw error..

    """
    params1 = copy.deepcopy(params)
    errorPlane = droiteError(params1, XvAvion, xPlane, model, norm)
    if norm == True:
        errorSun = epsilonNorm(params1, XvSoleil, xSun, model, c1 = c1)
    else :
        errorSun = epsilonModel(params1, XvSoleil, xSun, model, c1 = c1)
    errorTot =  errorSun + errorPlane*ponderation
    return errorTot

###############################################################################