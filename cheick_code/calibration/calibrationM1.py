#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 17:07:33 2021

@author: ngourgue
unuse function :
    vector3D : conversion 4 dimensions to array.
    vector2D : conversion 3 dimension to array.
    Tilde3D : conversion array 4 dimensions to array 3 dimensions
    Tilde2D : conversion array 3 dimensions to array 2 dimensions
    CamTransforme : conversion homogene or inhomogene coordonate
    conv3Dto2D : projection 3D spacoe to 2D space.
    
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
    f : model function 
    full : same function with all separate parameters
    fNorm : model function with normalise parameters. use to optimise.
    
constrain :
    not use so no documentation
    c1fNorm :
    c1f :
    c3f :
    ckf:
        
cost function :
    epsilonf :
    epsilonfO : 
    epsilonfC123 :
    droiteError :
    mixteSunPlane :
        
        
    
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
from .baseCalibration import (Cartesian2Spherical, initR, NormalisationCal, 
                              reversZoom, readCalParams, Spherical2Cartesian)
# from sympy import symbols, Eq, solve

#%%Forward camera model overview
#2.1.1 Projective transformation camera model

#%%part unuse (6 functions)
def vector3D (X, Y, Z, T):
    """
    transfrom 4 coordonate to a Vector 4D (in 3D space)

    Parameters
    ----------
    X : Float or Int
        First coordonate.
    Y : Float or Int
        Second coordonate.
    Z : Float or Int
        Third coordonate.
    T : Float or Int
        Fourth coordonate.

    Returns
    -------
    Xv : array
        vector 4D.

    """
    Xv = np.array([X, Y, Z, T])
    return Xv

###############################################################################

def vector2D (x, y, w):
    """
    transform 3 coordonate to a Vector 3D (in 2D space)

    Parameters
    ----------
    x : Float or Int
        First coordonate.
    y : Float or Int
        Second coordonate.
    w : Float or Int
        Third coordonate.

    Returns
    -------
    xv : array
        vector 3D.

    """
    xv = np.array([x, y, w])
    return xv

###############################################################################

def Tilde3D(Xv):
    """
    Conversion 4D vector to 3D vector. Normalise by T.

    Parameters
    ----------
    Xv : array
        Input vector.

    Returns
    -------
    XvTilde : array
        3D vector.

    """
    XvTilde = np.array([Xv[0]/Xv[3], Xv[1]/Xv[3], Xv[2]/Xv[3]])
    return XvTilde

###############################################################################

def Tilde2D(xv):
    """
    Conversion 3D vector to 2D vector. Normalise by w.

    Parameters
    ----------
    xv : array
        3D vector.

    Returns
    -------
    xvTilde : array
        2D vector.

    """
    xvTilde = np.array([xv[0]/xv[2], xv[1]/xv[2]])
    return xvTilde

###############################################################################

def CamTransforme(R, t, method = "", XvTilde = None, Xv = None):
    """
    Apply tranformation homogeenous or inhomogeneous.

    Parameters
    ----------
    R : array
        Matrix 3*3 to rotation.
    t : array
        Vector 3 to translation.
    method : string, optional
        Description of space. Value possible homogeneous or inhomogeneous. The default is "".
    XvTilde : array, optional
        Input vector in inhomogeneous cases. The default is None.
    Xv : array, optional
        Input vector in homogeneous cases. The default is None.

    Returns
    -------
    XvCam or XvCamTilde : array
        Vector output.

    """
    if method == "inhomogeneous":
        if not XvTilde:
            XvCamTilde = np.dot(R, XvTilde) + t
            return XvCamTilde
        else:
            print("error method inhomogeneous need XvTilde")
            return None
    elif method == "homogeneous":
        if not Xv is None:
            XvCam = np.dot(initP("Euvlidean", R = R, t = t), Xv)
            return XvCam
        else:
            print("error method homogeneous need Xv")
            return None
    else:
        print("error method unknow : %s .method possible : inhomogeneous, homogeneous"%method)
        return None

###############################################################################

def conv3Dto2D(XvCam):
    """
    Projection 3D space to 2D space. For this use matrix product between perspective matrix
    and 3D vector ( vector with 4 dimension)

    Parameters
    ----------
    XvCam : array
        Vector 4 dimension in 3D space.

    Returns
    -------
    xvChapeau : array
        Vector 3 dimensions in 2D space.

    """
    xvChapeau = np.dotmat(initP(method = "perspective"), XvCam)
    return xvChapeau 

###############################################################################

#%%part use (4 functions)

def initP (method ="", alphaX = None, alphaY = None, s = None, xo = None, yo = None, R=None, t= None):
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
        if not alphaX is None:
            P[0,0] = alphaX
        else :
            print("error method affine need alphaX")
            return None
        if not alphaY is None:
            P[1,1] = alphaY
        else :
            print("error method affine need alphaY")
            return None
        if not s is None:
            P[0,1] = s
        else:
            print("error method affine need s")
            return None
        if not xo is None:
            P[0,2] = xo
        else:
            print("error method affine need xo")
            return None
        if not yo is None:
            P[1,2] = yo
        else:
            print("error method affine need yo")
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
        if not t is None:
            P[:3,3] = t
        else :
            print("error method Euclidean need t")
            return None
    else :
        print("method is unknow : %s. method possible : affine, perspective, euclidean"%method)
        return None
    
    return P 

###############################################################################

def convxChapeautox(xvChapeau, alphaX, alphaY, s, xo, yo): 
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
    xv = np.dot(initP("affine", alphaX, alphaY, s, xo, yo), xvChapeau)
    return xv

###############################################################################

def initK (alphaX, alphaY, s, xo, yo):
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
    K = initP("affine", alphaX, alphaY, s, xo, yo)
    return K

###############################################################################

#initR

#%%2.1.2 Distortion model (3 functions)

def go(theta, method = ""):
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

def rChapeauForward (theta, k, method = ""):
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
    k2 =k[0]
    k3 =k[1]
    k4 =k[2]
    k5 =k[3]
    k6 =k[4]
    k7 =k[5]
    k8 =k[6]
    k9 =k[7]
    output = go(theta, method = method) + k2*theta**2 + k3*theta**3 + k4*theta**4 +\
              k5*theta**5 + k6*theta**6 + k7*theta**7 + k8*theta**8 + k9*theta**9
    return output

##############################################################################

def deltacForward (theta, phi, p, k, method = ""):
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
    p1 = p[0]
    p2 = p[1]
    p3 = p[2]
    p4 = p[3]
    # print(np.max(rChapeauForward(theta, k)))
    deltacx = (p1*(1+2*np.cos(phi)**2)+2*p2*np.sin(phi)*np.cos(phi))*\
              (rChapeauForward(theta, k, method)**2 + p3*rChapeauForward(theta, k, method)**4 +\
               p4*rChapeauForward(theta, k, method)**6)
    deltacy = (2*p1*np.sin(phi)*np.cos(phi)+p2*(1+2*np.sin(phi)**2))*\
              (rChapeauForward(theta, k, method)**2 + p3*rChapeauForward(theta, k, method)**4 +\
               p4*rChapeauForward(theta, k, method)**6)
    return  deltacx, deltacy

###############################################################################

#Cartesian2Spherical

#%%2.1.3 Forward camera overview (3 functions)

def EuclideanTransformation(Xv, R, t):
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
    XCam = np.dot(initP("Euclidean", R = R, t = t), Xv)
    return XCam

###############################################################################

def Distortion(psi, method, k, p):
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
    xTildeChapeau = rChapeauForward(theta, k, method)*np.cos(phi)+deltacForward(theta, phi, p, k, method)[0]
    yTildeChapeau = rChapeauForward(theta, k, method)*np.sin(phi)+deltacForward(theta, phi, p, k, method)[1]
    return xTildeChapeau, yTildeChapeau

###############################################################################

def AffineTransformation(R, xvTildeChapeau):
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

#Cartesian2Spherical

#%% Backward projection (8 functions)

def invK(K):
    """
    inversion of K matrix

    Parameters
    ----------
    K : array
        Input K matrix. 3*3 matrix.

    Returns
    -------
    Kinv : array
        Output inversion of K matrix. 3*3 matrix.

    """
    Kinv = np.linalg.inv(K)
    return Kinv

###############################################################################

def convxvToxvChapeauTilde(xv, Kinv):
    """
    Convertion to xv cooronate in image to xv coordonate global.

    Parameters
    ----------
    xv : array
        Vector 3 dimension (x, y, 1).
    Kinv : array
        Matrix inverse to K matrix. 3*3 matrix.

    Returns
    -------
    None.

    """
    xvChapeauTilde = np.dot(Kinv, xv)
    return(xvChapeauTilde)

###############################################################################

def Cartesian2Polar(xvChapeauTilde, xo = [383, 504]):
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

###############################################################################

def deltacBackward(r, phi, q):
    deltacx = (r**2+q[2]*r**4+q[3]*r**6+q[5]*r**8)*(q[0]*\
              (1+2*np.cos(phi)**2)+2*q[1]*np.sin(phi)*np.cos(phi))
    deltacy = (r**2+q[2]*r**4+q[3]*r**6+q[5]*r**8)*(q[0]*\
              (2*np.sin(phi)*np.cos(phi)+q[1]*(1+2*np.sin(phi)**2)))
    return(deltacx, deltacy)
    
def rChapeauBackward (xvChapeauTilde, r, phi, q ):
    rChapeau = np.sqrt((xvChapeauTilde[0] - deltacBackward(r, phi, q)[0])**2+\
                       (xvChapeauTilde[1] - deltacBackward(r, phi, q)[1])**2)
    return(rChapeau)

def invGo(X, method = ""):
    #perspective projection (not fisheye)
    if method == "perspective":
        return(np.arctan(X))
    #equidistant projection
    elif method == "equidistant" or method == "":
        return(X)
    #equisolid angle projection or stereographic projection
    elif method == "equisolid" or method == "stereographic":
        return(2*np.arcsin(X/2))
    #orthographic projection
    elif method == "orthographic":
        return(np.arcsin(X))
    else:
        print("arg method is unknow : %s. arg method possible : perspective,"+\
              "equidistant, equisolid, stereographic, orthographic." % (method))
        return(None)  
    
def p2b(k, theta, method):
    sommeB = rChapeauForward(theta, k, method) - go(theta, method)
    return(sommeB)

def psiBackward(xvChapeauTilde, method, r, phi, q, theta):
    rChap = rChapeauBackward(xvChapeauTilde, r, phi, q)
    theta = invGo(rChap - p2b(k, theta, method), method)
    phi   = np.arctan((xvChapeauTilde[1] - deltacBackward(r, phi, q)[1])/
                      (xvChapeauTilde[0] - deltacBackward(r, phi, q)[0]))
    return(theta, phi)

#Spherical2Cartesian

#%% function (3 functions)

def f (XvSoleil, alphaX, alphaY, s, xo, w, t, k, p, inversion = False):
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
    #1
    #convert w to R with skew symmetric matrix
    # R = np.zeros([3,3])
    # R[0,0] = np.cos(w[0])*np.cos(w[1])
    # R[1,0] = np.sin(w[0])*np.cos(w[1])
    # R[2,0] =-np.sin(w[1])
    # R[0,1] = np.cos(w[0])*np.sin(w[1])*np.sin(w[2]) - np.sin(w[0])*np.cos(w[2])
    # R[1,1] = np.sin(w[0])*np.sin(w[1])*np.sin(w[2]) + np.cos(w[0])*np.cos(w[2])
    # R[2,1] = np.cos(w[1])*np.sin(w[2])
    # R[0,2] = np.cos(w[0])*np.sin(w[1])*np.cos(w[2]) + np.sin(w[0])*np.sin(w[2])
    # R[1,2] = np.sin(w[0])*np.sin(w[1])*np.cos(w[2]) - np.cos(w[0])*np.sin(w[2])
    # R[2,2] = np.cos(w[1])*np.cos(w[2])
    R = initR(w)
    matEucli = initP("euclidean", R = R, t = t)
    XvCam = np.dot(matEucli, XvSoleil)
    #2
    psi = Cartesian2Spherical(XvCam[0], XvCam[1], XvCam[2])
    #3
    xTildeChapeau, yTildeChapeau = Distortion(psi, "", k, p) 
    #4
    K = initP("affine", alphaX, alphaY, s, xo[0], xo[1])
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

def full (XvSoleil, alphaX, alphaY, s, xo, yo, w1, w2, w3, tx, ty, tz, k2, 
          k3, k4, k5, k6, k7, k8, k9, p1,p2, p3, p4):
    """
    function f with detail of each parameters.

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
    xo : float
        DESCRIPTION.
    yo : float
        DESCRIPTION.
    w1 : float
        DESCRIPTION.
    w2 : float
        DESCRIPTION.
    w3 : float
        DESCRIPTION.
    tx : float
        DESCRIPTION.
    ty : float
        DESCRIPTION.
    tz : float
        DESCRIPTION.
    k2 : float
        DESCRIPTION.
    k3 : float
        DESCRIPTION.
    k4 : float
        DESCRIPTION.
    k5 : float
        DESCRIPTION.
    k6 : float
        DESCRIPTION.
    k7 : float
        DESCRIPTION.
    k8 : float
        DESCRIPTION.
    k9 : float
        DESCRIPTION.
    p1 : float
        DESCRIPTION.
    p2 : float
        DESCRIPTION.
    p3 : float
        DESCRIPTION.
    p4 : float
        DESCRIPTION.

    Returns
    -------
    xv : array
        Output vector. x image, y image, 1.
    """
    #mise en forme vecteur
    w = np.array([w1, w2, w3])
    t = np.array([tx, ty, tz])
    k = np.array([k2, k3, k4, k5, k5, k6, k7, k8, k9])
    p = np.array([p1, p2, p3, p4])
    xo = np.array([xo, yo])
    #appel de f
    xv = f (XvSoleil, alphaX, alphaY, s, xo, w, t, k, p)
    return xv

###############################################################################

def fNorm(XvSoleil, alphaX, alphaY, s, xo, w, t, k, p):
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
    alphaX = NormalisationCal(alphaX, "alphaX", normalise = False)
    alphaY = NormalisationCal(alphaY, "alphaY", normalise = False)
    s      = NormalisationCal(s,      "s",      normalise = False)
    xo[0]  = NormalisationCal(xo[0],  "xo",     normalise = False)
    xo[1]  = NormalisationCal(xo[1],  "yo",     normalise = False)
    w      = NormalisationCal(w,      "w",      normalise = False)
    t      = NormalisationCal(t,      "t",      normalise = False)
    k      = NormalisationCal(k,      "k",      normalise = False) 
    p      = NormalisationCal(p,      "p",      normalise = False)

    xv = f(XvSoleil, alphaX, alphaY, s, xo, w, t, k, p)
    
    return xv

###############################################################################

#%% constain (4 functions)
def c1fNorm(thetamax, k, method):
    k = NormalisationCal(k, name = 'k', normalise = False)
    c1 = c1f(thetamax, k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7])
    return c1

def c1f(thetamax, k2, k3, k4, k5, k6, k7, k8, k9, method = ""):
    if method == "orthographic":
        k = np.array([k2, k3, k4, k5, k6, k7, k8, k9])
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
        k = np.array([k2, k3, k4, k5, k6, k7, k8, k9])
        # thetamax = np.pi/2
        c1 = 0
        for n in range(2,10):
            c1 += k[n-2]*(1/n+2)*np.power(thetamax, n+2)
    return(c1)
        
def c3f(theta, k2, k3, k4, k5, k6, k7, k8, k9, method = ""):
    if method == "orthographic":
        c3 = np.cos(theta)
        k = np.array([k2, k3, k4, k5, k6, k7, k8, k9])
        for n in range(2, 10):
            c3 += k[n-2]*n*theta**(n-1)
    elif method == "":
        c3 = 1
    return(c3)

def ckf(k, k0, penalite):
    ck = 0
    if len(k) == len(penalite):
        for i, ki in enumerate(k):
            ck += ((ki-k0[i])*penalite[i])**2
    return ck

#%% cost function (6 functions)

def epsilonf(params, XvSoleil, xSun):
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
    alphaX = params1[0]
    alphaY = params1[1]
    s      = params1[2]
    xo     = [params1[3], params1[4]]
    w      = params1[5:8]
    t      = params1[8:11]
    k      = params1[11:19]
    p      = params1[19:]
    xvCal  = f(XvSoleil, alphaX, alphaY, s, xo, w, t, k, p)
    epsilonv =(xvCal[0,:]-xSun[0,:])**2+(xvCal[1,:]-xSun[1,:])**2
    epsilon = np.sum(epsilonv)
    return epsilon

###############################################################################

def epsilonfO(params, XvSoleil, xSun, c1 = True):
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
    params1 = copy.deepcopy(params)
    alphaX = params1[0]
    alphaY = params1[1]
    s      = params1[2]
    xo     = [params1[3], params1[4]]
    w      = params1[5:8]
    t      = params1[8:11]
    k      = params1[11:19]
    p      = params1[19:]
    xvCal  = f(XvSoleil, alphaX, alphaY, s, xo, w, t, k, p)
    if type(xSun[0]) is int:
        epsilonv =(xvCal[0]-xSun[0])**2+(xvCal[1]-xSun[1])**2
    else :
        epsilonv =(xvCal[0,:]-xSun[0,:])**2+(xvCal[1,:]-xSun[1,:])**2
        
    if c1 == True:
        c1v = 0
        c1f(1.5, k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], method = "")
        epsilon = np.sum(epsilonv) + c1v * len(epsilonv)
    else:
        epsilon = np.sum(epsilonv)
        
    return epsilon

###############################################################################

def epsilonfNorm(params, XvSoleil, xSun, tz = None, c1 = True, rand = None):
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
    alphaX = params1[0]
    alphaY = params1[1]
    s      = params1[2]
    xo     = [params1[3], params1[4]]
    w      = params1[5:8]
    if tz is None:
        t = params1[8:11]
        k = params1[11:19]
        p = params1[19:]
    else :
        t     = np.zeros(3)
        t[:2] = params1[8:10]
        t[2]  = tz
        k     = params1[10:18]
        p     = params1[18:]
    xvCal  = fNorm(XvSoleil, alphaX, alphaY, s, xo, w, t, k, p)
    if type(xSun[0]) is int:
        epsilonv =(xvCal[0]-xSun[0])**2+(xvCal[1]-xSun[1])**2
    else :
        epsilonv =(xvCal[0,:]-xSun[0,:])**2+(xvCal[1,:]-xSun[1,:])**2
    
    c1v = 0
    c1v = c1fNorm(np.pi/2, k, method = "")**2
    if c1 == True:
        epsilon = np.sum(epsilonv) + c1v*len(epsilonv)
    else :
        epsilon = np.sum(epsilonv)
    if not rand == None:
        epsilon += random.random()*rand
    # print("espsilon : %.2f c1 : %.2f"%(epsilon3, c1v))
    return epsilon

###############################################################################
    
def epsilonfC123(params, XvSoleil, xSun):
    """
    Cost function with integration constraint 1, 2, 3.

    Parameters
    ----------
    params : array
        Vector with all parameters for model.
    XvSoleil : array
        Matrix : data real world.
    xSun : array
        Matrix : annotations data.

    Returns
    -------
    epsilon : float
        Raw Error between projection data and annotations.

    """
    alphaX = params[0]
    alphaY = params[1]
    s      = params[2]
    xo     = [params[3], params[4]]
    w      = params[5:8]
    t      = params[8:11]
    k      = params[11:19]
    p      = params[19:]
    xvCal  = f(XvSoleil, alphaX, alphaY, s, xo, w, t, k, p)
    epsilonv =(xvCal[0,:]-xSun[0,:])**2+(xvCal[1,:]-xSun[1,:])**2
    c1 = c1f(XvSoleil, k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7])
    c3 = c3f(XvSoleil, np.pi/2, k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7])
    
    epsilon = np.sum(epsilonv)+c1**2+s**2+c3**2
    print("epsv : %f \n c1 : %f \n s : %f \n c3: %f"%(np.sum(epsilonv), c1, s, c3 ))

    return epsilon

###############################################################################

def droiteError(params, XvAvion, xPlane, norm= True):
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
    #uncompress params
    alphaX = params1[0]
    alphaY = params1[1]
    s      = params1[2]
    xo     = [params1[3], params1[4]]
    w      = params1[5:8]
    t      = params1[8:11]
    k      = params1[11:19]
    p      = params1[19:]

    #call function
    if norm:
        xvCal  = fNorm(XvAvion, alphaX, alphaY, s, xo, w, t, k, p)
    else:
        xvCal = f(XvAvion, alphaX, alphaY, s, xo, w, t, k, p)

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

def mixteSunPlane(params, XvSoleil, xSun, XvAvion, xPlane, ponderation = 1, c1 = True, norm = True):
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
    errorPlane = droiteError(params1, XvAvion, xPlane, norm)
    if norm == True:
        errorSun = epsilonfNorm(params1, XvSoleil, xSun, c1 = c1)
    else :
        errorSun = epsilonfO(params1, XvSoleil, xSun, c1 = c1)
    errorTot =  errorSun + errorPlane*ponderation
    return errorTot

###############################################################################

#%% function use calibration (5 functions)

def saveCalParams (alphaX, alphaY, s, xo, w, t, k, p, method = "csv", path = "/home/ngourgue/climavion/"):
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
        beta[0]     = alphaX
        beta[1]     = alphaY
        beta[2]     = s
        beta[3:5]   = xo
        beta[5:8]   = w
        beta[8:11]  = t
        beta[11:19] = k
        beta[19:23] = p
        df = pd.DataFrame(columns= ["params"], data = beta)
        df.to_csv(os.path.join(path, "calibration_parameters.csv"), index=False)
        return True
    else:
        print("paramters can't be save. method not function.")
        return False
    
###############################################################################

def plotParam (alphaX, alphaY, s, xo, w, t, k, p):
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
    print("alpha X : %.2f, Y : %.2f, s : %.3f"%(alphaX, alphaY, s))
    print("center xo : %.2f, yo %.2f"%(xo[0], xo[1]))
    print("w angle  alpha : %.5f, beta : %.5f, gamme : %.5f"%(w[0], w[1], w[2]))
    print("translation tx : %.6f, ty : %.6f, tz : %.8f"%(t[0], t[1], t[2]))
    print("k vector :", k)
    print("p vector :", p)
    
###############################################################################

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
        Coordonate

    """
    #read params
    alphaX, alphaY, s, xo, w, t, k, p = readCalParams(method = methodRead, site = site)
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
        if np.all(imageShape == np.array([768, 1024, 3])) and site == 'SIRTA':
            x, y, _= f(XPosition, alphaX, alphaY, s, xo, w, t, k, p)
            return(x, y)
        #transpose image
        elif np.all(imageShape == np.array([1024, 768, 3])) and site == 'SIRTA':
            x, y, _= f(XPosition, alphaX, alphaY, s, xo, w, t, k, p)
            return(y, x)
        #zoom or transforme image
        elif np.all(imageShape == np.array([901, 901, 3])) and site == 'SIRTA':
            x, y, _=f(XPosition, alphaX, alphaY, s, xo, w, t, k, p)
            x = x-68
            y = y-182
            xR, yR= reversZoom(x, y, szamax= 60, xmax =901, xRGBc = 339, yRGBc= 339)
            return(xR, yR)
        #cropped image old cal
        elif  np.all(imageShape == np.array([674, 674, 3])) and site == "SIRTA":
            x, y, _= f(XPosition, alphaX, alphaY, s, xo, w, t, k, p)
            x = x-66
            y = y-161
            return(x, y)
        #cropped image new cal
        elif  np.all(imageShape == np.array([678, 678, 3])) and site == 'SIRTA':
            x, y, _= f(XPosition, alphaX, alphaY, s, xo, w, t, k, p)
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

def convPara2Beta (alphaX, alphaY, s, xo, w, t, k, p):
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
    beta[0] = alphaX
    beta[1] = alphaY
    beta[2] = s
    beta[3] = xo[0]
    beta[4] = xo[1]
    beta[5:8]   = w
    beta[8:11]  = t
    beta[11:19] = k
    beta[19:]   = p
    return beta

###############################################################################

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

#NormalisationCal

#reversZoom
    
###############################################################################
#%%3 Solar Position
if __name__ == "__main__":
    verbose = True
    #psi sun estimation using NREL but we use ephem
    #calcul Xs with lamda =1 and t =0 R = np.eye(3)
    lamda = 1
    t = np.zeros([3,])
    R = np.eye(3)
    #use saturation for calculate xs and ys. xv = (xs, ys, 1)

    #%% 4 calibration
    
    #4.1 model init
    #beta = [alphaX, alphaY, s, xo, w, t, k, p]
    
    #alphaX, alphaY, s, xov need process each time camera is modify #intrinsec param
    #w and t need to performe each time camera is move              #extrinsec param
    #k and p ?                                                      #distortionparam
    k = np.zeros([8,1])
    p = np.zeros([4,1])

#4.1.1 intrinsic parameter estimation

    #%% solar position
    #--preparing call to ephem
    lon_sirta=2.208
    lat_sirta=48.713
    obs=ephem.Observer()
    obs.lon=lon_sirta*ephem.degree
    obs.lat=lat_sirta*ephem.degree
    sun=ephem.Sun()
    #input date 
    year = 2019
    month = 6
    day = 1
    hour = 11
    minute = 0
    second = 0
    dateDay = datetime.datetime(year, month, day, hour, minute, day)
    #psiSoleil = trans(thetaSoleil, phiSoleil)
    #xSoleilTilde = trans(xSoleil, ySoleil)
    #NREL use ephem
    #thetaSoleil : zenith angle
    #phiSoleil   : azimuth angle
    obs.date=dateDay
    sun.compute(obs)
    # thetaSoleil, phiSoleil = sun.alt, sun.az
    thetaSoleil, phiSoleil = np.pi/2.-sun.alt, sun.az*1.
    # landa  = 1
    # t = np.zeros(3)
    # R = np.eye(3)
    # XvSoleil = Spherical2Cartesian(R, t, theta, phi)

    #%%image init 
    #open image
    image = imread(os.path.join("/home/ngourgue/Images", "%04d"%year, "%02d"%month, "%04d%02d%02d"%(year, month, day),
                                "%04d%02d%02d%02d%02d%02d_01.jpg"%(year, month, day, hour, minute, second)))
    if verbose :
        plt.figure()
        plt.imshow(image[:,:,2], cmap = 'gray')
        plt.colorbar()
    #edges
    edges = canny(image[:,:,2], sigma = 2)
    if verbose :
        plt.figure()
        plt.imshow(edges, cmap='gray')
        plt.colorbar()
    #hough circular
    hough_radii = np.arange(300, 400, 10)
    hough_res = hough_circle(edges, hough_radii)
    
    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=1)
    
    # Draw them
    if verbose : 
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = circle_perimeter(center_y, center_x, radius,
                                            shape=image.shape)
            image[circy, circx] = (220, 20, 20)
        
        ax.imshow(image, cmap=plt.cm.gray)
        plt.show()
    #%%4.1.1 intrinsec parameter estimation 
    #Hough circle give
    #do mean for 100 images
    #xo center of image
    #yo center of image
    ximg = cx[0]
    yimg = cy[0]
    rimg = radii[0]
    if verbose : 
        print("ximg : %d, yimg : %d, radius : %d" %(ximg, yimg, radii))
    #alpha = rimg/sqrt(2)
    alpha = rimg/np.sqrt(2)
    alphaX = alpha
    alphaY = alpha
    s = 0
    Ko = np.zeros([3,3])
    Ko[0,0] = alphaX
    Ko[1,1] = alphaY
    Ko[0,1] = s
    Ko[0,2] = ximg
    Ko[1,2] = yimg
    Ko[2,2] = 1
    if verbose :
        print(Ko)

    #%%4.1.2
    #XCamTilde = [R|t]XvSoleil
    #XCamTilde : coordonnées inhomogènes de la camera
    #Xs        : coordonnées homogène du monde du Soleil
    
    # XCamTilde = np.dot(Rt, XvSoleil)
    from cloud import detect_sun
    imageSun = imread(os.path.join("/home/ngourgue/Images", "%04d"%year, "%02d"%month, "%04d%02d%02d"%(year, month, day),
                                "%04d%02d%02d%02d%02d%02d_03.jpg"%(year, month, day, hour, minute, second)))
    
    xs, ys = detect_sun(imageSun, method = "saturation")
    xv = np.array([xs, ys, 1])
    #Backward process
    XvSoleil = Spherical2Cartesian(R, t, thetaSoleil, phiSoleil)
    #%%
    #Forward process
    #chercher comment initalisé w
    w = np.array([1,1,1])
    beta = [alpha, alpha, 0, [ximg, yimg], w, t, k, p]
    #xvTilde = f(X, beta)
    xSun = np.array([xs, ys, 1])
    # def f (XvSoleil, beta):

#%% fit
    from lmfit import Minimizer, Parameters, report_fit
    
    params = Parameters()
    params.add("alphaX", value = alphaX)
    params.add("alphaY", value = alphaY)
    params.add("s" , value = s)
    params.add("xo", value = ximg)
    params.add("yo", value = yimg)
    
    params.add("w1", value = w[0])
    params.add("w2", value = w[1])
    params.add("w3", value = w[2])
    params.add("tx", value = t[0])
    params.add("ty", value = t[1])
    params.add("tz", value = t[2])
    
    params.add("k2", value = k[0])
    params.add("k3", value = k[1])
    params.add("k4", value = k[2])
    params.add("k5", value = k[3])
    params.add("k6", value = k[4])
    params.add("k7", value = k[5])
    params.add("k8", value = k[6])
    params.add("k9", value = k[7])
    
    params.add("p1", value = p[0])
    params.add("p2", value = p[1])
    params.add("p3", value = p[2])
    params.add("p4", value = p[3])
    
    minner = Minimizer(userfcn = epsilonf, params = params, fcn_args= (XvSoleil, xSun))
    #Perform the minimization
    fit_linear = minner.minimize()
    
    #%%#Get summary of the fit
    report_fit(fit_linear)
    

    
