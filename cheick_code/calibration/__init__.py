#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 12:47:52 2022

@author: ngourgue
"""

# from .calibration import (vector3D, vector2D, Tilde3D, Tilde2D, initP, CamTransforme,
#     conv3Dto2D, convxChapeautox, initK, Cartesian2Spherical, go, rChapeauForward,
#     deltacForward, EuclideanTransformation, Distortion, AffineTransformation, initR,
    
#     invK, convxvToxvChapeauTilde, Cartesian2Polar, deltacBackward, rChapeauBackward,
#     invGo, p2b, psiBackward, Spherical2Cartesian,
    
#     f, full, fNorm,
    
#     epsilonf, epsilonfO, epsilonfNorm, c1f, c1fNorm, c3f, epsilonfC123,
#     psiSun,
#     saveCalParams, readCalParams, world2image,
#     NormalisationCal, reducto, amplificatum,
#     reversZoom, droiteError, mixteSunPlane)

# __all__ = ['vector3D', 'vector2D', 'Tilde3D', 'Tilde2D', 'initP', 'CamTransforme',
#            'conv3Dto2D', 'convxChapeautox', 'initK', 'Cartesian2Spherical', 'go',
#            'rChapeauForward', 'deltacForward', 'EuclideanTransformation', 'Distortion',
#            'AffineTransformation', 'initR',
           
#            'invK', 'convxvToxvChapeauTilde', 'Cartesian2Polar', 'deltacBackward',
#            'rChapeauBackward', 'invGo', 'p2b', 'psiBackward', 'Spherical2Cartesian',
           
#            'f', 'full', 'fNorm',
           
#            'epsilonf', 'epsilonfO', 'epsilonfNorm', 'c1f', 'c1fNorm', 'c3f', 'epsilonfC123',
#            'psiSun',
#            'saveCalParams', 'readCalParams', 'world2image',
#            'NormalisationCal', 'reducto', 'amplificatum',
#            'reversZoom', 'droiteError', 'misteSunPlane']


from .baseCalibration   import (psiSun, readCalParams, reversZoom, initR, 
                                projCoord, zoom_image, cropped, projCoord2)
from .calibrationM1     import world2image as W2IM1
from .calibrationM1     import Spherical2Cartesian
from .calibrationFripon import world2image as W2IM2
from .calibrationFripon import convCartoGeo, convGeoCarto, image2world
from .useCalibration    import worldToImage, useCalibration, imageToWorld


__all__ = ['readCalParams', 'worldToImage', 'W2IM1', 'W2IM2', 'psiSun', 
           'Spherical2Cartesian', 'reversZoom', 'useCalibration', 'initR',
           'convCartoGeo', 'convGeoCarto', 'projCoord', 'zoom_image', 'cropped',
           'projCoord2', 'image2world', 'imageToWorld']