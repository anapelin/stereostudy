#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 14:52:01 2021

@author: ngourgue

code to run optimisation.

importation cell use to import package and function

input data cell use to input and split data in dataset to optimise : Fit and 
dataset to test : Test.
We have mutiple dataset :
    sun_zoonFull : all annotation in image.
    sun : automatic annotation (good first estimation but very imprecise)
    sun_zoonMean : annotation with mean by image.
    contrail_old : first dataset with contrail automatic annotation.
    contrail_new : second dataset with contrail automatic annotation
    contrail_mix : fusion with tow precedent dataset.

fitting function : obsolete function to make an optimisation. work only for M1 model.

fiiting
We have multiple fitting :
    __oldFitting__ : first code to optimise calibration using with model m1
    __FriponFitting__ : code to optimise calibration with model m2.
    __bassinHopping__ : last code tu make an optimisation with bassin hopping.
        this code is adapted to M1 and M2 model.

"""

#%% importation
import numpy  as np

import os, sys, random, copy#datetime, 

from scipy.optimize import minimize, basinhopping

if not "/home/ngourgue/climavion/detection_contrail/calibration" in sys.path:
    sys.path.append("/home/ngourgue/climavion/detection_contrail/calibration")
from calibrationM1 import (epsilonfO, saveCalParams, epsilonfNorm, droiteError, 
                           mixteSunPlane, plotParam, convBeta2Param, convPara2Beta)#, epsilonfC123, f)
from calibrationFripon import (epsiFrip, epsiFripNorm, droiteErrorFrip, mixteFripon, 
                               saveFripParams, plotFrip, convBeta2Frip, convFrip2Beta)
from baseCalibration import (readCalParams, normaliseBeta, 
                             UnnormaliseBeta, loadContrailData, loadSunData, separateData)#, psiSun

from useCalibration import (printError)

    



#%% input data
random.seed(5)
XvSoleil, xSun, indic   = loadSunData(2019, "sun_zoonFull", part = 'random')
XvSoleilAuto, xSunAuto  = loadSunData(2019, 'sun')

XvSoleilFit, XvSoleilTest, xSunFit, xSunTest = separateData(XvSoleil, xSun, indic)

XvSoleilMean, xSunMean, indice = loadSunData(2019, "sun_zoonMean", part='random')
XvSoleilMFit, XvSoleilMTest, xSunMFit, xSunMTest = separateData(XvSoleilMean, xSunMean, indice)

XvAvion, xPlane, indices = loadContrailData(2019, 'contrails_newAlt', part ='random')
XvAvionFit, XvAvionTest, xPlaneFit, xPlaneTest = separateData (XvAvion, xPlane, indices)

#%%fitting function
def fitting(beta, XvSoleil = None, xSun = None, XvAvion = None, xPlane = None, 
            withT = True, c1 = False, func = epsilonfNorm, pond = None):
    """
    to realise a simple otpimisation with M1 model.
    is an obsolet function.

    Parameters
    ----------
    beta : array
        vector needed optimisation.
    XvSoleil : array, optional
        Matrix sun data in real world. The default is None.
    xSun : array, optional
        Matrix sun data in image space. The default is None.
    XvAvion : array, optional
        Matrix plane data in real world. The default is None.
    xPlane : array, optional
        Matrix contrail equation in image space. The default is None.
    withT : bool, optional
        If you want to learn tz. When we study cost function we obserbe an iregularity 
        in tz cost function. So in first time to make an easly optimisation we not
        optimise tz. The default is True.
    c1 : bool, optional
        If you want to add first constraint based on k factors. The default is False.
    func : function, optional
        Function . The default is epsilonfNorm.
    pond : float, optional
        Value to ponderate plane error. The default is None.

    Returns
    -------
    beta2 : array
        Parameters after optimisation.

    """
    
    beta1 = normaliseBeta(beta)
    if withT == True :
        tz = None
    else :
        betaTmp = copy.deepcopy(beta1)
        tz = beta1[11]
        beta1 = np.zeros((22), dtype=np.float128)
        beta1[:11] = betaTmp[:11]
        beta1[11:] = betaTmp[12:]
    
    # cons = ({'type': 'eq', 'fun': c1f, 'args' : beta[11:19]})xSun, tz, c1
    methods = ['SLSQP', 'trust-constr', 'BFGS', 'L-BFGS-B']
    # bounds = [(-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), 
    #           (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), 
    #           (-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5),
    #           (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), 
    #           (None, None), (None, None), (None, None), (None, None)]
    
    if func == epsilonfNorm:
        error = epsilonfNorm(beta1, XvSoleil, xSun, tz = tz, c1 = c1)
        print("error before avant minimize : %.2f"%error)
        res=minimize(epsilonfNorm, beta1, args=(XvSoleil, xSun, tz, c1), method=methods[0], #bounds = bounds,
                 options = {'disp' : True, "maxiter" : 500})#, 'eps' : 2e-6})#, 'maxcor' : 100, 'gtol': 1e-11,
                            # 'ftol': 2.220446049250313e-11,'eps': 1e-11, 'maxls': 40})#, 'eps' : 1e-4})
                            #method 0 : eps 1e-8 maxiter 500
                            #method 2
    elif func == droiteError:
        error = droiteError(beta1, XvAvion, xPlane)
        print("error before minimize : %.2f"%error)
        res=minimize(droiteError, beta1, args=(XvAvion, xPlane), method=methods[2], #bounds = bounds,
                 options = {'disp' : True, "maxiter" : 500})#, 'maxcor' : 100, 'gtol': 1e-11,
                            # 'ftol': 2.220446049250313e-11,'eps': 1e-11, 'maxls': 40})#, 'eps' : 1e-4})
    elif func == mixteSunPlane:
        error = mixteSunPlane(beta1, XvSoleil, xSun, XvAvion, xPlane, c1 = c1, ponderation = pond)
        print("error before minimize : %.2f"%error)
        res=minimize(mixteSunPlane, beta1, args=(XvSoleil, xSun, XvAvion, xPlane, 
                                       pond, c1), method=methods[0], #bounds = bounds,
                 options = {'disp' : True, "maxiter" : 500})
    
    res['success']
    
    if withT == True :
        beta2 = UnnormaliseBeta(res['x'])
    else :
        betaTmp = copy.deepcopy(res['x'])
        beta2 = np.zeros((23), dtype=np.float128)
        beta2[:11] = betaTmp[:11]
        beta2[11] = tz
        beta2[12:] = betaTmp[11:]
    
    alphaX, alphaY, s, xo, w, t, k, p = convBeta2Param(beta2)
    
    plotParam(alphaX, alphaY, s, xo, w, t, k, p)
    
    return beta2
#%%fitting
__name__ = '__basinFitting__'
save = False

#%% old fitting
if __name__== '__oldFitting__':
    #%%
    alphaX, alphaY, s, xo, w, t, k, p = readCalParams(method = "csv",  path= "/home/ngourgue/climavion/calibration_parameters_bis.csv")
    beta = convPara2Beta(alphaX, alphaY, s, xo, w, t, k, p)
    
    beta1 = fitting(beta, XvSoleilMFit, xSunMFit, c1 =False, 
                    # func = epsilonfNorm)
                    XvAvion = XvAvionFit, xPlane = xPlaneFit, func = mixteSunPlane, pond = 0)
    beta2 = normaliseBeta(beta)
    beta3 = normaliseBeta(beta1)
    printError(beta = beta2, XvSoleilFit = XvSoleilFit, XvSoleilTest = XvSoleilTest, 
               xSunFit = xSunFit, xSunTest = xSunTest, funcSun =  epsilonfNorm, 
               beta2 = beta3, XvAvionFit = XvAvionFit, XvAvionTest = XvAvionTest, 
               xPlaneFit = xPlaneFit, xPlaneTest = xPlaneTest, funcCon = droiteError)
    if save == True:
        saveCalParams(alphaX, alphaY, s, xo, w, t, k, p, method = "csv", path = "/home/ngourgue/climavion/")
#%%fripon fitting
if __name__ =='__FriponFitting__':
    #%%
    # beta = readCalParams(method = "csv",  path= "/home/ngourgue/climavion/calibration_parameters_frip.csv")
    beta = np.array([0.3,0,0,0,0, 383,513, 0.055,0,0, 0,0 ], dtype = np.float128)
    beta1 = normaliseBeta(beta)
    methods = ['SLSQP', 'trust-constr', 'BFGS', 'L-BFGS-B']
    res=minimize(epsiFripNorm, beta1, args=(XvSoleilMFit[:3, :], xSunMFit[:2,:]), method=methods[0], #bounds = bounds,
                 options = {'disp' : True, "maxiter" : 500})
    beta2 = UnnormaliseBeta(res['x'])
    
    printError(beta = beta, XvSoleilFit = XvSoleilFit[:3, :], XvSoleilTest = XvSoleilTest[:3, :], 
               xSunFit = xSunFit[:2,:], xSunTest = xSunTest[:2,:], funcSun  = epsiFrip, 
               beta2 = beta2, XvAvionFit = XvAvionFit[:3, :], XvAvionTest = XvAvionTest[:3, :], 
               xPlaneFit = xPlaneFit, xPlaneTest = xPlaneTest, funcCon = droiteErrorFrip)
    if save == True:
        b, x0, y0, theta, K1, phi = convBeta2Frip(beta2)
        saveFripParams(b, x0, y0, theta, K1, phi)

#%% basin fitting
if __name__ ==  '__basinFitting__':
    #%%
    Fripon = True
    seed = 5
    pond = 25
    methods = ['SLSQP', 'trust-constr', 'BFGS', 'L-BFGS-B']
    funcs = [epsilonfNorm, droiteError, mixteSunPlane, mixteFripon]
    
    if Fripon==False:
        alphaX, alphaY, s, xo, w, t, k, p = readCalParams(method = "csv",  path= "/home/ngourgue/climavion/calibration_parameters_bis.csv")
        beta = convPara2Beta(alphaX, alphaY, s, xo, w, t, k, p)
        beta = normaliseBeta(beta)
        error = mixteSunPlane(beta, XvSoleilMFit, xSunMFit, XvAvionFit, xPlaneFit, c1 = False, ponderation = pond)

    
        res=basinhopping(funcs[2], beta, minimizer_kwargs={'args' : (XvSoleilMFit, xSunMFit, XvAvionFit, xPlaneFit, pond, False),
                                                                'method' : methods[0], 
                                                                'options' : {'disp' : True, "maxiter" : 500}},#, 'eps' : 1e-8}},
                                                               # 'bounds' : bounds},
                         niter = 20, disp= True, seed = seed, stepsize = 0.05)

        beta1 = UnnormaliseBeta(res['x'])
        alphaX, alphaY, s, xo, w, t, k, p = convBeta2Param(beta1)
        plotParam(alphaX, alphaY, s, xo, w, t, k, p)
        
        printError (beta = beta,  XvSoleilFit = XvSoleilMFit, XvSoleilTest = XvSoleilMTest, 
                    xSunFit = xSunMFit, xSunTest = xSunMTest, funcSun = epsilonfNorm, 
                    beta2 = res['x'], XvAvionFit = XvAvionFit,  XvAvionTest = XvAvionTest, 
                    xPlaneFit = xPlaneFit, xPlaneTest = xPlaneTest, funcCon = droiteError)
        
        
        if save == True:
            saveCalParams(alphaX, alphaY, s, xo, w, t, k, p)
    else:
        # if os.path.isfile("/home/ngourgue/climavion/calibration_parameters_frip.csv"):
        #     b, x0, y0, theta, K1, phi = readCalParams(site='SIRTA', method = "csv",  
        #                                               path= "/home/ngourgue/climavion/calibration_parameters_frip.csv")

        #     beta = convFrip2Beta(b, x0, y0, theta, K1, phi)
        #     # beta = np.array([0.3,0,0,0,0, 383,513, 0.055,0,0, 0,0 ], dtype = np.float128)
        # else :
        beta = np.array([222,0,0,0,0, 383,513, 0.055,0,0, 0,0 ])
        
        beta1 = normaliseBeta(beta)
        error = mixteFripon(beta1, XvSoleilFit[:3,:], xSunFit[:2, :], XvAvionFit[:3, :], xPlaneFit, ponderation = pond)
        res=basinhopping(funcs[3], beta1, minimizer_kwargs={'args' : (XvSoleilFit[:3, :], xSunFit[:2, :], XvAvionFit[:3, :], xPlaneFit, pond),
                                                                'method' : methods[0], 
                                                                'options' : {'disp' : True, "maxiter" : 500 }},# 'eps' : 1e-9}},
                                                               # 'bounds' : bounds},
                         niter = 20, disp= True, seed = seed)#, stepsize = 0.05)
        
        beta2 = UnnormaliseBeta(res['x'])
        b, x0, y0, theta, K1, phi = convBeta2Frip(beta2)
        plotFrip(b, x0, y0, theta, K1, phi)
        
        printError (beta = beta,  XvSoleilFit = XvSoleilMFit[:3, :], XvSoleilTest = XvSoleilMTest[:3, :], 
            xSunFit = xSunMFit[:2,:], xSunTest = xSunMTest[:2,:], funcSun = epsiFrip, 
            beta2 = beta2, XvAvionFit = XvAvionFit[:3, :],  XvAvionTest = XvAvionTest[:3, :], 
            xPlaneFit = xPlaneFit, xPlaneTest = xPlaneTest, funcCon = droiteErrorFrip)
        
        
        if save == True:
            saveFripParams(b, x0, y0, theta, K1, phi)
        
#%% plot cost function
# if __name__ == '__mai__':
#     from matplotlib import pyplt as plt

#     alphaX, alphaY, s, xo, w, t, k, p = readCalParams(method = "csv")
#     params2 = convPara2Beta(alphaX, alphaY, s, xo, w, t, k, p)
#     params1 = normaliseBeta(params2, model='M1')         
    
#     xt = np.arange(0.0,0.1, .001)
#     yt1 = []
#     yt2 = []
#     for x in xt:
        
#         params1[10]  = NormalisationCal(x,  "tz")
#         ytmp1 = epsilonfNorm(params1, XvSoleil, xSun, None)
#         yt1.append(ytmp1)
#         params2[10] = x
#         ytmp2 = epsilonfO(params2, XvSoleil, xSun)
#         yt2.append(ytmp2)                                                  
    
#     plt.figure()
#     plt.scatter(xt, yt1, s = 2, c ="red", label = "optimize Tz")
#     plt.scatter(xt, yt2, s = 2, c= "blue", label = "fixe Tz = 0.0225 ")
#     # plt.plot(xt, yt2)
#     plt.show()
    
#%%plot cost function
# if __name__ == '__mai__':
#     import time
#     from matplotlib import pyplot as plt
#     alphaX, alphaY, s, xo, w, t, k, p = readCalParams(method = "csv")
    
#     params2 = convPara2Beta(alphaX, alphaY, s, xo, w, t, k, p)
#     params2 = normaliseBeta(params2, model='M1')
    
#     xt = np.arange(-1, 1, 0.01)
#     yt1 = []
#     start_time = time.time()
#     beg_time = time.time()
#     for x in xt:
#         params2[19]  = NormalisationCal(x,  "p1")
#         ytmp1 = droiteError(params2, XvAvion[:,0:60], xPlane[0:60,:])
#         yt1.append(ytmp1)
        
#         if np.where(x == xt)[0][0]%100 == 0:
#             end_time = time.time()
#             print('x : %.2f, time : %.2f'%(x, end_time-start_time))
#             beg_time = time.time()
        
#     plt.figure()
#     # plt.scatter(xt, yt1, s = 2, c ="red", label = "error ")
#     plt.plot(xt, yt1)
#     plt.show()
         
