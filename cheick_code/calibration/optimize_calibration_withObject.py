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
from OriginalModel import OrigineModel
from OriginalModel import epsilonModel as epsiModO
from OriginalModel import epsilonNorm  as epsiNorO
from OriginalModel import droiteError  as epsiDroO
from OriginalModel import mixSunPlane  as epsiSPO

#from OriginalModel import convBeta2Param

from FriponModel import Fripon
from FriponModel import epsilonModel as epsiModF
from FriponModel import epsilonNorm  as epsiNorF
from FriponModel import droiteError  as epsiDroF
from FriponModel import mixSunPlane  as epsiSPF

from baseCalibration import (loadContrailData, loadSunData, separateData, 
                             UnnormaliseBeta, normaliseBeta)#, psiSun

from useCalibration import (printError)

    



#%% input data
random.seed(5)
XvSoleil, xSun, indic   = loadSunData(2019, "sun_zoonFull", part = 'random')
XvSoleilAuto, xSunAuto  = loadSunData(2019, 'sun')

XvSoleilFit, XvSoleilTest, xSunFit, xSunTest = separateData(XvSoleil, xSun, indic)

XvSoleilMean, xSunMean, indice = loadSunData(2019, "sun_zoonMean", part='random')
XvSoleilMFit, XvSoleilMTest, xSunMFit, xSunMTest = separateData(XvSoleilMean, xSunMean, indice)

XvAvion, xPlane, indices = loadContrailData(2019, 'contrails_mix', part ='random')
XvAvionFit, XvAvionTest, xPlaneFit, xPlaneTest = separateData (XvAvion, xPlane, indices)

#%%fitting function
def fitting(beta, XvSoleil = None, xSun = None, XvAvion = None, xPlane = None, 
            model = None, withT = True, c1 = False, func = epsiNorO, pond = None):
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
    
    if func == epsiNorO:
        error =func(beta1, XvSoleil, xSun, model, tz = tz, c1 = c1)
        print("error before avant minimize : %.2f"%error)
        res=minimize(func, beta1, args=(XvSoleil, xSun, model, tz, c1), method=methods[2], #bounds = bounds,
                 options = {'disp' : True, "maxiter" : 500, 'eps' : 2e-6})#, 'maxcor' : 100, 'gtol': 1e-11,
                            # 'ftol': 2.220446049250313e-11,'eps': 1e-11, 'maxls': 40})#, 'eps' : 1e-4})
                            #method 0 : eps 1e-8 maxiter 500
                            #method 2
    elif func == epsiDroO:
        error = func(beta1, XvAvion, xPlane, model)
        print("error before minimize : %.2f"%error)
        res=minimize(func, beta1, args=(XvAvion, xPlane, model), method=methods[2], #bounds = bounds,
                 options = {'disp' : True, "maxiter" : 500})#, 'maxcor' : 100, 'gtol': 1e-11,
                            # 'ftol': 2.220446049250313e-11,'eps': 1e-11, 'maxls': 40})#, 'eps' : 1e-4})
    elif func == epsiSPO:
        error = func(beta1, XvSoleil, xSun, XvAvion, xPlane, model, c1 = c1, ponderation = pond)
        print("error before minimize : %.2f"%error)
        res=minimize(func, beta1, args=(XvSoleil, xSun, XvAvion, xPlane, model,
                                       pond, c1), method=methods[2], #bounds = bounds,
                 options = {'disp' : True, "maxiter" : 500})
    else:
        return False
    
    res['success']
    
    if withT == True :
        beta2 = UnnormaliseBeta(res['x'])
    else :
        betaTmp = copy.deepcopy(res['x'])
        beta2 = np.zeros((22), dtype=np.float128)
        beta2[:11] = betaTmp[:11]
        beta2[12:] = betaTmp[11:]
        beta2[11] = tz
    
    model.setBeta(beta2)
    
    model.plotParam()
    
    return beta2
#%%fitting
__name__ = '__basinFitting__'
save = False

#%% old fitting
if __name__== '__oldFitting__':
    #%%
    modelOld = OrigineModel()
    modelOld.readParams(method = "csv",  path= "/home/ngourgue/climavion/calibration_parameters_bis.csv")
    beta = modelOld.convParam2Beta()
    
    beta1 = fitting(beta, XvSoleilFit, xSunFit, c1 =False, func = epsiNorO, model = modelOld)
                                                # XvAvion = XvAvionFit, xPlane = xPlaneFit, func = mixteSunPlane, pond = 1)
    
    printError(beta = beta, XvSoleilFit = XvSoleilFit, XvSoleilTest = XvSoleilTest, 
               xSunFit = xSunFit, xSunTest = xSunTest, funcSun =  epsiModO, 
               beta2 = beta1, XvAvionFit = XvAvionFit, XvAvionTest = XvAvionTest, 
               xPlaneFit = xPlaneFit, xPlaneTest = xPlaneTest, funcCon = epsiDroO,
               model = modelOld)
    if save == True:
        modelOld.setParams(beta1)
        modelOld.saveParams(method = "csv", path = "/home/ngourgue/climavion/")
#%%fripon fitting
if __name__ =='__FriponFitting__':
    #%%
    # beta = readCalParams(method = "csv",  path= "/home/ngourgue/climavion/calibration_parameters_frip.csv")
    beta = np.array([0.3,0,0,0,0, 383,513, 0.055,0,0, 0, 0 ])
    modelFri = Fripon('SIRTA')
    beta1 = normaliseBeta(beta)
    methods = ['SLSQP', 'trust-constr', 'BFGS', 'L-BFGS-B']
            
    res=minimize(epsiNorF, beta1, args=(XvSoleilFit[:3, :], xSunFit[:2,:], modelFri), method=methods[0], #bounds = bounds,
                 options = {'disp' : True, "maxiter" : 500})
    beta2 = UnnormaliseBeta(res['x'])
    modelFri.setBeta(beta2)
    modelFri.plotParams()
    
    printError(beta = beta, XvSoleilFit = XvSoleilFit[:3, :], XvSoleilTest = XvSoleilTest[:3, :], 
               xSunFit = xSunFit[:2,:], xSunTest = xSunTest[:2,:], funcSun  = epsiModF, 
               beta2 = beta2, XvAvionFit = XvAvionFit[:3, :], XvAvionTest = XvAvionTest[:3, :], 
               xPlaneFit = xPlaneFit, xPlaneTest = xPlaneTest, funcCon = epsiDroF, 
               model = modelFri)
    if save == True:
        modelFri.saveParams()

#%% basin fitting
if __name__ ==  '__basinFitting__':
    #%%
    model = Fripon('SIRTA')#OrigineModel()# Fripon() #
    seed = 2
    pond = 0.8
    methods = ['SLSQP', 'trust-constr', 'BFGS', 'L-BFGS-B']
    nameFunc = ['epsiMod', 'epsiNor', 'epsiDro', 'epsiSP']
    nbFunc = 3
    
    if model.name == 'Fripon':
        # model.readParams(method = "csv",  path= "/home/ngourgue/climavion/calibration_parameters_frip.csv")
        model.setParams(np.array([0.3, 0, 0, 0, 0]), 383, 513, np.array([0.055,0, 0]), 0, 0)
        XvSoleilFit = XvSoleilFit[:3,:]
        XvSoleilTest = XvSoleilTest[:3,:]
        XvAvionFit = XvAvionFit[:3, :]
        XvAvionTest = XvAvionTest[:3, :]
        xSunFit = xSunFit[:2, :]
        xSunTest = xSunTest[:2, :]
        
        func = locals()[nameFunc[nbFunc]+'F']
        funcSun = locals()[nameFunc[0]+'F']
        funcCon = locals()[nameFunc[2]+'F']
    
    elif model.name == 'Origine':
        model.readParams(method = "csv",  path= "/home/ngourgue/climavion/calibration_parameters_bis.csv")
        func = locals()[nameFunc[nbFunc]+'O']
        funcSun = locals()[nameFunc[0]+'O']
        funcCon = locals()[nameFunc[2]+'O']
    
    beta = model.convParam2Beta()
    beta1 = normaliseBeta(beta)
    error = func(beta, XvSoleilFit, xSunFit, XvAvionFit, xPlaneFit, model, c1 = False, ponderation = pond)
    res=basinhopping(func, beta1, minimizer_kwargs={'args' : (XvSoleilFit, xSunFit, XvAvionFit, xPlaneFit, model, pond, True),
                                                   'method' : methods[0], 
                                                   'options' : {'disp' : True, "maxiter" : 500}},# 'eps' : 1e-8}}, # 'bounds' : bounds},
                     niter = 20, disp= True, seed = seed)#, stepsize = 0.05)
    
    beta2 = UnnormaliseBeta(res['x'])
    model.setBeta(beta2)
    model.plotParams()
    
    printError (beta = beta,  XvSoleilFit = XvSoleilFit, XvSoleilTest = XvSoleilTest, 
                xSunFit = xSunFit, xSunTest = xSunTest, funcSun = funcSun, 
                beta2 = beta2, XvAvionFit = XvAvionFit,  XvAvionTest = XvAvionTest, 
                xPlaneFit = xPlaneFit, xPlaneTest = xPlaneTest, funcCon = funcCon,
                model = model)
    
    if save == True:
            model.saveParams()
            