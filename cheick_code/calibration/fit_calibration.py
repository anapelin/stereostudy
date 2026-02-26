#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 13:00:43 2021

@author: ngourgue
"""

#%%
import numpy  as np
import pandas as pd
import datetime, os

from skimage.io        import imread
from skimage.draw      import circle_perimeter
from skimage.feature   import canny
from skimage.transform import hough_circle, hough_circle_peaks

from matplotlib import pyplot as plt

from calibration import epsilonfC123, Spherical2Cartesian, epsilonfO, psiSun

from lmfit import Minimizer, Parameters, report_fit, Model
   
#%% init parameters
verbose = False
#%% init image
# # calcul alpha, center of image, 
# year  = 2019
# month = 6
# day   = 1
# hour  = 11
# minute= 0
# second= 0
# #open image
# image = imread(os.path.join("/home/ngourgue/Images", "%04d"%year, "%02d"%month, "%04d%02d%02d"%(year, month, day),
#                             "%04d%02d%02d%02d%02d%02d_01.jpg"%(year, month, day, hour, minute, second)))
# if verbose :
#     plt.figure()
#     plt.imshow(image[:,:,2], cmap = 'gray')
#     plt.colorbar()
# #edges
# edges = canny(image[:,:,2], sigma = 2)
# if verbose :
#     plt.figure()
#     plt.imshow(edges, cmap='gray')
#     plt.colorbar()
# #hough circular
# hough_radii = np.arange(300, 400, 10)
# hough_res = hough_circle(edges, hough_radii)

# # Select the most prominent 3 circles
# accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
#                                            total_num_peaks=1)
# # Draw them
# if verbose : 
#     fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
#     for center_y, center_x, radius in zip(cy, cx, radii):
#         circy, circx = circle_perimeter(center_y, center_x, radius,
#                                         shape=image.shape)
#         image[circy, circx] = (220, 20, 20)
    
#     ax.imshow(image, cmap=plt.cm.gray)
#     plt.show()
# rimg = radii[0]
# if verbose : 
#     print("ximg : %d, yimg : %d, radius : %d" %(cx[0], cy[0], radii))
# alpha = rimg/np.sqrt(2)

#%% import data
# dateDay = datetime.datetime(year, month, day)

# sunData = pd.read_csv(os.path.join("/home/ngourgue/Images", "%04d"%year, "%02d"%month, 
#                          "%04d%02d%02d"%(year, month, day), 
#                          "%04d%02d%02d"%(year, month, day)+"sun.csv"))
year = 2019
month = None
if month is None:
    sunData = pd.read_csv(os.path.join("/home/ngourgue/Images", "%04d"%year,"%04dsun.csv"%year))
else:
    sunData = pd.read_csv(os.path.join("/home/ngourgue/Images", "%04d"%year,"%02d"%month, 
                                       "%04d%02dsun.csv"%(year, month)))
#solar params
lamda = 1
t = np.zeros(3)
R = np.eye(3) 
thetaSoleil = sunData["theta"]
phiSoleil   = sunData["phi"]
XvSoleil = Spherical2Cartesian(R, t, thetaSoleil, phiSoleil)
xs = sunData["x"]
ys = sunData["y"]
xSun = np.array([xs, ys, np.ones([len(xs)])])
# xSun = np.array([xs, ys, 1])
#%%
#Beta params
from calibration import readCalParams
alphaX, alphaY, s, xo, w, t, k, p = readCalParams(method = "csv")
# alphaX = 350/np.pi*2
# alphaY = 350/np.pi*2
# s = 0
# ximg = 383
# yimg = 513
# t = np.array([0,0,0], dtype = np.float128)
# w = np.array([0,0,0], dtype = np.float128)
# w[0] = -0.07
# k = np.zeros([8], dtype = np.float128)
# p = np.zeros([4], dtype = np.float128)

#%%
params = Parameters()
params.add("alphaX", value = alphaX)
params.add("alphaY", value = alphaY)
params.add("s" , value = s)
params.add("xo", value = xo[0])
params.add("yo", value = xo[1])

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

minner = Minimizer(userfcn = epsilonfO, params = params, fcn_args= (XvSoleil, xSun))
#Perform the minimization
fit_linear = minner.minimize()

#%%#Get summary of the fit
report_fit(fit_linear)
#%% autre apprentissage
# mod = Model(full)
# pars = mod.make_params(alphaX = alphaX, alphaY = alphaY, s = s, xo = ximg, yo = yimg,
#                        w1 = 1, w2 = 0, w3 = 0, 
#                        tx = 0, ty = 0, tz = 0,
#                        k2 = k[0], k3 = k[1], k4 = k[2], k5 = k[3], k6 = k[4], k7 = k[5], 
#                        k8 = k[6], k9 = k[7], p1 = p[0], p2 = p[1], p3 = p[2], p4 = p[3])
# result = mod.fit(xSun, pars, XvSoleil = XvSoleil)
# print(result.fit_report())

# results = result.params
#%% olivier optimiser
from scipy.optimize import minimize
beta=np.zeros((23), dtype=np.float128)
# beta=np.zeros((2), dtype=np.float128)
beta[0] = alphaX
beta[1] = alphaY
beta[2] = s
beta[3] = xo[0]
beta[4] = xo[1]

beta[5:8]   = w
beta[8:11]  = t
beta[11:19] = k
beta[19:]   = p

# cons = ({'type': 'eq', 'fun': c1f, 'args' : beta[11:19]})
methods = ['SLSQP', 'trust-constr']
bounds = [(217, 227), (217, 227), (-0.1, 0.1), (378, 388), (508, 518), 
          (0.0, 0.14), (-0.07, 0.07), (-0.07, 0.07), 
          (-1, 1), (-1, 1), (-1, 1),
          (-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100), 
          (None, None), (None, None), (None, None), (None, None)]
res=minimize(epsilonfO, beta, args=(XvSoleil, xSun), method=methods[0])#, bounds = bounds)#,#  ), bounds = bounds,
                # options = {'disp' : True, 'eps' : 0.1})
res['success']
res_param=res['x']
res_param=res['progress']
#%%
# weight = pd.DataFrame(res['x'])
# weight.columns = ['x']
# weight.to_csv(os.path.join("/home/ngourgue/Images/", "%04d"%year,"%02d"%month, 
#                                   "results_calibration.csv"), index = False)

#%% save results
# results = fit_linear.params
# save_results = pd.DataFrame(columns=["name", "value", "stderr", "min", "max", "init_value"])
# list_param = list(results.valuesdict().keys())
# for i in range(len(list_param)):
#     param = results[list_param[i]]
#     save_results.loc[len(save_results)] = [param.name, param.value, param.stderr, param.min,
#                                             param.max, param.init_value]
# if month is None:
#     save_results.to_csv(os.path.join("/home/ngourgue/Images/", "%04d"%year,#"%02d"%dateDay.month, 
#                                   "results_calibration.csv"), index = False)
# else:
#     save_results.to_csv(os.path.join("/home/ngourgue/Images/", "%04d"%year,"%02d"%month, 
#                                   "results_calibration.csv"), index = False)


# # #results :
# [[Fit Statistics]]
#     # fitting method   = leastsq
#     # function evals   = 542
#     # data points      = 3253
#     # variables        = 23
#     chi-square         = 1.3355e+09
#     reduced chi-square = 413469.973
#     Akaike info crit   = 42091.8209
#     Bayesian info crit = 42231.8295
# [[Variables]]
#     alphaX: -69.9717284 +/- 275.565400 (393.82%) (init = 247.4874)
#     alphaY:  172.413520 +/- 678.039014 (393.26%) (init = 247.4874)
#     s:      -11.1948754 +/- 46.1442719 (412.19%) (init = 0)
#     xo:      565.424875 +/- 9.84765068 (1.74%) (init = 504)
#     yo:      591.385333 +/- 56.6015451 (9.57%) (init = 383)
#     w1:      1.20139296 +/- 1.82062368 (151.54%) (init = 1)
#     w2:      3.06733837 +/- 0.73089251 (23.83%) (init = 1)
#     w3:      1.56947324 +/- 0.90399332 (57.60%) (init = 1)
#     tx:     -2.38723199 +/- 1.54106970 (64.55%) (init = 0)
#     ty:     -3.43225287 +/- 2.51200463 (73.19%) (init = 0)
#     tz:     -2.83019411 +/- 5.07334582 (179.26%) (init = 0)
#     k2:     -0.14346925 +/- 11.2697638 (7855.18%) (init = 0)
#     k3:      0.34972685 +/- 5.02581959 (1437.07%) (init = 0)
#     k4:      0.01191124 +/- 13.5633059 (113869.80%) (init = 0)
#     k5:     -0.06997131 +/- 2.57256781 (3676.60%) (init = 0)
#     k6:      0.01866413 +/- 5.62910429 (30160.01%) (init = 0)
#     k7:     -0.04229643 +/- 1.19927063 (2835.39%) (init = 0)
#     k8:     -0.00540376 +/- 0.78809798 (14584.26%) (init = 0)
#     k9:      0.00904955 +/- 0.22711310 (2509.66%) (init = 0)
#     p1:     -0.00133718 +/- 0.01915878 (1432.78%) (init = 0)
#     p2:     -0.08584544 +/- 0.79834631 (929.98%) (init = 0)
#     p3:     -0.47423796 +/- 3.40686191 (718.39%) (init = 0)
#     p4:      0.28984148 +/- 3.16781665 (1092.95%) (init = 0)

# Output from spyder call 'get_namespace_view':
# [[Model]]
#     Model(full)
# [[Fit Statistics]]
#     # fitting method   = leastsq
#     # function evals   = 784
#     # data points      = 9759
#     # variables        = 23
#     chi-square         = 6081823.48
#     reduced chi-square = 624.673735
#     Akaike info crit   = 62843.8954
#     Bayesian info crit = 63009.1722
# [[Variables]]
#     alphaX:  203.099917 +/- 37.5875892 (18.51%) (init = 247.4874)
#     alphaY: -424.953790 +/- 57.1617901 (13.45%) (init = 247.4874)
#     s:      -117.618429 +/- 19.8264732 (16.86%) (init = 0)
#     xo:      568.562718 +/- 1.58218408 (0.28%) (init = 504)
#     yo:      394.414545 +/- 4.44730449 (1.13%) (init = 383)
#     w1:      0.49286148 +/- 0.04909646 (9.96%) (init = 1)
#     w2:      0.18613239 +/- 0.08840397 (47.50%) (init = 0)
#     w3:      0.55954406 +/- 0.14362584 (25.67%) (init = 0)
#     tx:     -0.28305907 +/- 0.09169507 (32.39%) (init = 0)
#     ty:     -0.34559759 +/- 0.11743981 (33.98%) (init = 0)
#     tz:     -0.37686323 +/- 0.07757996 (20.59%) (init = 0)
#     k2:      0.08372650 +/- 0.15505627 (185.19%) (init = 0)
#     k3:     -0.11033558 +/- 0.24293234 (220.18%) (init = 0)
#     k4:      0.21012479 +/- 0.14065509 (66.94%) (init = 0)
#     k5:     -0.09806998 +/- 0.33438169 (340.96%) (init = 0)
#     k6:     -0.00696046 +/- 0.16066725 (2308.28%) (init = 0)
#     k7:     -0.01665786 +/- 0.17668860 (1060.69%) (init = 0)
#     k8:     -0.02969124 +/- 0.05650754 (190.32%) (init = 0)
#     k9:      0.02088338 +/- 0.03609624 (172.85%) (init = 0)
#     p1:      0.01766883 +/- 0.01883628 (106.61%) (init = 0)
#     p2:     -0.25280753 +/- 0.05854142 (23.16%) (init = 0)
#     p3:      0.20026905 +/- 0.49201352 (245.68%) (init = 0)
#     p4:     -0.24252762 +/- 0.34559885 (142.50%) (init = 0)

#%%
# from detection_contrail.cloudy_detection import detect_sun
# from calibration import f
# import os, sys, datetime, ephem

# lon_sirta=2.208
# lat_sirta=48.713
# obs=ephem.ObserQQawwqqqqqqqqqqQQaqver()
# obs.lon=lon_sirta*ephem.degree
# obs.lat=lat_sirta*ephem.degree
# sun=ephem.Sun()
# minima =[]
# hours = np.arange(10,11)
# epsis = np.zeros([len(hours), 90])
# for j, hour in enumerate(hours):
#     datenow = datetime.datetime(2019, 6, 1, hour, 0, 0)
#     obs.date=datenow
#     sun.compute(obs)
#     theta, phi = psiSun(sun)
#     lamda = 1
#     t = np.zeros([3,1])
#     R = np.eye(3) 
#     XvSoleil = Spherical2Cartesian(R, t, theta, phi)
#     xo = [504, 383]
#     mini = 50000
#     alpha = 0
#     for i, alpha in enumerate(range(50,500,5)):
#         image_path = "/home/ngourgue/Images/2019/06/20190601/20190601"+"%02d"%hour+"0000_03.jpg"
#         image = imread(image_path)
#         image_R = image[:,::-1,:]
#         xs, ys = detect_sun(image, method = "saturation")
#         # output = f(XvSoleil, i, i, s, xo, w, t, k, p)
#         epsi = epsilonfO([alpha,alpha], XvSoleil, [xs, ys])
#         epsis[j, i] = epsi
#         # print("epsilon : %.3f, i : %d"%(epsi, i))
#         # if epsi < mini:
#         #     alphamini = alpha
#         #     mini = epsi
#     # minima.append([mini, alphamini])
# #%%
# from matplotlib import pyplot as plt
# plt.figure()
# plt.imshow(epsis)
# plt.colorbar()
# plt.show()    
#%%
# image[xs-2:xs+2, ys-2:ys+2,:] = [255,0, 0]
# image[xSun-2:xSun+2, ySun-2:ySun+2, :] = [255, 255, 0]
#%% new trying lmfit pas de convergence.
from calibration import f, readCalParams, full
from lmfit import Model, Parameter, report_fit

alphaX, alphaY, s, xo, w, t, k, p = readCalParams(method="csv")
#%%
model = Model(full, independent_vars=['XvSoleil'])

params = model.make_params()
params.add("alphaX", value = alphaX)
params.add("alphaY", value = alphaY)
params.add("s" , value = s)
params.add("xo", value = xo[0])
params.add("yo", value = xo[1])

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

result = model.fit(xSun, params, XvSoleil=XvSoleil)
report_fit(result.params)

#%% new trying scipy optimize
