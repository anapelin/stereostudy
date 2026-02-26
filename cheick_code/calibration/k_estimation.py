#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 17:28:22 2021

@author: ngourgue
"""

#importation

import pandas as pd
import numpy  as np

from matplotlib import pyplot as plt

from calibration import epsilonf, Spherical2Cartesian, f, Cartesian2Polar
from scipy.optimize import curve_fit
#%%
#extract data
sunData = pd.read_csv("/home/ngourgue/Images/2019/2019sun.csv")
addZeros = True
#%%
if addZeros:
   sunData.loc[len(sunData)] = [383, 513, None, 0, 0] 
#prepare question
lamda = 1
t = np.zeros(3)
R = np.eye(3) 
thetaSoleil = sunData["theta"]
phiSoleil   = sunData["phi"]
XvSoleil = Spherical2Cartesian(R, t, thetaSoleil, phiSoleil)

#prepare anwser
xs = sunData["x"]
ys = sunData["y"]
xSun = np.array([xs, ys, np.ones([len(xs)])])

#init param
alphaX = 350/np.pi*2
alphaY = 350/np.pi*2
s = 0.0
ximg = 383.0
yimg = 513.0
xo = [ximg, yimg]
t = np.array([0,0,0], dtype = np.float128)
w = np.array([0.075,0.0,0.0], dtype = np.float128)

k = np.zeros([8], dtype = np.float128)
p = np.zeros([4], dtype = np.float128)
# k = np.array([ -0.20588518,   4.28019682, -22.31093608,  53.68883314,
#         -70.4697131 ,  52.15086227, -20.50236026,   3.33895818])


#error
print("alphaX :", alphaX)
print("alphaY :", alphaY)
print("s :", s)
print("xo :", xo)
print("w : ", w)
print("t :",t)
print("k :", k)
print("p :", p)
xSoleil = f(XvSoleil, alphaX, alphaY, s, xo, w = w, t = t,  k=k, p=p)
epsilons = (xSoleil[0,:]-xSun[0,:])**2+(xSoleil[1,:]-xSun[1,:])**2
rSoleil = np.sqrt((xSoleil[0,:]-ximg)**2+(xSoleil[1,:]-yimg)**2)
rSun    = np.sqrt((xSun[0,:]-ximg)**2+(xSun[1,:]-yimg)**2)

#%% plot r(theta)
plt.figure()
plt.scatter(thetaSoleil, rSoleil/np.sqrt(alphaX**2+alphaY**2), s=2, c='g')
plt.xlabel("theta Soleil")
# maxi = np.max([thetaSoleil, rSun])
# plt.xlim(0,maxi)
# plt.ylim(0,maxi)
plt.ylabel("radius Soleil Calculation normalised")
plt.suptitle("rSoleil(theta)")
plt.plot([0, 1.4], [0, 1.4])
# plt.scatter(rSoleil, rFitSoleilFull, s=2, c='r')
plt.show()

#%% plot rSun(theta)
plt.figure()
plt.scatter(thetaSoleil, rSun/np.sqrt(alphaX**2+alphaY**2), s=2, c='g')
plt.xlabel("theta Soleil")
# maxi = np.max([thetaSoleil, rSun])
# plt.xlim(0,maxi)
# plt.ylim(0,maxi)
plt.ylabel("radius Soleil Saturation normelized")
plt.suptitle("rSat(theta)")
plt.plot([0, 1.4], [0, 1.4])
# plt.scatter(rSoleil, rFitSoleilFull, s=2, c='r')
plt.show()

#%% plot rSun(rSoleil)
plt.figure()
plt.scatter(rSoleil, rSun, s=2, c='g')
plt.xlabel("rSoleil")
# maxi = np.max([thetaSoleil, rSun])
# plt.xlim(0,maxi)
# plt.ylim(0,maxi)
plt.ylabel("radius Sun")
plt.suptitle("rSun(rSoleil)")
# plt.plot([1, maxi], [1, maxi])
# plt.scatter(rSoleil, rFitSoleilFull, s=2, c='r')
plt.show()

#%% plot f(thetaSoleil)
plt.figure()
plt.scatter(thetaSoleil, rSun/np.sqrt(alphaX**2+alphaY**2), s=2, c='g')
plt.xlabel("thetaSoleil")
# maxi = np.max([thetaSoleil, rSun])
# plt.xlim(0,maxi)
# plt.ylim(0,maxi)
plt.ylabel("theta  + somme knthetan")
plt.suptitle("f(thetaSoleil)")
# plt.plot([1, maxi], [1, maxi])
# plt.scatter(rSoleil, rFitSoleilFull, s=2, c='r')
plt.show()
# plt.figure()
# plt.scatter(rSoleil, epsilons, s=2)
# plt.xlabel("radius")
# plt.ylabel("error")
# plt.suptitle("error(radius)")
# plt.show()

#%% polyfit
def sumK(theta, k2, k3, k4, k5, k6, k7, k8, k9):
    x = theta + k2*theta**2 + k3*theta**3 + k4*theta**4 + k5*theta**5 + k6*theta**6 + \
        k7*theta**7 + k8*theta**8 + k9*theta**9
    return(x)

#calcul convert Sat coordonate to r(theta)
xr = (xs-xo[0])/alphaX
yr = (ys-xo[1])/alphaY
rSat = np.sqrt(xr**2+yr**2)

#filter end for stop saturation.
# index = np.where(thetaSoleil<1.3)[0]
thetaSoleilFit = thetaSoleil#[index]
rSatFit        = rSat#[index]

params, cov = curve_fit(sumK, thetaSoleilFit, rSatFit)

# fit = np.polyfit(rSoleil, rSun/np.sqrt(alphaX**2+alphaY**2)-thetaSoleil, 7)
#%% plot fit
# rFitSoleil = rSoleil + fit[7]*rSoleil**2+fit[6]*rSoleil**3+fit[5]*rSoleil**4+\
#                        fit[4]*rSoleil**5+fit[3]*rSoleil**6+fit[2]*rSoleil**7+\
#                        fit[1]*rSoleil**8+fit[0]*rSoleil**9#+fit[9] + fit[8]*rSoleil + 
                       
# rFitSoleilFull = fit[9] + fit[8]*rSoleil + fit[7]*rSoleil**2+fit[6]*rSoleil**3+fit[5]*rSoleil**4+\
#                           fit[4]*rSoleil**5+fit[3]*rSoleil**6+fit[2]*rSoleil**7+\
#                           fit[1]*rSoleil**8+fit[0]*rSoleil**9

                       
# rFitSoleilFull = fit[7] + fit[6]*rSoleil + fit[5]*rSoleil**2+fit[4]*rSoleil**3+fit[3]*rSoleil**4+\
#                           fit[2]*rSoleil**5+fit[1]*rSoleil**6+fit[0]*rSoleil**7


                                                    

plt.figure()
plt.scatter(thetaSoleil, rSat, s=2, c='g')
plt.xlabel("thetaSoleil")
# maxi = np.max([rSoleil, rSun])
# plt.xlim(0,maxi)
# plt.ylim(0,maxi)
plt.ylabel("sumK")
plt.suptitle("estimation of K")
# plt.plot([1, maxi], [1, maxi])
plt.scatter(thetaSoleil, sumK(thetaSoleil, params[0], params[1], params[2], params[3],
                              params[4], params[5], params[6], params[7]), s=2, c='r')
plt.show()

#%% plot error
plt.figure()
plt.scatter(thetaSoleil, (rSun-rSoleil)/np.max([rSun, rSoleil]), s=2)
plt.xlabel("thetaSoleil")
# maxi = np.max([rFitSoleilFull, rSun])
# plt.xlim(0,maxi)
# plt.ylim(0,maxi)
plt.ylabel("radius difference")
plt.suptitle("error(theta)")
# plt.plot([1, maxi], [1, maxi])
plt.show()

plt.figure()
plt.scatter(thetaSoleil, (ys-xSoleil[1,:])/(np.max([ys, xSoleil[1,:]])), s=2)
plt.xlabel("thetaSoleil")
# maxi = np.max([rFitSoleilFull, rSun])
# plt.xlim(0,maxi)
# plt.ylim(0,maxi)
plt.ylabel("y difference")
plt.suptitle("error(theta)")
# plt.plot([1, maxi], [1, maxi])
plt.show()



plt.figure()
plt.scatter(thetaSoleil, (xs-xSoleil[0,:])/(np.max([xs, xSoleil[0,:]])), s=2)
plt.xlabel("thetaSoleil")
# maxi = np.max([rFitSoleilFull, rSun])
# plt.xlim(0,maxi)
# plt.ylim(0,maxi)
plt.ylabel("x difference")
plt.suptitle("error(theta)")
# plt.plot([1, maxi], [1, maxi])
plt.show()

#%% measure phi error 0.075 radian so w[0]:0.75
phie = []
re = []
mini_phi = 10
maxi_phi = -10
for i in range(len(xs)):
    rs, phis = Cartesian2Polar([xs[i], ys[i]])
    rm, phim = Cartesian2Polar([xSoleil[0,i], xSoleil[1,i]])
    # if (phis >0):
    phie.append((phis-phim))#/(abs(phis)+abs(phim)))
    # print("phis : %.3f ; phim : %.3f"%(phis, phim))
    # print("xs : %d, ys : %d; xSol : %d, ySol : %d"%(xs[i], ys[i], xSoleil[0,i], xSoleil[1,i]))
    # if phis < mini_phi or phim < mini_phi:
    #     mini_phi = min(phim, phis)
    # if phis > maxi_phi or phim > maxi_phi:
    #     maxi_phi = max(phim, phis)
    re.append((rs-rm)/rs)

# #%%

# plt.close("all")
# plt.figure()
# plt.scatter(thetaSoleil, phie, s= 2)
# # plt.plot([0.4, 1.4], [2.02, 2.02], c= "r")
# # plt.ylim([1,15])
# plt.show()

# plt.figure()
# plt.hist(phie)
# plt.show()

# plt.figure()
# plt.scatter(thetaSoleil, re, s =2, c= "b")
# # plt.plot([0.4, 1.4], [2.02, 2.02], c= "r")
# # plt.ylim([1,15])
# plt.show()
#%%
# maxi = 10
# plt.figure()
# plt.scatter(xSun[0,:maxi], xSun[1,:maxi], s=1, cmap ='jet', c=np.arange(0, len(xSun[0,:maxi])))
# plt.colorbar()
# plt.scatter(xSoleil[0,:maxi], xSoleil[1,:maxi], s=1, cmap ='cool', c=np.arange(0, len(xSoleil[0,:maxi])))
# plt.colorbar()
# plt.xlim([0,768])
# plt.ylim([0, 1024])
# plt.plot([0,700], [504, 504])
# plt.show()
#%% epsilonf en fonction de chacun des 23 paramètres pour une variation raisonnable de chacun de ces paramètres
beta = np.zeros(23)
beta[0] = alphaX
beta[1] = alphaY
beta[2] = s
beta[3:5] = xo
beta[5:8] = w
beta[8:11] = t
beta[11:19] = k
beta[19:23] = p

xmin  = -0.018
xmax  = -0.010
xstep =  0.00001
ymin  = -0.007
ymax  = -0.005
ystep =  0.00001
X = np.arange(xmin, xmax, xstep)
Y = np.arange(ymin, ymax, ystep)
errors = np.zeros([len(Y), len(X)])
nameX = "w2"
nameY = "w3"

for i, varY in enumerate(Y):
    beta[7]=varY
    for j, varX in enumerate(X):
        beta[6]=varX
        errors[i, j] = epsilonf(beta, XvSoleil, xSun)
    
plt.figure()
plt.imshow(errors, cmap = "jet")
plt.xlabel("%s (%.3f, %3f)"%(nameX, xmin, xmax))
plt.ylabel("%s (%.4f, %.4f)"%(nameY, ymin, ymax))
plt.colorbar()
plt.suptitle("errors("+nameX+","+nameY+")")

plt.show()
    

