#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 11:36:04 2021

@author: ngourgue
"""
#%%
import os, sys, ephem, math
import numpy as np
# Commented out - AVION module not available and not needed for calibration
# if not '/home/ngourgue/climavion/detection_contrail/' in sys.path:
#     sys.path.insert(1,'/home/ngourgue/climavion/detection_contrail/')
# from AVION.angle import EKO, RTL, inv, world2cam, sph2car#, car2imrep, sincos2phi
# from conversion import convDateStringToInt
#%%
###############################################################################
def input_plane(SE_plane = 20, SE_plane_inside = 9, SE_contr = 2):
    loop = True
    while SE_plane_inside >= SE_plane or loop :
        SE_sizes =input('Enter the sizes of the structuring elements plane, plane_inside, contrail \n default: SE_plane %d SE_plane_inside %d SE_contr %d: '%(SE_plane, SE_plane_inside, SE_contr)).split(' ')
        if SE_sizes==['']:
            pass
        elif len(SE_sizes) == 3:
            SE_plane=int(SE_sizes[0])
            SE_plane_inside=int(SE_sizes[1])
            SE_contr=int(SE_sizes[2])
        else :
            SE_sizes = SE_sizes[0].split(",")
            if len(SE_sizes) == 3:
                SE_plane=int(SE_sizes[0])
                SE_plane_inside=int(SE_sizes[1])
                SE_contr=int(SE_sizes[2])
            else :
                SE_sizes =input('Enter the sizes of the structuring elements plane, plane_inside, contrail \n default: '+\
                        'SE_plane %d SE_plane_inside %d SE_contr %d: '%(SE_plane, SE_plane_inside, SE_contr)).split(' ')
        loop = False
    return ([SE_plane, SE_plane_inside, SE_contr])

###############################################################################

def input_datetime():
    nbdates = input('Are there different dates [default: y] ? ')
    if nbdates == 'n':
        date1 =input("Please enter the date (yyyymmdd):")
        while int(date1[4:6])>12:
            date1=input('There is a problem with the month. Please enter the date again (yyyymmdd):')
        while int(date1[6:])==0:
            date1=input('There is a problem with the date of the day. Please enter the date again (yyyymmdd): ')
        while int(date1[6:])>31:
            date1=input('There is a problem with the date of the day. Please enter the date again (yyyymmdd): ')
        #begin of the time
        timeb = str(input('Please enter the time when beginning (hhmm):'))
        #
        #end of the time	
        timee =str(input('Please enter the time when it ends (hhmm):'))
        while int(timee)<int(timeb):
            timeb=input('There is a probleme with the time. Please enter the time of beginning again (hhmm): ')
            timee=input('There is a probleme with the time. Please enter the time of ending again (hhmm): ')
        date2=date1+timeb+timee
        dates=[date2]
    else:
        source= open('date_time','r')
        dates=source.readlines()
        source.close()
    return(dates)

###############################################################################

def set_dir_path(SE_plane_inside, SE_plane, SE_contr, path_home = None, DirIma = None,
                         DirRadar = None, DirWind = None, DirOutput = None):
        
    if path_home is None:
        path_home = os.path.join('/homedata', os.environ['USER'])
    if not os.path.isdir(path_home):
        print("Error path home doesn't exist :"+path_home)
        return(-1)
    
    if DirIma is None:
        DirIma=os.path.join(path_home,'IMAGES')
    if not os.path.isdir(DirIma):
        while(not os.path.isdir(DirIma) and not DirIma == False):
            DirIma_new = input('DirIma not exist :'+DirIma+' enter False to pass, True to create it or  enter a correct path :')
            if DirIma_new == 'True':
                os.mkdir(DirIma)
                print("DirImages not exist so we create : "+DirIma)
            elif DirIma_new == 'False':
                DirIma = False
            else:
                DirIma = DirIma_new
            
                
    if DirRadar is None:
        DirRadar=os.path.join(path_home, 'ADSB')
    if not os.path.isdir(DirRadar):
        while(not os.path.isdir(DirRadar) and not DirRadar == False):
            DirRadar_new = input('DirRadar not exist :'+DirRadar+' enter False to pass, True to create it or enter a correct path :')
            if DirRadar_new == 'True':
                os.mkdir(DirRadar)
                print("DirRadar not exist so we create : "+DirRadar)
            elif DirRadar_new == 'False':
                DirRadar = False
            else:
                DirRadar = DirRadar_new
                
    if DirWind is None:
        DirWind=os.path.join(path_home, 'WIND')
    if not os.path.isdir(DirWind):
        while(not os.path.isdir(DirWind) and not DirWind == False):
            DirWind_new = input('DirWind not exist :'+DirWind+' enter False to pass, True to create it or enter a correct path :')
            if DirWind_new == 'True':
                os.mkdir(DirWind)
                print("DirWind not exist so we create : "+DirWind)
            elif DirWind_new == 'False':
                DirWind = False
            else:
                DirWind = DirWind_new
                
    if DirOutput is None:
        DirOutput = os.path.join(path_home, 'COMP')
        name_path=str(SE_plane)+'_'+str(SE_plane_inside)+'_'+str(SE_contr)
        DirOutput = os.path.join(DirOutput,name_path)
        if not os.path.isdir(DirOutput):
            while(not os.path.isdir(DirOutput) and not DirOutput == False):
                DirOutput_new = input('DirOutput not exist :'+DirOutput+' enter False to pass, True to create it or enter a correct path :')
                if DirOutput_new == 'True':
                    os.makedirs(DirOutput)
                    print("DirOutput not exist so we create : "+DirOutput)
                elif DirOutput_new == 'False':
                    DirOutput = False
                else:
                    DirOutput = DirOutput_new
            
    return([DirIma, DirRadar,DirWind, DirOutput])
    
##############################################################################

def init_variable (dates, minut_step = None, second = '00', imtype = '01', addmin = 1,
                   last = 3, lin = False, newcal = True):
    #variables
    if minut_step is None:
        yrbeg, mthbeg, daybeg, yrend, mthend, dayend, hrbeg, minbeg, hrend, minend = \
            convDateStringToInt(dates[0])
        if (yrbeg <2017 and mthbeg <12 and daybeg <21):
            minut_step = 1
        else :
            minut_step = 2
            
    if not second in ['00']:
        second = input('second is not correct value :'+second+' enter a correct value :')
    
    if not imtype in ['01', '03']:
        if not imtype in  ['normal', 'underexposed']:
            print("imtype value is not correct. correction : imtype  = 01")
            imtype = '01'
            
    #--flight trajectories
    addmin=1
    while(not last in  [3, 4]):
        last = input('last is not correct value. correct value is in 3, 4 : %d enter a correct value :'%last)
    #
    while not type(lin) is bool:
        lin = input('lin is not correct value. correct value is in True or False :'+str(lin)+' enter a correct value :')
        if lin in ["True", "False"]:
            lin = bool(lin)
    #
    while not type(newcal) is bool:
        newcal = input('newcal is not correct value. correct value is in True or False :'+str(newcal)+' enter a correct value :')
        if lin in ["True", "False"]:
            lin = bool(lin)
            
    EKOmod=EKO(newcal=newcal)
    
    return([minut_step, second, imtype, addmin, last, lin, newcal, EKOmod])
    
###############################################################################

def init_euler (newcal = True):
    #--from loopmoon_RTL4.py
    #--could be provided directly later
    if not newcal:
       euler=[0.00066214,0.0259516,0.04081559]
    else:
       #--original fit
       euler=[0.03912114,0.00788915,0.05074268]
    #
    #--rotation matrix
    rotate=RTL(euler)
    invrotate=inv(RTL(euler))
    return ([rotate, invrotate])

###############################################################################

def call_ephem (dateandtime):
    lon_sirta=2.208
    lat_sirta=48.713
    
    newcal=True
    EKOmod=EKO(newcal=newcal)
    sun=ephem.Sun()
    obs=ephem.Observer()
    obs.lon=lon_sirta*ephem.degree
    obs.lat=lat_sirta*ephem.degree
    obs.date = "%04d" % (dateandtime.year)+'/'+"%02d" %(dateandtime.month)+'/'+ "%02d" % (dateandtime.day)+' '+ \
                "%02d" % (dateandtime.hour)+":"+"%02d" %(dateandtime.minute)+":"+"%02d" % (dateandtime.second)
    # obs.date = dateandtime
    sun.compute(obs)
    print(obs.date)
    print(sun.alt, sun.az)
    theta, phi = np.pi/2.-sun.alt*1., sun.az*1.
    print(theta, phi)
    xsuncal, ysuncal = world2cam(sph2car(1.0,theta,phi)*[-1.,1.,1.],
                              EKOmod,cropped=True,newcal=newcal)
    print(xsuncal, ysuncal)
    return(obs, sun, xsuncal, ysuncal)

##############################################################################

def init_resize (xmax = 901, szamax= 60.0, SE_edge = 8):
    #xmax should be an odd number
    xmax=901
    #--mid point of image
    cx=int(xmax/2)
    cy=int(xmax/2)
    #
    #--limit of SZA for using image
    szamax=60.0
    #--Size of structuring elements for edge
    SE_edge = 8
    return([xmax, cx, cy, szamax, SE_edge])

##############################################################################

def init_threshold(level = "new"):
    #--Hough transform criteria
    #--threshold, min length, max gap and
    #--look at +/- dtheta (in degrees) around flight dir
    if level == "new":
        threshold      = 3 
        houghthreshold = 20
        hminlength     = 10
        #hmaxgap        = 25
        hmaxgap        = 10
        dtheta         = 10.0
        hough          = "line"
        #--max number of directions for Hough transform
        nthetamax      = 20
        
    elif level == "old":
        #--more stringent criteria for old contrails
        threshold      = 11
        houghthreshold = 50  #--has to be larger than houghthreshold
        hminlength     = 30  #--has to be larger  than hminlength
        hmaxgap        = 25  #--has to be smaller than hmaxgap
        dtheta         = 6.0 #--has to be smaller than dtheta
        hough          = "proba"
        nthetamax  = 2*dtheta
        #
    dist2planemax=20
    
    return([threshold, houghthreshold, hminlength, hmaxgap, dtheta, hough, nthetamax, dist2planemax])
    
###############################################################################

def init_output():
    #--open empty dictionary for outputs
    dico={}
    crossing={}
    contrail={}
    x_inter=[]
    y_inter=[]
    x_exact=[]
    y_exact=[]
    return([dico, crossing, contrail, x_inter, y_inter, x_exact, y_exact])

###############################################################################

def init_area (xmax = 901, szamax = 60, verbose = ['']):
    #--surface area of one pixel
    #-- pi/4 * xmax*xmax pixels = pi R**2 = pi H**2 tan(szamax)**2
    #--with H height in km
    km2m=1000.0
    alt10km=10.0
    area_one_pixel_10km=4.*alt10km**2.0*np.tan(math.degrees(szamax))**2./float(xmax)**2.0
    length_one_pixel_at_10km_alt=area_one_pixel_10km**0.5*km2m
    if 'all' in verbose or 'info_pixel' in verbose:
        print('Area of 1 reprojected pixel at 10 km height = %.1f m2' % (area_one_pixel_10km*km2m**2.0))
        print('Length of 1 reprojected pixel at 10 km height = %.1f m' % length_one_pixel_at_10km_alt)
    return [km2m, alt10km, area_one_pixel_10km, length_one_pixel_at_10km_alt]

###############################################################################

def init_mask (imBW, xmax, SE_edge):
    #--initialising masks
    #--in principle x=y=xmax
    if len(imBW.shape) == 3:
        x, y, _ = imBW.shape
    elif len(imBW.shape) == 2:
        x, y = imBW.shape
        
    if (x!= y or x!= xmax):
        print('Attention xmax, x and y don t have same size. x: %d y: %d, xmax : %d' % (x,y, xmax))
      
    #--maskIM = image w/o edges
    maskIM = np.ones((x,y),dtype=np.uint8)
    maskIM[0:SE_edge,:]  =0
    maskIM[x-SE_edge:x,:]=0
    maskIM[:,1:SE_edge]  =0
    maskIM[:,y-SE_edge:y]=0
    
    return(maskIM)
