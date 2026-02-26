#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:31:36 2022

@author: ngourgue
"""

import os, datetime


from save_picture2 import save_image
from image import readOldImage
import numpy as np
import pandas as pd

rootData = '/homedata'
rootHome = '/home'

def getRootData():
    return rootData

def getRootHome():
    return rootHome

login = os.environ['USER']
pathLoginData = os.path.join(rootData, login)
pathLoginUser = os.path.join(rootHome, login)

def getHomeUser():
    return pathLoginUser

def getHomeData():
    return pathLoginData

pathImages = os.path.join(pathLoginData, 'Images')
pathADSB   = os.path.join(pathLoginData, 'ADSB')
pathWind   = os.path.join(pathLoginData, 'WIND')
pathCOMP   = os.path.join(pathLoginData, 'COMP')



#%% class path
class pathBuilder():
    #%%init
    ##########--init--#########################################################
    def __init__(self, homeData = '/homedata', home = '/home', 
                 bddPLFolderStable = '/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_PL',
                 bddPLFolderTMP = '/bdd/ERA5-RT/NETCDF/GLOBAL_025/hourly/AN_PL',
                 bddSFFolderStable = '/bdd/ERA5/NETCDF/GLOBAL_025/hourly/AN_SF',
                 bddSFFolderTMP = '/bdd/ERA5-RT/NETCDF/GLOBAL_025/hourly/AN_SF',
                 bddADSB = '/bdd/SIRTA/priv/basesirta_HN/kinetic/',
                 bddssh = '/run/user/40395/gvfs/sftp:host=spiritx2.ipsl.fr,user=ngourgue'):
        self.rootData = homeData
        self.rootHome = home
        self.bddPLFolderStable = bddPLFolderStable
        self.bddPLFolderTMP = bddPLFolderTMP
        self.bddSFFolderStable = bddSFFolderStable
        self.bddSFFolderTMP = bddSFFolderTMP
        self.bddADSB = bddADSB
        self.bddssh = bddssh
        
        self.login = os.environ['USER']
        
        #link to bdd
        if 'SESSION_MANAGER' in list(os.environ.keys()) :
            if 'port-562:' in os.environ['SESSION_MANAGER']:
                self.link2bdd = False
            else:
                print('problèmes')
        elif 'DISPLAY' in list(os.environ.keys()):
            if 'spiritx1' in os.environ['DISPLAY'].split('.'):
                self.link2bdd = True
            elif ':0' in os.environ['DISPLAY']:
                print('fail Session manager')
                self.link2bdd = False
            else:
                print('problèmes')
        else:
            print('not implemented')
            ValueError('implement algorithme for a new machine.')
       
    #%% set parameters        
    ##########--Set parameters--###############################################
    def setAuto(self):
        self.setImageFolder(os.path.join(self.rootData, self.login, 'Images'))
        self.setADSBFolder(os.path.join(self.rootData, self.login, 'ADSB'))
        self.setWindFolder(os.path.join(self.rootData, self.login, 'WIND'))
        self.setCompFolder(os.path.join(self.rootData, self.login, 'COMP'))
        self.setPresFolder(os.path.join(self.rootData, self.login, 'PRES'))
        self.setTempFolder(os.path.join(self.rootData, self.login, 'TEMP'))
        
    def setSite(self, site):
        if site == 'SIRTA':
            self.site = site
        elif site in ['FRIPON', 'Orsay', 'PARIS', 'URANO']:
            self.site = site
        else:
            print('site unkonw. Site possible SIRTA, FRIPON. site input :', site)
            return False
        return True
        
    def setImageFolder(self, ImageFolder = 'Images'):
        self.ImageFolder = ImageFolder
        
    def setADSBFolder(self, ADSBFolder = 'ADSB'):
        self.ADSBFolder = ADSBFolder
        
    def setWindFolder(self, WindFolder = 'WIND'):
        self.WindFolder = WindFolder
        
    def setCompFolder(self, COMPFolder = 'COMP'):
        self.CompFolder = COMPFolder
        
    def setPresFolder(self, PRESFolder = 'PRES'):
        self.PresFolder = PRESFolder
        
    def setTempFolder(self, TEMPFolder = 'TEMP'):
        self.TempFolder = TEMPFolder
        
    def setDateDay(self, dateDay):
        self.dateDay = dateDay 
        
    def createFolder(self, path):
        #case path exist and is dir
        if os.path.exists(path) and os.path.isdir(path):
        #do nothing
            print('your path is already exist')
        #case path exist but is file
        elif os.path.exists(path) and os.path.isfile(path):
        #do noting
            print('your path exist but is a file')
        #case path not existe
        elif not os.path.exists(path):
        #test parent path
            #extract parents path
            splitPath = path.split('/')
            parentPath = os.path.join('/', *splitPath[:-1])
            if os.path.exists(parentPath):
                #create folder
                os.makedirs(path)
            else:
                #create parent folder
                # print('problème a corrigé')
                self.createPath(parentPath)
                os.makedirs(path)
                
    def createPath(self, parentPath):
        parentParentPath = '/'.join(parentPath.split('/')[:-1])
        if os.path.exists(parentParentPath):
            #create folder
            os.makedirs(parentPath)
        else:
            #create parent folder
            self.createPath(parentParentPath)
            os.makedirs(parentPath)
        
    def buildData(self):
        self.createFolder(self, os.path.join(self.rootData, self.login, 'Images'))
        self.createFolder(self, os.path.join(self.rootData, self.login, 'ADSB'))
        self.createFolder(self, os.path.join(self.rootData, self.login, 'WIND'))
        self.createFolder(self, os.path.join(self.rootData, self.login, 'COMP'))
        self.createFolder(self, os.path.join(self.rootData, self.login, 'PRES'))
        self.createFolder(self, os.path.join(self.rootData, self.login, 'TEMP'))
    
    #%% get general
    ##########--get general folder/ information --##########        
        
    def getHomeUser(self):
        return os.path.join(self.rootHome, self.login)
    
    def getHomeData(self):
        return os.path.join(self.rootData, self.login)
     
    def getFolderImages(self):
        return os.path.join(self.getHomeData(), self.ImageFolder)
    
    def getFolderADSB(self):
        return os.path.join(self.getHomeData(), self.ADSBFolder)
    
    def getFolderWind(self):
        return os.path.join(self.getHomeData(), self.WindFolder)
    
    def getFolderComp(self):
        return os.path.join(self.getHomeData(), self.CompFolder)
    
    def getFolderPres(self):
        return os.path.join(self.getHomeData(), self.PresFolder)
    
    def getFolderTemp(self):
        return os.path.join(self.getHomeData(), self.TempFolder)
    
    def getSite(self):
        return self.site    
        
    def getBddPLStable(self):
        if self.link2bdd:
            return self.bddPLFolderStable
        else:
            return self.bddssh+self.bddPLFolderStable
    
    def getBddPLTMP(self):
        return self.bddPLFolderTMP
    
    def getBddSFStable(self):
        return self.bddSFFolderStable
    
    def getBddSFTMP(self):
        return self.bddSFFolderTMP
    
    def getBddADSB(self):
        return self.bddADSB
    
    def getBddSSH(self):
        return self.bddssh
    
    
    #%%get Input
    ##########--get Images Input--##########
    
    def getPathImage(self, dateDay = None):
        if dateDay is None:
            dateDay = self.dateDay
        pathFolderImage = self.getPathDay(dateDay)
        if self.site == 'SIRTA':
            imageName = "%04d%02d%02d%02d%02d%02d"%(dateDay.year, dateDay.month, dateDay.day,
                                                      dateDay.hour, dateDay.minute, dateDay.second)+\
                              "_01.jpg"
        elif self.site in ['FRIPON', 'Orsay']:
            imageName = "FRIF02_%04d%02d%02dT%02d%02d%02d"%(dateDay.year, dateDay.month, dateDay.day,
                                                      dateDay.hour, dateDay.minute, dateDay.second)+\
                              "_UT-0.jpg"
        return os.path.join(pathFolderImage, imageName)
    
    def getPathImagePasted(self, dateDay = None):
        if dateDay is None:
            dateDay = self.dateDay
        dateDay = dateDay - datetime.timedelta(seconds = 120)
        filename = self.getPathImage(dateDay)
        return filename
    
    def getPathFolder(self, dateDay = None, deep = 'day'):
        if dateDay is None:
            dateDay = self.dateDay
        if deep == 'site':
            return os.path.join(self.getFolderImages(), self.site)
        if deep == 'year':
            return os.path.join(self.getFolderImages(), self.site, "%04d"%dateDay.year)
        if deep == 'month':
            return os.path.join(self.getFolderImages(), self.site, "%04d"%dateDay.year,
                                "%02d"%dateDay.month)
        if deep == 'day':
            return os.path.join(self.getFolderImages(), self.site, "%04d"%dateDay.year,
                                "%02d"%dateDay.month, "%04d%02d%02d"%(dateDay.year, 
                                                                      dateDay.month, dateDay.day))
        
    def getPathYear(self, dateDay = None):
        if dateDay is None:
            dateDay = self.dateDay
        return self.getPathFolder(dateDay, deep = 'year')
    
    def getPathDay(self, dateDay = None):
        if dateDay is None:
            dateDay = self.dateDay
        return self.getPathFolder(dateDay, deep = 'day')
    
    def getDateDay(self):
        return self.dateDay
    
    def findSecond(self, dateDay = None):
        if dateDay is None:
            dateDay = self.dateDay
        pathMinute = self.getPathImage(dateDay)
        nameMinute = pathMinute.split("/")
        nameMinute = nameMinute[-1]
        nameMinute = nameMinute[:nameMinute.find('.')]  
        nameMinute = nameMinute.split("_")
        nameMinute = nameMinute[-2]
        nameMinute = nameMinute[:-2]       
        
        #faire un os.listDir sur le path
        pathFolderMinute = self.getPathDay(dateDay)
        list_images = os.listdir(pathFolderMinute)
        #faire un filtre sur la liste avec nameMinute
        pathNames = []
        for image_name in list_images:
            if nameMinute in image_name:
                pathNames.append(image_name)
                
        if len(pathNames) == 1:
            pathName = pathNames[0]
        elif len(pathNames) == 2:
            #cas du sirta avec image sous expose
            if self.site == 'SIRTA':
                if '01.jpg' in pathName[0] and '03.jpg' in pathName[1]:
                    pathName = pathName[0]
                elif '01.jpg' in pathName[1] and '03.jpg' in  pathName[0]:
                    pathName = pathName[1]
                else:
                    raise ValueError('problem more than one image with date pattern.')
        else:
            #print('problème il y a plusieurs images pour cette minute')
            raise ValueError('problem more than one image with date pattern.')
            
        #extract second
        pathName = pathName[:pathName.find('.')]  
        pathName = pathName.split("_")
        pathName = ''.join(pathName[:-1])
        nameSec  = pathName[-2:]      
        dateSec  = datetime.datetime(dateDay.year, dateDay.month, dateDay.day, 
                                    dateDay.hour, dateDay.minute, int(nameSec))
        return dateSec
    
    def getBddPathImages(self, dateDay = None):
        if dateDay == None:
            dateDay = self.dateDay
        if self.link2bdd == False:
            return False
        if self.site == 'SIRTA':
            pathBddImages = '/bdd/SIRTA/priv/basesirta/0a/srf02/'
            #add date
            pathBddDate = os.path.join(pathBddImages, "%04d"%dateDay.year,
                                       "%02d"%dateDay.month, 
                                       "%02d"%dateDay.day)
            if os.path.isdir(pathBddDate):
                name_file = os.listdir(pathBddDate)
                return os.path.join(pathBddDate, name_file[0])
            else:
                print('not link to bdd')
                return 'Fail'
            
        else:
            print('this site is not on bdd')
            return 'Fail'
        
    #%% get radar
    ########--get radar--############
    def getPathRadarTime(self, last, dateDay = None):
        #get path to radar file after extraction.
        if dateDay == None:
            dateDay = self.dateDay
        if self.site == 'SIRTA':
            pathRadar = os.path.join(self.getFolderADSB(), "%04d"%dateDay.year,
                                     "%02d"%dateDay.month, dateDay.strftime("%Y%m%d"),
                                     '%04d%02d%02d_hr%02d_min%02d_last%dmin.bst'%
                                     (dateDay.year, dateDay.month, dateDay.day, 
                                      dateDay.hour, dateDay.minute+1, last))
            return pathRadar
        
        elif self.site == 'Orsay':
            pathRadar = os.path.join(self.getFolderADSB(), "%04d"%dateDay.year,
                                     "%02d"%dateDay.month, dateDay.strftime("%Y%m%d"),
                                     '%04d%02d%02d_hr%02d_min%02d_last%dmin.bst'%
                                     (dateDay.year, dateDay.month, dateDay.day, 
                                      dateDay.hour, dateDay.minute+1, last))
            return pathRadar
        else:
            print('site has not ADSB file')
    
    def getPathRadarHour(self, last, method = 'hour', dateDay = None):
        if dateDay == None:
            dateDay = self.dateDay
        if self.site == 'SIRTA':
            if method == 'hour':
                pathRadar = os.path.join(self.getFolderADSB(), "%04d"%dateDay.year,
                                     "%02d"%dateDay.month, dateDay.strftime("%Y%m%d"),
                                     '%04d%02d%02d_hr%02d_last%dmin.bst'%
                                     (dateDay.year, dateDay.month, dateDay.day, 
                                      dateDay.hour, last))
            else:
                print('method not implemented')
            return pathRadar
        
        elif self.site == 'Orsay':
            if method =='hour':
                pathRadar = os.path.join(self.getFolderADSB(), "%04d"%dateDay.year,
                                     "%02d"%dateDay.month, dateDay.strftime("%Y%m%d"),
                                     '%04d%02d%02d_hr%02d_last%dmin.bst'%
                                     (dateDay.year, dateDay.month, dateDay.day, 
                                      dateDay.hour, last))
            else:
                print('method not implemented')
            return pathRadar 
        else:
            print('site has not ADSB file')
    
    def getPathFolderRadar(self, dateDay = None):
        #get path to radar folder
        if dateDay == None:
            dateDay = self.dateDay
        if self.site in ['SIRTA', 'Orsay']:
            pathRadar = os.path.join(self.getFolderADSB(), "%04d"%dateDay.year,
                                     "%02d"%dateDay.month, dateDay.strftime("%Y%m%d"))
            return pathRadar
        else:
            print('site has not ADSB folder')
    
    def getBddPathFileRadar(self, dateDay = None):
        #get path to radar file in bdd folder
        if dateDay == None:
            dateDay = self.dateDay
        if self.link2bdd == False:
            return False
        if self.site == 'SIRTA' or 'Orsay':
            
            pathRadar = self.searchName(self.getBddADSB()+'%04d'%dateDay.year)
            return pathRadar
        else:
            print('site has not ADSB file')     
        
    def getPathFileRadarOrigin(self, dateDay = None, gz = False):
        #get path file before extraction
        if dateDay == None:
            dateDay = self.dateDay
        if self.site in ['SIRTA', 'Orsay']:
            pathFolder = self.getPathFolderRadar(dateDay= dateDay )
            #search name

            pathRadar = self.searchName(pathFolder, gz = gz)
            if type(pathRadar) == bool:
                return pathRadar
            elif type(pathRadar) == str:
                if pathRadar[-3:] == '.gz' and gz == False:
                    #we have .gz file and we don't want
                    pathRadar = pathRadar[:-3]
                elif pathRadar[-3:] != '.gz' and gz == True:
                    #we don't have .gz file and we want
                    pathRadar =  pathRadar + '.gz'
                elif pathRadar[-3:] == '.gz' and gz == True:
                    #we have .gz file and we want
                    pass
                elif pathRadar[-3:] != '.gz' and gz == False:
                    #we don't have .gz file and we don't want
                    pass
            elif type(pathRadar) == list:
                pass
            
            return pathRadar
        else:
            print('site has not ADSB file') 
    
    def searchName(self, path, dateDay = None, time = False, gz = True):
        if dateDay == None:
            dateDay = self.dateDay
        if os.path.exists(path):
            list_File = os.listdir(path)
        else:
            return False
        list_Date = []
        for file in list_File:
            if dateDay.strftime('%Y%m%d') in file:
                list_Date.append(file)
        fileGz = []
        for file in list_Date:
            if '.gz' in file and gz == True and not 'last' in  file:
                fileGz.append(file)
            elif '.bst' in file and gz == False and not '.gz' in file and not 'last' in file:
                fileGz.append(file)
        if  len(fileGz) == 0:
            # return os.path.join(path, fileGz)
            return False
        elif len(fileGz) == 1:
            return os.path.join(path, fileGz[0])
        elif len(fileGz) >1:
            outputPath = []
            for file in fileGz:
                outputPath.append(os.path.join(path, file))
            return outputPath
        else:
            if len(list_Date) == 1:
                return  os.path.join(path, list_Date[0])
            else:
                return False
         
    #%%get save data
    ##########--get saved data--###############################################
            
    def getSavedata(self, data, name = None, dateDay = None):
        if dateDay == None:
            dateDay = self.dateDay
        #folder = self.getSaveFolder(dateDay = dateDay)
        if type(data) == np.ndarray:
            #image case
            save_image(self, date = dateDay, name = name, image = data, extension= ".png")
        elif type(data) == pd.core.frame.DataFrame:
            pass
            
    def getSaveFolder(self, dateDay = None):
        if dateDay == None:
            dateDay = self.dateDay
        path = os.path.join(self.CompFolder, dateDay.strftime('%Y'), dateDay.strftime('%m'),
                            dateDay.strftime('%Y%m%d'))
        return path
            
    def getSavePath(self, name, dateDay = None):
        if dateDay == None:
            dateDay = self.dateDay
        pathName = os.path.join(self.getSaveFolder(dateDay = dateDay), 
                                "%s_%s.csv"%(dateDay.strftime('%Y%m%d_%H%M%S'), name))
        return pathName
    
    def getSaveDataPasted(self, name, dateDay = None):
        if dateDay == None:
            dateDay = self.dateDay
        dateDay = dateDay - datetime.timedelta(seconds = 120)
        pathName = os.path.join(self.getSaveFolder(dateDay = dateDay), 
                                "%04d%02d%02d_%02d%02d00_%s.csv"%(dateDay.year, dateDay.month,
                                   dateDay.day, dateDay.hour, dateDay.minute, name))
        return pathName
    
    def getSaveImage(self, name, dateDay = None):
        if dateDay == None:
            dateDay = self.dateDay
        dateDay = dateDay - datetime.timedelta(seconds = 120)
        pathName = os.path.join(self.getSaveFolder(dateDay = dateDay), 
                                "%04d%02d%02d_%02d%02d00_%s.png"%(dateDay.year, dateDay.month,
                                   dateDay.day, dateDay.hour, dateDay.minute, name))
        return pathName
    
    def getPathOldContrail(self, ext = 'csv', dateDay = None):
        if dateDay == None:
            dateDay = self.dateDay
        
        if ext == 'csv':
            pathName = os.path.join(self.CompFolder, dateDay.strftime('%Y'), dateDay.strftime('%m'),
                                    dateDay.strftime('%Y%m%d'), 
                                    "%04d%02d%02d_%02d%02d00_%s.csv"%(dateDay.year, dateDay.month,
                                   dateDay.day, dateDay.hour, dateDay.minute-2, "contrail"))
        return pathName
        
    def getDatedayTargz(self, dateDay = None, comp = True):
        if dateDay == None:
            dateDay = self.dateDay
        
        if comp == True:
            pathTarz = os.path.join(self.CompFolder, dateDay.strftime('%Y'), dateDay.strftime('%m'),
                                    dateDay.strftime('%Y%m%d')+'.tar.gz')
        else:
            pathTarz = os.path.join(self.CompFolder, dateDay.strftime('%Y'), dateDay.strftime('%m'),
                                    dateDay.strftime('%Y%m%d')+'.tar')
        return pathTarz
    
    def getSaveImageNow(self, name, dateDay = None):
        if dateDay == None:
            dateDay = self.dateDay
        pathName = os.path.join(self.getSaveFolder(dateDay = dateDay), 
                                "%04d%02d%02d_%02d%02d00_%s.png"%(dateDay.year, dateDay.month,
                                   dateDay.day, dateDay.hour, dateDay.minute, name))
        return pathName
    
    def getTarFilename(self, name, dateDay = None):
        if dateDay == None:
            dateDay = self.dateDay
        namefile = '/'.join(self.getSavePath(name, dateDay).split('/')[-2:])
        return namefile
    
    def getTarFilenamePasted(self, name, dateDay = None):
        if dateDay == None:
            dateDay = self.dateDay
        dateDay = dateDay - datetime.timedelta(seconds = 120)
        namefile = self.getTarFilename(name, dateDay = dateDay)
        return namefile
    
    def getTarImage(self, name, dateDay = None):
        if dateDay == None:
            dateDay = self.dateDay
        namefile = '/'.join(self.getSaveImageNow(name, dateDay).split('/')[-2:])
        return namefile
    
    def getTarImagePasted(self, name, dateDay = None):
        if dateDay == None:
            dateDay = self.dateDay
        # dateDay = dateDay - datetime.timedelta(seconds = 120)
        namefile = '/'.join(self.getSaveImage(name, dateDay).split('/')[-2:])
        return namefile
    
    def getImageOutputExist(self, dateDay = None):
        if dateDay == None:
            dateDay = self.dateDay
            
        #check image exist raw
        if os.path.isfile(self.getSaveImageNow(name = 'final_0_type01')):
            return True
        #check tarfile exist raw
        elif os.path.isfile(self.getDatedayTargz(dateDay=dateDay, comp = False)):
            output = readOldImage(self, name = 'final_0_type01' )
            if type(output) == np.ndarray:
                return True
            elif type(output) == bool and output == False:
                return False
            else:
                ValueError('type of output unknow')
        else:
            return None
    #%%get wind
    ###########--get wind--####################################################
    def getPathFolderWind(self, dateDay = None):
        if dateDay == None:
            dateDay = self.dateDay
        if self.site in ['SIRTA', 'Orsay']:
            pathWind = os.path.join(self.getFolderWind(), "%04d"%dateDay.year)
            return pathWind
        else:
            print('site has not Wind folder')
            
    def getBddPathFileWind(self, dateDay = None):
        #get path to radar file in bdd folder
        if dateDay == None:
            dateDay = self.dateDay
        if self.site == 'SIRTA' or 'Orsay':
            bddWindFileU = os.path.join(self.getBddPLStable(),  dateDay.strftime('%Y/u.%Y%m.ap1e5.GLOBAL_025.nc'))
            bddWindFileV = os.path.join(self.getBddPLStable(),  dateDay.strftime('%Y/v.%Y%m.ap1e5.GLOBAL_025.nc'))
                
            if os.path.isfile(bddWindFileU) and os.path.isfile(bddWindFileV):
                if self.link2bdd:
                    os.system(os.path.join(self.getFolderWind(), 'newYear.sh %04d'%(self.getDateDay().year)))
                else:
                    print('extract data on spirit')
                    return False
                windFile  = os.path.join(self.getFolderWind(), '%s_%s.nc'%(dateDay.strftime('%Y/wind_%Y%m'), 
                                                                           self.getSite()))
                if os.path.isfile(windFile):
                    return windFile
                else:
                    return False
                print('process wind file need to be tested')
                
            else:
                #test if temporary file existe
                bddWindFileU = os.path.join(self.getBddPLTMP(),  dateDay.strftime('%Y/u.%Y%m%d.ap1e5t.GLOBAL_025.nc'))
                bddWindFileV = os.path.join(self.getBddPLTMP(),  dateDay.strftime('%Y/u.%Y%m%d.ap1e5t.GLOBAL_025.nc'))
                if os.path.isfile(bddWindFileU) and os.path.isfile(bddWindFileV):
                    os.system(os.path.join(self.getFolderWind(), 'newDay.sh %s'%(self.getDateDay().strftime('%Y%m%d'))))
                    windFile  = os.path.join(self.getFolderTemp(), '%s_%s.nc'%(dateDay.strftime('%Y/wind_%Y%m%d'), 
                                                                               self.getSite()))
                    if os.path.isfile(windFile):
                        return windFile
                    else:
                        return False
                    print('process pressure file need to be tested')
            return pathWind
        else:
            print('site has not ADSB file')             
    
    def getWindFileName(self, dateDay = None):
        if dateDay == None:
            dateDay = self.dateDay
        #test local file
        if os.path.isfile(os.path.join(self.getFolderWind(), 
                                       'wind_%04d%02d_%s.nc'%(dateDay.year, 
                                                              dateDay.month, 
                                                              self.site))):
            windfile = os.path.join(self.getFolderWind(), 
                                    'wind_%04d%02d_%s.nc'%(dateDay.year, 
                                                           dateDay.month, 
                                                           self.site))
        elif os.path.isfile(os.path.join(self.getFolderWind(),"%04d"%dateDay.year, 
                                         'wind_%04d%02d_%s.nc'%(dateDay.year, 
                                                                dateDay.month, 
                                                                self.site))):
            windfile = os.path.join(self.getFolderWind(), "%04d"%dateDay.year,
                                    'wind_%04d%02d_%s.nc'%(dateDay.year, 
                                                           dateDay.month, 
                                                           self.site))
        elif os.path.isfile(os.path.join(self.getFolderWind(), '%s_%s.nc'%(dateDay.strftime('%Y/%m/wind_%Y%m%d'),
                                                                            self.site))):
            windfile = os.path.join(self.getFolderWind(), '%s_%s.nc'%(dateDay.strftime('%Y/%m/wind_%Y%m%d'),
                                                                                self.site))

        else:
            #there are no file in local
            if self.link2bdd == True:
                #test if stable file existe
                bddWindUFile = os.path.join(self.bddPLFolderStable, '%04d'%dateDay.year, 
                                            'u.%04d%02d.ap1e5.GLOBAL_025.nc'%(dateDay.year, 
                                                                              dateDay.month))
                bddWindVFile = os.path.join(self.bddPLFolderStable, '%04d'%dateDay.year, 
                                            'v.%04d%02d.ap1e5.GLOBAL_025.nc'%(dateDay.year, 
                                                                              dateDay.month))
                if os.path.isfile(bddWindUFile) and os.path.isfile(bddWindVFile):
                    os.system(os.path.join(self.getFolderWind(), 'newYear.sh %04d'%(self.getDateDay().year)))
                    windfile = os.path.join(self.getFolderWind(), "%04d"%dateDay.year,
                                            'wind_%04d%02d_%s.nc'%(dateDay.year, 
                                                                   dateDay.month, 
                                                                   self.site))
                    if os.path.isfile(windfile):
                        return windfile
                    else:
                        return False
                    print('process wind file need to be tested')
                else:
                    #test if temporary file existe
                    
                    bddWindUFile = os.path.join(self.bddPLFolderTMP, '%04d'%dateDay.year, 
                                                'u.%04d%02d.ap1e5.GLOBAL_025.nc'%(dateDay.year, 
                                                                                  dateDay.month))
                    bddWindVFile = os.path.join(self.bddPLFolderTMP, '%04d'%dateDay.year, 
                                                'v.%04d%02d%02d.ap1e5t.GLOBAL_025.nc'%(dateDay.year, 
                                                                                       dateDay.month,
                                                                                       dateDay.day))
                    if os.path.isfile(bddWindUFile) and os.path.isfile(bddWindVFile):
                        os.system(os.path.join(self.getFolderWind(), 
                                               'newDay.sh %04d%02d%02d'%(dateDay.strftime('%Y%m%d'))))
                        windfile = os.path.join(self.getFolderWind(), "%04d"%dateDay.year,
                                                "%02d"%dateDay.month, "%s"%(dateDay.strftime('%Y%m%d')),
                                                'wind_%s_%s.nc'%(dateDay.strftime('%Y%m%d'), 
                                                                       self.site))
                        if os.path.isfile(windfile):
                            return windfile
                        else:
                            return False
                        print('process wind file need to be tested')
                    else:
                        print('wind file not exist')
                        return False
                
            else:
                return False
            
        return windfile
    
    #%% get pressure
    ##########--get pres--#####################################################
    def getPresFileName(self, dateDay = None):
        if dateDay == None:
            dateDay = self.dateDay
        
        if os.path.isfile(os.path.join(self.getFolderPres(), '%s_%s.nc'%(dateDay.strftime('%Y/pres_%Y%m'), 
                                                                   self.getSite()))):
            presFile  = os.path.join(self.getFolderPres(), '%s_%s.nc'%(dateDay.strftime('%Y/pres_%Y%m'), 
                                                                       self.getSite()))
            
        elif os.path.isfile(os.path.join(self.getFolderPres(), '%s_%s.nc'%(dateDay.strftime('%Y/%m/pres_%Y%m%d'), 
                                                                   self.getSite()))):
            presFile  = os.path.join(self.getFolderPres(), '%s_%s.nc'%(dateDay.strftime('%Y/%m/pres_%Y%m%d'), 
                                                                       self.getSite()))
        else:
            #there are no file in local
            if self.link2bdd == True:
                #test if stable file existe
                bddPresFile = os.path.join(self.bddSFFolderStable,  dateDay.strftime('%Y/sp.%Y%m.as1e5.GLOBAL_025.nc'))
                if os.path.isfile(bddPresFile):
                    os.system(os.path.join(self.getFolderPres(), 'newYear.sh %04d'%(self.getDateDay().year)))
                    presFile  = os.path.join(self.getFolderPres(), '%s_%s.nc'%(dateDay.strftime('%Y/pres_%Y%m'), 
                                                                               self.getSite()))
                    if os.path.isfile(presFile):
                        return presFile
                    else:
                        return False
                    print('process pressure file need to be tested')
                    
                else:
                    #test if temporary file existe
                    bddPresFile = os.path.join(self.bddSFFolderTMP,  dateDay.strftime('%Y/sp.%Y%m%d.as1e5t.GLOBAL_025.nc'))
                    if os.path.isfile(bddPresFile):
                        os.system(os.path.join(self.getFolderPres(), 'newDay.sh %s'%(self.getDateDay().strftime('%Y%m%d'))))
                        presFile  = os.path.join(self.getFolderPres(), '%s_%s.nc'%(dateDay.strftime('%Y/pres_%Y%m%d'), 
                                                                                   self.getSite()))
                        if os.path.isfile(presFile):
                            return presFile
                        else:
                            return False
                        print('process pressure file need to be tested')
            else:
                return False
            
        return presFile
    
    #%% get temperature
    ###########--get temp #####################################################
    def getTempFileName(self, dateDay = None):
        if dateDay == None:
            dateDay = self.dateDay
            
        if os.path.isfile(os.path.join(self.getFolderTemp(), '%s_%s.nc'%(dateDay.strftime('%Y/temp_%Y%m'), 
                                                                   self.getSite()))):
            tempFile  = os.path.join(self.getFolderTemp(), '%s_%s.nc'%(dateDay.strftime('%Y/temp_%Y%m'), 
                                                                       self.getSite()))

        elif os.path.isfile(os.path.join(self.getFolderTemp(), '%s_%s.nc'%(dateDay.strftime('%Y/%m/temp_%Y%m%d'), 
                                                                   self.getSite()))):
            tempFile  = os.path.join(self.getFolderTemp(), '%s_%s.nc'%(dateDay.strftime('%Y/%m/temp_%Y%m%d'), 
                                                                       self.getSite()))
        else:
            #there are no file in local
            if self.link2bdd == True:
                #test if stable file existe
                bddTempFile = os.path.join(self.bddPLFolderStable,  dateDay.strftime('%Y/ta.%Y%m.ap1e5.GLOBAL_025.nc'))
                if os.path.isfile(bddTempFile):
                    os.system(os.path.join(self.getFolderTemp(), 'newYear.sh %04d'%(self.getDateDay().year)))
                    tempFile  = os.path.join(self.getFolderTemp(), '%s_%s.nc'%(dateDay.strftime('%Y/temp_%Y%m'), 
                                                                               self.getSite()))
                    if os.path.isfile(tempFile):
                        return tempFile
                    else:
                        return False
                    print('process pressure file need to be tested')
                    
                else:
                    #test if temporary file existe
                    bddTempFile = os.path.join(self.bddPLFolderTMP,  dateDay.strftime('%Y/ta.%Y%m%d.ap1e5t.GLOBAL_025.nc'))
                    if os.path.isfile(bddTempFile):
                        os.system(os.path.join(self.getFolderTemp(), 'newDay.sh %s'%(self.getDateDay().strftime('%Y%m%d'))))
                        tempFile  = os.path.join(self.getFolderTemp(),'%s_%s.nc'%(dateDay.strftime('%Y/%m/temp_%Y%m%d'), 
                                                                                   self.getSite()))
                        if os.path.isfile(tempFile):
                            return tempFile
                        else:
                            return False
                        print('process pressure file need to be tested')
            else:
                return False
            
        return tempFile
        
    #%% end
        
        
if __name__ == '__main__':
    nico = pathBuilder()
    nico.createFolder('/home/ngourgue/climavion/toto/tata')