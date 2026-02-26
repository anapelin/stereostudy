 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 12:20:37 2021

@author: ngourgue

different fonction about pre open, open en transformed image before detection process.

function : 
    imagetime : extract exact time of picture.
    
    readimage : open and read image.
    
    pre_processing : realise the different transformation to raw image.
    
    zoom_image :
        zoom_image : function to zoom on raw image, delete black strip and defrome image
        
        zoom_with_loop : old function for calculate new pixel with raw image with double loop for.
                         very colstly in time.
        zoom_without_loop : new function for calculate new pixel with raw image with one loop for
                            and matrix calculation.
                            
    channel_image : choice the channnel to conserve. red green blue gray.
    
    otsu processing : function which use local otsu threshold and substrat this thres to original
                      image to delete variation of contrast.
                      
    cropped : delete black strip arround image.
    
    diff : make difference between last image and currently image.
    

objet :
    croopedSize : save value of cropped
"""
#%% importation
import time, copy, os, sys, datetime, subprocess, tarfile, zipfile, shutil
import numpy as np
import pandas as pd
import zipfile as zp

from skimage.morphology import disk
from skimage.filters    import rank
from skimage.io         import imread
from skimage.color      import rgb2hsv
from PIL                import Image

if not '/home/ngourgue/.conda/envs/climavion/lib/python3.6/site-packages' in sys.path:
    sys.path.insert(1,'/home/ngourgue/.conda/envs/climavion/lib/python3.6/site-packages')
import exifread

from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

# from check_file2 import unZipImage#, check_ImageUnzip
from cloud       import cloud_segmentation#, cloudy_or_notCloudy
# from deco import timeDuration  # Commented out - library not available
from calibration import zoom_image, cropped
#%% extrat name
def secFripon(name_image, DirImages, name, date, timeImage):
    if DirImages in name_image:
        start = name_image.find(DirImages)
        name_image = name_image[start+len(DirImages)+1:]
    if name in name_image:
        start = name_image.find(name)
        name_image = name_image[start+len(name):]
    if date in name_image:
        start = name_image.find(date)
        name_image = name_image[start+len(date)+1:]
    if timeImage in name_image:
        start = name_image.find(timeImage)
        name_image = name_image[start+len(timeImage):]
        realsec = int(name_image[:2])
        return realsec
    return 'Fail'

#%% extract exact image time
# @timeDuration  # Commented out - library not available
def imagetime(path, imtype, site = 'SIRTA', cle = False, verbose = ['']):
    """
    extract exact second of image.

    Parameters
    ----------
    DirImages : String
        Path Images root folder.
    imageDateInput : String
        Image's name.
    imtype : Int
        Number about exposition. 1 images is normal, 3 image is unexposed.
    Site : String, optional
        SIRTA or FRIPON. Folder where are image.  The default is SIRTA.
    cle : thearding.RLock, optional
        Lock to limit write and read acces to one thread.  The default is False.
    verbose : list of String, optional
        List of things to printing. The default is [''].

    Returns
    -------
    Float
        Second exact to take image.
    !bool
        Avertissement if différent time in image.

    """
    
    # timeMax = 7200
    if site == 'SIRTA' :
        #extract metadata
        if True :
        # if cle == False : 
        #     if os.path.isfile(path.getPathImages()):
        #         tags = exifread.process_file(open(path.getPathImages(),'rb'))
        #     else: #--second chance
        #         #imageDateInput = imageDateInput[:-2]+"%02d"%(int(imageDateInput[-2:])+1)
        #         try :
        #             tags=exifread.process_file(open(path.getPathImages(
        #                 path.getDateDay()+ datetime.timedelta(seconds = 1)),'rb'))
        #         except:
        #             try :
        #                 #extract pathFolder
        #                 pathFolderFull = path.getPathFolder().split('/')
        #                 pathFolder = '/'.join(pathFolderFull[:-1])
        #                 if os.path.isdir(pathFolder):
        #                     listFiles = os.listdir(pathFolder)
        #                     listZips = []
        #                     for file in listFiles:
        #                         if '.zip' in file:
        #                             listZips.append(file)
        #                     listDay = []
        #                     for zipFile in listZips:
        #                         if pathFolderFull[-1] in zipFile:
        #                             listDay.append(zipFile)
        #                     if len(listDay) == 1:
        #                         fileZip = listDay[0]
        #                         try:
        #                             with zp.ZipFile(os.path.join(pathFolder, fileZip), 'r') as zipF:
        #                                 dateTime = path.getDateDay()
        #                                 imageDate =  "%04d%02d%02d%02d%02d%02d"%(dateTime.year, dateTime.month,  dateTime.day,
        #                                       dateTime.hour, dateTime.minute, dateTime.second)
        #                                 with zipF.open(imageDate[:8]+'/'+imageDate[:-2]+'00_'+imtype+'.jpg') as myImage:
        #                                     # image = Image.open(myImage)
        #                                     tags = exifread.process_file(myImage)

        #                         except:
        #                             print(path.getPathImages(), ' not found')
        #                     else:
        #                         print(path.getPathImages(), ' not found')
        #                 else:
        #                     print(path.getPathImages(), ' not found')
        #             except:
        #                 return None, 'Fail'
        # else:   
        #     cle.acquire(timeout = timeMax)
            if os.path.isfile(path.getPathImage()):
                tags = exifread.process_file(open(path.getPathImage(),'rb'))
            else: #--second chance
                # imageDateInput = imageDateInput[:-2]+"%02d"%(int(imageDateInput[-2:])+1)
                try :
                    tags=exifread.process_file(open(path.getPathImage(
                        path.getDateDay()+ datetime.timedelta(seconds = 1)),'rb'))
                except:
                    #search if folder day exist
                    if os.path.isdir(path.getPathFolder(deep = 'month')+'/%02d'%path.getDateDay().day):
                        #open unzip and delete
                        unZipImage([path.getDateDay()], path.getPathFolder(deep = 'site'))
                    else:
                        #search if file day exist
                        pathMonth = path.getPathFolder(deep = 'month')
                        if not os.path.isdir(pathMonth):
                            os.makedirs(pathMonth)
                        filesAndFolders = os.listdir(pathMonth)
                        fileDay = None
                        for fileOrFolder in filesAndFolders:
                            if path.getDateDay().strftime('%Y%m%d') in fileOrFolder\
                                and 'zip' in fileOrFolder:
                                fileDay = fileOrFolder
                        if not fileDay is None:
                            #unzip method
                            # os.system('unzip -d '+path.getPathFolder(deep = 'month')+' '+\
                            #            os.path.join(path.getPathFolder(deep = 'month'), fileDay))
                            # os.remove(os.path.join(path.getPathFolder(deep = 'month'), fileDay))
                            
                            #extract tag in zip
                            fullNameImage = path.getPathImage()
                            fullListImage = fullNameImage.split('/')
                            lastNameImage = '/'.join(fullListImage[-2:])
                            with zp.ZipFile(os.path.join(pathMonth, fileDay), 'r') as zipF:
                                try :
                                    with zipF.open(lastNameImage) as myImage:
                                        tags = exifread.process_file(myImage)
                                except :
                                    try:
                                        with zipF.open(lastNameImage[:21]+'01'+lastNameImage[23:]) as myImage:
                                            tags = exifread.process_file(myImage)
                                    except :
                                        try:
                                            with zipF.open(lastNameImage[:21]+'02'+lastNameImage[23:]) as myImage:
                                                tags = exifread.process_file(myImage)
                                        except:
                                            return None, 'Fail'
                        else:
                            #search in bdd
                            if os.path.isfile(path.getBddPathImages()):
                                # check_ImageUnzip(path, verbose)
                                #extract tag in zip
                                os.system('cp '+path.getBddPathImages()+' '+path.getPathFolder(deep='month')+'/')
                                
                                filesAndFolders = os.listdir(pathMonth)
                                fileDay = None
                                for fileOrFolder in filesAndFolders:
                                    if path.getDateDay().strftime('%Y%m%d') in fileOrFolder\
                                        and 'zip' in fileOrFolder:
                                        fileDay = fileOrFolder
                                if not fileDay is None:
                                    #unzip method
                                    # os.system('unzip -d '+path.getPathFolder(deep = 'month')+' '+\
                                    #            os.path.join(path.getPathFolder(deep = 'month'), fileDay))
                                    # os.remove(os.path.join(path.getPathFolder(deep = 'month'), fileDay))
                                    
                                    #extract tag in zip
                                    fullNameImage = path.getPathImages()
                                    fullListImage = fullNameImage.split('/')
                                    lastNameImage = '/'.join(fullListImage[-2:])
                                    with zp.ZipFile(os.path.join(pathMonth, fileDay), 'r') as zipF:
                                        with zipF.open(lastNameImage) as myImage:
                                            tags = exifread.process_file(myImage)
                                else:
                                    return None, 'Fail'
                            else:
                                return None, 'Fail'
                    # tags=exifread.process_file(open(path.getPathImages(
                    #         path.getDateDay()+ datetime.timedelta(seconds = 1)),'rb'))
            # cle.release()
        
        #comparaison date
        date =path.getDateDay().strftime("%Y%m%d")
        date1=str(tags['Image DateTime'])
        date2=str(tags['EXIF DateTimeOriginal']) 
        date3=str(tags['EXIF DateTimeDigitized']) 
        warn=False
        if not date1==date2==date3: 
            warn=True
            if not 'no' in verbose:
                print('Warning EXIF dates=', date1, date2, date3)
        if date1[0:4]+date1[5:7]+date1[8:10] != date: 
            warn=True
            if not 'no' in verbose:
                print('Warning date inconsistent=', date1[0:4]+date1[5:7]+date1[8:10], date)
        if date1[11:13] != "%02d"%path.getDateDay().hour:
            warn=True
            if not 'no' in verbose:
                print('Warning hour inconsistent=', date1[11:13], "%02d"%path.getDateDay().hour)
        if date1[14:16] != "%02d"%path.getDateDay().minute:
            warn=True
            if not 'no' in verbose :
                print('Warning minute inconsistent=', date1[14:16], "%02d"%path.getDateDay().minute)
            
        realsec = float(date1[17:19])
        
    elif site == 'Orsay':
        name = 'FRIF04_'
        date = path.getDateDay().strftime("%Y%m%d")
        timeImage = '%02d%02d'%(path.getDateDay().hour, path.getDateDay().minute)
        name_images = subprocess.check_output('ls '+os.path.join(path.getPathFolder(), name+date+'T'+timeImage+'*'), 
                                              shell=True).decode('utf-8').strip().split('\n')
        if len(name_images) == 1:
            name_image = name_images[0]
            realsec = secFripon(name_image, path.getPathFolder(), name, date, timeImage)
            if realsec == 'Fail':
                warn = realsec
            else:
                warn = False
        else:
            for name_Image in name_images:
                if '.jpg' in name_Image:
                    name_image = name_Image
                    realsec = secFripon(name_image, path.getPathFolder(), name, date, timeImage)
                    if realsec == 'Fail':
                        warn = realsec
                    else:
                        warn = False
    
    else:
        raise ValueError('Site is not know. Site :', site)
        
    return realsec, warn #--return exact second of image

###############################################################################

#%% read image
# @timeDuration  # Commented out - library not available
def readimage(DirectoryImage, imageDate, imtype, site = 'SIRTA', cle = False, verbose = ['']):
    """
    Function to read image. 
    
    With path of folder of image the function build all path with arborencence.

    Parameters
    ----------
    DirectoryImage : String
        Path full to image.
    imageDate : String
        Image's name.
    imtype : Int
        Is 1 if image is normal 3 if image is unexposed.
    site : String, optional
        SIRTA or FRIPON. Folder where are image.  The default is SIRTA.
    cle : thearding.RLock,
        Lock to limit write and read acces to one thread.
    verbose : list of String, optional list of string
        List of things to printing. The default is [''].

    Returns
    -------
    found : TYPE bool
        DESCRIPTION found image or not.
    imRGB : TYPE array
        DESCRIPTION image open and read.

    """
    # timeMax = 7200
    #--Load the images
    #--Modify these directories to adapt
    #--The directory for the images
    if site == 'SIRTA':
        pathImage = os.path.join(DirectoryImage, imageDate+'_'+imtype+'.jpg')
        #--read image
        if os.path.isfile(pathImage): 
            imRGB = imread(pathImage)
            found = True
            if 'all' in verbose or 'debug' in verbose or 'path' in verbose:
                print(pathImage)
        else:
            #--a second chance to read image with an extra second
            imageDate = imageDate[:-1]+"1"
            pathImage = os.path.join(DirectoryImage, imageDate+'_'+imtype+'.jpg')
            if os.path.isfile(pathImage): 
                if True:
                # if cle == False:
                #     imRGB = imread(pathImage)
                # else:
                #     cle.acquire(timeout = timeMax)
                    imRGB = imread(pathImage)
                    # cle.release()
                found = True
                if 'all' in verbose or 'debug' in verbose or 'path' in verbose:
                    print(pathImage)
            else:
                #extract pathFolder
                pathFolder = DirectoryImage.split('/')
                pathFolder = '/'.join(pathFolder[:-1])
                if os.path.isdir(pathFolder):
                    listFiles = os.listdir(pathFolder)
                    listZips = []
                    for file in listFiles:
                        if '.zip' in file:
                            listZips.append(file)
                    listDay = []
                    for zipFile in listZips:
                        if imageDate[:8] in zipFile:
                            listDay.append(zipFile)
                    if len(listDay) == 1:
                        fileZip = listDay[0]
                        try:
                            with zp.ZipFile(os.path.join(pathFolder, fileZip), 'r') as zipF:
                                with zipF.open(imageDate[:8]+'/'+imageDate[:-2]+'00_'+imtype+'.jpg') as myImage:
                                    image = Image.open(myImage)
                                    imRGB = np.array(image)
                                    found = True
                        except:
                            if not 'no' in verbose:
                                print(pathImage, ' not found')
                            imRGB = 0
                            found=False
                    else:
                        if not 'no' in verbose :
                            print(pathImage, ' not found')
                        imRGB = 0
                        found=False
                else:
                    if not 'no' in verbose:
                        print(pathImage, ' not found')
                    imRGB = 0
                    found=False
        
    elif site == 'Orsay':
        name_images = subprocess.check_output('ls '+os.path.join(DirectoryImage, '*'+ imageDate[:8]+'T'+imageDate[8:12]+'*'),
                                     shell=True).decode('utf-8').strip().split('\n')
        if len(name_images) == 1 :
            pathImage = name_images[0]
            try :
                image_file = get_pkg_data_filename(pathImage)
                imRGB = fits.getdata(image_file, ext=0)
                imRGB = np.transpose(imRGB)
                found = True
            except :
                #copy file open and delete.
                #make physical link to take less size
                pathTmp = os.getcwd()+'/image'
                newPath = pathTmp+'/'+pathImage.split('/')[-1]
                command = 'ln '+pathImage+' '+newPath
                os.system(command)
                image_file = get_pkg_data_filename(newPath)
                imRGB = fits.getdata(image_file, ext=0)
                found = True
                imRGB = np.transpose(imRGB)
                os.system('rm '+newPath)
        else:
            for name_Image in name_images:
                if '.fit' in name_Image:
                    pathImage = name_Image
                    os.system('ln '+pathImage+' '+'image/'+os.path.split(pathImage)[-1])
                    image_file = get_pkg_data_filename(os.path.join(os.getcwd(),'image',  os.path.split(pathImage)[-1]))
                    imRGB = fits.getdata(image_file, ext=0)
                    #imRGB = imRGB[::-1, :]
                    #imRGB = imread(pathImage)
                    found = True
                    break
                elif '500x500' in name_Image:
                    continue
                elif '.jpg' in name_Image:
                    pathImage = name_Image
                    imRGB = imread(pathImage)
                    found = True
                    
    if 'all' in verbose or 'debug' in verbose or 'path' in verbose:
        print(pathImage)
        
    return found, imRGB
###############################################################################

def readOpenZip(path, pathFolder, fileZip, verbose = ['']):
    try:
        with zp.ZipFile(os.path.join(pathFolder, fileZip) , 'r') as zipF:
            with zipF.open(os.path.join(path.getDateDay().strftime('%Y%m%d'),
                                        path.getPathImage().split('/')[-1])) as myImage:
                image = Image.open(myImage)
                imRGB = np.array(image)
                found = True
    except:
        if not 'no' in verbose:
            print(path.getPathImage(), ' not found')
        imRGB = 0
        found=False
    return found, imRGB
    
    
###############################################################################

def readimage2(path, imtype, cle = False, verbose = ['']):
    """
    Function to read image. 
    
    With path of folder of image the function build all path with arborencence.

    Parameters
    ----------
    DirectoryImage : String
        Path full to image.
    imageDate : String
        Image's name.
    imtype : Int
        Is 1 if image is normal 3 if image is unexposed.
    site : String, optional
        SIRTA or FRIPON. Folder where are image.  The default is SIRTA.
    cle : thearding.RLock,
        Lock to limit write and read acces to one thread.
    verbose : list of String, optional list of string
        List of things to printing. The default is [''].

    Returns
    -------
    found : TYPE bool
        DESCRIPTION found image or not.
    imRGB : TYPE array
        DESCRIPTION image open and read.

    """
    #--Load the images
    if path.getSite() == 'SIRTA':
        pathImage = path.getPathImage()
        #--read image
        if os.path.isfile(pathImage): 
            imRGB = imread(pathImage)
            found = True
            if 'all' in verbose or 'debug' in verbose or 'path' in verbose:
                print(pathImage)
        else:
            #--a second chance to read image with an extra second
            path1 = copy.deepcopy(path)
            path1.setDateDay(datetime.datetime(path.getDateDay().year,
                                               path.getDateDay().month,
                                               path.getDateDay().day,
                                               path.getDateDay().hour,
                                               path.getDateDay().minute,
                                               path.getDateDay().second + 1))
            pathImage = path1.getPathImage()
            if os.path.isfile(pathImage): 
                imRGB = imread(pathImage)
                found = True
                if 'all' in verbose or 'debug' in verbose or 'path' in verbose:
                    print(pathImage)
            else:
                #extract pathFolder
                pathFolder = path.getPathDay()
                if os.path.isdir(pathFolder):
                    listFiles = os.listdir(pathFolder)
                    #use dataFrame to test without loop
                    dfFile = pd.DataFrame(listFiles, columns=['filename'])
                    filenames = dfFile[dfFile['filename'].str.contains('.zip')]
                    filename = filenames[filenames['filename'].str.contains(path.getDateDay().strftime('%Y%m%d'))]
                    if filename.shape[0] == 1:
                        fileZip = filename.values[0, 0]
                        # try:
                        #     with zp.ZipFile(os.path.join(pathFolder, fileZip) , 'r') as zipF:
                        #         with zipF.open(os.path.join(path.getDateDay().strftime('%Y%m%d'),
                        #                                     path.getPathImage().split('/')[-1])) as myImage:
                        #             image = Image.open(myImage)
                        #             imRGB = np.array(image)
                        #             found = True
                        # except:
                        #     if not 'no' in verbose:
                        #         print(pathImage, ' not found')
                        #     imRGB = 0
                        #     found=False
                        found, imRGB = readOpenZip(path, pathFolder, fileZip, verbose = [''])
                    else:
                        if not 'no' in verbose :
                            print('not implemented')
                            print(pathImage, ' not found')
                        imRGB = 0
                        found=False
                else:
                    #test in folder month
                    pathFolder  = path.getPathFolder(deep='month')
                    if os.path.isdir(pathFolder):
                        listFiles = os.listdir(pathFolder)
                        dfFile = pd.DataFrame(listFiles, columns=['filename'])
                        filenames = dfFile[dfFile['filename'].str.contains('.zip')]
                        filename = filenames[filenames['filename'].str.contains(path.getDateDay().strftime('%Y%m%d'))]
                        if filename.shape[0] == 1:
                            fileZip = filename.values[0, 0]
                            # try:
                            #     with zp.ZipFile(os.path.join(pathFolder, fileZip) , 'r') as zipF:
                            #         with zipF.open(os.path.join(path.getDateDay().strftime('%Y%m%d'),
                            #                                     path.getPathImage().split('/')[-1])) as myImage:
                            #             image = Image.open(myImage)
                            #             imRGB = np.array(image)
                            #             found = True
                            # except:
                            #     if not 'no' in verbose:
                            #         print(pathImage, ' not found')
                            #     imRGB = 0
                            #     found=False
                            found, imRGB = readOpenZip(path, pathFolder, fileZip, verbose = [''])
                        else:
                            if not 'no' in verbose :
                                print('not implemented')
                                print(pathImage, ' not found')
                            imRGB = 0
                            found=False
                    else:
                        if not 'no' in verbose:
                            print(pathImage, ' not found')
                        imRGB = 0
                        found=False
        
    else:
        print('not implemented')
        ValueError('need to be impletemed')
                    
    if 'all' in verbose or 'debug' in verbose or 'path' in verbose:
        print(pathImage)
        
    return found, imRGB

#%% fripon
def delSunFripon(image):
    image1 = copy.deepcopy(image)
    if image1.shape[0] == 1280 and image1.shape[1] == 960:
        counts, bins = np.histogram(image1, bins=1000)
        maxi = counts.argsort()
        if 999 in maxi[-5:]:
            #test if saturation is in 5 more importante.
            image1[image1>bins[999]] = 0
        thres = np.max(np.where(counts > 100))
        thresIm = bins[thres]
        image1[image1>thresIm] = 0
        
        image1 = image1/np.max(image1)*255
        image1 = np.array(image1, dtype = np.uint8)
        
        return image1
            
    else:
        raise ValueError('Image size not good. expected 960*1280')
        
###############################################################################

#%% pre_processing
# @timeDuration  # Commented out - library not available
def pre_processing (image, processing, path = False, cle = False, verbose = ['']):
    """
    Apply different transformation about raw image.
    
    Tansformation like cropped, zoom, otsu filter, diff.
    
    cropped : delete stramp band at left and right. image out size 678*678.
    zoom : can remove after szamax on image. inclue cropped. image out size xmax*xmax.
        zoom include a defomration on original image to restaure straight line.
    otsu : substract ostu local threshold to the current image. delete contract but 
        increase artefact reponse to top hat.
    diff : substract last image to currently image. Function well with blue sky. 
        but every cloud can become artefact. More interesting if minute_step descrease.                                            

    Parameters
    ----------
    image : Array
        I mage raw input.
    processing : TYPE list of dic.
        DESCRIPTION list of filter to apply. for each filter contain a dictionnay
        with all parameters like color, radius.
    cle : thearding.RLock,
        Lock to limit write and read acces to one thread.
    verbose : list of String, optional list of string
        List of things to printing. The default is [''].

    Returns
    -------
    image1 : Array
        Image after filter.

    """
    # start_time=time.time()
    
    image1 = copy.deepcopy(image)
    #loop to do each processus
    for processus in processing:
        #otsu substract local otsu threshold
        if "otsu" == processus['name']:
            if "color" in processus.keys():
                image1 = channel_image(image1, processus['color'])
                image1 = otsu_processing(image1, processus['radius'], verbose)
            else:
                image1 = otsu_processing(image1, processus['radius'], verbose)
        #zoom on the original image
        elif "zoom" == processus['name']:
            image1 = zoom_image(image = image1, xmax = processus['xmax'], szamax = processus['szamax'], 
                                verbose = verbose,  with_loop=False)
        #cropped, delete black stipe left and right.
        elif "cropped" == processus['name']:
            image1 = cropped(image = image1)
        
        # #diff, substract last image to the currently image
        # elif "diff" == processus['name']:
        #     image1 = diff(image = image1, path = path, cle = cle, verbose = verbose)
        elif "color" == processus['name']:
            image1 = channel_image(image1, processus['color'])
            
    # dtime=time.time()-start_time
    # if 'all' in verbose or 'time' in verbose:
    #     print("duration of pre processing image : %.2f s" %(dtime))
    return image1

###############################################################################

# #%% zoom_image
# def zoom_with_loop(image, xmax, szamaxrad, xIm, yIm, xImc, yImc, lin = False, verbose= ['']):
#     """
#     make zoom with double loop for.
#     it's costly in time but it's the original transformation.

#     Parameters
#     ----------
#     image : Array
#         Input image.
#     xmax : Int
#         Size of output image.
#     szamaxrad : Float
#         Angle in radian to zoom.
#     xIm : Int
#         Width util of input image.
#     yIm : Int
#         Height util of input image.
#     xImc : Int
#         X center of input image.
#     yImc : Int
#         Y center of input image.
#     lin : Bool, optional 
#         The default is False.
#     verbose : List of string, optional 
#         The default is [''].

#     Returns
#     -------
#     imz : Array
#         Image with zoom.

#     """
    
#     # start_time = time.time()
    
#     imz = np.zeros([xmax, xmax, 3], dtype=np.uint8)
#     for ix in range(0,xmax):
#         for iy in range(0,xmax):
#             dx=ix-int(xmax/2)
#             dy=iy-int(xmax/2)
#             dr=(dx**2.0+dy**2.0)**0.5
#             if dr < xmax/2.: 
#                 if dr != 0: 
#                     cosalpha=dx/dr
#                     sinalpha=dy/dr
#                 else: 
#                     cosalpha=1.0
#                     sinalpha=0.0
#                 if lin:
#                     #regridding to a hemispheric map
#                     theta=dr*szamaxrad*2./float(xmax)
#                 else:
#                     #regridding to a flat map
#                     theta=np.arctan(dr*np.tan(szamaxrad)*2./float(xmax))
#                 #--find pixels
#                 iix=int(round(xImc+theta/np.pi*float(xIm)*cosalpha))
#                 iiy=int(round(yImc+theta/np.pi*float(yIm)*sinalpha))
#                 #--regrid
#                 imz[ix,iy,:] = np.uint8(image[iix,iiy,:])
                
#     # dtime = time.time() - start_time
#     # if 'all' in verbose or 'time' in verbose:
#     #     print("time for loop zoom : %.2f"%dtime)
        
#     return imz


# ###############################################################################

# def zoom_wihout_loop(image, xmax, szamaxrad, xIm, yIm, xImc, yImc, verbose = ['']):
#     """
#     zoom on image without loop. it's near to instantane and the result is same
#     than zoom with loop.

#     Parameters
#     ----------
#     image : Array
#         Input image.
#     xmax : Int
#         Size width and height of output image.
#     szamaxrad : Float
#         Angle in radian about zoom.
#     xIm : Int
#         Width until to input image.
#     yIm : Int
#         Height until to input image.
#     xImc : Int
#         Width center to input image.
#     yImc : Int
#         Height center to input image.
#     verbose : , optional
#         The default is [''].

#     Returns
#     -------
#     imageZoom : Array
#         Image with zoom.

#     """
    
#     # start_time = time.time()
    
#     image1 = copy.deepcopy(image)
#     #create matrix
#     matImage = np.zeros([xmax, xmax, 8])
    
#     #create coordoante x,y
#     #discusion if int is necessary
#     for i in range(xmax):
#         matImage[i,:, 0] = i-int(xmax/2)
#         matImage[:,i, 1] = i-int(xmax/2)
        
#     #create radius
#     matImage[:,:, 2] = (matImage[:,:, 0]**2.0+matImage[:,:,1]**2.0)**0.5
    
#     #create alpha
#     #cas radius >0
#     indicOmega = np.where(matImage[:,:,2] > 0)
#     #cas radius < xmax/2
#     indicAlpha = np.where(matImage[indicOmega[0], indicOmega[1],2] < xmax/2)[0]
#     matImage[indicOmega[0][indicAlpha], indicOmega[1][indicAlpha], 3] = \
#     matImage[indicOmega[0][indicAlpha], indicOmega[1][indicAlpha], 0]/matImage[indicOmega[0][indicAlpha], 
#                                                                         indicOmega[1][indicAlpha], 2]
#     matImage[indicOmega[0][indicAlpha], indicOmega[1][indicAlpha], 4] = \
#     matImage[indicOmega[0][indicAlpha], indicOmega[1][indicAlpha], 1]/matImage[indicOmega[0][indicAlpha], 
#                                                                         indicOmega[1][indicAlpha], 2]
#     #cas radius = 0
#     indicOmicron = np.where(matImage[:,:,2] == 0)
#     matImage[indicOmicron[0], indicOmicron[1], 3] = 1
#     matImage[indicOmicron[0], indicOmicron[1], 4] = 0
    
#     #delete radius > xmax/2
#     indicBeta = np.where(matImage[:,:,2] > xmax/2)
#     matImage[indicBeta[0], indicBeta[1], 3:] = 0
    
#     #create theta, theta max = np.pi/3
#     matImage[:,:,5] = np.arctan(matImage[:,:,2]*np.tan(szamaxrad)*2./float(xmax))
#     matImage[indicBeta[0], indicBeta[1], 3:] = 0
    
#     matImage[:,:,6] = np.round(xImc+matImage[:,:, 5]*float(xIm)*matImage[:,:, 3]/np.pi)
#     matImage[:,:,7] = np.round(yImc+matImage[:,:, 5]*float(yIm)*matImage[:,:, 4]/np.pi)
#     matImage[indicBeta[0], indicBeta[1], 3:] = 0
#     matCoord = np.array(matImage[:,:,6:], dtype = np.int16)
#     # #set zeros for indice >xmax/2
#     # indicBeta = np.where(mat1[:,:,2] > xmax/2)
#     # mat2[indicBeta[0], indicBeta[1]] = 0
    
#     if len(image1.shape) == 3:
#         if image1.dtype == 'uint8':
#             imageZoom = np.uint8(image1[matCoord[:,:,0], matCoord[:,:,1],:])
#         elif image1.dtype == 'uint16':
#             imageZoom = np.uint16(image1[matCoord[:,:,0], matCoord[:,:,1],:])

#     elif len(image1.shape) == 2:
#         if image1.dtype == 'uint8':
#             imageZoom = np.uint8(image1[matCoord[:,:,0], matCoord[:,:,1]])
#         elif image1.dtype == 'uint16':
#             imageZoom = np.uint16(image1[matCoord[:,:,0], matCoord[:,:,1]])
        
#     # dtime = time.time() - start_time
#     # if 'all' in verbose or 'time' in verbose:
#     #     print('time for zoom witout loop : %.2f'%dtime)
        
#     return imageZoom


# ###############################################################################
# @timeDuration
# def zoom_image (image, xmax = 901, szamax = 60, verbose = [''], with_loop = False):
#     """
#     zoom on image, resize with xmax and delete perimeter after szamax.
#     Zoom restaure straight line to have plane trajectories as straight line.

#     Parameters
#     ----------
#     image : array,
#         Input image.
#     xmax : int, optinal
#         Size new image. The default is 901
#     szamax : int, optinal
#         Angle to filter image. The default is 60.
#     verbose : list of stirng, optional
#         List of element to print . The default is [''].
#     with_loop : bool, optional
#         Calcul resize with loop or with matrix. The default is False.

#     Raises
#     ------
#     ValueError
#         Image size is not correct.

#     Returns
#     -------
#     imageZoom : array
#         Output image.

#     """
    
#     # start_time=time.time()
    
#     szamaxrad = np.radians(szamax)
#     imageZoom = np.zeros((xmax, xmax, 3), np.uint8)
    
#     #extract size and dimension input
#     if len(image.shape)==3:
#         xImo, yImo, nbcolor = image.shape
#     elif len(image.shape)==2:
#         xImo, yImo = image.shape

#     #determine  if image is cropped
#     if xImo == 768 and yImo == 1024:
#         #we have original image
#         croppedParam = croppedSize()
#         xImmax, xImmin = croppedParam.xmax, croppedParam.xmin
#         xIm = xImmax-xImmin
#         yImmax, yImmin = croppedParam.ymax, croppedParam.ymin
#         yIm = yImmax-yImmin
#     elif xImo == 678 and yImo == 678:
#         #we have cropped image
#         xImmax, xImmin = 678, 0
#         xIm = xImmax-xImmin
#         yImmax, yImmin = 678, 0
#         yIm = yImmax-yImmin
#     elif yImo == 1280 and xImo == 960:
#         croppedParam = croppedSize('Orsay')
#         xImmax, xImmin = croppedParam.xmax, croppedParam.xmin
#         xIm = xImmax-xImmin
#         yImmax, yImmin = croppedParam.ymax, croppedParam.ymin
#         yIm = yImmax-yImmin
#     elif xImo == 1280 and yImo == 960:
#         croppedParam = croppedSize('Orsay')
#         yImmax, yImmin = croppedParam.xmax, croppedParam.xmin
#         yIm = yImmax-yImmin
#         xImmax, xImmin = croppedParam.ymax, croppedParam.ymin
#         xIm = xImmax-xImmin
#     else :
#         #we have an unknow cas
#         # sys.exit('Image size does not exist')
#         raise ValueError("image's shape is not correct shpae :"+','.join("%d"%i for i in image.shape))

#     #--finding center
#     #case same size X and Y
#     if xIm == yIm:
#         xImc = int(xImmin+xIm/2)
#         yImc = int(yImmin+yIm/2)
#     #case different size Y and Y
#     else:
#         sizemax = max(xIm, yIm)
#         center = int(sizemax/2)
#         # case Saturate min
#         if xImmin == 0 or yImmin == 0:
#             xImc = xImmax - center
#             yImc = yImmax - center
#         #case Saturate max
#         elif xImmax == xImo or yImmax == yImo:
#             xImc = xImmin + center
#             yImc = yImmin + center
            
#     #zoom without for
#     if with_loop is True:
#         imageZoom = zoom_with_loop(image, xmax, szamaxrad, xIm, yIm, xImc, yImc, verbose = verbose)
#     else :
#         imageZoom = zoom_wihout_loop(image, xmax, szamaxrad, xIm, yIm, xImc, yImc, verbose = verbose)

#     #--reverse image to get orientation right
#     if xImo in [768, 678] and yImo in [1024, 678]:
#         if len(image.shape) == 3:
#             imageZoom=imageZoom[:,::-1,:]
#         elif len(image.shape) == 2:
#             imageZoom=imageZoom[:,::-1]
    

#     # dtime=time.time()-start_time
#     # if 'all' in verbose or 'time' in verbose:
#     #     print("duration of zoom image : %.2f s" %(dtime))
#     return imageZoom

# ###############################################################################

#%% channel_image
def channel_image(image, color, verbose = ['']):
    """
    converstion 3D image to 2D image.

    Parameters
    ----------
    image : Array
        Image input.
    color : String
        Color for conversion.

    Returns
    -------
    image 2D

    """
    if color == "gray":
        imBW = np.dot(image[:,:,0:3],[0.299,0.587,0.114]).astype(np.uint8)
    elif color == "red":
        imBW = np.dot(image[:,:,0:3],[1,0,0]).astype(np.uint8)
    elif color == "green":
        imBW = np.dot(image[:,:,0:3],[0,1,0]).astype(np.uint8)
    elif color == "blue":
        imBW = np.dot(image[:,:,0:3],[0,0,1]).astype(np.uint8)
    elif color == 'hsv':
        imBW = rgb2hsv(image)
        imBW = np.array(imBW*255, dtype = np.uint8)
    elif color == 'hsV':
        imBW = rgb2hsv(image)[:,:,2]
        imBW = np.array(imBW*255, dtype = np.uint8)
    else :
        raise ValueError('color uncknow. color : %s'%color)
    
    if 'debug' in verbose :
        print("you choice %s channel in image"%color)
        
    return(imBW)

###############################################################################

#%% otsu_processing
def otsu_processing(image, radius, verbose = ['']):
    """
    substract local otsu threshold for delete contrast.

    Parameters
    ----------
    image : Array
        Array image 2D or RGB.
    processus : Int
        Radius.
    verbose : list of string, optional
        List of things to printing. The default is [''].

    Returns
    -------
    image1 : Array
        Image with substraction of contrast.

    """
    start_time=time.time()
    
    image1 = copy.deepcopy(image)
    if len(image1.shape) == 3:
        for i in range(image1.shape[2]):
            image1[:,:,i] = otsu_processing(image1[:,:,i], radius, verbose)
    else :
        #-- local otsu to delete sun contrast
        selem = disk(radius)
        local_otsu = rank.otsu(image1, selem)
        image1 = np.array(image1, dtype=(np.int16))
        local_otsu = np.array(local_otsu, dtype = (np.int16))
        image1 = image1 - local_otsu
        #normalisation
        image1 = (image1-np.min(image1))/(np.max(image1)-np.min(image1))*255
        image1 = np.array(image1, dtype= np.uint8)
        
    dtime=time.time()-start_time
    if 'all' in verbose or 'time' in verbose:
        print("duration of otsu processing image : %1.f s" %(dtime))
        
    return image1

###############################################################################        
            
#%% diff
def diff(image, path, cle = False, verbose = ['']):
    """
    

    Parameters
    ----------
    image : Array,
        Input image.
    timestamp : Datetime.datetime,
        Datetime image.
    minute_step : Int,
        Duration in minute between two image.
    cle : threading.RLock, optional
        Lock to limit write and read acces to one thread. The default is False.
    verbose : list of String, optional list of string
        List of things to printing. The default is [''].

    Raises
    ------
    ValueError
        Problem image shape.

    Returns
    -------
    image : Array
        Output image.

    """
    # dateTime = path.getDateDay()
    # minute_step = 2
    # timeMax = 7200
    # print('fonction non fini')
    # dTO = dateTime - datetime.timedelta(seconds = minute_step*60)
    # imageOld_name = os.path.join('/homedata/ngourgue/Images/SIRTA/', "%04d"%dTO.year, "%02d"%dTO.month, 
    #                                    "%04d%02d%02d"%(dTO.year, dTO.month, dTO.day),
    #                                    "%04d%02d%02d%02d%02d%02d_01.jpg"%(dTO.year, dTO.month, dTO.day, 
    #                                                                       dTO.hour, dTO.minute, dTO.second))
    imageOld_name = path.getPathImagePasted()
    if os.path.exists(imageOld_name):
        # if True :
        # if cle == False:
        #     if os.path.isfile(imageOld_name):
        #         imageOld = imread(imageOld_name)
        #     else:
        #         return image, False
        # else:
        #     cle.acquire(timeout = timeMax)
            if os.path.isfile(imageOld_name):
                imageOld = imread(imageOld_name)
            else:
                # cle.release()
                return image, False       
            # cle.release()
    elif os.path.exists(path.getDatedayTargz(comp = False)):
        # tar_name = path.getDatedayTargz(comp = False)
        imageOld = readOldImage(path, 'ref', cle = False)
        if type(imageOld) == bool:
            return image, False
        ratio_cloudy, imRBR = cloud_segmentation(imageOld)
        if ratio_cloudy > 0.05:
            return image, False
        
        
        
    image1 = copy.deepcopy(image)
    image1 = np.array(image1, dtype = np.int32)
    
    if   np.all(image.shape == (768, 1024, 3)) and np.all(imageOld.shape == (768, 1024, 3)):
        pass
    elif np.all(image.shape == (901, 901,  3)) and np.all(imageOld.shape == (768, 1024, 3)):
        imageOld = zoom_image(imageOld, image.shape[0], 60)
        
    elif np.all(image.shape == (901, 901,  3)) and np.all(imageOld.shape == (901, 901,  3)):
        pass
    else:
        raise ValueError('image size not conform', image.shape)
    imageOld = np.array(imageOld, np.int32)
    imageOut = image1-imageOld
    imageOut = (imageOut - np.min(imageOut))/(512)
    imageOut = np.array(imageOut*255, dtype = np.int8)
    
    return imageOut, True

###############################################################################

#%% read old image output
def readOldImage(path, name, cle = False):  
    contrail_image = False
    contrail_image_name = path.getSaveImage(name)
    if os.path.exists(contrail_image_name):
        contrail_image = imread(contrail_image_name)
    elif os.path.exists(path.getDatedayTargz(comp = False)):
        # tar_name = path.getDatedayTargz(comp = False)
        with tarfile.open(path.getDatedayTargz(comp = False), 'r') as compFileDay:
            members = compFileDay.getmembers()
            for member in members:
                if path.getTarImagePasted(name) == member.get_info()['name']:
                    with compFileDay.extractfile(member) as imagefile:
                        contrail_image = imread(imagefile)
    else:
        return False
        
    return contrail_image
#%% read old csv
def readOldCsv(path, name, maskPL = None, flightname = None, cle = False):
    contrail_csv = False
    if os.path.exists(path.getSaveDataPasted(name)):
            contrail_csv_name = path.getSaveDataPasted(name)
            contrail_csv = pd.read_csv(contrail_csv_name, sep=',', header=0, index_col=0)
    elif os.path.exists(path.getDatedayTargz(comp = False)):
        # tar_name = path.getDatedayTargz(comp = False)
        with tarfile.open(path.getDatedayTargz(comp = False), 'r') as compFileDay:
            members = compFileDay.getmembers()
            for member in members:
                if path.getTarFilenamePasted(name) == member.get_info()['name']:
                    with compFileDay.extractfile(member) as csvfile:
                        contrail_csv = pd.read_csv(csvfile, sep=',', header=0, index_col=0)
    else:
        if np.max(maskPL)==1:
            return maskPL
        elif np.max(maskPL)==0:
            return False
    
    return contrail_csv


###############################################################################

#%%
def unZipImage(dateDays, folderInput):
    for dateDay in dateDays:
        #search file
        files = os.listdir(os.path.join(folderInput, "%04d"%dateDay.year, "%02d"%dateDay.month, 
                                        "%02d"%dateDay.day))
        filesDate = []
        for file in files:
            if "%04d%02d%02d"%(dateDay.year, dateDay.month, dateDay.day) in file:
                filesDate.append(file)
        fileszip = []
        for file in filesDate:
            if ".zip" in file:
                fileszip.append(file)
        if len(fileszip)==1:
            filezip = fileszip[0]
            
        fileZip = zipfile.ZipFile(os.path.join(folderInput, "%04d"%dateDay.year, "%02d"%dateDay.month, 
                                               "%02d"%dateDay.day, filezip))
        fileZip.extractall(os.path.join(folderInput, "%04d"%dateDay.year, "%02d"%dateDay.month))
        shutil.rmtree(os.path.join(folderInput, "%04d"%dateDay.year, "%02d"%dateDay.month, "%02d"%dateDay.day))

