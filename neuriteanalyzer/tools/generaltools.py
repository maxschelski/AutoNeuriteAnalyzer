# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:08:06 2019

@author: schelskim
"""

from scipy.spatial import distance
import numpy as np
import cv2

from skimage import morphology as morph
from scipy import ndimage
import copy
from skimage.morphology import disk
from matplotlib import pyplot as plt
#import pyopencl as cl


import ctypes

class generalTools():

    @staticmethod
    def getPointByDist(Coords1,Points2,mode):
        allPoints = [Coords1[0],Coords1[1]]
        allPoints = np.transpose(allPoints)
        
        distances = distance.cdist(allPoints,Points2)
        if(mode == 'min'):
            targetDistNb = distances.argmin(axis=0)
        elif(mode == 'max'):   
            targetDistNb = distances.argmax(axis=0)
        targetPoint = allPoints[targetDistNb,:]
#        targetDistance = distances[targetDistNb]
        return targetPoint[0]
    
    @staticmethod
    def showThresholdOnImg(threshold,img,nb = 1):
        signal = np.max(img) * 1.1
        for a in range(0,nb):
            plt.figure()
        newImg = copy.copy(img)
        newImg[threshold == 1] = signal
        plt.imshow(newImg)
    
    @staticmethod
    def getFurthestPoint(Coords1,Points2):
        furthestPoint = generalTools.getPointByDist(Coords1,Points2,'max')
        return furthestPoint
    
#    @staticmethod
#    def GPUmorph(img,operation,kernelSize):
#        #source: https://github.com/githubharald/GPUImageProcessingArticle/blob/master/main.py
#        platforms = cl.get_platforms()
#        platform = platforms[1]
#        print(platforms)
#        devices = platform.get_devices(cl.device_type.GPU)
#        print(devices)
#        device = devices[0]
#        context = cl.Context([device])
#        queue = cl.CommandQueue(context,device)
#        imgOutput = np.empty_like(img)
#        imgShape = img.shape
#        
#        imgBuffer = cl.Image(context, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8), shape=imgShape)
#        imgOutputBuffer = cl.Image(context, cl.mem_flags.WRITE_ONLY, cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8), shape = imgShape)
#        
#        program = cl.Program(context, open('tools\\kernels\\morph_kernel.cl').read()).build()
#        
#        xKernelSize = kernelSize[0]
#        yKernelSize = kernelSize[1]
#        
#        kernel = cl.Kernel(program, 'morphologicalProcessing')
#        kernel.set_arg(0,imgBuffer)
#        kernel.set_arg(1,imgOutputBuffer)
#        kernel.set_arg(2,np.uint32(operation))
#        kernel.set_arg(3,np.uint32(xKernelSize))
#        kernel.set_arg(4,np.uint32(yKernelSize))
#        
#        cl.enqueue_copy(queue, imgBuffer, img, origin=(0,0), region=imgShape, is_blocking=False)
#        cl.enqueue_nd_range_kernel(queue, kernel, imgShape, None)
#        cl.enqueue_copy(queue, imgOutput,imgOutputBuffer,origin=(0,0),region=imgShape,is_blocking=True)
#    
#        return imgOutput
    
    @staticmethod
    def getMidCoords(image):
        moments = cv2.moments(image)
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        return cX, cY
    
    @staticmethod
    def getClosestPoint(Coords1,Points2):
        closestPoint = generalTools.getPointByDist(Coords1,Points2,'min')
        return closestPoint
    
    @staticmethod
    def microseconds():
        #code by Gabriel Staples, www.ElectricRCAircraftGuy.com
        tics = ctypes.c_int64()
        freq = ctypes.c_int64()
        
        ctypes.windll.Kernel32.QueryPerformanceCounter(ctypes.byref(tics))
        ctypes.windll.Kernel32.QueryPerformanceFrequency(ctypes.byref(freq))
    
        time = tics.value * 1e6/freq.value
        time = time / 1000
        return time
    
    
    
    @staticmethod
    def get_diff_of_two_arrays(a1,a2):
        diff = generalTools.compare_two_arrays(a1,a2,"diff")
        return diff
    
    @staticmethod
    def get_intersect_of_two_arrays(a1,a2):
        intersect = generalTools.compare_two_arrays(a1,a2,"intersect")
        return intersect
    
    def compare_two_arrays(a1,a2,mode):
        a1_rows,a1 = generalTools.get_array_as_row(a1)
        a2_rows,a2 = generalTools.get_array_as_row(a2)
        if mode == "diff":
            invert_val = True        
        elif mode == "intersect":
            invert_val = False
        comparison = a1_rows[np.isin(a1_rows,a2_rows,assume_unique=True,invert=invert_val)].view(a1.dtype).reshape(-1, a1.shape[1])
        return comparison
            
    @staticmethod
    def get_array_as_row(a):
        a = np.array(a)
        a_rows = a.view([('', a.dtype)] * a.shape[1])
        return a_rows, a
    


    @staticmethod
    def cropImage(image,borderVals=[],padding=0):
        if len(borderVals) == 0:
            coords = np.where(image > 0)
            xMin = np.min(coords[0])-padding 
            if xMin < 0:
                xMin = 0
            xMax = np.max(coords[0])+padding
            if xMax > image.shape[0]:
                xMax = image.shape[0]
            yMin = np.min(coords[1])-padding
            if yMin < 0:
                yMin = 0             
            yMax = np.max(coords[1])+padding
            if yMax > image.shape[1]:
                yMax = image.shape[1]
#            print("{} - {} - {} - {}".format(xMin,xMax,yMin,yMax))
            imgShape = image.shape
            borderVals = [xMin,xMax,yMin,yMax,imgShape]
            croped = image[xMin:xMax,yMin:yMax]
        else:
            croped = image[borderVals[0]:borderVals[1],borderVals[2]:borderVals[3]]
        return croped, borderVals
    
    def crop2Images(image1,image2,padding=0):
        cropedImage1,borderVals1 = generalTools.cropImage(image1,[],padding)
        cropedImage2,borderVals2 = generalTools.cropImage(image2,[],padding)
        borderVals = []
        borderVals.append(np.min([borderVals1[0],borderVals2[0]]))
        borderVals.append(np.max([borderVals1[1],borderVals2[1]]))
        borderVals.append(np.min([borderVals1[2],borderVals2[2]]))
        borderVals.append(np.max([borderVals1[3],borderVals2[3]]))
        borderVals.append(image1.shape)
        cropedImage1, borderVals = generalTools.cropImage(image1,borderVals)
        cropedImage2, borderVals = generalTools.cropImage(image2,borderVals)
        return cropedImage1,cropedImage2, borderVals

    @staticmethod
    def uncropImage(image,borderVals):
        dataType = image.dtype
        uncroped = np.zeros(borderVals[4])
        uncroped[borderVals[0]:borderVals[1],borderVals[2]:borderVals[3]] = image
        uncroped = uncroped.astype(dataType)
        return uncroped
    
    @staticmethod
    def getImageFromPoints(points,shape):
        coords = np.transpose(points)
        image = generalTools.getImageFromCoords(coords,shape)
        return image
    
    @staticmethod
    def getImageFromCoords(coords,shape):
        image = np.zeros(shape)
        image[coords[0],coords[1]] = 1
        return image
    
    @staticmethod
    def convert_points_to_point_list(point):
        if (len(point) == 2) & (type(np.transpose(point)[0]) == np.int64):
            point_list = []
            point_list.append(point)
            point_list = np.array(point_list)
        else:
            point_list = point
        return point_list
    
    @staticmethod
    def smoothenImage(image,dilationForSmoothing,gaussianForSmoothing):
        iterationsForSmoothing = int(np.round(dilationForSmoothing/2,0))
        dilationForSmoothing = int(np.round(dilationForSmoothing/iterationsForSmoothing,0))
        #crop image for speed
        image,borderVals = generalTools.cropImage(image,[],20)
        
        #smoothen neurite by dilating, blurring, thresholding and then thinning (skeletonizing would take away ends of neurite!)
        image_smooth = ndimage.binary_dilation(image,disk(dilationForSmoothing),iterations=iterationsForSmoothing)
        image_smooth = image_smooth.astype(int)
        image_smooth = image_smooth*255
        image_smooth = ndimage.gaussian_filter(image_smooth,gaussianForSmoothing)
        image_smooth = image_smooth > 0
        image_smooth = morph.thin(image_smooth)
        
        #uncrop image
        image_smooth = generalTools.uncropImage(image_smooth,borderVals)
        return image_smooth

    @staticmethod
    def overlapImgs(img1,img2,dilationForOverlap,overlapMultiplicator,reducedOverlap=0,twoWayDiff = 0):
        img1, img2, borderVals = generalTools.crop2Images(img1,img2,10)
        
        if reducedOverlap:
            dilationForOverlap = dilationForOverlap
        multiplicator = overlapMultiplicator
        img1 = img1.astype(int)
        img2 = img2.astype(int)
        sumOfImgs = np.add(img1,img2)
#        plt.figure()
#        plt.figure()
#        plt.figure()
#        plt.imshow(sumOfImgs)
        
        overlapOfImgs = np.zeros_like(sumOfImgs)
        overlapOfImgs[sumOfImgs == 2] = 1
        overlapOfImgs = morph.thin(overlapOfImgs)
        
        diffOfImgs = np.subtract(img1,img2)
        
#        plt.figure()
#        plt.figure()
#        plt.imshow(diffOfImgs)
#        plt.figure()
#        plt.imshow(img1)
#        plt.figure()
#        plt.imshow(img2)
        if twoWayDiff == 0:
            diffOfImgs[diffOfImgs == -1] = 1 
#            diffOfImgs = morph.binary_erosion(diffOfImgs,disk((dilationForOverlap)))
            diffOfImgs = ndimage.binary_erosion(diffOfImgs,disk((dilationForOverlap)/multiplicator),iterations=multiplicator)
            diffOfImgs = morph.thin(diffOfImgs)
            
            diffOfImgs = generalTools.uncropImage(diffOfImgs,borderVals)
            overlapOfImgs = generalTools.uncropImage(overlapOfImgs,borderVals)
            
        elif twoWayDiff == 1:
            gainOfImgs = np.zeros_like(diffOfImgs)
            gainOfImgs[diffOfImgs == -1] = 1
#            plt.figure()
#            plt.figure()
#            plt.figure()
#            plt.imshow(gainOfImgs)
#            gainOfImgs = morph.binary_erosion(gainOfImgs,disk((dilationForOverlap)))
            gainOfImgs = ndimage.binary_erosion(gainOfImgs,disk((dilationForOverlap)/multiplicator),iterations=multiplicator)
#            plt.figure()
#            plt.imshow(gainOfImgs)
            gainOfImgs = ndimage.morphology.binary_closing(gainOfImgs,disk(10))
#            plt.figure()
#            plt.imshow(gainOfImgs)
#            plt.figure()
#            plt.imshow(morph.binary_dilation(gainOfImgs,disk(3)))
            gainOfImgs = morph.thin(gainOfImgs)
            lossOfImgs = np.zeros_like(diffOfImgs)
            lossOfImgs[diffOfImgs == 1] = 1
            
            
#            plt.figure()
#            plt.figure()
#            plt.imshow(lossOfImgs)
            lossOfImgs_mix = copy.copy(lossOfImgs)
            lossOfImgs_mix[img2 == 1] = 1
            
#            lossOfImgs_mix = morph.binary_erosion(lossOfImgs_mix,disk((dilationForOverlap)))
            lossOfImgs_mix = ndimage.binary_erosion(lossOfImgs_mix,disk((dilationForOverlap)/multiplicator),iterations=multiplicator)
            lossOfImgs[lossOfImgs_mix == 0] = 0
#            plt.figure()
#            plt.imshow(lossOfImgs)
            
#            plt.figure()
#            plt.imshow(morph.binary_dilation(lossOfImgs,disk(3)))
            lossOfImgs = ndimage.morphology.binary_closing(lossOfImgs,disk(10))
#            plt.figure()
#            plt.imshow(lossOfImgs)
#            plt.figure()
#            plt.imshow(morph.binary_dilation(lossOfImgs,disk(3)))
            
            lossOfImgs = morph.thin(lossOfImgs)
            
            
            gainOfImgs = generalTools.uncropImage(gainOfImgs,borderVals)
            lossOfImgs = generalTools.uncropImage(lossOfImgs,borderVals)
            overlapOfImgs = generalTools.uncropImage(overlapOfImgs,borderVals)
            
            diffOfImgs = [lossOfImgs,gainOfImgs]
        return diffOfImgs,overlapOfImgs