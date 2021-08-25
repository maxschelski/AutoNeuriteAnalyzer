# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 09:51:49 2019

@author: schelskim
"""

from tools.sortpoints import sortPoints
from tools.somatools import SomaTools
from tools.generaltools import generalTools


import numpy as np
from skimage import morphology as morph
from scipy import ndimage
from skimage import measure
import copy
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt
from skimage.filters import median
from scipy.signal import medfilt2d as median2
from skimage.filters import scharr
from skimage.morphology import disk
from scipy.ndimage.filters import gaussian_filter1d as gaussian
from skimage.filters import threshold_otsu as otsu




class ThresholdNeuron:
    
    
    @staticmethod
    def start(img,cX,cY,grainSizeToRemove,dilationForOverlap,overlapMultiplicator,neuron,minEdgeVal,minBranchSize,percentileChangeForThreshold, maxThresholdChange,minSomaOverflow,somaExtractionFilterSize,maxToleratedSomaOverflowRatio,nb1Dbins,minNbOfPxPerLabel,objectSizeToRemoveForSomaExtraction,distanceToCover,filterBackgroundPointsFactor,openingForThresholdEdge,medianForThresholdEdge,closingForPresoma,mask=np.zeros([10,10]),starting_threshold=0):

        #higher dilation than normaly, 
        #since neurite path can fluctuate during thresholding a lot 
#        dilationForOverlap = dilationForOverlap * 2
#        overlapMultiplicator = overlapMultiplicator * 2
        
        Ints = img[img >0].ravel()
        sortedInts = np.sort(Ints)
        thresholdVal = np.nan
        
        
        
        if starting_threshold > 0:
            percentile = starting_threshold
            thresholdVal = ThresholdNeuron.getValueOfPercentile(sortedInts,percentile)
            img_threshold = img > thresholdVal
        else:
            percentile = np.nan
            Ints = img[img >0].ravel()
            sortedInts = np.sort(Ints)
            
            img_for_threshold_edge = ThresholdNeuron.getEdgeImage(img,
                                                                  openingForThresholdEdge,
                                                                  medianForThresholdEdge)
    
            #threshold edge image using histogram based thresholding
            sortedInts = np.sort(img_for_threshold_edge.ravel())
            MAX = np.max(sortedInts[0:int(np.round(len(sortedInts)*0.999,0))])
            allBins = np.logspace(np.log10(0.1),np.log10(MAX),num=25)
            edgeThresh = otsu(img_for_threshold_edge)
            
            
            if len(np.where(mask > 0)[0]) > 0:
                #DOES THE MASK WORK IN THE CURRENT SYSTEM?
                img[mask == True] = 0
                
            #check if any threshold was found, if not don't continue
            if np.isnan(edgeThresh):
                print("edge threshold didnt work...")
                thresholdVal = np.nan
                threshold_percentile = np.nan
                img_filtered = np.zeros_like(img)
            else:
                img_edges = img_for_threshold_edge > edgeThresh
                
                img_edges,fillingSuccessfull = ThresholdNeuron.fillSomaFromEdges(img_edges,grainSizeToRemove,objectSizeToRemoveForSomaExtraction)
                
                
                #if thresholding & filling of edges didnt work, try more conservative thresholding
                if (len(np.where(img_edges == 1)[0]) == 0) | (not fillingSuccessfull):
                    edgeThresh = ThresholdNeuron.getInitialThresholdValue(img_for_threshold_edge,allBins,nb1Dbins,True)
                    img_edges = img_for_threshold_edge > edgeThresh
                    img_edges,fillingSuccessfull = ThresholdNeuron.fillSomaFromEdges(img_edges,grainSizeToRemove,objectSizeToRemoveForSomaExtraction)
                
                if len(np.where(img_edges == 1)[0]) == 0:
                    print("initial threshold of edges didn't work")
                    img_filtered = img
                else:
                    Ints = img[img >0].ravel()
        
                    img_threshold, percentile, thresholdVal, cX, cY, Ints, sortedInts = ThresholdNeuron.getThresholdBasedOnSomaEdge(img,img_edges,minSomaOverflow,somaExtractionFilterSize,objectSizeToRemoveForSomaExtraction,maxToleratedSomaOverflowRatio,closingForPresoma)
                    if np.isnan(thresholdVal):
                        print("initial threshold didn't work")
                        img_filtered = img
        
        if not np.isnan(percentile):
                    #changed last parameter from imgOfWholeNeuron to img_threshold
                    nbOfLabels,nbPixHoles,lastImg,thresholdVal = ThresholdNeuron.evalThreshold(img,Ints,sortedInts,percentile,grainSizeToRemove,img_threshold)
                   
#    
                        #define one soma area that will not be included in skeleton analysis
                    soma,cX,cY,lastImg_clean = ThresholdNeuron.getSomaFromThresholdedImage(lastImg,minSomaOverflow,somaExtractionFilterSize,cX,cY,objectSizeToRemoveForSomaExtraction,closingForPresoma)
                   
                    #if no soma was found, then cX and cY are nan
                    if np.isnan(cX) | (len(np.where(soma == 1)[0]) == 0):
                        print("no soma was found")
                        img_filtered=img
                        thresholdVal = np.nan
                        threshold_percentile = np.nan
                        cX = np.nan
                        cY = np.nan
                    else:
                        img_filtered, img_filtered_forBackgound, thresholdVal,threshold_percentile = ThresholdNeuron.refineThreshold(img,percentile,lastImg,lastImg_clean,copy.copy(cX),copy.copy(cY),soma,Ints,sortedInts,maxThresholdChange,percentileChangeForThreshold,dilationForOverlap,overlapMultiplicator,minBranchSize,grainSizeToRemove)   
                        
        if not np.isnan(thresholdVal):
            backgroundVal = np.mean(img[(img_filtered_forBackgound == 0) & (img != 0)])
            img_filtered = ThresholdNeuron.removeSingleBackgroundPoints(img_filtered,distanceToCover,filterBackgroundPointsFactor)
        else:
            img_filtered = np.zeros_like(img_filtered)
            thresholdVal = np.nan
            threshold_percentile = np.nan
            backgroundVal = np.nan
        return img_filtered,thresholdVal,threshold_percentile, backgroundVal, cX, cY
    
    @staticmethod
    def isolateNeuronBasedOnSoma(img_threshold,cX,cY):
        #cx and cy need to be defined!
        #initiate img of whole neuron
        imgOfWholeNeuron = np.zeros_like(img_threshold)
        imgOfWholeNeuron_labeled,nbOfLabels = ndimage.label(img_threshold,structure=[[1,1,1],[1,1,1],[1,1,1]])
        neuronLabel = imgOfWholeNeuron_labeled[cY,cX]
#        img_threshold[self.cY,self.cX] = 1
        imgOfWholeNeuron[imgOfWholeNeuron_labeled == neuronLabel] = 1
        return imgOfWholeNeuron
    
    
        
    @staticmethod
    def getSomaFromThresholdedImage(img_thresholded,minSomaOverflow,somaExtractionFilterSize,cX,cY,objectSizeToRemoveForSomaExtraction,closingForPresoma):
        #isolate soma from thresholded image and set mid point of soma
        #include grains around soma in soma image to account for unspecific grains

        #----------------CROP IMAGES FOR SPEED
        preSoma,borderVals = generalTools.cropImage(img_thresholded,[],10)
        img_thresholded,borderVals = generalTools.cropImage(img_thresholded,borderVals)

        preSoma = ndimage.morphology.binary_closing(img_thresholded,disk(closingForPresoma))
        preSoma = preSoma.astype(np.uint8)
        preSoma[preSoma > 0] = 255
        soma, img_thresholded = SomaTools.getSoma(preSoma,minSomaOverflow,somaExtractionFilterSize,cX,cY,img_thresholded,objectSizeToRemoveForSomaExtraction)
#        soma = ndimage.binary_dilation(soma,disk(self.somaExtractionFilterSize/6),iterations=12)
        
        #----------------UNCROP IMAGES
        soma = generalTools.uncropImage(soma,borderVals)
        img_thresholded = generalTools.uncropImage(img_thresholded,borderVals)
        
        soma_forMid = np.zeros_like(soma)
        soma_forMid[soma == True] = 1
        soma_forMid = np.array(soma_forMid,dtype=np.uint8)
        if len(np.where(soma == 1)[0]) > 0:
            cX, cY = generalTools.getMidCoords(soma_forMid)
        return soma, cX, cY, img_thresholded
    
    @staticmethod
    def getValueOfPercentile(sortedInts,percentile):
        val = sortedInts[int(len(sortedInts)/(1/(1-percentile)))-1]
        return val
    
    @staticmethod
    def removeSingleBackgroundPoints(image,distanceToCover,filterBackgroundPointsFactor):
        img_labeled,nbLabels = ndimage.label(image,structure=[[1,1,1],[1,1,1],[1,1,1]])
        minimumNumberInDistance = distanceToCover*filterBackgroundPointsFactor
        for label in np.unique(img_labeled):
            labelCoords = np.where(img_labeled == label)
            if len(labelCoords[0]) < 10:
                totalNumberOfXPx = 0
                for x in np.unique(labelCoords[0]):
                    yStart = labelCoords[1][np.where(labelCoords[0] == x)[0][0]]
                    imageSection = image[x,yStart-distanceToCover:yStart+distanceToCover]
                    numberOfPx = len(np.where(imageSection == 1)[0])
                    totalNumberOfXPx = totalNumberOfXPx + numberOfPx
                totalNumberOfYPx = 0
                for y in np.unique(labelCoords[1]):
                    xStart = labelCoords[0][np.where(labelCoords[1] == y)[0][0]]
                    imageSection = image[xStart-distanceToCover:xStart+distanceToCover,y]
                    numberOfPx = len(np.where(imageSection == 1)[0])
                    totalNumberOfYPx = totalNumberOfYPx + numberOfPx
                totalNumberOfPx = np.max([totalNumberOfXPx,totalNumberOfYPx])
                totalNumberOfPx = totalNumberOfPx - len(labelCoords[0])
                if totalNumberOfPx < minimumNumberInDistance:
                    image[img_labeled == label] = 0
        return image
        
    @staticmethod
    def evalThreshold(img,Ints,sortedInts,percentile,grainSizeToRemove,imgOfWholeNeuron):
        
        thresholdVal = ThresholdNeuron.getValueOfPercentile(sortedInts,percentile)
        img_thresh = img > thresholdVal
        #remove obvious grains
        img_thresh = morph.remove_small_objects(img_thresh,grainSizeToRemove)
        img_thresh_labeled,nbOfLabels = ndimage.label(img_thresh,structure=[[1,1,1],[1,1,1],[1,1,1]])
        
        #if img of whole neuron exists, check for each group in thresholded image if it is part of whole neuron image
        if len(np.unique(imgOfWholeNeuron)) > 1:
            for label in np.unique(img_thresh_labeled):
                singleLabel = np.where(img_thresh_labeled == label)
                overlapWithNeuron = imgOfWholeNeuron[singleLabel]
                #if there is no overlap (no 1 in array) 
                if len(np.where(np.unique(overlapWithNeuron) == 1)[0]) == 0:
                    img_thresh[singleLabel] = 0
        
        img_inv = np.invert(img_thresh)
        img_inv_dil = ndimage.morphology.binary_closing(img_inv,morph.square(2))
        nbPixHoles = len(np.where(img_inv_dil == True)[0]) - len(np.where(img_inv == True)[0])
        img_labeled,nbOfLabels = ndimage.label(img_thresh,structure=[[1,1,1],[1,1,1],[1,1,1]])
        nbOfLabels= len(np.unique(img_labeled))
#        img_filtered = morph.remove_small_objects(img_filtered,self.objectSizeToRemove,2)
        return nbOfLabels,nbPixHoles,img_thresh,thresholdVal
    

    @staticmethod
    def getEdgeImage(img,openingForThresholdEdge,medianForThresholdEdge):
        
        #create image of edge to get shape of soma (place of highest contrast in neuron)
        img_for_threshold_edge = ndimage.grey_opening(img,structure=disk(openingForThresholdEdge))
        img_for_threshold_edge = median(img_for_threshold_edge,disk(medianForThresholdEdge))
        img_for_threshold_edge = img_for_threshold_edge/np.max(img_for_threshold_edge) * 255
        img_for_threshold_edge = np.abs(scharr(img_for_threshold_edge))
        #don't scale! otherwise dynamic for thresholding depends on dynamic range of edge values 
        #two objects with very different contrast will mess it up!
        return img_for_threshold_edge
    
    @staticmethod
    def fillSomaFromEdges(img_edges,grainSizeToRemove,objectSizeToRemoveForSomaExtraction):
        img_edges = morph.remove_small_objects(img_edges,grainSizeToRemove,1)
        #check how many pixels need to be closed in edge image to fully enclose the soma
        allSizesOfFilling = []
        for closingDiameter in np.linspace(1,32,7):
            img_edges_test = ndimage.morphology.binary_closing(img_edges,disk(closingDiameter))
            startingSizeOfFilling = len(np.where(img_edges_test == 1)[0])
            img_edges_test = ndimage.morphology.binary_fill_holes(img_edges_test)
            sizeOfFilling = len(np.where(img_edges_test == 1)[0]) - startingSizeOfFilling
            allSizesOfFilling.append(int(sizeOfFilling))
        allSizesOfFilling_smooth = gaussian(allSizesOfFilling,1)
        allSizesOfFilling_Maxs = argrelextrema(allSizesOfFilling_smooth,np.greater)[0]
        if len(allSizesOfFilling_Maxs) > 0:
            firstMaxOfFilling = allSizesOfFilling[allSizesOfFilling_Maxs[0]]
        else:
            firstMaxOfFilling = np.max(allSizesOfFilling)
        finalClosingSize = np.where(allSizesOfFilling == np.int64(firstMaxOfFilling))[0][0]
        img_edges = ndimage.morphology.binary_closing(img_edges,disk(1+(finalClosingSize*5)))
        img_edges = ndimage.morphology.binary_fill_holes(img_edges)
        img_edges = img_edges.astype(int)
        img_edges = morph.remove_small_objects(img_edges,objectSizeToRemoveForSomaExtraction)
        if finalClosingSize == 0:
            fillingSuccessfull = False
        else:
            fillingSuccessfull = True
        return img_edges,fillingSuccessfull
    
    @staticmethod
    def getInitialThresholdValue(img,allBins,nb1Dbins,smoothenHistogram=False):
        #calculate bin size in x and y direction
        xBin = int(np.round(img.shape[0]/nb1Dbins,0))
        yBin = int(np.round(img.shape[1]/nb1Dbins,0))
        
#        img = gaussian(img,2)
        #local thresholding: based on first local minimum in 1D graph of pixel histogram in subset of pixels
        allMinVals = []
        startingPoints = [0,20]
        for startingPoint in startingPoints:
            for a in range(1,nb1Dbins+1):
                for b in range(1,nb1Dbins+1):
                    partImg = img[startingPoint+(a-1)*xBin:a*xBin,startingPoint+(b-1)*yBin:b*yBin]
                    partInts = partImg[partImg != 0].ravel()
                    
                    partHisto = np.histogram(partInts,bins=allBins)
                    if smoothenHistogram:
                        partHisto_vals = gaussian(partHisto[0],0.5)
                        partHisto_smooth=[]
                        partHisto_smooth.append(partHisto_vals)
                        partHisto_smooth.append(partHisto[1])
                        partHisto = partHisto_smooth
#                        print(partHisto)
                    minimum = argrelextrema(partHisto[0],np.less)[0]
#                    print("MINIMUMs: {}".format(minimum))
                    if len(minimum) > 1:
                        #randomly sometimes the one of the two first bins would be the chosen one -> always too early!
                        minimumVal = partHisto[1][minimum[-1]]
#                        print(minimumVal)
                        
                        allMinVals.append(minimumVal)
    #                    plt.figure()
    #                    partImgThresh = partImg > minimumVal
    #                    plt.imshow(partImgThresh)
#                    plt.figure()
#                    plt.hist(partInts,log=True,bins=allBins)
#        print(np.median(allMinVals))
#        
#        print(np.mean(allMinVals))
        #take second smallest of the locally determined thresholdvalues (thereby exclude outliers)
#        print(allMinVals)
        if len(allMinVals) > 0:
            sortedMinVals = np.sort(allMinVals)
            if len(allMinVals) > 2:
                    #choose third value in all min values to ignore very low values
                    newThresholdVal = sortedMinVals[2]
            elif len(allMinVals) > 1:
                newThresholdVal = sortedMinVals[1]
            else:
                newThresholdVal = sortedMinVals[0]
        else:
            newThresholdVal = np.nan
        return newThresholdVal    
 
    @staticmethod
    def getThresholdBasedOnSomaEdge(img,img_edges,minSomaOverflow,somaExtractionFilterSize,objectSizeToRemoveForSomaExtraction,maxToleratedSomaOverflowRatio,closingForPresoma):
        
        percentile = 0.5
        thresholdVal = 0
        lastSoma = np.zeros_like(img)
        lastSoma[:,:] = 1
        somaOverflowRatio = 1
        sizeRatio=20
        continueRefiningThreshold = True
        img_for_threshold = copy.copy(img)
#        start = generalTools.microseconds()
        while continueRefiningThreshold:
            
            if sizeRatio > 10:
                percentile = percentile * 0.4
            elif sizeRatio > 4:
                percentile = percentile * 0.7
            elif sizeRatio > 2:
                percentile = percentile * 0.8
            elif sizeRatio > 1.5:
                percentile = percentile * 0.95
            elif somaOverflowRatio > 0.8:
                percentile = percentile * 0.97
            elif somaOverflowRatio > 0.65:
                percentile = percentile * 0.98
            else:
                percentile = percentile * 0.99
            Ints = img_for_threshold[img>0].ravel()
            sortedInts = np.sort(Ints)
            thresholdVal = ThresholdNeuron.getValueOfPercentile(sortedInts,percentile)
#                thresholdVal, percentile, best,img_for_threshold = self.thresholdByNormDistr(percentile,Ints,img_for_threshold,img,thresholdVal)

            if ~np.isnan(thresholdVal):
                
                img_threshold = img > thresholdVal
                soma,cX,cY, img_threshold = ThresholdNeuron.getSomaFromThresholdedImage(img_threshold,minSomaOverflow,somaExtractionFilterSize,np.nan,np.nan,objectSizeToRemoveForSomaExtraction,closingForPresoma)
                

                if len(np.where(soma == 1)[0]) == 0:
                    continueRefiningThreshold = False
                else:
                    somaEroded = ndimage.morphology.binary_erosion(soma,disk(2))
                    somaBorder = np.zeros_like(soma)
                    somaBorder[(soma == 1) & (somaEroded == 0)] = 1
                    somaBorder = morph.skeletonize(somaBorder)
                    
                    somaDifference = np.subtract(img_edges,somaBorder)
                    sizeRatio = len(np.where(soma == 1)[0]) / len(np.where(img_edges == 1)[0])
                    somaOverflowRatio = len(np.where(somaDifference == -1)[0])/len(np.where(somaBorder == 1)[0])
                    
                    if somaOverflowRatio <= maxToleratedSomaOverflowRatio:
                        continueRefiningThreshold = False
            else:
                continueRefiningThreshold = False
        return img_threshold, percentile, thresholdVal,cX,cY, Ints, sortedInts

    @staticmethod
    def refineThreshold(img,percentile,lastImg,lastImg_clean,cX,cY,soma,Ints,sortedInts,maxThresholdChange,percentileChangeForThreshold,dilationForOverlap,overlapMultiplicator,minBranchSize,grainSizeToRemove):
        finalThresholdVal = np.nan
        
        
        #--------------------CROP IMAGES FOR SPEED
        lastImg_clean,borderVals = generalTools.cropImage(lastImg_clean,[],10)
        lastImg,borderVals = generalTools.cropImage(lastImg,borderVals)
        img,borderVals = generalTools.cropImage(img,borderVals)
        soma,borderVals = generalTools.cropImage(soma,borderVals)
        
        cX = cX - borderVals[2]
        cY = cY - borderVals[0]

        #create reference image to check for parts being connected to the main neuron
        lastImg_forNeuron = ndimage.morphology.binary_closing(lastImg,disk(2))
        lastImg_forNeuron = ndimage.morphology.binary_dilation(lastImg_forNeuron,disk(2))
        imgOfWholeNeuron = ThresholdNeuron.isolateNeuronBasedOnSoma(lastImg_forNeuron,cX,cY)
        iteration = 0
        firstIteration = True
        optimalThresh = False
        soma = soma.astype(bool)
        thresholdVal = 0
        while (optimalThresh == False) & (iteration < 30):
            #in first iteration, thresholded image was already generated
            if firstIteration:
                firstIteration = False
            else:
                nbOfLabels,nbPixHoles,lastImg,thresholdVal = ThresholdNeuron.evalThreshold(img,Ints,sortedInts,percentile,grainSizeToRemove,imgOfWholeNeuron)
                lastImg_copy = copy.copy(lastImg)
            
            iteration += 1
            if iteration == 1:
                pxOpening = 0
                lastImg_copy = ndimage.morphology.binary_opening(lastImg,morph.disk(pxOpening))
                lastImg_skel = morph.skeletonize(lastImg_copy)
                nbOfPx = len(np.where(lastImg_skel == 1)[0])
                lastImg_skel = ndimage.binary_dilation(lastImg_skel,disk(dilationForOverlap/overlapMultiplicator),iterations=overlapMultiplicator)
               
                lastImg_skel[soma] = False
                difference= 0
                openingIteration = 0
                if nbOfPx > minBranchSize/8:
                    #test how many px opening can be applied before some parts of neurites are lost
                    while (difference < minBranchSize/8) & (openingIteration < 10):
                        pxOpening += 1
                        openingIteration += 1
                        #open by two more px to be a bit more conservative
                        
                        lastImg_copy_test = ndimage.morphology.binary_opening(lastImg,disk(pxOpening))
                        lastImg_skel_test = morph.skeletonize(lastImg_copy_test)
                        lastImg_skel_test[soma] = False
                        lastImg_skel_test = morph.remove_small_objects(lastImg_skel_test,minBranchSize,connectivity=2)
                        lastImg_skel_test = ndimage.binary_dilation(lastImg_skel_test,disk(dilationForOverlap/overlapMultiplicator),iterations=overlapMultiplicator)


                        if len(np.where(lastImg_skel_test)[0]) > 0:
                            diffOfImgs,overlapOfImgs = generalTools.overlapImgs(lastImg_skel,lastImg_skel_test,dilationForOverlap,overlapMultiplicator,1,0)
                            diffOfImgs = morph.remove_small_objects(diffOfImgs,minBranchSize/8,connectivity=2)
                            difference = len(np.where(diffOfImgs > 0)[0])
                        else:
                            difference = minBranchSize
                    pxOpening -= 1
                lastImg_copy = ndimage.morphology.binary_opening(lastImg,morph.disk(pxOpening))
                #smoothen image a lot to skeletonize
            lastImg_copy = ndimage.morphology.binary_closing(lastImg_copy,disk(int(np.round(minBranchSize/8,0))))
            lastImg_skel = morph.skeletonize(lastImg_copy)
            #set soma area to true to prevent irrelevant/random changes in soma area after skeletonization

            lastImg_skel[soma] = False
            
#                        lastImg_skel = morph.remove_small_objects(lastImg_skel,self.minBranchSize,connectivity=2)
            lastImg_skel_labeled = measure.label(lastImg_skel)

            for label in np.unique(lastImg_skel_labeled):
                if label != 0:
                    allCoords = np.where(lastImg_skel_labeled == label)
                    startPoint = generalTools.getClosestPoint(allCoords,[[cY,cX]])
                    sortedPoints = [startPoint]
                    allSortedPoints, length_tmp, neuriteCoords_tmp = sortPoints.startSorting(allCoords,sortedPoints,0,minBranchSize,keepLongBranches=True)
                    
                    lastImg_skel[lastImg_skel_labeled == label] = False
                    lastImg_skel[np.transpose(allSortedPoints)[0],np.transpose(allSortedPoints)[1]] = True
            
            lastImg_skel_dil = ndimage.binary_dilation(lastImg_skel,disk(dilationForOverlap/overlapMultiplicator),iterations=overlapMultiplicator)
            
            lastImg_skel_dil[soma] = True

            
            if iteration == 1:
                firstImg_skel_dil = lastImg_skel_dil
            diffOfImgs,overlapOfImgs = generalTools.overlapImgs(firstImg_skel_dil,lastImg_skel_dil,dilationForOverlap,overlapMultiplicator,1,1)

            realDiff = len(np.where(diffOfImgs[0] == 1)[0])-len(np.where(diffOfImgs[1] == 1)[0])

            if (realDiff >= (maxThresholdChange)):
                optimalThresh = True
                finalThresholdVal = percentile/percentileChangeForThreshold
            if optimalThresh == False:
                percentile = percentile * percentileChangeForThreshold
                img_filtered = lastImg_clean
                img_filtered_forBackgound = lastImg
            #if reference image in first iteration has too few pixels (very early stage or dying neuron), stop thresholding
            if iteration == 1:
                nbOfPx = len(np.where(lastImg_skel == 1)[0])
                if nbOfPx < minBranchSize/8:
                    optimalThresh = True
        if (optimalThresh == False) & ~np.isnan(finalThresholdVal):
            percentile = finalThresholdVal
            nbOfLabels,nbPixHoles,lastImg,thresholdVal = ThresholdNeuron.evalThreshold(img,Ints,sortedInts,percentile,grainSizeToRemove,imgOfWholeNeuron)
            img_filtered = lastImg
        if np.isnan(finalThresholdVal):
            thresholdVal = np.nan
        else:
            thresholdVal = ThresholdNeuron.getValueOfPercentile(sortedInts,finalThresholdVal)
        
        #-------------------UNCROP IMAGES-------------------------
        img_filtered = generalTools.uncropImage(img_filtered,borderVals)
        img_filtered_forBackgound = generalTools.uncropImage(img_filtered_forBackgound,borderVals)
        
        return img_filtered, img_filtered_forBackgound, thresholdVal,finalThresholdVal