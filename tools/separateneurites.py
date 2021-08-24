# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:07:09 2019

@author: schelskim
"""

from tools.generaltools import generalTools

import sys
import os
import numpy as np
import pandas as pd
from skimage import io
from skimage import morphology as morph
from scipy import ndimage
from skimage import measure
import cv2
import copy
from scipy.spatial import distance
from matplotlib import pyplot as plt
from skimage.filters import median
from skimage.morphology import disk
from skimage.morphology import square
import warnings



class SeparateNeurites:
    
    @staticmethod
    def separateNeuritesByOpening(timeframe_neurites,maxNeuriteOpening,timeframe_soma,img,grainSizeToRemove,contrastThresholdToRemoveLabel,maxFractionToLoseByOpening):
        timeframe_neurites, borderVals = generalTools.cropImage(timeframe_neurites,[],10)
        timeframe_soma, borderVals = generalTools.cropImage(timeframe_soma,borderVals)
        img, borderVals = generalTools.cropImage(img,borderVals)
        
        timeframe_neurites[timeframe_soma == True] = False;
        timeframe_labeled,nbOfLabels = ndimage.label(timeframe_neurites,structure=[[1,1,1],[1,1,1],[1,1,1]])
        
        #open each neurite to see if neurites can be seperated 
        for label in np.unique(timeframe_labeled):
            if label != 0:
                oneNeurite = np.zeros_like(timeframe_neurites)
                oneNeurite[timeframe_labeled == label] = 1
                oneNeurite[timeframe_soma == True] = 1
                applyOpening = False
                forceApplyOpening = True
                removedOneLabel = False
                originalNeuriteSize = len(np.where(oneNeurite)[0])
                for a in range(1,maxNeuriteOpening+1):
                    openingSize = a
#                    print("erosion: {}".format(a))
                    oneNeurite_test = ndimage.morphology.binary_opening(oneNeurite,square(openingSize))
                    oneNeurite_test[timeframe_soma == True] = 0
                    oneNeurite_labeled,nbOfLabels_oneNeurite = ndimage.label(oneNeurite_test,structure=[[1,1,1],[1,1,1],[1,1,1]])
                    
                    neuriteSizeAfterOpening = len(np.where(oneNeurite_test)[0])
                    sizeLostByOpening = 1-(neuriteSizeAfterOpening/originalNeuriteSize)
                    
                    #don't apply opening if neurite Size has been constantly reduced, without hitting another break in the loop
                    if sizeLostByOpening > maxFractionToLoseByOpening:
                        applyOpening = False
                        break
                        
                    if nbOfLabels_oneNeurite > 1:
                        oneNeurite_test[timeframe_soma == True] = 1
                        oneNeurite_labeled,nbOfLabels_oneNeurite = ndimage.label(oneNeurite_test,structure=[[1,1,1],[1,1,1],[1,1,1]])
                        
                        if nbOfLabels_oneNeurite == 1:
                            applyOpening = True
                            openingSizeToApply = openingSize
                            break
                        else:
                            pxToKeep,pxToRemove,pxToRemove_contrast, removedOneLabel,forceApplyOpening = SeparateNeurites.checkWhichPxDontSeparateNeurites(oneNeurite_test,oneNeurite,timeframe_soma,timeframe_labeled,label,img)        
                            if removedOneLabel:
                                applyOpening = False
                                break
                    else:
                        oneNeurite_test[timeframe_soma == True] == 1
                        oneNeurite_labeled,nbOfLabels_oneNeurite = ndimage.label(oneNeurite_test,structure=[[1,1,1],[1,1,1],[1,1,1]])
                        if nbOfLabels_oneNeurite == 1:
                            if forceApplyOpening:
                                applyOpening = True
                                openingSizeToApply = openingSize
                        else:
                            break
                    
                    #if end of for loop is reached without hitting any stop, don't apply the opening!
                    if openingSize == maxNeuriteOpening:
                        applyOpening = False
                        
                if applyOpening:
                    oneNeurite[timeframe_soma == True] = 1
                    oneNeurite = ndimage.morphology.binary_opening(oneNeurite,square(openingSizeToApply))
                    
                    timeframe_neurites[(oneNeurite == 0) & (timeframe_labeled == label)] = 0
                if removedOneLabel:
                    #go through each island in pxToRemove, remove if nb of labels with soma is still 1
                    #if nb of labels with soma is > 1 & if something else was removed already:
                    #check which had the higher contrast, remove only that one
                    maxContrast = np.max(pxToRemove_contrast)
                    if maxContrast/100 > contrastThresholdToRemoveLabel:
                        oneNeurite[pxToRemove_contrast == maxContrast] = 0
                    else:
                        #check each label for which one separates neurites most equally
                        oneNeurite = SeparateNeurites.separateNeuritesEvenly(oneNeurite,pxToRemove)
                    timeframe_neurites[(oneNeurite == 0) & (timeframe_labeled == label)] = 0
        timeframe_neurites[timeframe_soma == True] = True;

        timeframe_neurites = generalTools.uncropImage(timeframe_neurites,borderVals)
        return timeframe_neurites
    
    @staticmethod
    def checkWhichPxDontSeparateNeurites(oneNeurite_test,oneNeurite,timeframe_soma,timeframe_labeled,label,img):
        

        #-------------CROP IMAGES FOR SPEED-------------
        oneNeurite,borderVals = generalTools.cropImage(oneNeurite,[],30)
        oneNeurite_test,borderVals = generalTools.cropImage(oneNeurite_test,borderVals)
        timeframe_soma,borderVals = generalTools.cropImage(timeframe_soma,borderVals)
        timeframe_labeled,borderVals = generalTools.cropImage(timeframe_labeled,borderVals)
        img,borderVals = generalTools.cropImage(img,borderVals)
        

        #if more labels were present, check each part of px that were removed and their intensity
        #measure intensity as fold change from neighbouring px (take all sites at which it touches px separately)
        
        forceApplyOpening = True
        removedOneLabel = False
        sizeOneNeurite = len(np.where(oneNeurite == 1)[0])
        oneNeurite_removedPx = np.zeros_like(oneNeurite)
        oneNeurite_removedPx[(oneNeurite == 1) & (oneNeurite_test == 0)] = 1
        oneNeurite_removedPx_labeled, nbLabels = ndimage.label(oneNeurite_removedPx,structure=[[1,1,1],[1,1,1],[1,1,1]])
        pxToKeep = np.zeros_like(oneNeurite)
        pxToRemove = np.zeros_like(oneNeurite)
        pxToRemoveContrast = np.zeros_like(oneNeurite)
        pxToRemoveContrast = pxToRemoveContrast.astype(int)
        oneNeurite[timeframe_soma == 1] = 0
        for label_removed in np.unique(oneNeurite_removedPx_labeled):
            if label_removed > 0:
                #test if removing the current label from original image of neurite, increases nb of labels (seperates neurites)
                
                oneLabel = np.zeros_like(oneNeurite)
                oneLabel[oneNeurite_removedPx_labeled == label_removed] = 1
                
                oneLabel = ndimage.morphology.binary_dilation(oneLabel,disk(2))
                oneLabel[timeframe_labeled != label] = 0
                
                oneNeurite_removedLabel = copy.copy(oneNeurite)
                oneNeurite_removedLabel[oneLabel == 1] = 0
                oneNeurite_removedLabel_labeled, nbLabelsRemovedLabel = ndimage.label(oneNeurite_removedLabel,structure=[[1,1,1],[1,1,1],[1,1,1]])
                
                if nbLabelsRemovedLabel > 1:
                    #prevent default application of opening 
                    #since things that were remoced actually led to splitting and therefore loss of some part/s of neurites
                    forceApplyOpening = False
                    
                    #get contast of island to neighhboring pixels
                    #contrast helps to remove only parts of mini-neurites between bigger neurites without separating bigger neurites in the middle
                    averagePxVal = np.median(img[oneLabel == 1]) 
                    oneLabel_dilated = ndimage.morphology.binary_dilation(oneLabel,disk(10))
                    oneLabel_neighbors = np.zeros_like(oneNeurite)
                    oneLabel_neighbors[((oneLabel_dilated == 1) & (oneNeurite == 1)) & (oneLabel == 0)] = 1
                    oneLabel_neighbors = ndimage.morphology.binary_erosion(oneLabel_neighbors,disk(2))
                    oneLabel_neighbors_labeled, nbLabels = ndimage.label(oneLabel_neighbors,structure=[[1,1,1],[1,1,1],[1,1,1]])
                    maxContrast = 0
                    for neighbor_label in np.unique(oneLabel_neighbors_labeled):
                        if neighbor_label != 0:
                            averagePxValOfNeighbor = np.median(img[oneLabel_neighbors_labeled == neighbor_label])
                            contrast = averagePxValOfNeighbor / averagePxVal
                            if contrast > maxContrast:
                                maxContrast = contrast
                    #check if removal of label led to separation of a piece at least 10% or neurite
                    maxSizeOfLabels = SeparateNeurites.getMaxSizeOfLabels(oneNeurite_removedLabel_labeled)
                    
                    #also check whether removing these pixels does not lead to more pieces when adding soma back in
                    oneNeurite_removedLabel_soma = copy.copy(oneNeurite_removedLabel)
                    oneNeurite_removedLabel_soma[timeframe_soma == 1] = 1
                    oneNeurite_removedLabel_soma_labeled,nbLabelsWithSoma = ndimage.label(oneNeurite_removedLabel_soma,structure=[[1,1,1],[1,1,1],[1,1,1]])
               
                    
                    if (nbLabelsWithSoma > 1) | (maxSizeOfLabels > 0.9 * sizeOneNeurite):
                        pxToKeep[oneLabel == 1] = 1
                    else:
                        pxToRemove[oneLabel == 1] = 1
                        pxToRemoveContrast[oneLabel == 1] = int(np.round(maxContrast*100,0))
                        removedOneLabel = True
                        
                        
        #-------------UNCROP IMAGES-------------
        pxToKeep = generalTools.uncropImage(pxToKeep,borderVals)
        pxToRemove = generalTools.uncropImage(pxToRemove,borderVals)
        pxToRemoveContrast = generalTools.uncropImage(pxToRemoveContrast,borderVals)
            
        return pxToKeep,pxToRemove, pxToRemoveContrast,removedOneLabel, forceApplyOpening

    @staticmethod
    def separateNeuritesEvenly(oneNeurite,pxToRemove):
        pxToRemove_labeled,nbLabels = ndimage.label(pxToRemove,structure=[[1,1,1],[1,1,1],[1,1,1]])
        allSizeRatios = []
        allLabelToRemove = []
        for labelToRemove in np.unique(pxToRemove_labeled):
            if labelToRemove != 0:
                oneNeurite_test = copy.copy(oneNeurite)
                oneNeurite_test[pxToRemove_labeled == labelToRemove] = 0
                oneNeurite_test_labeled,nbLabels = ndimage.label(oneNeurite_test,structure=[[1,1,1],[1,1,1],[1,1,1]])
                
                allNeuriteSizes = []
                for oneNeuriteLabel in np.unique(oneNeurite_test_labeled):
                    if oneNeuriteLabel != 0:
                        neuriteSize = len(np.where(oneNeurite_test_labeled == oneNeuriteLabel)[0])
                        allNeuriteSizes.append(neuriteSize)
                allNeuriteSizes = np.sort(allNeuriteSizes)
                if len(allNeuriteSizes) > 1:
                    sizeRatio = allNeuriteSizes[-1]/allNeuriteSizes[-2]
                    allSizeRatios.append(sizeRatio)
                    allLabelToRemove.append(labelToRemove)
        if len(allSizeRatios) > 1:
            #minimal size ratio is the one closest to 1
            finalLabelToRemove = allLabelToRemove[np.where(allSizeRatios == np.min(allSizeRatios))[0][0]]
            oneNeurite[pxToRemove_labeled == finalLabelToRemove] = 0
        return oneNeurite

    @staticmethod
    def getMaxSizeOfLabels(thresholdedImg_labeled):
        maxSize = 0
        for label in np.unique(thresholdedImg_labeled):
            if label != 0:
                size = len(np.where(thresholdedImg_labeled == label)[0])
                if size > maxSize:
                    maxSize = size
        return maxSize

    @staticmethod
    def separateNeuritesBySomaDilation(timeframe_neurites,timeframe_soma,timeframe_thresholded,maxSomaDilation):
        timeframe_neurites[timeframe_soma == 1] = 0
        timeframe_labeled,nbOfLabels = ndimage.label(timeframe_neurites,structure=[[1,1,1],[1,1,1],[1,1,1]])
        timeframe_finalSoma = copy.copy(timeframe_soma)
        for a in range(1,maxSomaDilation):
            timeframe_labeled,nbOfLabels = ndimage.label(timeframe_neurites,structure=[[1,1,1],[1,1,1],[1,1,1]])
            timeframe_soma_dilated = ndimage.morphology.binary_dilation(timeframe_soma,disk(a))
            for label in np.unique(timeframe_labeled):
                if (label != 0):
                    oneLabel = np.zeros_like(timeframe_neurites)
                    oneLabel[timeframe_labeled == label] = 1
                    oneLabel_labeled, nbOfLabelsBefore = ndimage.label(oneLabel,structure=[[1,1,1],[1,1,1],[1,1,1]])
                    oneLabel_noSoma = copy.copy(oneLabel)
                    oneLabel_noSoma[timeframe_soma_dilated == 1] = 0
                    oneLabel_labeled, nbOfLabelsAfter = ndimage.label(oneLabel_noSoma,structure=[[1,1,1],[1,1,1],[1,1,1]])
                    if nbOfLabelsAfter > nbOfLabelsBefore:
                        #open oneLabel and check which parts are removed there and with dilation of soma
                        oneLabel_opened = ndimage.morphology.binary_opening(oneLabel,disk(a/2+2))
#                        timeframe_neurites[timeframe_labeled == label] = 0
#                        timeframe_neurites[oneLabel_noSoma == 1] = 1
#                        testImg = np.zeros_like(timeframe_neurites)
#                        testImg[(timeframe_soma == 0) & (timeframe_soma_dilated == 1) & (oneLabel_opened == 0) & (oneLabel == 1)] = 1
                        timeframe_finalSoma[(timeframe_soma == 0) & (timeframe_soma_dilated == 1) & (oneLabel_opened == 0) & (oneLabel == 1)] = 1
                        timeframe_neurites[(timeframe_soma == 0) & (timeframe_soma_dilated == 1) & (oneLabel_opened == 0) & (oneLabel == 1)] = 0
                        
        return timeframe_neurites,timeframe_finalSoma