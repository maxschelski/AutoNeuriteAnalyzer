# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:47:17 2019

@author: schelskim
"""
from .generaltools import generalTools

import numpy as np
from skimage import morphology as morph
from scipy import ndimage
from matplotlib import pyplot as plt
import copy
from skimage.morphology import disk



class SomaTools():

    @staticmethod
    def getSoma(preSoma, minSomaOverflow, somaExtractionFilterSize,
                img_thresholded, objectSizeToRemoveForSomaExtraction,
                cX=None, cY=None):
        soma_base = ndimage.filters.gaussian_filter(preSoma,(somaExtractionFilterSize))
        preSoma = preSoma > 0
        preSoma = preSoma.astype(int)
        somaThreshold = 240
        currentSomaOverflow = minSomaOverflow
        #start with lower soma threshold, work up if overflow is lower, work down if overflow is higher
        while (currentSomaOverflow <= minSomaOverflow) & (somaThreshold >= 100):
#            print("soma thresh:{}".format(somaThreshold))
            all_soma = soma_base > somaThreshold
            soma = SomaTools.extractSoma(all_soma,img_thresholded,cX,cY,objectSizeToRemoveForSomaExtraction)
            soma = soma.astype(int)
            somaDifference = np.subtract(preSoma,soma)
            currentSomaOverflow = len(np.where(somaDifference == -1)[0])
#            currentSomaSize = len(np.where(soma == True)[0])
            somaThreshold -= 5
        
        #remover other somas, if they are not attached to the "main" soma
        #remove small objects to prevent slowing down of algorithm for early threshold iterations

        image_thresholded_labeled, nbLabels = ndimage.label(morph.remove_small_objects(img_thresholded,objectSizeToRemoveForSomaExtraction),structure=[[1,1,1],[1,1,1],[1,1,1]])
        soma_labels = np.unique(image_thresholded_labeled[soma == 1])
        maxSomaLabelSize = 0
        soma_label = np.nan
        for label in soma_labels:
            somaLabelSize = len(np.where((image_thresholded_labeled == label) & (soma == 1))[0])
            if somaLabelSize > maxSomaLabelSize:
                maxSomaLabelSize = somaLabelSize
                soma_label = label
        if not np.isnan(soma_label):
            img_thresholded[((all_soma == 1) & (soma == 0)) & (image_thresholded_labeled != soma_label)] = 0
        else:
            img_thresholded = np.zeros_like(img_thresholded)
            soma = np.zeros_like(soma)
        return soma, img_thresholded
    
    @staticmethod
    def extractSoma(soma,image_thresholded,cX,cY,objectSizeToRemoveForSomaExtraction,min_branch_size=40):
        #choose one soma out of more possibilities
        #remove small additional objects which otherwise could be mistakenly assumed to be the soma
        soma = morph.remove_small_objects(soma,50)
        soma_labeled, nbOfLabels = ndimage.label(soma,structure=[[1,1,1],[1,1,1],[1,1,1]])
        soma_labels = np.unique(soma_labeled)
        #if more than one soma label is present, reduce to one label
        maxNbOfNewLabels = np.nan
        alternative_maxNbOfNewLabels = np.nan
        if len(np.unique(soma_labeled)) > 2:
            #if cX and cY were defined during thresholding from prototypic soma, use as reference to find soma 
            if type(cX) != type(None):
                somaLabelPoint = generalTools.getClosestPoint(np.where(soma == True),[[cY,cX]])
                somaLabel = soma_labeled[somaLabelPoint[0],somaLabelPoint[1]]
            else:
                #if cX has not been defined, check which label is bigger (more pixels)
                image_thresholded = morph.remove_small_objects(image_thresholded,objectSizeToRemoveForSomaExtraction)
                image_labeled, nbOfLabelsBefore = ndimage.label(image_thresholded,structure=[[1,1,1],[1,1,1],[1,1,1]])
                #extract soma which increases number of labels the most after being removed (indicates more neurites attached to it)
                for label in soma_labels:
                    if label != 0:
                        image_thresholded_removedSoma = copy.copy(image_thresholded)
                        thisSoma = np.zeros_like(soma)
                        thisSoma[soma_labeled == label] = 1
                        thisSoma = ndimage.binary_dilation(thisSoma,disk(1),iterations=40)
                        image_thresholded_removedSoma[thisSoma == 1] = 0
                        image_thresholded_removedSoma = morph.remove_small_objects(image_thresholded_removedSoma,objectSizeToRemoveForSomaExtraction/10)
                        image_thresholded_removed_soma_skel = morph.skeletonize(image_thresholded_removedSoma)
                        image_thresholded_removed_soma_skel = morph.remove_small_objects(image_thresholded_removed_soma_skel,min_branch_size)
                        image_labeled_after, nbOfLabelsAfter = ndimage.label(image_thresholded_removed_soma_skel,structure=[[1,1,1],[1,1,1],[1,1,1]])

                        alternaative_image_labeled_after, alternative_nbOfLabelsAfter = ndimage.label(image_thresholded_removedSoma,structure=[[1,1,1],[1,1,1],[1,1,1]])
                        
                        nbOfLabelsDiff = nbOfLabelsAfter - nbOfLabelsBefore
                        
                        if np.isnan(maxNbOfNewLabels):
                            maxNbOfNewLabels= nbOfLabelsDiff
                            somaLabel = label
                        elif nbOfLabelsDiff > maxNbOfNewLabels:
                            maxNbOfNewLabels= nbOfLabelsDiff
                            somaLabel = label
                            
                        #if neurites are too short, it could be that they are removed during skeletonization
                        #for this possibility also get soma label without skeletonization etc
                        if np.isnan(alternative_maxNbOfNewLabels):
                            alternative_maxNbOfNewLabels = alternative_nbOfLabelsAfter
                            alternative_somaLabel = label
                        if alternative_nbOfLabelsAfter > alternative_maxNbOfNewLabels:
                            alternative_maxNbOfNewLabels = alternative_nbOfLabelsAfter
                            alternative_somaLabel = label
            
            if not np.isnan(maxNbOfNewLabels):
                if maxNbOfNewLabels <= 1:
                    somaLabel = alternative_somaLabel
                
            #remove labels which are not the soma
            for label in soma_labels:
                if (label != 0) & (label != somaLabel):
                    soma[soma_labeled == label] = False
        return soma

        
    
