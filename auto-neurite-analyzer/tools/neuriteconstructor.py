# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:18:16 2019

@author: schelskim
"""

from tools.sortpoints import sortPoints
from tools.generaltools import generalTools

import numpy as np
import pandas as pd
from skimage import morphology as morph
from scipy.spatial import distance
from scipy import ndimage
import copy
from skimage.morphology import disk
from skimage.morphology import square
from matplotlib import pyplot as plt
from skimage.draw import line




class neuriteConstructor():
    
    
    def __init__(self,allNeuritePoints,labeledGroupImage,cX,cY,possibleOriginsColumns,allNeuritesColumns,overlappingOriginsColumns,identity,allNeurites,dilationForOverlap,overlapMultiplicator,maxChangeOfNeurite,minBranchSize,testNb,optimalThreshold,maxOverlapOfNeurites,img,toleranceForSimilarity,dilationForSmoothing,gaussianForSmoothing):
        
        self.labeledGroupImage = labeledGroupImage
        self.allNeuritePoints = allNeuritePoints
        
        self.cX = cX
        self.cY = cY
        self.possibleOriginsColumns = possibleOriginsColumns
        self.allNeuritesColumns = allNeuritesColumns
        self.overlappingOriginsColumns = overlappingOriginsColumns
        self.identity = identity
        self.allNeurites = allNeurites
        self.dilationForOverlap = dilationForOverlap
        self.overlapMultiplicator = overlapMultiplicator
        self.maxChangeOfNeurite = maxChangeOfNeurite
        self.minBranchSize = minBranchSize
        self.testNb = testNb
        self.optimalThreshold = optimalThreshold
        self.maxOverlapOfNeurites = maxOverlapOfNeurites
        self.toleranceForSimilarityOfNeurites = toleranceForSimilarity
        
        self.branchNb = 0
        self.overlappingOrigins = pd.DataFrame(columns=self.overlappingOriginsColumns)
        
        self.newNeuritesDf = pd.DataFrame(columns=self.allNeuritesColumns)
        self.neuriteNb = 0
        self.neuriteOrigin = np.nan
        self.continueWithNeurite = True
        self.iterations = 0
        
        #smoothen stronger than for initial smoothing
        self.dilationForSmoothing = dilationForSmoothing*2
        self.gaussianForSmoothing = gaussianForSmoothing*2
        
        
        self.img = img
        
    def constructNeurites(self):
        
        for neuritePoints in self.allNeuritePoints:
            if len(neuritePoints) > self.minBranchSize:
                # CHECK WHICH NEURITE WAS JUST BUILD - 
                #REFERENCE TO OTHER TIME FRAMES
                #discontinue separation if following code is 
                #repeatedly execute (> 2 times) without changing neurite structure
                
                self.constructedNeuriteImage = generalTools.getImageFromPoints(neuritePoints,self.img.shape)
                
                self.constructedNeuriteImage = generalTools.smoothenImage(self.constructedNeuriteImage,self.dilationForSmoothing,self.gaussianForSmoothing)
                
                
                self.allCoords = np.where(self.constructedNeuriteImage == 1)
                
                startPoint = generalTools.getClosestPoint(self.allCoords,[[neuritePoints[0][0],neuritePoints[0][1]]])
                
                #sort all points of neurite starting with the point closest to cell body (endpoint)
                keepLongBranches = False

                (self.sortedPoints, 
                 length,
                 neuriteCoords_tmp) = sortPoints.startSorting(self.allCoords,
                                                              [startPoint],0,
                                                              self.minBranchSize,
                                                              [],keepLongBranches,
                                                              -1,
                                                              distancePossible=False)
                

                self.sortedArray = np.transpose(self.sortedPoints)
                
                (self.neuriteOrigin, bestOrigin, 
                 neuriteBranch, start_branch,
                 self.overlappingOrigins) = self.getOriginFromPreviousTimeframes(self.overlappingOrigins,
                                                              self.neuriteOrigin)
                
                
                avgIntOfPoints = np.mean(self.img[self.sortedArray[0],self.sortedArray[1]])
                self.testNb += 1
                self.sortedArray = np.transpose(
                                    generalTools.convert_points_to_point_list(
                                        np.transpose(self.sortedArray))
                                    )
                self.newNeuritesDf.loc[self.neuriteNb] = [self.identity[0],
                                                          self.identity[1],
                                                          self.identity[2],
                                                          self.identity[3],
                                                          self.identity[4],
                                                          self.neuriteOrigin,
                                                          neuriteBranch,
                                                          start_branch,
                                                          np.array([
                                                              self.sortedArray[0][0],
                                                              self.sortedArray[1][0]]),
                                                          np.array([
                                                              self.sortedArray[0][-1],
                                                              self.sortedArray[1][-1]]),
                                                          bestOrigin.loc['pxdifference'],
                                                          bestOrigin.loc['diffRatio'],
                                                          bestOrigin.loc['pxSimilar'],
                                                          np.array([self.cX,self.cY]),
                                                          self.sortedArray[0],
                                                          self.sortedArray[1],
                                                          bestOrigin.loc['gain_x'],
                                                          bestOrigin.loc['gain_y'],
                                                          bestOrigin.loc['loss_x'],
                                                          bestOrigin.loc['loss_y'],
                                                          length,0,
                                                          self.optimalThreshold,
                                                          avgIntOfPoints]
                
                self.branchNb += 1
                self.neuriteNb += 1        

        
        if len(self.newNeuritesDf) > 0:
            #set origin number similar to previous timeframes
            self.neuriteOrigin, self.newNeuritesDf = self.setOriginIfNoneWasFound(self.neuriteOrigin,self.newNeuritesDf)

            self.newNeuritesDf,allOrigins = self.setRemainingOriginsOfNeurites(self.newNeuritesDf,self.neuriteOrigin)
            
            #set branch number similar to previous timeframes
            self.newNeuritesDf,allOrigins = self.checkIfBranchesWithSameOriginOverlap(self.newNeuritesDf)
            
            #set remaining branches which were not found
            #in previous timeframes
            self.newNeuritesDf = self.setRemainingBranchesOfNeurites(allOrigins,self.newNeuritesDf,neuriteBranch)
    
            self.newNeuritesDf = self.removeDoubledBranches(allOrigins,self.newNeuritesDf,self.allNeurites)
    
            #exclude neurites with different origins which are overlapping
            #this would probably indicate crossing points used as branch points
            self.allNeurites, self.newNeuritesDf = self.checkForOverlapBetweenDifferentOrigins(self.overlappingOrigins,self.allNeurites,self.newNeuritesDf)       
            
            #remove branches that overlap 70% starting from second timeframe
            if len(self.allNeurites) > 0:
                self.newNeuritesDf = self.remove_overlapping_branches(allOrigins,self.newNeuritesDf)
    
            #at this point, all origins and branches are set                             

        return self.allNeurites,self.newNeuritesDf,self.testNb

    def checkIfBranchesWithSameOriginOverlap(self,newNeuritesDf):
        #check if al branches for one origin show overlap. 
        #For branches which do not show overlap, create new origins.
        allOrigins = np.unique(newNeuritesDf['origin'])                        
        for origin in allOrigins:
            oneOrigin = newNeuritesDf.loc[newNeuritesDf['origin'] == origin]
            
            allOverlappingGroups = self.groupBranchesByOverlap(oneOrigin)
            
            #set different origins for different groups
            if len(allOverlappingGroups) > 1:
                #get next origin which has not been used
                nextOrigin = np.nan                
                for group in allOverlappingGroups:
                        #only define first next origin to use from scratch, then each iteration increment next origin
                    if np.isnan(nextOrigin):
                        nextOrigin = self.getNextOrigin(newNeuritesDf,origin)
                    else:
                        nextOrigin += 1
                    for index in group:
                        newNeuritesDf.loc[index,'origin'] = nextOrigin
                        
        allOrigins = np.unique(newNeuritesDf['origin'])
        return newNeuritesDf, allOrigins
    
    def getNextOrigin(self,newNeuritesDf,origin):
        if len(self.allNeurites) > 0:
            nextOriginAllNeurites = np.max(self.allNeurites['origin'])+1
        else:
            nextOriginAllNeurites = 1
        newNeuritesDfForNextOrigin = newNeuritesDf.loc[newNeuritesDf['origin'] != origin]
        nextOrigin = max(nextOriginAllNeurites,np.max(newNeuritesDfForNextOrigin['origin'])+1)
        return nextOrigin
        
    
    def groupBranchesByOverlap(self,oneOrigin,min_branch_overlap_factor=-1):
        
        #go through each row by index, combine index of all rows with overlap in index groups
        allOverlappingGroups= []        
        for oneIndex in oneOrigin.index.values:
            #check if index is already present in sorted overlapping groups
            indexAlreadySorted = self.checkIfIndexIsAlreadySorted(oneIndex,allOverlappingGroups)
            firstRow = oneOrigin.loc[int(oneIndex)]
            #px similar needs to be 0, indicating that no corresponding neurite was found
            if min_branch_overlap_factor == -1:
                no_corresponding_neurite = firstRow['pxSimilar'] == 0
            else:
                no_corresponding_neurite = True
            if (not indexAlreadySorted) & (no_corresponding_neurite):
                allOverlappingGroups.append([oneIndex])
                firstCoords = [firstRow['x'],firstRow['y']]
                firstImg = generalTools.getImageFromCoords(firstCoords,self.img.shape)
                firstImg = ndimage.binary_dilation(firstImg,disk(self.dilationForOverlap/self.overlapMultiplicator),iterations=self.overlapMultiplicator)
                for otherIndex in oneOrigin.index.values:
                    indexAlreadySorted = self.checkIfIndexIsAlreadySorted(otherIndex,allOverlappingGroups)
                    secondRow = oneOrigin.loc[otherIndex]
                    secondCoords = [secondRow['x'],secondRow['y']]
                    if min_branch_overlap_factor == -1:
                        no_corresponding_neurite = secondRow['pxSimilar'] == 0
                    else:
                        no_corresponding_neurite = True
                    if (not indexAlreadySorted) & (otherIndex != oneIndex) & (no_corresponding_neurite):
                        secondImg = generalTools.getImageFromCoords(secondCoords,self.img.shape)
                        secondImg = ndimage.binary_dilation(secondImg,disk(self.dilationForOverlap/self.overlapMultiplicator),iterations=self.overlapMultiplicator)
                        diffOfImgs,overlapOfImgs = generalTools.overlapImgs(firstImg,secondImg,self.dilationForOverlap,self.overlapMultiplicator)
                        if min_branch_overlap_factor == -1:
                            min_branch_overlap = self.minBranchSize*0.8
                        else:
                            max_length = max(len(firstCoords[0]),len(secondCoords[0]))
                            min_branch_overlap = max_length*min_branch_overlap_factor
#                        generalTools.showThresholdOnImg(firstImg,self.img,2)
#                        generalTools.showThresholdOnImg(secondImg,self.img)
#                        plt.figure()
#                        plt.imshow(np.zeros((521,512)))
#                        plt.text(40,40,str(len(np.where(overlapOfImgs==1)[0])))
#                        plt.text(80,80,str(min_branch_overlap))
                        if len(np.where(overlapOfImgs==1)[0]) >= (min_branch_overlap):
                            allOverlappingGroups[-1].append(int(otherIndex))
        return allOverlappingGroups
    
    
    def checkIfIndexIsAlreadySorted(self,oneIndex,allOverlappingGroups):
        indexAlreadySorted = False
        for overlappingGroup in allOverlappingGroups:
            if oneIndex in overlappingGroup:
                indexAlreadySorted = True
        return indexAlreadySorted

    def getOriginFromPreviousTimeframes(self,overlappingOrigins,neuriteOrigin):                        
        possibleOrigins = pd.DataFrame(columns=self.possibleOriginsColumns)
        
        continueSearch = True
        frameTimer = 1
                
        # only check for what origin neurite has, if origin is not already determined for this group of neurites
                        
        while continueSearch:
            #if there is no earlier frame to look at, end here
            if(self.identity[4]-frameTimer) <= 0:
                continueSearch = False
            else:
                frameTimerNeurites = self.allNeurites[(((self.allNeurites.date == self.identity[0]) & (self.allNeurites.experiment == self.identity[1])) & ((self.allNeurites.neuron == self.identity[2]) & (self.allNeurites.channel == self.identity[3]))) & (self.allNeurites.time == (self.identity[4]-frameTimer))].reset_index()                        
                
                possibleOrigins,overlappingOrigins = self.getPossibleOriginFromList(self.sortedArray,frameTimerNeurites,possibleOrigins,overlappingOrigins,len(self.newNeuritesDf),self.dilationForOverlap,self.overlapMultiplicator,frameTimer)
                frameTimer += 1
            if (len(possibleOrigins) > 4) | (frameTimer >= 10):
                continueSearch = False
        
#        plt.figure()
#        plt.imshow(np.zeros((512,512)))
#        plt.text(40,40,str(possibleOrigins[['origin','branch','pxdifference','pxSimilar']]))
        
        neuriteBranch = np.nan
        start_branch = np.nan
                    
        #if any possible neurite origin was found
        if(len(possibleOrigins) > 0):
            #get origin that matches the best
            bestOrigin = neuriteConstructor.getBestNeuriteBasedOnSimilarityAndFrequency(possibleOrigins,self.allNeurites)
#            bestOrigin = self.getRowWithMaximumVal(possibleOrigins,'pxSimilar')
            neuriteOrigin = int(bestOrigin['origin'])
            neuriteBranch = int(bestOrigin['branch'])
            start_branch = int(bestOrigin['start_branch'])
        # --------------- end of finding origin & branch
        
        #set value for bestOrigin in no similar neurites were found
        if 'bestOrigin' not in locals():
            bestOrigin = pd.DataFrame(columns=self.possibleOriginsColumns)
            bestOrigin.loc[0] = [np.nan,np.nan,np.nan,np.nan,np.nan,[],[],[],[],self.maxChangeOfNeurite*2,self.maxChangeOfNeurite*2,0,0]
            bestOrigin = bestOrigin.loc[0]
                        
        return neuriteOrigin, bestOrigin, neuriteBranch,start_branch, overlappingOrigins     

                      
    def setOriginIfNoneWasFound(self,neuriteOrigin,newNeuritesDf):
        #----------------- set origin as new origin if no origin was found for any branch
        #neuriteorigin is only nan if no origin at all was found yet (it is carried over from branch to branch)
        if np.isnan(neuriteOrigin):
            if len(self.allNeurites) != 0:
                allFrameNeurites = self.allNeurites[((self.allNeurites.date == self.identity[0]) & (self.allNeurites.experiment == self.identity[1])) & ((self.allNeurites.neuron == self.identity[2]) & (self.allNeurites.channel == self.identity[3]))].reset_index()
                neuriteOrigin = allFrameNeurites.origin.max()+1
            else:
                neuriteOrigin = 1
            #if no origin at all was found yet, all origins are np.nan -> can all be set together
            newNeuritesDf['origin'] = neuriteOrigin
        return neuriteOrigin, newNeuritesDf


    def setRemainingOriginsOfNeurites(self,newNeuritesDf,neuriteOrigin):
        
        #check which different origins were identified already
        allOrigins = np.unique(newNeuritesDf['origin'])
        nbOrigins = len(allOrigins)
        IDsToDelete = []
        for i in range(0,nbOrigins):
            if np.isnan(allOrigins[i]):
                IDsToDelete.append(i)
        IDsToDelete.sort(reverse=True)
        for i in IDsToDelete:
                allOrigins = np.delete(allOrigins,i)
        
        #if only one neurite is found, then origin of all remaining np.nan neurites can be set together
        if len(allOrigins) == 1:
            newNeuritesDf['origin'] = neuriteOrigin
        else:
            #do new fitting to determine to which origin neurites with np.nan origin fits best
            for rowIndex in newNeuritesDf.index.values:
                thisNeurite = newNeuritesDf.loc[rowIndex]
                if np.isnan(thisNeurite['origin']):
                    thisNeuriteImg = np.zeros_like(self.img)
                    thisNeuriteImg[[thisNeurite['x'],thisNeurite['y']]] = 1
                    possibleOrigins = pd.DataFrame(columns=self.possibleOriginsColumns)  
                    overlappingOrigins_temp = pd.DataFrame(columns=self.overlappingOriginsColumns)
                    
                    for thisOrigin in allOrigins:
                        
                        thisOriginNewNeurites = newNeuritesDf[newNeuritesDf.origin == thisOrigin]
                        #set threshold for accepting possible origin very low
                        possibleOrigins,overlappingOrigins_temp = self.getPossibleOriginFromList([thisNeurite['x'],thisNeurite['y']],thisOriginNewNeurites,possibleOrigins,
                                                                                      overlappingOrigins_temp,rowIndex,self.dilationForOverlap,self.overlapMultiplicator,0,self.constructedNeuriteImage.shape[0],-1,0)
                        
                    if len(possibleOrigins) > 0:
                        bestOrigin = neuriteConstructor.getRowWithMaximumVal(possibleOrigins,'pxSimilar')['origin']
                    else:
                        bestOrigin = self.getNextOrigin(newNeuritesDf,-1)
                    newNeuritesDf.loc[rowIndex,'origin'] = bestOrigin
                    
        return newNeuritesDf, allOrigins
    
    @classmethod
    def getNextBranchNb(self,thisOrigin,newNeuritesDf,allNeurites,identity):
        thisOriginNeurites = allNeurites[(((allNeurites.date == identity[0]) & (allNeurites.experiment == identity[1])) & ((allNeurites.neuron == identity[2]) & (allNeurites.channel == identity[3]))) & (allNeurites.origin == thisOrigin)].reset_index()
        oldNeuriteBranch = thisOriginNeurites.branch.max() + 1
        thisOriginNewNeurites = newNeuritesDf[newNeuritesDf.origin == thisOrigin].reset_index()
        newNeuriteBranch = thisOriginNewNeurites.branch.max() + 1
        if (not np.isnan(oldNeuriteBranch)) & (not np.isnan(newNeuriteBranch)):
            neuriteBranch = np.max([oldNeuriteBranch,newNeuriteBranch])
        elif (not np.isnan(oldNeuriteBranch)):
            neuriteBranch = oldNeuriteBranch
        elif (not np.isnan(newNeuriteBranch)):
            neuriteBranch = newNeuriteBranch
        else:
            neuriteBranch = 0
        return neuriteBranch

    def setRemainingBranchesOfNeurites(self,allOrigins,newNeuritesDf,neuriteBranch):
        #if neurites of different origins were determined, treat neurites for each origin seperately
        for thisOrigin in allOrigins:
            #set branch for each neurite where it was not set even if origin was found (for new branchs that emerged)
            #set starting neuriteBranch as first not used branch number      
            if len(self.allNeurites) != 0:
                neuriteBranch = self.getNextBranchNb(thisOrigin,self.newNeuritesDf,self.allNeurites,self.identity)
                
            else:
                neuriteBranch = 0
            if np.isnan(neuriteBranch):
                neuriteBranch = 0
            thisOriginNewNeurites = newNeuritesDf[newNeuritesDf.origin == thisOrigin]
            for newNeuritesRowNb in thisOriginNewNeurites.index.values:
                thisRow = thisOriginNewNeurites.loc[newNeuritesRowNb]
                if np.isnan(thisRow['branch']):
                    newNeuritesDf.loc[newNeuritesRowNb,'branch'] = neuriteBranch
                    newNeuritesDf.loc[newNeuritesRowNb,'start_branch'] = neuriteBranch
                    neuriteBranch += 1
        return newNeuritesDf
    
    @staticmethod
    def getBestNeuriteBasedOnSimilarityAndFrequency(allNeuritesToCompare,allNeurites,checkFrequency=True):
        maxPxSimilar = max(allNeuritesToCompare.loc[:,'pxSimilar'])
        #find all rows within tolerance of max nb of similar px
        #if frequency is checked, the most important criteria is the frequency, only if similar px deviate a lot (>20%) choose the not most frequently occuring neurite!
        if checkFrequency:
            max_deviation = 0.2
        else:
            max_deviation = 0.1
        mostSimilarNeurites = allNeuritesToCompare.loc[allNeuritesToCompare['pxSimilar'] >= (maxPxSimilar * (1 - max_deviation))]
        frequencyCheckFailed  = False
        if (len(allNeurites) > 0) & checkFrequency:
            #check which origin/branch combination is present in most timeframes
            maxFrequency = 0
            mostCommonNeurite = [np.nan]
            for rowNb in mostSimilarNeurites.index.values:
                oneNeurite = mostSimilarNeurites.loc[rowNb]
                origin = oneNeurite['origin']
                branch = oneNeurite['branch']
                frequency = len(allNeurites.loc[(allNeurites['origin'] == origin) & (allNeurites['branch'] == branch)])
                if frequency > maxFrequency:
                    maxFrequency = frequency
                    mostCommonNeurite = [origin,branch]
            if np.isnan(mostCommonNeurite[0]):
                frequencyCheckFailed = True
            else:
                bestBranch = mostSimilarNeurites.loc[(mostSimilarNeurites['origin'] == mostCommonNeurite[0]) & (mostSimilarNeurites['branch'] == mostCommonNeurite[1])].reset_index().iloc[0]
        if (not checkFrequency) | frequencyCheckFailed:
            #if more than one branch have highest nb of similar -- !+/- tolerance! (self.toleranceForSimilarityOfNeurites) 
            #then choose the branch with the smallest number of different px
            minPxDifference = min(mostSimilarNeurites.loc[:,'pxdifference'])
            mostSimilarNeurites = mostSimilarNeurites.loc[mostSimilarNeurites['pxdifference'] <= (minPxDifference + 5)]
            #if they are similar still, choose neurite with highest average intensity
            bestBranch = neuriteConstructor.getRowWithMaximumVal(mostSimilarNeurites,'avIntOfPoints')
        return bestBranch
        
    def remove_overlapping_branches(self,all_origins,newNeuritesDf):
        #remove branches that overlap in 70% of their px 
        if len(newNeuritesDf) != 0:
            for this_origin in all_origins:     
                thisOriginNewNeurites = newNeuritesDf[newNeuritesDf.origin == this_origin]
                #group all branches together that show 70% overlap of their px
                min_branch_overlap_factor = 0.7
                grouped_branches = self.groupBranchesByOverlap(thisOriginNewNeurites,min_branch_overlap_factor)

                for group in grouped_branches:
                    this_group_df = pd.DataFrame(columns = newNeuritesDf.columns)
                    for index in group:
                        new_rows = thisOriginNewNeurites.loc[index].to_frame().transpose()
                        this_group_df = pd.concat([this_group_df,new_rows])
                    checkFrequency = True
                    bestBranch = neuriteConstructor.getBestNeuriteBasedOnSimilarityAndFrequency(this_group_df,self.allNeurites,checkFrequency)
                    #each branch in same group than best branch should be removed from newneurites df
                    for index in group:
                        if index != bestBranch['index']:
                            newNeuritesDf = newNeuritesDf.drop(index,axis=0)
        return newNeuritesDf
        
    @classmethod
    def removeDoubledBranches(self,allOrigins,newNeuritesDf,allNeurites):
        #check if branches are doubled for same origin, only keep longest branch
        for thisOrigin in allOrigins:     
            if len(newNeuritesDf) != 0:
                thisOriginNewNeurites = newNeuritesDf[newNeuritesDf.origin == thisOrigin]
                usedBranches = []
                for newNeuritesRowNb in thisOriginNewNeurites.index.values:
                    thisRow = thisOriginNewNeurites.loc[newNeuritesRowNb]
                    thisBranch = thisRow['branch']
                    if thisBranch not in usedBranches:
                        usedBranches.append(thisBranch)
                    else:
                        #branch was used already, find out which is the branch with the highest number os similar px
                        thisBranchNewNeurites = newNeuritesDf[(newNeuritesDf.branch == thisBranch) & (newNeuritesDf.origin == thisOrigin)]
                        checkFrequency = False
                        bestBranch = neuriteConstructor.getBestNeuriteBasedOnSimilarityAndFrequency(thisBranchNewNeurites,allNeurites,checkFrequency)
                                
                        #remove bet branch from this branch neurites - "branch" wont be changed for that neurite
                        thisBranchNewNeurites = thisBranchNewNeurites.drop(bestBranch['index'],axis=0)
                        
#                        plt.figure()
#                        plt.imshow(np.zeros((512,512)))
#                        plt.text(40,40,str(bestBranch[['time','origin','branch','pxdifference','pxSimilar']]))
#                        plt.figure()
#                        plt.imshow(np.zeros((512,512)))
#                        plt.text(40,40,str(mostSimilarNeurites))
#                        plt.figure()
#                        plt.imshow(np.zeros((512,512)))
#                        plt.text(40,40,str(thisBranchNewNeurites))
                        identity = []
                        identity.append(newNeuritesDf['date'].iloc[0])
                        identity.append(newNeuritesDf['experiment'].iloc[0])
                        identity.append(newNeuritesDf['neuron'].iloc[0])
                        identity.append(newNeuritesDf['channel'].iloc[0])
                        #delete all rows which do not belong to the longest neurite
                        for newNeuritesBranchRowNb in thisBranchNewNeurites.index.values:
                            nextNeuriteBranch = self.getNextBranchNb(thisOrigin,newNeuritesDf,allNeurites,identity)
#                            plt.figure()
#                            plt.imshow(np.zeros((512,512)))
#                            plt.text(40,40,str("something was changed!"))
#                            plt.text(80,80,str(nextNeuriteBranch))
                            newNeuritesDf.loc[newNeuritesBranchRowNb,'branch'] = nextNeuriteBranch
                            newNeuritesDf.at[newNeuritesBranchRowNb,'loss_x'] = []
                            newNeuritesDf.at[newNeuritesBranchRowNb,'loss_y'] = []
                            newNeuritesDf.at[newNeuritesBranchRowNb,'gain_x'] = []
                            newNeuritesDf.at[newNeuritesBranchRowNb,'gain_y'] = []
        return newNeuritesDf
    
    
    def checkForOverlapBetweenDifferentOrigins(self,overlappingOrigins,allNeurites,newNeuritesDf):
        #now check if there is too much overlap between neurites of different origins
        for overlappingOriginsRowNb in overlappingOrigins.index.values:
            overlappingOrigin = overlappingOrigins.loc[overlappingOriginsRowNb]
            #check whether indices where already excluded due to duplicate-branches
            if overlappingOrigin['newNeuriteNb']:
                comparedNeurite = newNeuritesDf.loc[overlappingOrigin['newNeuriteNb']]
                if overlappingOrigin['origin'] != comparedNeurite['origin']:
                    allNeurites.loc[((((allNeurites.date == overlappingOrigin['date']) & (allNeurites.neuron == overlappingOrigin['neuron'])) & ((allNeurites.time == overlappingOrigin['time']) & (allNeurites.channel == overlappingOrigin['channel']))) & ((allNeurites.origin == overlappingOrigin['origin']) & (allNeurites.branch == overlappingOrigin['branch']))),'overlap'] = 1
                    newNeuritesDf.loc[overlappingOrigin['newNeuriteNb'],'overlap'] = 1 
        return allNeurites,newNeuritesDf      
    

    def isThereOverlap(self,x1,y1,x2,y2):
        minX1 = np.min(x1)-self.dilationForOverlap
        maxX1 = np.max(x1)+self.dilationForOverlap
        minY1 = np.min(y1)-self.dilationForOverlap
        maxY1 = np.max(y1)+self.dilationForOverlap
        minX2 = np.min(x2)-self.dilationForOverlap
        maxX2 = np.max(x2)+self.dilationForOverlap
        minY2 = np.min(y2)-self.dilationForOverlap
        maxY2 = np.max(y2)+self.dilationForOverlap
        isOverlap = False
        xOverlap = ((((minX1 >= minX2) & (minX1 <= maxX2)) | ((maxX1 >= minX2) & (maxX1 <= maxX2))) | ((minX1 <= minX2) & (maxX1 >= maxX2)))
        yOverlap = ((((minY1 >= minY2) & (minY1 <= maxY2)) | ((maxY1 >= minY2) & (maxY1 <= maxY2))) | ((minY1 <= minY2) & (maxY1 >= maxY2)))
        if xOverlap & yOverlap:
            isOverlap = True
        return isOverlap
    
    
    def getPossibleOriginFromList(self,currentNeuriteCoords,targetNeuritesDf,possibleOrigins,overlappingOrigins,currentNeuriteNb,dilationForOverlap,overlapMultiplicator,frameDifference=0,threshold=-1,overlapThreshold=-1,minNbSimilarPx=-1):
#        print("get possible origin from list")
        if threshold == -1:
            threshold=self.maxChangeOfNeurite
        if overlapThreshold == -1:
            overlapThreshold=self.maxOverlapOfNeurites
        if minNbSimilarPx == -1:
            minNbSimilarPx = self.minBranchSize
        maxDilationForOverlap = self.dilationForOverlap
        startDilationForOverlap = int(np.round(dilationForOverlap-1,0))
        multiplicator = 1
        oneOriginFound = False
        currentNeuriteImage = np.zeros_like(self.labeledGroupImage)
        currentNeuriteImage[currentNeuriteCoords[0],currentNeuriteCoords[1]] = 1
        currentNeuriteCoords = np.transpose(generalTools.convert_points_to_point_list(np.transpose(currentNeuriteCoords)))
        startPointOfCurrentNeurite = [currentNeuriteCoords[0][0],currentNeuriteCoords[1][0]]
        endPointOfCurrentNeurite = [currentNeuriteCoords[0][-1],currentNeuriteCoords[1][-1]]
        endPointImageOfCurrentNeurite = np.zeros_like(currentNeuriteImage)
        endPointImageOfCurrentNeurite[endPointOfCurrentNeurite[0],endPointOfCurrentNeurite[1]] = 1
        
        #crop image for speed
        endPointImageOfCurrentNeurite,borderVals1 = generalTools.cropImage(endPointImageOfCurrentNeurite,[],20)
        endPointImageOfCurrentNeurite = morph.binary_dilation(endPointImageOfCurrentNeurite,disk(6))
        #uncrop
        endPointImageOfCurrentNeurite = generalTools.uncropImage(endPointImageOfCurrentNeurite,borderVals1)
        
        lengthOfCurrentNeurite = len(currentNeuriteCoords[0])
        
        for dilationForOverlap in range(startDilationForOverlap,maxDilationForOverlap+1):
            if len(targetNeuritesDf) > 0:
                currentNeuriteImgCopy = copy.copy(currentNeuriteImage)

                #crop for increased speed and after one operation uncrop
                currentNeuriteImgCopy,borderVals2 = generalTools.cropImage(currentNeuriteImgCopy,[],20)
                currentNeuriteImgCopy = ndimage.binary_dilation(currentNeuriteImgCopy,disk(dilationForOverlap/multiplicator),iterations=multiplicator)
                currentNeuriteImgCopy = generalTools.uncropImage(currentNeuriteImgCopy,borderVals2)
                
            for neuriteNb in targetNeuritesDf.index.values:
                neurite = targetNeuritesDf.loc[neuriteNb]
                neuritesOverlap = self.isThereOverlap(neurite['x'],neurite['y'],currentNeuriteCoords[0],currentNeuriteCoords[1])
                if neuritesOverlap:
                    thisNeuriteImage = np.zeros_like(currentNeuriteImgCopy)
                    thisNeuriteImage[[neurite['x'],neurite['y']]] = 1
                    
                    
#                    plt.figure()
#                    plt.imshow(thisNeuriteImage)
                    #add some of the px that were gained or lost in image to compare to 
                    #gained or lost relative to the image which was the best compared image for the image currently compared to
                    thisNeuriteImage = self.remove_gained_and_add_lost_px(thisNeuriteImage,neurite)
#                    plt.figure()
#                    plt.imshow(thisNeuriteImage)       
                    
                    
                    #crop for speed, uncroping only necessary for currentNeuriteImgCopy & endPointImageOfCurrentNeurite since it is used in next iteration again
                    thisNeuriteImage,currentNeuriteImgCopy,borderVals3 = generalTools.crop2Images(thisNeuriteImage,currentNeuriteImgCopy,20)
                    endPointImageOfCurrentNeurite,borderVals3 = generalTools.cropImage(endPointImageOfCurrentNeurite,borderVals3)
                    
                    thisNeuriteImage = ndimage.binary_dilation(thisNeuriteImage,disk(dilationForOverlap/multiplicator),iterations=multiplicator)
                    
                    two_way_diff = 1
                    diffOfImgs,overlapOfImgs = generalTools.overlapImgs(thisNeuriteImage,currentNeuriteImgCopy,dilationForOverlap,overlapMultiplicator,0,two_way_diff)
                    gain_of_imgs = diffOfImgs[1]
                    loss_of_imgs = diffOfImgs[0]
                    
                    
                    #draw line between startpoints of current neurite and neurite to compare to, dilate line afterwards
                    startPointOfThisNeurite = [neurite['x'][0]-borderVals3[0],neurite['y'][0]-borderVals3[2]]
                    lineConnectingStartPoints = line(startPointOfCurrentNeurite[0]-borderVals3[0],startPointOfCurrentNeurite[1]-borderVals3[2],startPointOfThisNeurite[0],startPointOfThisNeurite[1])
                    lineImage = np.zeros_like(thisNeuriteImage)
                    
                    lineImage[lineConnectingStartPoints[0],lineConnectingStartPoints[1]] = 1
                    lengthOfLine = len(lineConnectingStartPoints[0])
                    lineImage_dil = ndimage.binary_dilation(lineImage,disk(3))
                    
                    
                    #if lineImage coincides with difference of image very well, add it to overlap and remove it from difference
                    diffOfImgs = (diffOfImgs[0] == 1) | (diffOfImgs[1] == 1)
                    diffData = np.where(diffOfImgs == 1)
                    origDiffLength = len(diffData[0])
                    diffOfImgs_noLine = copy.copy(diffOfImgs)
                    diffOfImgs_noLine[lineImage_dil == 1] = 0
    #                plt.figure()
    #                plt.imshow(diffOfImgs_noLine)
                    noLineDiffLength = len(np.where(diffOfImgs_noLine == 1)[0])
                    if (origDiffLength - noLineDiffLength) > lengthOfLine:
                        overlapOfImgs[lineImage == 1] = 1
                        diffOfImgs[lineImage_dil == 1] = 0
                    
                    #label all differences in image, delete differences that can be associated with end point of either neurite
                    diffOfImgs_labeled, nbLabels = ndimage.label(diffOfImgs,structure=[[1,1,1],[1,1,1],[1,1,1]])
                    
                    endPointOfThisNeurite = [neurite['x'][-1]-borderVals3[0],neurite['y'][-1]-borderVals3[2]]
                    endPointImageOfThisNeurite = np.zeros_like(thisNeuriteImage)
                    endPointImageOfThisNeurite[endPointOfThisNeurite[0],endPointOfThisNeurite[1]] = 1
                    
                    #crop for speed and after one operation uncrop
                    endPointImageOfThisNeurite,borderVals4 = generalTools.cropImage(endPointImageOfThisNeurite,[],20)
                    endPointImageOfThisNeurite = morph.binary_dilation(endPointImageOfThisNeurite,disk(6))
                    endPointImageOfThisNeurite = generalTools.uncropImage(endPointImageOfThisNeurite,borderVals4)
                    
                    endPointLabelThisNeurite = np.unique(diffOfImgs_labeled[endPointImageOfThisNeurite == 1])
                    endPointLabelThisNeurite = endPointLabelThisNeurite[endPointLabelThisNeurite > 0]
                    if len(endPointLabelThisNeurite) > 0:
                        endPointLabelThisNeurite = endPointLabelThisNeurite[0]
                    
                    endPointLabelCurrentNeurite = np.unique(diffOfImgs_labeled[endPointImageOfCurrentNeurite == 1])
                    endPointLabelCurrentNeurite = endPointLabelCurrentNeurite[endPointLabelCurrentNeurite > 0]
                    if len(endPointLabelCurrentNeurite) > 0:
                        endPointLabelCurrentNeurite = endPointLabelCurrentNeurite[0]
                    
                    
                    diffOfImgs[diffOfImgs_labeled == endPointLabelCurrentNeurite] = 0
                    diffOfImgs[diffOfImgs_labeled == endPointLabelThisNeurite] = 0
                    
                    gain_of_imgs[(diffOfImgs_labeled != endPointLabelCurrentNeurite) & (diffOfImgs_labeled != endPointLabelThisNeurite)] = 0
                    loss_of_imgs[(diffOfImgs_labeled != endPointLabelCurrentNeurite) & (diffOfImgs_labeled != endPointLabelThisNeurite)] = 0
                    
#                    plt.figure()
#                    plt.imshow(overlapOfImgs)
#                    plt.figure()
#                    plt.imshow(diffOfImgs)
                    
                    lengthOfThisNeurite = len(neurite['x'])
                    
                    overlapData = np.where(overlapOfImgs == 1)
                    
        #            print("difference is {}; max is is {}; nb of islands: {}".format(len(diffData[0])+(dilationForOverlap*2*(nbOfIslands-1)),pxDifference,nbOfIslands))
                    overlapRatios = []
                    #calculate ratio and px differences of length of overlap to both neurites that were overlapped
                    overlapRatios.append(len(overlapData[0])/lengthOfThisNeurite)
                    overlapRatios.append(len(overlapData[0])/lengthOfCurrentNeurite)
                    #choose smallest ratio (worst fit)
                    overlapRatio = np.min(overlapRatios)
                    pxDifferences = []
                    pxDifferences.append(abs(len(overlapData[0]) - lengthOfThisNeurite))
                    pxDifferences.append(abs(len(overlapData[0]) - lengthOfCurrentNeurite))
    
                    #calcuate how many independent islands there are in differences
                    labeledDiffOfImgs,nbOfIslands = ndimage.label(diffOfImgs,structure=[[1,1,1],[1,1,1],[1,1,1]])
                    #increase number of difference points due to erosion, for each island 2x erosion more (subtracting one for background) 
                    #dilation for erosion is dilationForOverlap
                    pxDifferences.append(len(diffData[0])+(dilationForOverlap*2*(nbOfIslands-1)))
                    #choose biggest difference (worst fit)
                    pxDifference = np.max(pxDifferences)
                    pxOverlap = len(overlapData[0])
                    
    #                plt.figure()
    #                plt.imshow(diffOfImgs)
#                    plt.figure()
#                    plt.imshow(np.zeros((512,512)))
#                    plt.text(40,40,pxOverlap)
    
                    #uncrop image to be ready for next loop iteration
                    currentNeuriteImgCopy = generalTools.uncropImage(currentNeuriteImgCopy,borderVals3)
                    endPointImageOfCurrentNeurite = generalTools.uncropImage(endPointImageOfCurrentNeurite,borderVals3)
                    
                    #uncrop images for saving coord data
                    gain_of_imgs = generalTools.uncropImage(gain_of_imgs,borderVals3)
                    loss_of_imgs = generalTools.uncropImage(loss_of_imgs,borderVals3)
    
    
                    if (pxOverlap > overlapThreshold) & (frameDifference < 4):
                        overlappingOriginsNb = len(overlappingOrigins)
                        overlappingOrigins.loc[overlappingOriginsNb] = [currentNeuriteNb,neurite['date'],neurite['experiment'],neurite['neuron'],neurite['channel'],neurite['time'],neurite['origin'],neurite['branch'],neurite['x'],neurite['y'],pxOverlap]


                    #check if similarities are at least around the size of minimum required branch size
                    if (pxOverlap > (minNbSimilarPx*0.8)):
                        
                        #sort gain and loss coords before adding to Dataframe
                        ref_coords = [neurite['x'],neurite['y']]
                        if len(np.unique(gain_of_imgs)) > 1:
                            sorted_gain_coords = self.get_sorted_coords_from_img(gain_of_imgs,ref_coords,self.minBranchSize)
                        else:
                            sorted_gain_coords = [[],[]]
                        if len(np.unique(loss_of_imgs)) > 1:
                            sorted_loss_coords = self.get_sorted_coords_from_img(loss_of_imgs,ref_coords,self.minBranchSize)                        
                        else:
                            sorted_loss_coords = [[],[]]
                        
                        possibleOriginsNb = len(possibleOrigins)
                        possibleOrigins.loc[possibleOriginsNb] = [neurite['origin'],neurite['branch'],neurite['start_branch'],neurite['x'],neurite['y'],sorted_gain_coords[0],sorted_gain_coords[1],sorted_loss_coords[0],sorted_loss_coords[1],pxDifference,overlapRatio,pxOverlap,neurite['avIntOfPoints']]
                        oneOriginFound = True
            if oneOriginFound:
                break
                        
        return possibleOrigins,overlappingOrigins
    
    
    def remove_gained_and_add_lost_px(self,this_neurite_image,neurite):
        #add some of the px that were gained or lost in image to compare to 
        #gained or lost relative to the image which was the best compared image for the image currently compared to
        nb_px_to_remove = int(np.round(min(len(neurite['x'])*0.1,self.minBranchSize*0.5),0))
        nb_lost_px = len(neurite['loss_x'])
        nb_gained_px = len(neurite['gain_x'])
        if nb_lost_px > 0:
            if nb_lost_px > nb_px_to_remove:
                x_to_add = neurite['loss_x'][0:nb_lost_px-nb_px_to_remove]
                y_to_add = neurite['loss_y'][0:nb_lost_px-nb_px_to_remove]
                this_neurite_image[x_to_add,y_to_add] = 1
        if nb_gained_px > 0:
            if nb_gained_px > nb_px_to_remove:
                x_to_remove = neurite['gain_x'][nb_gained_px-nb_px_to_remove:-1]
                y_to_remove = neurite['gain_y'][nb_gained_px-nb_px_to_remove:-1]
                this_neurite_image[x_to_remove,y_to_remove] = 0
        return this_neurite_image
    
    
    def get_sorted_coords_from_img(self,image,ref_coords,minBranchSize):
        coords = np.where(image == 1)
        closest_point = generalTools.getClosestPoint(coords,[[ref_coords[0][0],ref_coords[1][0]]])
        sorted_points, length, new_coords = sortPoints.startSorting(coords, [closest_point],0,minBranchSize)
        sorted_points = generalTools.convert_points_to_point_list(sorted_points)
        sorted_coords = np.transpose(sorted_points)
        return sorted_coords
        
    @staticmethod
    def getRowWithMaximumVal(allRows,column):
        maxVal = max(allRows.loc[:,column])
        bestRow = allRows[allRows[column] == maxVal].reset_index().iloc[0]
        return bestRow
    
    def getRowWithMinimalVal(self,allRows,column):
        minVal = min(allRows.loc[:,column])
        bestRow = allRows[allRows[column] == minVal].reset_index().iloc[0]
        return bestRow