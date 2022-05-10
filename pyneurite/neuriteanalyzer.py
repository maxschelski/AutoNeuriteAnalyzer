# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:17:25 2017

@author: Max
"""

from .tools.neuriteconstructor import neuriteConstructor
from .tools.thresholdneuron import ThresholdNeuron
from .tools.analyzers import Analyzers
from .tools.dataframecleanup import DataframeCleanup
from .tools.sortpoints import sortPoints
from .tools.somatools import SomaTools
from .tools.connectneurites import connectNeurites
from .tools.generaltools import generalTools
from .tools.separateneurites import SeparateNeurites


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
from scipy.signal import medfilt2d as median2
import glob

from matplotlib import pyplot as plt
from skimage.filters import median
from skimage.morphology import disk
from skimage.morphology import square
import warnings
import time
import cProfile
from io import StringIO
import pstats
from skimage.filters import scharr
from skimage.filters import threshold_otsu as otsu
import shutil


from scipy.ndimage import filters


class NeuriteAnalyzer():

    def __init__(self,path="",skelChannel=None,image=None, image_thresh=None,
                 analysisChannel=None,folders="",conditions=None,
                 cpuCores=1,multiThreading=False, 
                 continueAnalysis=False,create_skeleton=True,
                 always_clean_dataframe=False,always_re_clean_dataframes=False,
                 always_save_skeletons=False,
                 always_save_completed_skeletons=False,
                 continue_with_incomplete_analysis=False,
                 analyze_intensities=True,first_timepoint_to_analyze=1):

        #how many um per pixel
        self.UMperPX = 0.22

        #---------------------GENERAL-------------------------
        #necessary for setting large strings of arrays in dataframe!
        self.folders = folders
        self.timePoint = 1
        self.timeframe = image
        self.timeframe_thresholded = image_thresh
        self.conditions = conditions
        self.continueAnalysis = continueAnalysis
        self.create_skeleton = create_skeleton
        self.always_clean_dataframe = always_clean_dataframe
        self.always_save_skeletons = always_save_skeletons
        self.always_re_clean_dataframes = always_re_clean_dataframes
        self.analyze_intensities = analyze_intensities
        self.always_save_completed_skeletons = always_save_completed_skeletons
        self.continue_with_incomplete_analysis = continue_with_incomplete_analysis
        self.first_timepoint_to_analyze = first_timepoint_to_analyze
        np.set_printoptions(threshold=sys.maxsize)


        self.label_structure = [[1,1,1],[1,1,1],[1,1,1]]

        sys.setrecursionlimit(10**5)
        warnings.filterwarnings("once")
        self.multiThreading = multiThreading
        self.nbCPUcores = cpuCores

        #channel used for neurite skeletonization & analysis
        self.channelUsedForSkel = skelChannel
        self.channelsToAnalyze = analysisChannel


        #to analyze growth of neurites
        self.minChangeInNeuriteLength = 20


        #median filter size applied to image before any downstream procedure
        self.imgMedianFilterSize = 1

        #min size of px (non diagonaly connected) to not be removed
        self.grainSizeToRemove = 2

        #px were optimized for 0.22 um/px,conversion factor to adjust all px values
        self.convFactor = 0.22 / self.UMperPX

        #minimum size of branch to be not deleted
        self.minBranchSize = 40

        #px dilation used to check for overlap of neurites
        self.dilationForOverlap = 6
        self.overlapMultiplicator = 2

        #max square diameter in px for separating neurites by opening
        self.maxNeuriteOpening = 10

        #among others minimum size for endpoint branches to be not deleted
        self.branchLengthToKeep = 20

        #contrast of px to remove compared to neighboring pixel, px with a contrast equal or higher than this will be removed to separate neurites
        self.contrastThresholdToRemoveLabel = 3

        #separate neurites by opening: maximum accepted fraction of size lost by opening
        self.maxFractionToLoseByOpening = 0.8

        #max disk size of soma dilation for separating neurites
        self.maxSomaDilationForSeparation = 10

        #parameters for smoothing neurite image
        self.gaussianForSmoothing = 2
        self.dilationForSmoothing = 4

        #minimum size of branchpoints to calculate sumintensities image
        self.min_branchpoint_size = 10

        #for evaluating curvature of different branches (searching for branch with lowest curvature)
        self.distToLookForCurvature = 10
        self.pxToJumpOverForCurvature = 5

        #max. number of px deviation from longest starting branch
        self.px_tolerance_for_sorting_branches_by_starting_segments = 5

        #nb of px at tip of neurite to check to look for overlap with other neurites
        self.nb_last_px_to_check = 20

        self.max_nb_of_branchpoints_before_overload = 100

        # radius of disk for opening image before skeletonization
        # to remove artificial branches in skeleton which are just due to
        # small bumps in the thresholded image
        self.open_for_skeletonization = 2


        #---------------------NEURITE CONSTRUCTOR-------------------------
        #set max growth & retraction of a neurite (in px)
        #1um = 4.5px, max change of 20um per neurite is safe => 90px
        self.maxChangeOfNeurite = 90
        #max amount of pixels different neurites are allowed to overlap, 5um => 22.5px, round down => 20px
        self.maxOverlapOfNeurites = 20
        #max nb of similar px (to neurite compared to) in which different branches can differ without choosing one of them over the other based on the number of similar px
        self.toleranceForSimilarity = 10


        #---------------------THRESHOLD NEURON-------------------------
        #initial processing of grey value image by opening
        self.openingForThresholdEdge = 5
        #median filter applied to grey value image after opening
        self.medianForThresholdEdge = 10

        #maximum tolerated number of pixels for soma overflow during thresholding (of soma found with edges and soma found with current threshold)
        self.maxToleratedSomaOverflowRatio = 0.6
        self.minSomaOverflow = 50

        #minimum value to be considered for thresholding of edge image - step size of histogram for initial thresholding function
        self.minEdgeVal = 0.5

        #nb of bins in X and Y for subdividing image for initial threshold val
        self.nb1Dbins = 3

        #filter out background points
        #distance in which to look for other thresholded points
        self.distanceToCover = 20

        #factor of how many more number of thresholded points necessary in distance relative to distance (2 means 2x distance necessary)
        self.filterBackgroundPointsFactor = 2

        #threshold parameter for exclusion of found initial threshold value for imaging - minimum number of pixel for not connected pieces
        self.minNbOfPxPerLabel = 20

        #max number of pixels in difference of thresholded image and initial image allowed during iterative thresholding
        self.maxThresholdChange = 5

        #how much the percentile of px is multiplied by for each iteration of thresholding
        self.percentileChangeForThreshold = 0.8

        #disk size for closing thresholded images to obtain presoma for thresholding
        self.closingForPresoma = 6

        #maximum contrast allowed to connect islands in image
        #contrast = mean signal of thresholded image divided by mean signal of rest of image
        self.maxContrastForConneting = 5


        #------------------------ SORT POINTS -----------------------------

        #minimum length of filopodia
        self.minFilopodiaLength = 5
        self.maxFilopodiaLength = 40
        #min inverse contrast of filopodia
        self.minContrastFilopodia = 2
        #minimal contrast allowed during iterations of finding filopodia
        self.minAllowedContrastForFindingFilopodia = 0.8

        #---------------------CONNECT NEURITES-------------------------
        #maximum allowed fold difference of length for connecting islands when comparing distance to closest point and intensity based found distance
        self.maxRelDifferenceOfLengths = 1.6+0.1

        #maximum relative local difference of last point of connection and point they connect to
        self.maxLocalConnectionContrast = 10

        #for finding connection path
        #in which distance from currentpoint values should be compared to find next point
        self.distanceToCheck = 3

        #minimum contrast of newly drawn connection compared to background
        self.minContrast = 1.05



        #---------------------GETTING BRANCH AND END POINTS-------------------------
        #radius from endpoints in which labels will be considered
        self.endpointlabelRadius = 5
        #radius from branchpoint which will be cut to subsequently label
        self.branchPointRadius = 6
        #radius from branchpoint in which labels will be considered
        self.branchPointRefRadius = 4



        #---------------------GET SOMA-------------------------
        #px of increase in soma size
        self.somaExtractionFilterSize = 20

        #reduction of dilation of soma
        self.reductionOfSomaDilation = 15

        #SOMA extraction - what size of objects will be excluded when extracting soma
        self.objectSizeToRemoveForSomaExtraction = 1000



        #---------------------DATA FRAME CLEAN UP-------------------------
        #minimum fraction of all timepoints in which neurite is present (gaps are allowed) for neurite to be considered continuous
        self.min_nb_of_timeframes_factor = 0.3
        #maximum number of consecutive timeframes allowed without neurite
        self.maxGapSize = 2

        #the px radius in which the fraction of start points is quantified for choosing start point with a minimum nb of other start points closeby
        self.radius_of_start_points = 10
        #minimum fraction of all start points which need to be in radius above of currently looked at start point to be choosen
        self.min_fraction_of_start_points_closeby = 0.2

        #---------------------ANALYZERS-------------------------
        #minimum length of whole neurite before not considered
        self.minNeuriteLength = 40

        self.AW_maxIntFac = 2

        #---------------------INITIATE PARAMETERS-------------------------
        #x and y coordinates of mid point of soma
        self.cX = None
        self.cY = None
        self.last_total_nb_branchpoints = np.nan
        self.optimalThreshold = np.nan
        self.deletedSortedPoints = []
        self.upperPath = path
        self.overlappingOriginsColumns = ('newNeuriteNb','date','experiment','neuron','channel','time','origin','branch','x','y','overlap')
        self.possibleOriginsColumns = ('origin','branch','start_branch','x','y','gain_x','gain_y','loss_x','loss_y','pxdifference','diffRatio','pxSimilar','avIntOfPoints')
        self.allNeuritesColumns = ('date','experiment','neuron','channel','time','origin','branch','start_branch','start','end','pxdifference','diffRatio','pxSimilar','neuronMid','x','y','gain_x','gain_y','loss_x','loss_y','length','overlap','threshold','avIntOfPoints')
        self.allNeurites = pd.DataFrame()
        self.backgroundVal = np.nan

    def convertAllConstants(self):

        self.open_for_skeletonization = self.convert(self.open_for_skeletonization)

        self.radius_of_start_points = self.convert(self.radius_of_start_points)
        self.distToLookForCurvature = self.convert(self.distToLookForCurvature)
        self.pxToJumpOverForCurvature = self.convert(self.pxToJumpOverForCurvature)
        self.px_tolerance_for_sorting_branches_by_starting_segments = self.convert(self.px_tolerance_for_sorting_branches_by_starting_segments)

        self.maxThresholdChange = self.convert(self.maxThresholdChange)
        self.dilationForOverlap = self.convert(self.dilationForOverlap)
        self.maxThresholdChange = self.convert(self.maxThresholdChange)
        self.minSomaOverflow = self.convert(self.minSomaOverflow)

        self.openingForThresholdEdge = self.convert(self.openingForThresholdEdge)
        self.medianForThresholdEdge = self.convert(self.medianForThresholdEdge)

        self.grainSizeToRemove = self.convert(self.grainSizeToRemove)

        self.minChangeInNeuriteLength = self.convert(self.minChangeInNeuriteLength)

        self.closingForPresoma = self.convert(self.closingForPresoma)

        self.maxNeuriteOpening = self.convert(self.maxNeuriteOpening)

        self.endpointlabelRadius = self.convert(self.endpointlabelRadius)
        self.branchPointRadius = self.convert(self.branchPointRadius)
        self.branchPointRefRadius = self.convert(self.branchPointRefRadius)

        self.distanceToCover = self.convert(self.distanceToCover)
        self.minBranchSize = self.convert(self.minBranchSize)
        self.somaExtractionFilterSize = self.convert(self.somaExtractionFilterSize)
        self.reductionOfSomaDilation = self.convert(self.reductionOfSomaDilation)
        self.maxChangeOfNeurite = self.convert(self.maxChangeOfNeurite)
        self.maxOverlapOfNeurites = self.convert(self.maxOverlapOfNeurites)
        self.minNeuriteLength = self.convert(self.minNeuriteLength)
        self.minNbOfPxPerLabel = self.convert(self.minNbOfPxPerLabel)
        self.objectSizeToRemoveForSomaExtraction = self.convert(self.objectSizeToRemoveForSomaExtraction)


    def convert(self,nb):
        #change number according to conversion factor , round to whole px
        nb = int(round(nb*self.convFactor,0))
        return nb


    def goDeeper(self,props,this_path,a,max_a,conditions,folders):
        a += 1
        for newFolder in os.listdir(this_path):
             if conditions[a-1] != "":
                 if newFolder.replace(conditions[a-1],"") != newFolder:
                     useFolder = True
                 else:
                     useFolder = False
             else:
                 useFolder = True
             if useFolder:
                 
                 path_new = this_path + "\\" + newFolder
                 
                 props_new = copy.copy(props)
                 props_new[folders[a-1]] = newFolder
                 if a == (max_a):
                     if os.path.isdir(path_new):
                         path_new = os.path.abspath(path_new)
                         self.processFolder(props_new, path_new)
                 else:
                     if os.path.isdir(path_new):
                         self.goDeeper(props_new,path_new,a,max_a,conditions,folders)

    def traverseFolders(self,folders,basePath,conditions):
        props = {}
        self.nb = 0
        self.goDeeper(props,basePath,0,len(folders),conditions,folders)


    def startAnalysis(self,thread):
        self.thread = thread
        self.convertAllConstants()
        folders  = self.folders
        conditions= self.conditions
        self.traverseFolders(folders,self.upperPath,conditions)

    def processFolder(self,props,path):

        if self.nb < self.nbCPUcores:
            self.nb += 1
        else:
            self.nb = 1

        if self.multiThreading:
            self.doAnalysis = self.nb == self.thread
        else:
            self.doAnalysis = True

        if self.doAnalysis:
            warnings.filterwarnings("ignore")

            self.neuron = props['neuron']
            self.date = props['date']
            self.experiment = props['experiment']
            self.iteration = ""
    #        doAnalysis = neuron == "cell0031"
            print("{} - {} - {} - {} - {}".format(self.thread,self.date,self.experiment,self.iteration,self.neuron))

            self.allNeurites = pd.DataFrame(columns=self.allNeuritesColumns)
            self.baseFileName = os.path.join(path,"\\"+self.date+"_"+self.experiment+"_allNeurites_skeletons_"+str(self.neuron))
            all_csv_files = glob.glob(os.path.join(path, "*.csv"))
            #get all relevant csv files - all that end in the neuron_folder name + .csv
            relevant_csv_files = [csv_file for csv_file in all_csv_files if csv_file.find(str(self.neuron)+".csv") != -1]

            #sort relevant csv files by timestamp and take the newest file
            relevant_csv_files = sorted(relevant_csv_files, key=lambda x:os.path.getmtime(x), reverse=True)

            if self.continueAnalysis:
                if len(relevant_csv_files) > 0:
                    self.allNeurites = pd.read_csv(relevant_csv_files[0])
                    self.baseFileName = relevant_csv_files[0].replace(".csv","")
    #                self.allNeurites.y = self.allNeurites.y.astype(list)
            self.allNeurites = self.createSkeleton(path)
            if os.path.exists(self.baseFileName+".csv"):
                self.allNeurites = pd.read_csv(self.baseFileName+".csv")
            if (self.doAnalysis) & (len(self.allNeurites) > 0):
                self.allNeurites = self.convert_columns_from_string(self.allNeurites)
                clean_dataframe = True
                save_skeletons = True
                if os.path.exists(self.baseFileName+"_corr.csv"):
                    all_neurites_corr = pd.read_csv(self.baseFileName+"_corr.csv")
                    if all_neurites_corr['time'].max() == self.allNeurites['time'].max():
                        if (not self.always_re_clean_dataframes) & (not self.always_clean_dataframe):
                            clean_dataframe = False
                        if (not self.always_save_completed_skeletons) & (not self.always_save_skeletons):
                            save_skeletons = False
                    else:
                        if (not self.always_re_clean_dataframes) & (not self.always_clean_dataframe):
                            clean_dataframe = False
                        if (not self.always_save_completed_skeletons) & (not self.always_save_skeletons):
                            save_skeletons = False
                if (not self.all_timepoints_analyzed) & (not self.continue_with_incomplete_analysis):
                    clean_dataframe = False
                    save_skeletons = False
                if clean_dataframe:
                    self.allNeurites = DataframeCleanup.remove_double_branches_from_all_neurites(self.allNeurites)
                    self.allNeurites = DataframeCleanup.removeNonContinuousOrigins(self.allNeurites,
                                                                                   self.maxGapSize,
                                                                                   self.min_nb_of_timeframes_factor)
#                    self.allNeurites = DataframeCleanup.adjustStartPoints(self.allNeurites,self.radius_of_start_points ,self.min_fraction_of_start_points_closeby)
                    self.allNeurites.to_csv(self.baseFileName+"_corr.csv")
                continue_analysis = False
                if os.path.exists(self.baseFileName+"_corr.csv"):
                    self.allNeurites = pd.read_csv(self.baseFileName+"_corr.csv")
                    if len(self.allNeurites) > 0:
                        continue_analysis = True
                if continue_analysis:
                    if save_skeletons:
                        self.saveSkeletons(self.allNeurites,path,
                                           self.date,
                                           self.experiment,
                                           self.neuron,
                                           self.image_shape)
                    self.allNeurites = pd.read_csv(self.baseFileName+"_corr.csv")
                    if self.analyze_intensities:
                        self.allNeurites.date = self.allNeurites.date.astype(str)
                        if not os.path.exists(self.baseFileName+"_corr_int.csv"):
                            self.allNeurites = Analyzers.analyzeIntensities(self.date,
                                                                            self.experiment,
                                                                            self.neuron,path,
                                                                            self.allNeurites,
                                                                            self.channelUsedForSkel,
                                                                            self.channelsToAnalyze,
                                                                            self.allNeuritesColumns,
                                                                            self.minNeuriteLength
                                                                            ,self.AW_maxIntFac)
                            self.allNeurites.to_csv(self.baseFileName+"_corr_int.csv")
                        else:
                            print("Thread {} - {} - {}: Intensities were analyzed already.".format(self.thread, 
                                                                                                   self.date, 
                                                                                                   self.neuron))
                        #commented out since not working properly yet.
#                        self.allNeurites = pd.read_csv(self.baseFileName+"_corr_int.csv")
#                        self.allNeurites = Analyzers.analyzeGrowth(self.date,self.experiment,self.neuron,path,self.allNeurites,self.allNeuritesColumns,self.minChangeInNeuriteLength)
#                        self.allNeurites.to_csv(self.baseFileName+"_corr_int.csv")
#                    self.allNeurites = pd.read_csv(self.baseFileName+"_corr_int.csv")
#                    self.allNeurites.date = self.allNeurites.date.astype(str)
#                    self.allNeurites = Analyzer.analyzeActinWaves(self.allNeurites)
#                    self.allNeurites.to_csv(self.baseFileName+"_corr_int_AW.csv")
#                self.allNeurites = self.allNeurites.reset_index().set_index(['date','experiment','neuron','channel','origin','branch','time']).sort_index()

    def convert_columns_from_string(self,all_neurites):
        all_neurites.date = all_neurites.date.astype(str)
        all_neurites['x'] = all_neurites['x'].apply(Analyzers.stringToNbArray)
        all_neurites['y']= all_neurites['y'].apply(Analyzers.stringToNbArray)
        all_neurites['start'] = all_neurites['start'].apply(Analyzers.stringToNbArray)
        all_neurites['end'] = all_neurites['end'].apply(Analyzers.stringToNbArray)
        if "gain_x" in all_neurites.columns:
            all_neurites['gain_x'] = all_neurites['gain_x'].apply(Analyzers.stringToNbArray)
            all_neurites['gain_y'] = all_neurites['gain_y'].apply(Analyzers.stringToNbArray)
            all_neurites['loss_x'] = all_neurites['loss_x'].apply(Analyzers.stringToNbArray)
            all_neurites['loss_y'] = all_neurites['loss_y'].apply(Analyzers.stringToNbArray)
        return all_neurites


    def isolateBranchInMostTimeframesPerOrigin(self,allNeurites):
        #clean up allneurites df - only keep one branch per neurite (the one that is present in most timeframes and longest compared to others in similar nb of timeframes)
        allOrigins  = allNeurites.reset_index().drop_duplicates(['origin'])
        for originsRowNb in allOrigins.index.values:
            #get data from single timepoint
            originsID = allOrigins.loc[originsRowNb,['time','origin','branch']]
            originsData = allNeurites.loc[allNeurites['origin'] == originsID[1]]
            allBranchesOfOrigin = originsData.reset_index().drop_duplicates(['branch'])
            if type(originsData) == pd.core.series.Series:
                originsData = pd.DataFrame(originsData).transpose().reset_index()

            #get the max number of timeframes in which one branch in present
            maxNbOfFrames = 0
            for branchRowNb in allBranchesOfOrigin.index.values:
                branchID = allBranchesOfOrigin.loc[branchRowNb,['branch']]
                timeframesWithBranch = len(originsData.loc[originsData['branch'] == branchID[0]])
                maxNbOfFrames = max(maxNbOfFrames,timeframesWithBranch)

            #get all branches which are present in at least 90% of the maximum number of timeframes
            branchesInMostFrames = []
            for branchRowNb in allBranchesOfOrigin.index.values:
                branchID = allBranchesOfOrigin.loc[branchRowNb,['branch']]
                timeframesWithBranch = len(originsData.loc[originsData['branch'] == branchID[0]])
                if timeframesWithBranch >= (0.9 * maxNbOfFrames):
                    branchesInMostFrames.append(branchID[0])

            #check which branch has the highest average length
            highestAvBranchLength = 0
            for branch in branchesInMostFrames:
                branchData = originsData.loc[originsData['branch'] == branch]
                if type(branchData) == pd.core.series.Series:
                    branchData = pd.DataFrame(branchData).transpose().reset_index()
                avBranchLength = branchData['length'].mean()
                if avBranchLength > highestAvBranchLength:
                    highestAvBranchLength = avBranchLength
                    bestBranch = branch

            allNeurites = allNeurites[(allNeurites['origin'] != originsID[1]) | ((allNeurites['origin'] == originsID[1]) & (allNeurites['branch'] == bestBranch))]

        return allNeurites


    def saveSkeletons(self,allNeurites,neuronPath,date,experiment,neuron,imageShape):
        
        allNeurites.set_index(['date','experiment','neuron','channel','time'],inplace=True)

        #save all skeletons
        allTimePointIDs = allNeurites.reset_index().drop_duplicates(['time'])
        allTimePointIDs.sort_values(['time'],inplace=True)
        lastTimePoint = 0
        imagePath = neuronPath+"\\c9999\\"

        if os.path.exists(imagePath):
            nb_skeleton_images = len(os.listdir(imagePath))
        else:
            nb_skeleton_images = 0

        if (nb_skeleton_images != np.max(allTimePointIDs['time'])) | (self.always_save_skeletons) | (self.always_save_completed_skeletons):

            if os.path.exists(imagePath):
                shutil.rmtree(imagePath,ignore_errors=True)
            if not os.path.exists(imagePath):
                for retry in range(100):
                    try:
                        os.mkdir(imagePath)
                        break
                    except:
                        print("creating folder failed, retrying...")
                    if retry == 99:
                        print("folder could not be created")
            for timePointRowNb in allTimePointIDs.index.values:
                skeletonImage = np.zeros(imageShape)
                skeletonImage = skeletonImage.astype(np.uint32)

                #get data from single timepoint
                timePointID = allTimePointIDs.loc[timePointRowNb,['date','experiment','neuron','channel','time']]
                timePoint = timePointID[4]
                print("save skeletons - time: {}".format(timePoint))

                #if there are timepoints in between the last and current timepoints without neurite length data, still save blank image of timepoint
                for missingTimePoint in range(lastTimePoint+1,timePoint):
                    imageName = str(date) + "_" + str(experiment) + "_" + str(neuron) + "_" + str(missingTimePoint) + "_skeleton.tif"
                    io.imsave(imagePath+imageName,skeletonImage)

                lastTimePoint = timePoint

                timePointData = allNeurites.loc[timePointID[0],timePointID[1],timePointID[2],timePointID[3],timePointID[4]]
                

                if (type(timePointData) == pd.core.series.Series):
                    timePointData = pd.DataFrame(timePointData).transpose().reset_index()

                all_origin_start_branches = timePointData.drop_duplicates(["origin","start_branch"]).reset_index()

                for origin_start_branch_index in all_origin_start_branches.index.values:
                    origin_start_branch_ID = all_origin_start_branches.loc[origin_start_branch_index,['origin','start_branch']]
                    origin_start_branch = timePointData.loc[(timePointData['origin'] == origin_start_branch_ID['origin']) & (timePointData['start_branch'] == origin_start_branch_ID['start_branch'])].reset_index()

                    for branch_index in origin_start_branch.index.values:
                        branch_data = origin_start_branch.loc[branch_index]
                        Xs = Analyzers.stringToNbArray(branch_data.loc["x"])
                        Ys = Analyzers.stringToNbArray(branch_data.loc["y"])
                        origin = branch_data.loc['origin']
                        branch = branch_data.loc['branch']
                        pxValue = origin*100000 + branch

                        skeletonImage[Xs,Ys] = pxValue
                skeletonImage = morph.dilation(skeletonImage,disk(4))
                skeletonImage = skeletonImage.astype(np.uint32)
                imageName = str(date) + "_" + str(experiment) + "_" + str(neuron) + "_" + str(timePoint) + "_skeleton.tif"
                io.imsave(imagePath+imageName,skeletonImage)


    def getNearbyRefs(self,xPos,yPos,groupImage,endpointlabelRadius):
            xStart = xPos-endpointlabelRadius
            if xStart < 0:
                xStart = 0
            elif xStart >= groupImage.shape[0]:
                xStart = groupImage.shape[0]
            xEnd = xPos+endpointlabelRadius+1
            if xEnd < 0:
                xEnd = 0
            elif xEnd >= groupImage.shape[0]:
                xEnd = groupImage.shape[0]
            yStart = yPos-endpointlabelRadius
            if yStart < 0:
                yStart = 0
            elif yStart >= groupImage.shape[0]:
                yStart = groupImage.shape[0]
            yEnd = yPos+endpointlabelRadius+1
            if yEnd < 0:
                yEnd = 0
            elif yEnd >= groupImage.shape[0]:
                yEnd = groupImage.shape[0]
            endPointImage = groupImage[xStart:xEnd,yStart:yEnd]
            endPointImage = copy.copy(endPointImage)
            endPointMid = endpointlabelRadius
            endPointImage[endPointMid,endPointMid] = False
            endPointImageLabeled,nbOfLabels = ndimage.label(endPointImage,structure=[[1,1,1],[1,1,1],[1,1,1]])
            endPointImageData = np.where(endPointImage != False)
            references = np.where(((endPointImageData[0] < (endPointMid+2)) & (endPointImageData[0] > (endPointMid-2))) & ((endPointImageData[1] < (endPointMid+2)) & (endPointImageData[1] > (endPointMid-2))))
            references = endPointImageLabeled[endPointImageData[0][references],endPointImageData[1][references]]
            references = np.unique(references)
            return references


    def createSkeleton(self,neuronPath):
        maskPath = neuronPath+'\\mask.tif'
        timePoint = 0
        maskSet = False
        maskImData_thresh = np.zeros((1,1))
        if os.path.exists(maskPath):
            maskImData = io.imread(maskPath)
            maskImData_inv = np.invert(maskImData)
            maskImData_thresh = maskImData_inv > 200
            maskSet = True
        if os.path.isdir(neuronPath):
            for channel in os.listdir(neuronPath):
                self.channel = channel
                channelPath = neuronPath+"\\"+channel
                if (channel.replace("c0","") != channel):
                    if (int(channel.replace("c0","")) == self.channelUsedForSkel):
                        if os.path.isdir(channelPath):
                            timePoint = 0
                            allImageNames = os.listdir(channelPath)
                            if len(self.allNeurites) > 0:
                                last_timepoint_analyzed = max(self.allNeurites['time'])
                            else:
                                last_timepoint_analyzed = 0
                            nb_of_timepoints = len(allImageNames)
                            if (nb_of_timepoints > 10):
                                self.image_shape = io.imread(channelPath+"\\"+allImageNames[0]).shape
                                if (last_timepoint_analyzed < nb_of_timepoints):
                                    self.all_timepoints_analyzed = False
                                else:
                                    self.all_timepoints_analyzed = True
                                if ((not self.all_timepoints_analyzed)) & self.create_skeleton:
                                        self.allNeurites = self.convert_columns_from_string(self.allNeurites)
                                        timePoint = last_timepoint_analyzed
                                        allImageNames = allImageNames[last_timepoint_analyzed:]
                                        for timeframeImageName in allImageNames:
                                            analyze = True
                                            timePoint += 1
                                            if self.continueAnalysis:
                                                timePointData = self.allNeurites.loc[self.allNeurites['time'] == timePoint]
                                                if len(timePointData) > 0:
                                                    analyze = False

                                            if timePoint < self.first_timepoint_to_analyze:
                                                analyze = False

                                            if analyze:
                                                print("thread {} - neuron {} - time {}".format(self.thread,self.neuron,timePoint))
                                                imagePath = channelPath+"\\"+timeframeImageName
                                                
                                                self.processTimePoint(imagePath,timePoint,maskImData_thresh,maskSet)

                                                self.allNeurites.to_csv(self.baseFileName+".csv")
                                                if (timePoint == nb_of_timepoints):
                                                    self.all_timepoints_analyzed = True

        return self.allNeurites


    def threshold_neuron(self, image=None, mask=None, starting_threshold=0):

        # use supplied image by default
        # otherwise use image of neuriteanalyzer object
        if type(image) == type(None):
            image = self.timeframe

        image = median(image, disk(self.imgMedianFilterSize))
        self.timeframe = image

        if type(mask) == type(None):
            mask = np.zeros_like(image)
        
        neuron_nb = 0
            
        output = ThresholdNeuron.start(image,self.cX,self.cY,
                                       self.grainSizeToRemove,
                                       self.dilationForOverlap,
                                       self.overlapMultiplicator,
                                       neuron_nb,self.minEdgeVal,
                                       self.minBranchSize,
                                       self.percentileChangeForThreshold, 
                                       self.maxThresholdChange,
                                       self.minSomaOverflow,
                                       self.somaExtractionFilterSize,
                                       self.maxToleratedSomaOverflowRatio,
                                       self.nb1Dbins,
                                       self.minNbOfPxPerLabel,
                                       self.objectSizeToRemoveForSomaExtraction,
                                       self.distanceToCover,
                                       self.filterBackgroundPointsFactor,
                                       self.openingForThresholdEdge,
                                       self.medianForThresholdEdge,
                                       self.closingForPresoma,
                                       mask,
                                       starting_threshold)

        (self.timeframe_thresholded,
         self.optimalThreshold, 
         self.threshold_percentile,
         self.backgroundVal, 
         self.cX,self.cY) = output

        self.optimalThreshold = self.optimalThreshold + self.backgroundVal
        
        return None
            
    def get_soma(self):
        timeframe_invMask = self.timeframe_thresholded.astype(np.uint8)
        timeframe_invMask[timeframe_invMask > 0] = 255
        nbOfLabels = 2
        #extract soma and threshold image
        (self.timeframe_soma,
         self.timeframe_thresholded) = SomaTools.getSoma(timeframe_invMask,
                                                       self.minSomaOverflow,
                                                       self.somaExtractionFilterSize,
                                                       self.timeframe_thresholded,
                                                       self.objectSizeToRemoveForSomaExtraction,
                                                      self.cX, self.cY)

        # dilate soma a little bit to account for soma which are a bit soo small (leave some border of the thresholded soma)
        timeframe_somaLabels, nbOfLabels = ndimage.label(self.timeframe_soma,
                                                         structure=
                                                         [[1, 1, 1], [1, 1, 1],
                                                          [1, 1, 1]])
        timeframe_soma_forMid = np.zeros_like(self.timeframe_thresholded,
                                              dtype=np.uint8)
        timeframe_soma_forMid[self.timeframe_soma == True] = 1

        self.cX, self.cY = generalTools.getMidCoords(timeframe_soma_forMid)

        # isolate middle region as soma
        if timeframe_somaLabels[self.cY, self.cX] == 0:
            allSomaCoords = np.where(self.timeframe_soma == 1)
            closestSomaPoint = generalTools.getClosestPoint(allSomaCoords,
                                                            [[self.cY, self.cX]])
            somaLabel = timeframe_somaLabels[closestSomaPoint[0],
                                             closestSomaPoint[1]]
        else:
            somaLabel = timeframe_somaLabels[self.cY, self.cX]

        self.timeframe_soma[timeframe_somaLabels != somaLabel] = False

        self.timeframe_somaborder = morph.binary_dilation(self.timeframe_soma,
                                                          morph.square(3))
        self.timeframe_somaborder[self.timeframe_soma == True] = False


    def connect_islands(self):
        # remove pixel border around image - can lead to confusion in connectislands algorithm if not connected to neurite
        self.timeframe_thresholded[0:2, :] = False
        self.timeframe_thresholded[:, 0:2] = False
        self.timeframe_thresholded[self.timeframe_thresholded.shape[0] - 3:
                                self.timeframe_thresholded.shape[0] - 1, :] = False
        self.timeframe_thresholded[:, self.timeframe_thresholded.shape[1] - 3:
                                   self.timeframe_thresholded.shape[1] - 1] = False

        # dilation important pre processing step for connecting islands, otherwise halo around thresholded area misleads connection
        self.timeframe_thresholded = morph.remove_small_objects(
            self.timeframe_thresholded, 5, 2)

        # steps of connecting islands, if a low intensity part in the middle was left out before (also to reconnect cut parts by cutter)
        timeframe_islands = copy.copy(self.timeframe_thresholded)
        # check whether size of all neurites together is at least as big as the minimum branch size
        timeframe_islands_test = morph.skeletonize(timeframe_islands)
        timeframe_islands_test[self.timeframe_soma] = False

        if (len(np.where(timeframe_islands_test == 1)[0]) <=
                self.minBranchSize):
            return

        # fill in gaps in neurites
        # that might have happened during thresholding
        timeframe_islands = connectNeurites.start(timeframe_islands,
                                                  self.timeframe,
                                                  self.timeframe_thresholded,
                                                  [self.cX, self.cY],
                                                  self.timeframe_somaborder,
                                                  self.maxRelDifferenceOfLengths,
                                                  self.minBranchSize,
                                                  self.minContrast,
                                                  self.maxLocalConnectionContrast,
                                                  self.distanceToCheck,
                                                  self.backgroundVal)

        self.timeframe_thresholded[timeframe_islands == True] = True

    def remove_not_connected_to_soma_from_neurites(self):
        # remove soma from neurites image
        timeframe_labeled,_ = ndimage.label(self.timeframe_thresholded,
                                            structure=self.label_structure)
        self.timeframe_thresholded[timeframe_labeled !=
                                   timeframe_labeled[self.cY,self.cX]] = 0

    def remove_soma_from_neurites(self):
        # remove soma from neurites image
        timeframe_neurites_labeled,_ = ndimage.label(self.timeframe_neurites,
                                                     structure=self.label_structure)
        self.timeframe_neurites[timeframe_neurites_labeled ==
                                timeframe_neurites_labeled[self.cY,self.cX]] = 0

    def separate_neurites(self, separate_neurites_by_opening=True):
        output = SeparateNeurites.separateNeuritesBySomaDilation(self.timeframe_thresholded,
                                                                self.timeframe_soma,
                                                                self.maxSomaDilationForSeparation)
        self.timeframe_neurites, self.timeframe_soma = output

        if (len(np.where(self.timeframe_neurites > 0)[0]) <= 0) | \
                (len(np.where(self.timeframe_soma > 0)[0]) <= 0):
            return False, "Separating neurites failed."

        if separate_neurites_by_opening:
            self.timeframe_neurites = SeparateNeurites.separateNeuritesByOpening(self.timeframe_neurites,
                                                                            self.maxNeuriteOpening,
                                                                            self.timeframe_soma,
                                                                            self.timeframe,
                                                                            self.grainSizeToRemove,
                                                                            self.contrastThresholdToRemoveLabel,
                                                                            self.maxFractionToLoseByOpening)

    def get_clean_thresholded_image(self, starting_threshold=0,
                                    maskImData_thresh=None,
                                    find_threshold=True,
                                    connect_islands=True,
                                    separate_neurites=True,
                                    separate_neurites_by_opening=True):

        if find_threshold:
            self.threshold_neuron(mask=maskImData_thresh,
                                  starting_threshold=starting_threshold)

            if np.isnan(self.optimalThreshold):
               return False, "Initial thresholding failed."

        self.get_soma()

        if len(np.where(self.timeframe_soma == 1)[0]) <= 0:
            return False, "Finding soma failed."

        if connect_islands:
            self.connect_islands()

        self.remove_not_connected_to_soma_from_neurites()

        if separate_neurites:
            self.separate_neurites(separate_neurites_by_opening)

        self.remove_soma_from_neurites()

        self.timeframe_neurites[self.timeframe_soma == True] = False

        self.timeframe_neurites[self.timeframe_soma == True] = 1

        # project changes made by separation on thresholded image
        self.timeframe_thresholded[(self.timeframe_neurites == 0) &
                                   (self.timeframe_soma == 0)] = 0

        self.timeframe_thresholded_neurites = copy.copy(self.timeframe_thresholded)
        self.timeframe_thresholded_neurites[self.timeframe_soma == 1] = 0

        return True, ""

    def get_long_skeleton_to_keep(self, initial_skeleton, image_opened,
                                  skeleton_opened_image):
        """
        1) Reconnect all skeletons outside of the opened image
         always keep parts of overhang larger than min branch size!
        """
        # create new image with
        # old skeleton which goes beyond opened thresholded image
        skeleton_overhang = copy.copy(initial_skeleton)
        skeleton_overhang[image_opened == True] = False

        skeletons_overhang_keep = copy.copy(skeleton_overhang)

        #create image with all overhangs longer than minbranhsize
        skeletons_overhang_keep = morph.remove_small_objects(skeletons_overhang_keep,
                                                             self.minBranchSize,
                                                             connectivity=2)

        # add all possible connectors to overhangs to keep
        skeletons_overhang_keep_connected = copy.copy(skeletons_overhang_keep)
        # possible connectors are the parts of the skeleton that were removed
        # in the skeleton of the opened thresholded image
        possible_connectors = copy.copy(initial_skeleton)
        possible_connectors[skeleton_opened_image] = False
        skeletons_overhang_keep_connected[possible_connectors] = True
        # only keep labels of the connector which are part of the
        # overhang to keep (before adding connectors)
        overhang_connected_labeled, _ = ndimage.label(skeletons_overhang_keep_connected,
                                                      structure=self.label_structure)
        labels_of_overhangs_to_keep = np.unique(overhang_connected_labeled[skeletons_overhang_keep])
        connected_overhangs_to_keep = np.zeros_like(skeleton_opened_image)
        for label in labels_of_overhangs_to_keep:
            connected_overhangs_to_keep[overhang_connected_labeled == label] = True

        return connected_overhangs_to_keep

    def get_lost_branch_tips_of_skeleton(self, initial_skeleton,
                                         skeleton_opened_image):
        """
        # 2) Connect tips of skeleton, if they are not branches which are just
        #    present in the initial skeleton
        """
        # find tips by getting the difference of the initial skeleton
        # and the skeleton of the opened image
        skeleton_difference = initial_skeleton & np.invert(skeleton_opened_image)
        skeleton_difference_labeled, _ = ndimage.label(skeleton_difference,
                                                       structure=self.label_structure)
        # It will include some elongation of tips
        # and also some added branches
        # To find all branches,
        # the difference of the skeletons is dilated by 1 px
        skeleton_difference_dil = morph.binary_dilation(skeleton_difference,
                                                        [[1,1,1],
                                                         [1,1,1],
                                                         [1,1,1]])
        # and then remove each single part from the initial skeleton
        skeleton_difference_dil_labeled, _ = ndimage.label(skeleton_difference_dil,
                                                    structure=self.label_structure)
        difference_labels = np.unique(skeleton_difference_dil_labeled)
        _, initial_nb_labels = ndimage.label(initial_skeleton,
                                              structure=self.label_structure)
        skeleton_tips = np.zeros_like(initial_skeleton)
        for label in difference_labels:
            initial_skeleton_test = copy.copy(initial_skeleton)
            initial_skeleton_test[skeleton_difference_dil_labeled == label] = False
            _, new_nb_labels = ndimage.label(initial_skeleton_test,
                                             structure=self.label_structure)
            # if this leads to more labeled regions, then it was a branch
            # (since there was no distance between the main part and the part
            # removed from it)
            # only add skeleton parts where this was not the case
            if new_nb_labels == initial_nb_labels:
                # do not add the dilated part but the undilated part
                # first find labels of undilated part to add, then add that
                labels_to_add = np.unique(skeleton_difference_labeled[skeleton_difference_dil_labeled == label])
                label_to_add = [label_to_add
                                for label_to_add in labels_to_add
                                if label_to_add != 0 ]
                skeleton_tips[skeleton_difference_labeled == label_to_add[0]] = True

        return skeleton_tips


    def get_skeleton_connections(self, skeleton, initial_skeleton,
                                 skeleton_opened_image):
        """
        3) At the end check for each lost part of the skeleton whether
            adding it will connect some parts of the skeleton. If so, keep it.
        """
        # now go through each connector and test whether adding it reduces
        # the number of labels
        possible_connectors = copy.copy(initial_skeleton)
        possible_connectors[skeleton_opened_image] = False
        skeleton_connected = copy.copy(skeleton)
        connectors_labeled, _ = ndimage.label(possible_connectors,
                                              structure = self.label_structure)
        connector_labels = np.unique(connectors_labeled)
        _, current_nb_labels = ndimage.label(skeleton,
                                             structure = self.label_structure)
        for label in connector_labels:
            if label == 0:
                continue
            skeleton_connected_test = copy.copy(skeleton_connected)
            skeleton_connected_test[connectors_labeled == label] = True
            _, new_nb_labels = ndimage.label(skeleton_connected_test,
                                             structure=self.label_structure)
            if new_nb_labels < current_nb_labels:
                # if the nb of labels was reduced,
                # set the new nb of labels as the current number of labels
                current_nb_labels = new_nb_labels
                # and accept changes to the connected skeleton
                skeleton_connected = skeleton_connected_test
        return skeleton_connected


    def get_neurite_skeletons(self):
        """
        Create skeleton of neurites from clean thresholded image.
        Prevent skeletons to form in thick sections to small uneven
        parts of the neurite, open the neurite image to smoothen it
        However, opening the images might lead to loss of parts of neurites
        (either complete branches or a significant portion at the tip),
        it might also separate a neurite into two parts of which only one is
        connected to the soma, therefore the not connected part would
        be removed
        Re-connect the correct neurites by three different ways:
        1) Reconnect ALL skeletons outside of the opened image;
        2) Connect tips of skeleton, if they are not branches which are only
            present in the initial skeleton (and no part of them present in
            skeleton of opened image)
        3) At the end check for each lost part of the skeleton whether
            adding it will connect some parts of the skeleton. If so, keep it.
        """
        initial_skeleton = morph.skeletonize(self.timeframe_neurites)

        # Open image to smoothen it and thereby to remove artifial branches
        # created from small bumps in thresholded image
        image_opened = morph.binary_opening(self.timeframe_neurites,
                                            disk(self.open_for_skeletonization))

        skeleton_opened_image = morph.skeletonize(image_opened)

        connected_overhangs_to_keep = self.get_long_skeleton_to_keep(initial_skeleton,
                                                                     image_opened,
                                                                     skeleton_opened_image)

        skeleton_tips = self.get_lost_branch_tips_of_skeleton(initial_skeleton,
                                                              skeleton_opened_image)

        skeleton = copy.copy(skeleton_opened_image)
        skeleton[skeleton_tips] = 1
        skeleton[connected_overhangs_to_keep] = 1

        skeleton_connected = self.get_skeleton_connections(skeleton,
                                                           initial_skeleton,
                                                           skeleton_opened_image)

        self.timeframe_neurites = skeleton_connected
        self.timeframe_neurites[self.timeframe_soma == True] = False

    def processTimePoint(self, imagePath, timePoint, maskImData_thresh,
                         maskSet, starting_threshold= 0):
        self.imagePath = imagePath
        self.timePoint = timePoint
        self.maskImData_thresh = maskImData_thresh
        self.maskSet = maskSet
        self.starting_threshold = starting_threshold
        self.timeframe = io.imread(imagePath)
        if(len(np.unique(self.timeframe)) <= 1):
            return

        #if mask is defined, subtract mask from image to isolate neuron from surrounding cells
        if maskSet == False:
            maskImData_thresh = np.zeros_like(self.timeframe)

        thresholding_worked = self.get_clean_thresholded_image(starting_threshold,
                                                               maskImData_thresh)

        if not thresholding_worked:
            return

        self.get_neurite_skeletons()

        if len(np.where(self.timeframe_neurites == 1)[0]) <= self.minBranchSize:
            return

        self.process_neurites()

    def get_labeled_neurite_skeletons_based_on_thresholded_labels(self):
        # get labels from thresholded image of neurites
        (self.timeframe_thresholded_neurites_labeled,
         _) = ndimage.label(self.timeframe_thresholded_neurites,
                            structure=self.label_structure)

        # then go through each label in the labeled image of neurite skeletons
        # and set their label to the same label in the labeled thresolded image
        (self.timeframe_neurites_labeled, _) = ndimage.label(self.timeframe_neurites,
                                                            structure= self.label_structure)
        new_timeframe_neurites_labeled = np.zeros_like(self.timeframe_neurites, dtype=np.uint8)
        neurite_labels = np.unique(self.timeframe_neurites_labeled)
        for neurite_label in neurite_labels:
            # ignore the background label
            if neurite_label == 0:
                continue
            # get the label of the labeled thresholded image of neurites
            # at the position(s) of the skeleton of the current neurite
            new_labels = np.unique(self.timeframe_thresholded_neurites_labeled[self.timeframe_neurites_labeled ==
                                                                    neurite_label],
                                   return_counts = True)
            # create list of counts of nonzero labels
            nonzero_label_counts = [new_labels[1][label_nb]
                                    for label_nb, label in enumerate(new_labels[0])
                                    if label != 0]
            nonzero_labels = [label
                              for label in new_labels[0]
                              if label != 0]

            max_count_nonzero_label_idx = np.argmax(nonzero_label_counts)

            max_count_nonzero_label = nonzero_labels[max_count_nonzero_label_idx]

            new_timeframe_neurites_labeled[self.timeframe_neurites_labeled == neurite_label] = max_count_nonzero_label

        self.timeframe_neurites_labeled = new_timeframe_neurites_labeled

    def get_neurites(self):

        self.get_labeled_neurite_skeletons_based_on_thresholded_labels()

        self.backgroundInt = np.mean(self.timeframe[self.timeframe_neurites_labeled == 0])

        # if backgroundVal was not set (only done during thresholding,
        # but if thresholded image is directly supplied, no thresholding is
        # done)
        # set the backgroundval was backgroundint (average value of not
        # thresholded area)
        if np.isnan(self.backgroundVal):
            self.backgroundVal = self.backgroundInt

        maskToExcludeFilopodia = self.createMaskWithoutFilopodia(self.timeframe,
                                                                 self.timeframe_thresholded)

        self.branchPointsCols = ('x','y','refs','used')

        neurite_labels = np.unique(self.timeframe_neurites_labeled)
        # get sorted points of all neurites in dict with label number of neurite
        # as key
        new_timeframe_neurites_labeled = np.zeros_like(self.timeframe_neurites,
                                                       dtype=np.uint8)
        all_sorted_points = {}
        self.all_number_of_branchpoints = {}

        for group in neurite_labels:
            if group == 0:
                continue

            group_image = self.get_skeleton_of_current_neurite(group)

            # if there is only one value in the current group
            # (no neurite with minimum neurite length left)
            # then continue to next group
            if len(np.unique(group_image)) == 1:
                continue

            (group_image,
             group_image_branches) = self.remove_small_branches_at_end(group_image)

            (total_nb_branchpoints,
             group_image_branches_labeled) = self.get_nb_of_branchpoints(group_image_branches,
                                                                        group_image)
            self.all_number_of_branchpoints[group] = total_nb_branchpoints

            sorted_points = self.get_sorted_neurite_points(group_image,
                                                          group_image_branches_labeled,
                                                          maskToExcludeFilopodia)

            for sorted_points_branch in sorted_points:
                new_timeframe_neurites_labeled[sorted_points_branch[:, 0],
                                               sorted_points_branch[:, 1]] = group

            all_sorted_points[group] = sorted_points

        self.timeframe_neurites_labeled = new_timeframe_neurites_labeled
        return all_sorted_points

    def createMaskWithoutFilopodia(self,timeframe,timeframe_thresholded):
        openingForSmoothenedNeuriteImg = 2
        timeframe_opened = morph.opening(timeframe,square(openingForSmoothenedNeuriteImg))
        timeframe_edges = scharr(timeframe_opened)
        timeframe_edges[timeframe_thresholded == 0] = 0
        thresh = otsu(timeframe_edges)*0.5
        timeframe_edge_thresh = timeframe_edges > thresh
        timeframe_edge_thresh = morph.binary_closing(timeframe_edge_thresh,disk(10))
        return timeframe_edge_thresh

    def get_skeleton_of_current_neurite(self, group):

        groupData = np.where(self.timeframe_neurites_labeled == group)
        if((group < 0) | (len(groupData[0]) < self.minNeuriteLength)):
            return np.zeros_like(self.timeframe_neurites)

        # create binary image of the skeleton of the current neurite
        groupImage = self.timeframe_neurites_labeled == group

        #close small holes in neurite structure which lead to artifially complicated neurites
        groupImage = morph.binary_closing(groupImage,disk(1))
        groupImage = morph.skeletonize(groupImage)
        return groupImage

    def remove_small_branches_at_end(self, groupImage):
        #get all branch points of neurite, separate neurite by deleting branch points
        groupImage_branches, branchPoints = self.getBranchPoints(groupImage,
                                                                 self.branchPointRadius,
                                                                 self.branchPointRefRadius)

        (groupImage_branches_labeled,
         nbOfBranches) = ndimage.label(groupImage_branches,
                                       structure=self.label_structure)

        #create image of all branchpoints
        group_image_branchpoints = np.zeros_like(groupImage_branches_labeled)
        group_image_branchpoints[(groupImage_branches == 0) & (groupImage == 1)] = 1
        (group_image_branchpoints_labeled,
         total_nb_branchpoints) = ndimage.label(group_image_branchpoints,
                                                structure=self.label_structure)

        self.last_total_nb_branchpoints = total_nb_branchpoints
        soma_labeled ,nb_labels_soma = ndimage.label(self.timeframe_soma,
                                                     structure=self.label_structure)

        #remove all labels at end of branch (only connected to one branchpoint) which are too small
        for label in np.unique(groupImage_branches_labeled):
            one_label = np.zeros_like(groupImage_branches_labeled)
            one_label[groupImage_branches_labeled == label] = 1
            one_label_dilated = morph.binary_dilation(one_label,square(3))
            #subtract 1 from number of branchpoints due to 0 in np.unique output (background)
            nb_branchpoints = len(np.unique(group_image_branchpoints_labeled[one_label_dilated == 1]))-1
            if nb_branchpoints < 2:
                if len(np.where(one_label == 1)[0]) < self.branchLengthToKeep:
                    #check if branch is attached to soma, if so, do not remove it
                    neurite_plus_soma = copy.copy(self.timeframe_soma)
                    neurite_plus_soma[one_label_dilated == 1] = 1
                    (neurite_plus_soma_labeled,
                     nb_labels_with_soma) = ndimage.label(neurite_plus_soma,
                                                          structure=self.label_structure)
                    #if same number of labels after adding neurite to soma, they are connected
                    if nb_labels_with_soma > nb_labels_soma:
                        groupImage[one_label == 1] = 0
                        groupImage_branches_labeled[one_label == 1] = 0
                        groupImage_branches[one_label == 1] = 0
        return groupImage, groupImage_branches

    def get_nb_of_branchpoints(self, groupImage_branches, groupImage):
        (groupImage_branches,
         branchPoints) = self.getBranchPoints(groupImage_branches,
                                              self.branchPointRadius,
                                              self.branchPointRefRadius)

        (groupImage_branches_labeled,
         nbOfBranches) = ndimage.label(groupImage_branches,
                                       structure=self.label_structure)

        #create image of all branchpoints
        group_image_branchpoints = np.zeros_like(groupImage_branches_labeled)
        group_image_branchpoints[(groupImage_branches == 0) & (groupImage == 1)] = 1
        (groupImage_branches_labeled, total_nb_branchpoints) = ndimage.label(group_image_branchpoints,
                                                   structure=self.label_structure)

        return total_nb_branchpoints, groupImage_branches_labeled

    def get_sorted_neurite_points(self, groupImage,
                                  groupImage_branches_labeled,
                                  maskToExcludeFilopodia):

        sumIntensitieGroupImg = copy.copy(self.timeframe)
        #subtract background from intensity image (timeframe)
        background_val_int = int(np.round(self.backgroundVal,0))
        timeframeForNeurite = copy.copy(self.timeframe)
        timeframeForNeurite[timeframeForNeurite <
                            int(np.round(self.backgroundVal,0))] = background_val_int
        timeframeForNeurite = timeframeForNeurite - background_val_int
        constructNeurite = True
        distancePossible =False
        keepLongBranches = True
        length = 0
        neuriteCoords = np.where(groupImage == 1)

        closestNeuritePoints = self.getStartPointsOfNeurite(groupImage,
                                                            copy.copy(self.timeframe_soma),
                                                            self.timeframe)

        sortedArrays = []
        for closestNeuritePoint in closestNeuritePoints:
            #don't remove filopodia in first time frame
            if self.timePoint == 1:
                min_filopodia_length = 0
                max_filopodia_length = 1
            else:
                min_filopodia_length = self.minFilopodiaLength
                max_filopodia_length = self.maxFilopodiaLength

            (sortedArray_tmps,
             length, _) = sortPoints.startSorting(neuriteCoords,
                                                         [closestNeuritePoint],
                                                         length,
                                                         self.minBranchSize,
                                                         [],
                                                         keepLongBranches,
                                                         self.branchLengthToKeep,
                                                         distancePossible,
                                                         constructNeurite,
                                                         timeframeForNeurite,
                                                         groupImage_branches_labeled,
                                                         sumIntensitieGroupImg,
                                                         groupImage,
                                                         maskToExcludeFilopodia,
                                                         min_filopodia_length,
                                                         max_filopodia_length,
                                                         self.minContrastFilopodia,
                                                         self.minAllowedContrastForFindingFilopodia)
            for oneSortedArray in sortedArray_tmps:
                if len(oneSortedArray) > self.minNeuriteLength:
                    sortedArrays.append(oneSortedArray)

            if len(sortedArrays) == 0:
                return []

            # if more than one start point was present,
            # remove resulting neurites which are too similar
            if (len(sortedArrays) > 1):
                sortedArrays = self.sortOutBranches(sortedArrays,
                                                    sumIntensitieGroupImg)
        return sortedArrays

    def process_neurites(self,group,maskToExcludeFilopodia):

        # rewrite function to create sets of sortedarrays
        # for each neurite
        # then process all neurites through neurite construction
        # final structure should be ONE image which contains the thresholded
        # neurites and a dict with sorted points of each neurite with a key
        # of the label in the thresholded image

        all_sorted_points = self.get_neurites()

        if len(self.allNeurites) > 0:
            self.testNb = np.max(self.allNeurites.index.values)+1
        else:
            self.testNb = 0

        for group, sorted_points in all_sorted_points.items():
            group_image = self.timeframe_neurites_labeled == group
            total_nb_branchpoints = self.all_number_of_branchpoints[group]

            # very high nb of branchpoints indicates suboptimal thresholding
            # (too low threshold)
            # start entire process over with higher threshold
            if (total_nb_branchpoints >=
                    self.max_nb_of_branchpoints_before_overload):

                if self.starting_threshold > 0:
                    starting_threshold = self.starting_threshold * 0.8
                else:
                    starting_threshold = self.threshold_percentile * 0.8
                self.processTimePoint(self.imagePath,
                                      self.timePoint,
                                      self.maskImData_thresh,
                                      self.maskSet, starting_threshold)
                return


            # ensure that current neurite (after removing too small end branches)
            # still is long enough
            if (len(np.where(group_image == 1)[0]) <= self.minNeuriteLength):
                return

            sumIntensitieGroupImg = copy.copy(self.timeframe)

            if len(sorted_points) == 0:
                return

            self.process_neurite(group_image, sorted_points)

    def process_neurite(self, group_image, sorted_points):

        labeled_group_image,nbOfLabels = ndimage.label(group_image,
                                                     structure=self.label_structure)

        #(self,allNeuritePoints,startPoint,labeledGroupImage,cX,cY,
        # possibleOriginsColumns,allNeuritesColumns,
        # overlappingOriginsColumns,identity,allNeurites,
        # dilationForOverlap,overlapMultiplicator,maxChangeOfNeurite,
        # minBranchSize,testNb,optimalThreshold,maxOverlapOfNeurites,img):s
        self.identity = [self.date,self.experiment,self.neuron,
                         self.channel,self.timePoint]
        constructor = neuriteConstructor(sorted_points,
                                         labeled_group_image,
                                         self.cX,self.cY,
                                         self.possibleOriginsColumns,
                                         self.allNeuritesColumns,
                                         self.overlappingOriginsColumns,
                                         self.identity,
                                         self.allNeurites,
                                         self.dilationForOverlap,
                                         self.overlapMultiplicator,
                                         self.maxChangeOfNeurite,
                                         self.minBranchSize,
                                         self.testNb,
                                         self.optimalThreshold,
                                         self.maxOverlapOfNeurites,
                                         self.timeframe,
                                         self.toleranceForSimilarity,
                                         self.dilationForSmoothing,
                                         self.gaussianForSmoothing)
        self.allNeurites,self.newNeuritesDf,self.testNb = constructor.constructNeurites()
        #remove rows with NA values to prevent program error (can happen for < 1% cases by cross skeletonization between neurites)
        self.newNeuritesDf = self.newNeuritesDf.dropna()
        self.newNeuritesDf['length_um'] = self.newNeuritesDf['length'] * self.UMperPX


        if 'length_um' not in self.allNeurites.columns:
            self.allNeurites['length_um'] = self.allNeurites['length'] * self.UMperPX

        #quick fix to exclude columns that were added
        diff_col_nb = abs(len(self.allNeurites.columns) - len(self.newNeuritesDf.columns))
        if diff_col_nb > 0:
            self.allNeurites = pd.concat([self.allNeurites.iloc[:,diff_col_nb:],self.newNeuritesDf])
        else:
            self.allNeurites = pd.concat([self.allNeurites,self.newNeuritesDf])


    def sortOutBranches(self,sortedArrays,sumIntensitieGroupImg):

        sortedArrays = self.remove_overlapping_branches(sortedArrays)

        #sort neurites in groups by start points
        distanceForGrouping = 10
        startPointGroupedSortedArrays = self.sortBranchesInGroupsByPoints(sortedArrays,"start",distanceForGrouping)


        #only use the neurite/s of each group with the highest startPoint intensity
        sortedArrays = []
        for groupNb, group in enumerate(startPointGroupedSortedArrays):
            highestIntensity = 0
            for sortedArray in group:
                startPointIntensity = sumIntensitieGroupImg[sortedArray[0][0],sortedArray[0][1]]
                highestIntensity = max(startPointIntensity,highestIntensity)
            for sortedArray in group:
                startPointIntensity = sumIntensitieGroupImg[sortedArray[0][0],sortedArray[0][1]]
                if startPointIntensity >= highestIntensity*0.8:
                    sortedArrays.append(sortedArray)



        sorted_arrays_grouped_by_start_segments = self.sort_branches_in_groups_by_start_segments(sortedArrays,
                                                                                                 self.branchLengthToKeep*2)

        #sort out branches with similar start segments that show too few differences
        sortedArrays = []
#        print("FIRST  LENGTHS!!!")
        for group in sorted_arrays_grouped_by_start_segments:
            max_len = 0
            for a, oneBranch in enumerate(group):
                if len(oneBranch) > max_len:
                    longestBranch = oneBranch
                    longestBranchNb = a
                    max_len = len(oneBranch)
            del group[longestBranchNb]
            constructNeurite = True
            group = sortPoints.checkWhichBranchesToKeep(group,self.branchLengthToKeep*2,constructNeurite,longestBranch,sumIntensitieGroupImg)
            for oneBranch in group:
                sortedArrays.append(oneBranch)

        return sortedArrays

    def remove_overlapping_branches(self,sortedArrays):
        #remove branches if last 20px overlap with other neurites - also happens with circled neurites
        new_sorted_arrays = []
        nb_last_px_to_check = self.nb_last_px_to_check
        for one_branch in sortedArrays:
            last_px_of_branch = one_branch[-nb_last_px_to_check:]
            similarity_found = False
            for other_branch in sortedArrays:
                other_branch = other_branch[0:-nb_last_px_to_check]
                similar_px = generalTools.get_intersect_of_two_arrays(other_branch,last_px_of_branch)
                if len(similar_px) == nb_last_px_to_check:
                    similarity_found = True
                    break
            if not similarity_found:
                new_sorted_arrays.append(one_branch)

        sortedArrays = new_sorted_arrays
        return sortedArrays

    def sort_branches_in_groups_by_start_segments(self,sorted_arrays,branch_length_to_keep):
        #group branches by similarity in first x pixels
        #last px (nb = branch_length_to_keep) is excluded to allow for differences in end of branch
        grouped_sorted_arrays = []
        for one_branch in sorted_arrays:
            found_group = False
            first_points = one_branch[0:-branch_length_to_keep]
            for group_nb, one_group in enumerate(grouped_sorted_arrays):
                max_length = 0
                for other_branch in one_group:
                    if len(other_branch) > max_length:
                        max_length = len(other_branch)
                        longest_branch_of_group = other_branch
                other_first_points = longest_branch_of_group[0:-branch_length_to_keep]
                similar_starts = set(tuple(map(tuple,other_first_points))).intersection(set(tuple(map(tuple,first_points))))
                minSimilarity = min(len(first_points),len(other_first_points))
                if len(similar_starts) >= minSimilarity-self.px_tolerance_for_sorting_branches_by_starting_segments:
                    found_group = True
                    grouped_sorted_arrays[group_nb].append(one_branch)
                    break
            if not found_group:
                grouped_sorted_arrays.append([])
                grouped_sorted_arrays[-1].append(one_branch)

        return grouped_sorted_arrays

    def sortBranchesInGroupsByPoints(self,sortedArrays,mode,maxDistOfEndPoints = 20):
        #sort out further branches
        if mode == "start":
            pos = 0
        elif mode == "end":
            pos = -1
        #first group branches according to using the same endpoint
        groupedSortedArrays = []
        #go through each branch
        for sortedBranch in sortedArrays:
            currentEndPoint = sortedBranch[pos]
            #go through each group of already grouped sorted Branches
            avDistOfAllGroups = []
            for oneGroupSortedBranches in groupedSortedArrays:
                allPointsInRange = True
                #go through each branch of group and check whether any of its endpoints is further away from currentendPoint than allowed
                #save distance of currentendpoint to each other endpoint
                allDistOfGroup = []
                for sortedGroupedBranch in oneGroupSortedBranches:
                    otherEndPoint = sortedGroupedBranch[pos]
                    oneDist = distance.cdist([currentEndPoint],[otherEndPoint])[0][0]
                    if oneDist > maxDistOfEndPoints:
                        allPointsInRange = False
                        break
                    allDistOfGroup.append(oneDist)
                #calculate average distance for group
                if allPointsInRange:
#                    print(allDistOfGroup)
                    averageDist = np.mean(allDistOfGroup)
                    avDistOfAllGroups.append(averageDist)
                    break
                else:
                    avDistOfAllGroups.append(maxDistOfEndPoints+1)
            smallestAvDist = maxDistOfEndPoints + 1
#            print(avDistOfAllGroups)
            #add sortedbranch to group with lowest average distance of all endpoints
            bestGroupFound= False
            for groupNb, averageDist in enumerate(avDistOfAllGroups):
                if averageDist < smallestAvDist:
                    smallestAvDist = averageDist
#                    print(len(groupedSortedArrays))
#                    print(groupNb)
                    groupedSortedArrays[groupNb].append(sortedBranch)
                    bestGroupFound = True
            if not bestGroupFound:
                groupedSortedArrays.append([])
                groupedSortedArrays[-1].append(sortedBranch)
        return groupedSortedArrays


    def getLowestCurvatureBranchFromEachGroup(self,groupedSortedArrays,distToLookForCurvature,pxToJumpOverForCurvature):
        sortedArraysFromGroups = []
        #quantify curvature of each neurite
        for oneGroupSortedArrays in groupedSortedArrays:
            sortedArraysFromGroups.append([])
            lowestAvCurvature = np.nan
            for sortedArray in oneGroupSortedArrays:
                branchHasLowestCurvature = False
                allCurvaturesOfBranch = []
                #for each 10th point in branch, check curvature
                sortedArrayImg = np.zeros((512,512))
                sortedArrayCoords = np.transpose(sortedArray)
                sortedArrayImg[sortedArrayCoords[0],sortedArrayCoords[1]] = 1
                generalTools.showThresholdOnImg(sortedArrayImg,self.timeframe,1)
                lastAngle = np.nan
                for pointNb, point in enumerate(sortedArray):
                    if (pointNb % (pxToJumpOverForCurvature)) == 0:
                        if (pointNb + distToLookForCurvature) < len(sortedArray):
                            #calculate curvature by calculating ratio of x to y distance
                            laterPoint = sortedArray[pointNb+9]
                            #calculate angle of direction
                            angle = connectNeurites.getAngleOfLine(point,laterPoint)
                            if not np.isnan(lastAngle):
                                angleDiff = abs(lastAngle - angle)
                                if angleDiff > 180:
                                    angleDiff = 360 - angleDiff
                                #angle differences below 10 are probably just noise due to uneven neurites generated
                                if angleDiff < 10:
                                    angleDiff = 0
                                allCurvaturesOfBranch.append(angleDiff)
                            lastAngle = angle
                #if branch has lowest curvature, add branch to sortedArray from groups
                avCurvature = np.mean(allCurvaturesOfBranch)
                if np.isnan(lowestAvCurvature):
                    branchHasLowestCurvature = True
                else:
                    if avCurvature < lowestAvCurvature:
                        branchHasLowestCurvature = True
                plt.figure()
                plt.imshow(np.zeros((512,512)))
                plt.text(40,40,str(allCurvaturesOfBranch))
                plt.text(200,200,str(avCurvature))
                if branchHasLowestCurvature:
                    lowestAvCurvature = avCurvature
                    sortedArraysFromGroups[-1] = sortedArray
        return sortedArraysFromGroups


    def getPossibleStartPoints(self,groupImage,timeframe_soma):
        possibleStartPoints = [[]]
        dilationDiskDiameter = 0
        furtherSteps = 0
        startPointsFound = False
        finalPossibleStartPoints = []
        finalPossibleStartPoints.append([])
        finalPossibleStartPoints.append([])
        finalSoma_border = [[]]

        timeframe_soma,borderVals1 = generalTools.cropImage(timeframe_soma,[],60)
        groupImage,borderVals1 = generalTools.cropImage(groupImage,borderVals1)
        soma_border_copy = [[]]
        groupImage_tmp = copy.copy(groupImage)
        while ((len(possibleStartPoints[0]) == 0) | (furtherSteps < 5)) & (dilationDiskDiameter < 30):
            timeframe_soma_tmp = morph.binary_dilation(timeframe_soma,disk(dilationDiskDiameter))
            dilationDiskDiameter += 1
            timeframe_soma_dilated = morph.binary_dilation(timeframe_soma_tmp,disk(1))
            soma_border = np.zeros_like(timeframe_soma_dilated)
            soma_border[(timeframe_soma_dilated == 1) & (timeframe_soma_tmp == 0)] = 1
            soma_border = morph.skeletonize(soma_border)
            #create copy of soma border in first iteration
            if len(soma_border_copy) == 1:
                soma_border_copy = soma_border
            #check for intersection of groupImage with soma_border to find points that reach until soma
            possibleStartPoints = np.where((groupImage_tmp == 1) & (soma_border == 1))
            #coount iterations after startpoints were found
            if startPointsFound:
                furtherSteps += 1
            #for each iteration where new endpoints are found, add them to finalPossibleStartPoints
            if(len(possibleStartPoints[0]) > 0):
                if len(finalSoma_border) == 1:
                    finalSoma_border = soma_border
                furtherSteps += 1
                startPointsFound = True
                for b in range(0,len(possibleStartPoints[0])):
                    finalPossibleStartPoints[0].append(possibleStartPoints[0][b]+borderVals1[0])
                    finalPossibleStartPoints[1].append(possibleStartPoints[1][b]+borderVals1[2])

                    #create image of each startpoint, dilate and remove from group image (to not find start points close to it)
                    startPointImage = np.zeros_like(groupImage)
                    startPointImage[possibleStartPoints[0][b],possibleStartPoints[1][b]] = 1
                    startPointImage,borderVals2 = generalTools.cropImage(startPointImage,[],20)
                    startPointImage = morph.dilation(startPointImage,disk(5))
                    startPointImage = generalTools.uncropImage(startPointImage,borderVals2)
                    startPointCoords = np.where(startPointImage == 1)
                    groupImage_tmp[startPointCoords[0],startPointCoords[1]] = 0

        if len(finalSoma_border) == 1:
            finalSoma_border = soma_border_copy
        finalSoma_border = generalTools.uncropImage(finalSoma_border,borderVals1)
        possibleStartPoints = finalPossibleStartPoints
        return possibleStartPoints, finalSoma_border


    def removeStartPointsOfNotPerpendicularStartingBranches(self,
                                                            possibleStartPoints,
                                                            groupImage,
                                                            soma_border,
                                                            timeframe,
                                                            borderVals):
        #go through each border-intersecting point, check which point going in direction of neurite tip moves away from soma border faster
        startPointsOfPerpendicularBranch = []
        minDistanceRatioOfStartBranch = 0.4
        soma_border_coords = np.where(soma_border == 1)
        if len(possibleStartPoints[0]) > 1:
            for a in range(0,len(possibleStartPoints[0])):
                possibleStartPoint= [possibleStartPoints[0][a],possibleStartPoints[1][a]]
                possibleStartBranch = np.zeros_like(groupImage)
                possibleStartBranch[possibleStartPoint[0],possibleStartPoint[1]] = 1

                #get how fast neurite is moving away from soma by enlarging, getting furthest point and calculate distance between furthest point and closest soma border point
                pointSource = True
                possibleStartBranch_enlarged = sortPoints.enlargeBranch(possibleStartBranch,soma_border,groupImage,timeframe,20,pointSource)
                possibleStartBranch_coords = np.where(possibleStartBranch_enlarged == 1)
                furthestPoint = generalTools.getFurthestPoint(possibleStartBranch_coords,[possibleStartPoint])
                closestSomaBorderPoint = generalTools.getClosestPoint(soma_border_coords,[furthestPoint])
                distanceTraveledByEnlarging = distance.cdist([furthestPoint],[possibleStartPoint])[0][0]
                distanceTraveledAwayFromSoma = distance.cdist([furthestPoint],[closestSomaBorderPoint])[0][0]
                distanceRatioOfStartBranch = distanceTraveledAwayFromSoma / distanceTraveledByEnlarging

                #add all points that move at a close enough to perpendicular angle (specific minimum traveled distance from soma per traveled total distance)

                if distanceRatioOfStartBranch > minDistanceRatioOfStartBranch:
                    possibleStartPoint[0] = possibleStartPoint[0] + borderVals[0]
                    possibleStartPoint[1] = possibleStartPoint[1] + borderVals[2]
                    startPointsOfPerpendicularBranch.append(possibleStartPoint)

            if len(startPointsOfPerpendicularBranch) == 0:
                for a in range(0,len(possibleStartPoints[0])):
                    possibleStartPoint= [possibleStartPoints[0][a],possibleStartPoints[1][a]]
                    possibleStartPoint[0] = possibleStartPoint[0] + borderVals[0]
                    possibleStartPoint[1] = possibleStartPoint[1] + borderVals[2]
                    startPointsOfPerpendicularBranch.append(possibleStartPoint)
        else:
            possibleStartPoint= [possibleStartPoints[0][0],possibleStartPoints[1][0]]
            possibleStartPoint[0] = possibleStartPoint[0] + borderVals[0]
            possibleStartPoint[1] = possibleStartPoint[1] + borderVals[2]
            startPointsOfPerpendicularBranch.append(possibleStartPoint)
        return startPointsOfPerpendicularBranch


    def getStartPointsOfNeurite(self,groupImage,timeframe_soma,timeframe):
        #get start point by finding all points of groupimage that get to the soma
        (groupImage,
         timeframe_soma,
         borderVals) = generalTools.crop2Images(groupImage, timeframe_soma, 20)
        timeframe,borderVals = generalTools.cropImage(timeframe,borderVals)

        timeframe = median(timeframe,disk(4))
        possibleStartPoints,soma_border = self.getPossibleStartPoints(groupImage,
                                                                      timeframe_soma)

        if len(possibleStartPoints[0]) <= 1:
            startPointsOfPerpendicularBranch = possibleStartPoints
            return startPointsOfPerpendicularBranch

        startPointsOfPerpendicularBranch = self.removeStartPointsOfNotPerpendicularStartingBranches(possibleStartPoints,
                                                                                                    groupImage,
                                                                                                    soma_border,
                                                                                                    timeframe,
                                                                                                    borderVals)
        return startPointsOfPerpendicularBranch



    def getSumIntensityImg(self,groupImage,group_image_branches,img):
        #create image of group where each px corresponds to sum of all intensities along the whole diameter of neurite

        branchpoints_image = copy.copy(groupImage)
        branchpoints_image[group_image_branches == 1] = 0

        branchpoints_image_dilated = morph.binary_dilation(branchpoints_image,disk(2))
        branchpoints_image_labeled, nbLabels = ndimage.label(branchpoints_image_dilated,structure=[[1,1,1],[1,1,1],[1,1,1]])

        for label in np.unique(branchpoints_image_labeled):
            if label != 0:
                branchpoint_image = np.zeros_like(groupImage)
                branchpoint_image[branchpoints_image_labeled == label] = 1
                branchpoint_size = len(np.where(branchpoint_image == 1)[0])
                while branchpoint_size < self.min_branchpoint_size:
                    branchpoint_image = morph.dilation(branchpoint_image,disk(1))
                    branchpoint_size = len(np.where(branchpoint_image == 1)[0])
                branchpoints_image_dilated[branchpoint_image == 1] = 1

#        generalTools.showThresholdOnImg(branchpoints_image_dilated,img)

        branches_image = copy.copy(groupImage)
        branches_image[branchpoints_image_dilated == 1] = 0

        branches_image_labeled, nbBranches = ndimage.label(branches_image,structure=[[1,1,1],[1,1,1],[1,1,1]])

        full_branches_image_labeled, nbBranches = ndimage.label(group_image_branches,structure=[[1,1,1],[1,1,1],[1,1,1]])

        sumIntImg = np.zeros_like(img)

        for branch_label in np.unique(branches_image_labeled):
            if branch_label != 0:
                branch_image = np.zeros_like(branches_image)
                branch_image[branches_image_labeled == branch_label] = 1

                groupImageCoords = np.where(branch_image == 1)
                allGroupPoints = np.c_[groupImageCoords[0],groupImageCoords[1]]
#                widthToCheck = np.nan
                pointInt = np.nan
                for point in allGroupPoints:
                    #create local image of point, sort corresponding coordinates to have template for getting neurite int along whole neurite diameter
                    #local sorted coords are necessary since this is the template to check for the perpendicular axis
                    pointImage = np.zeros_like(groupImage)
                    pointImage[point[0],point[1]] = 1
                    allNeighbors = sortPoints.getAllNeighborsAtMaxDistance(point,groupImageCoords,[point],5)[0]
                    if len(allNeighbors) > 0:
                        pointImage[np.transpose(allNeighbors)[0],np.transpose(allNeighbors)[1]] = 1
                        pointCoords = np.where(pointImage == 1)
                        startpointForSorting = generalTools.getFurthestPoint(pointCoords,[point])
                        #    def startSorting(neuriteCoords, sortedPoints, length,minBranchSize,decisivePoints = [], keepLongBranches = False
                        sortedPoints, length,neuriteCoords = sortPoints.startSorting(pointCoords,[startpointForSorting],0,self.minBranchSize,[],True)
                        sortedCoords = np.transpose(sortedPoints)
                        pointIndex = np.where((sortedCoords[0] == point[0]) & (sortedCoords[1] == point[1]))[0]
                        if len(pointIndex) > 0:
                            pointIndex = pointIndex[0]
                            self.maxNeuriteWidth = 18
                            pointInt = Analyzers.getNeuriteIntAtPoint(sortedPoints,pointIndex,[img],self.maxNeuriteWidth,'sum',5, self.optimalThreshold)
                            sumIntImg[point[0],point[1]] = pointInt[0]
                av_int_of_branch = np.mean(sumIntImg[(branch_image == 1) & (sumIntImg != 0)])
                full_branch_labels = np.unique(full_branches_image_labeled[branch_image == 1])
                for full_branch_label in full_branch_labels:
                    if full_branch_label != 0:
                        break
                sumIntImg[(full_branches_image_labeled == full_branch_label) & (branch_image == 0)] = av_int_of_branch

        return sumIntImg


    def getPreBranchPoints(self,groupData,groupImage):
        #create new dataframes for endpoints and branchpoints
        preBranchPoints = pd.DataFrame(columns=self.branchPointsCols)
        branchPointNb = 0;

        for point in range(0,len(groupData[1])):
            xPos = groupData[0][point]
            yPos = groupData[1][point]
            neighbours = np.where(((groupData[0] < (xPos+2)) & (groupData[0] > (xPos-2))) & ((groupData[1] < (yPos+2)) & (groupData[1] > (yPos-2))))
            if(len(neighbours[0]) > 3):
                preBranchPoints.loc[branchPointNb] = [int(xPos),int(yPos),[],0]
                branchPointNb += 1

        preBranchPoints.loc[:,'x'] = pd.to_numeric(preBranchPoints.loc[:,'x'])
        preBranchPoints.loc[:,'y'] = pd.to_numeric(preBranchPoints.loc[:,'y'])
        preBranchPoints.loc[:,['x','y']] = preBranchPoints.loc[:,['x','y']].astype(int)
        #delete branch point to isolate all neurite branches from each other
        groupImage[preBranchPoints.loc[:,'x'],[preBranchPoints.loc[:,'y']]] = False

        return groupImage,preBranchPoints


    def getBranchPoints(self,groupImage,branchPointRadius,branchPointRefRadius):
        groupImage = copy.copy(groupImage)
        groupData = np.where(groupImage == 1)
        groupImage, preBranchPoints = self.getPreBranchPoints(groupData,groupImage)

        branchPoints = pd.DataFrame(columns=self.branchPointsCols)

        groupImage[preBranchPoints[:]['x'],[preBranchPoints[:]['y']]] = True
        #go through each pre-branchpoint and check whether it really is a branchpoint
        branchPointNb = 0
        for branchpoint in range(0,len(preBranchPoints)):
            bxPos = preBranchPoints.loc[branchpoint,'x']
            byPos = preBranchPoints.loc[branchpoint,'y']
            xStart = max(0,bxPos-branchPointRadius)
            xEnd = min(groupImage.shape[0],bxPos+branchPointRadius+1)
            yStart = max(0,byPos-branchPointRadius)
            yEnd = min(groupImage.shape[1],byPos+branchPointRadius+1)
            branchPointImage = groupImage[xStart:xEnd,yStart:yEnd]
            branchPointImage = copy.copy(branchPointImage)
            #remove area around branchpoint due to unprecise positioning (could be a few px off)
            #-1 to +2 beacuse ":" operator excludes last number in sequence (number right of :), last number is number after : -1 (like range)
            branchPointImage[branchPointRadius-1:branchPointRadius+2,branchPointRadius-1:branchPointRadius+2] = False

#            branchPointImage = morph.remove_small_objects(branchPointImage,2,connectivity=2)
#                                                        branchPointImage = morph.remove_small_objects(branchPointImage,branchPointRadius/2,connectivity=2)

            labeledBranchPointImage = measure.label(branchPointImage,background=False,connectivity=2)

            labeledBranchPointData = np.where(labeledBranchPointImage > 0)
            #only check references in small radius from branchpoint
            references = np.where(((labeledBranchPointData[0] < (branchPointRadius+branchPointRefRadius)) &
                                   (labeledBranchPointData[0] > (branchPointRadius-branchPointRefRadius))) &
                                ((labeledBranchPointData[1] < (branchPointRadius+branchPointRefRadius)) &
                                 (labeledBranchPointData[1] > (branchPointRadius-branchPointRefRadius))))

            references = labeledBranchPointImage[labeledBranchPointData[0][references],labeledBranchPointData[1][references]]
            references = np.unique(references)
            if(len(references) > 2):
                branchPoints.loc[branchPointNb] = [int(bxPos),int(byPos),(),0]
                preBranchPoints.loc[branchpoint,'used'] = 1
                branchPointNb += 1

        resetedBranchPointIDs = np.where(preBranchPoints.used == 1)
        resetedBranchPointCoords = preBranchPoints.loc[resetedBranchPointIDs]

        groupImage[resetedBranchPointCoords[:]['x'],resetedBranchPointCoords[:]['y']] = False
        groupImage = morph.remove_small_objects(groupImage,2,connectivity=2)

        return groupImage, branchPoints
