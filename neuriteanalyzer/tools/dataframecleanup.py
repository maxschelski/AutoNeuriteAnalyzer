
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:27:05 2019

@author: schelskim
"""

from .generaltools import generalTools
from .analyzers import Analyzers
from .neuriteconstructor import neuriteConstructor
#from tools.dataframecleanup import DataframeCleanup


import numpy as np
import copy
import pandas as pd
from scipy.spatial import distance
from skimage.draw import line_aa
import sys

from ast import literal_eval


class DataframeCleanup():
    
    
    @staticmethod
    def remove_double_branches_from_all_neurites(all_neurites):
        all_timepoints = all_neurites['time'].drop_duplicates()
        for timepoint in all_timepoints.values:
            timepoint_data = all_neurites.loc[all_neurites['time'] == timepoint]
            all_origins = np.unique(timepoint_data['origin'])
            timepoint_data = neuriteConstructor.removeDoubledBranches(all_origins,timepoint_data,all_neurites)
            all_neurites.loc[all_neurites['time'] == timepoint] = timepoint_data
        return all_neurites
    
    @staticmethod
    def removeNonContinuousOrigins(allNeurites,maxGapSize,min_nb_of_timeframes_factor):
            
            allTimePointIDs = allNeurites.reset_index().drop_duplicates(['date','experiment','neuron','channel','time'])
            allTimePointIDs.sort_values(['date','experiment','neuron','channel','time'],inplace=True)
            
            allNeurites.set_index(['date','experiment','neuron','channel','time','origin','branch'],inplace=True)
            
            #check whether neurites are continuously identified in timeframe
            allNeurites['continuous'] = 1
            maxTimeFrame = np.max(allNeurites.reset_index().time)
            #initiate continuousOrigins dictionary: contains one key for each origin-branch, and for each timepoint it is present one entry in list
            continuousOrigins = {}
            min_nb_of_timeframes = len(allTimePointIDs)*min_nb_of_timeframes_factor
            for timePointRowNb in allTimePointIDs.index.values:
                #get data from single timepoint
                timePointID = allTimePointIDs.loc[timePointRowNb,['date','experiment','neuron','channel','time']]
                
                timePointData = allNeurites.loc[timePointID[0],timePointID[1],timePointID[2],timePointID[3],timePointID[4]].reset_index()
                
                allNeurites, continuousOrigins = DataframeCleanup.checkIfBranchIsContinuous(continuousOrigins,timePointData,allNeurites,timePointID,maxGapSize,min_nb_of_timeframes,maxTimeFrame)
                    
                for rowIndex in timePointData.index.values:
                    timePointNeurite = timePointData.loc[rowIndex]
                    if tuple([timePointNeurite['origin'],timePointNeurite['branch']]) not in continuousOrigins:
                        #only add entry if not last timeframe
                        if maxTimeFrame > timePointID[4]:
                            continuousOrigins[tuple([timePointNeurite['origin'],timePointNeurite['branch']])] = [timePointID[4]]
               
                if timePointID[4] == maxTimeFrame:
                    timePointData = pd.DataFrame(columns=("origin","branch"))
                    allNeurites, continuousOrigins = DataframeCleanup.checkIfBranchIsContinuous(continuousOrigins,timePointData,allNeurites,timePointID,maxGapSize,min_nb_of_timeframes,maxTimeFrame)        
            #exclude all non continuous neurites - not done anymore, was part of earlier version 
#            allNeurites = allNeurites.loc[allNeurites.overlap == 0]
            allNeurites = allNeurites[(allNeurites.continuous == 1)].reset_index()
            return allNeurites
    
    
    
    @staticmethod        
    def checkIfBranchIsContinuous(continuousOrigins,timePointData,allNeurites,timePointID,maxGapSize,minNbOfTimeframes,maxTimeFrame):
        #fuse origin and branch to have a unique identifier
        timePointData['origin-branch'] = timePointData.loc[:,['origin','branch']].values.tolist()                    
        continuousOrigins_tmp = copy.copy(continuousOrigins)    
        for key in continuousOrigins:
            origin = key[0]
            branch = key[1]
            #test whether entry in continuousorigins is in current timepoint
            isIn = False
            for element in timePointData['origin-branch']:
                if (element[0] == origin) & (element[1] == branch):
                    isIn = True
            #if is in, add new timepoint to entry
            if isIn:
                continuousOrigins_tmp[key].append(timePointID[4])
            if ~isIn:
                #check gapsize (how many timeframes origin-branch was not present)
                lastTimeFrame = np.max(continuousOrigins_tmp[key])
                gapSize = timePointID[4] - lastTimeFrame
                #also trigger if we are in last timeframe- all neurites should then be checked for continuity
                if (gapSize > maxGapSize) | (timePointID[4] == maxTimeFrame):
                    #check if origin-branch was present for enough timeframes to be considered continuous
                    if len(continuousOrigins[key]) < minNbOfTimeframes:
                        allTimeFrames = continuousOrigins[key]
                        for timeFrameNb in allTimeFrames:
                            allNeurites.loc[(timePointID[0],timePointID[1],timePointID[2],timePointID[3],timeFrameNb,origin,branch),'continuous'] = 0
                    #remove entry since gapsize to big
                    continuousOrigins_tmp.pop(key)  
        return allNeurites, continuousOrigins_tmp
    
    @staticmethod
    def adjustStartPoints(allNeurites,radius_of_start_points,min_fraction_of_start_points_closeby):   
        #set start pointn of origin-branch for all timeframes to same point (closest to mid of neuron) & adjust length accordingly
        allNeuriteNbs = allNeurites[['date','experiment','neuron','channel','origin','branch']].drop_duplicates().reset_index()
        allNeurites.set_index(['date','experiment','neuron','channel','origin','branch','time'],inplace=True)
#        allNeurites['start'].apply(print)
        for currentNeuriteRowID in allNeuriteNbs.index.values:
            allStartpoints = []
            #get all timeframes of origin-branch
            currentNeuriteNb = allNeuriteNbs.loc[currentNeuriteRowID,['date','experiment','neuron','channel','origin','branch']]
            allTimeFramesOfNeurite = allNeurites.loc[(currentNeuriteNb[0],currentNeuriteNb[1],currentNeuriteNb[2],currentNeuriteNb[3],currentNeuriteNb[4],currentNeuriteNb[5])].reset_index()
            
            #collect all startpoints of origin-branch
            for rowIndex in allTimeFramesOfNeurite.index.values:
                oneTimeFrameOfNeuriteStart = allTimeFramesOfNeurite.loc[rowIndex,'start']
                if len(allStartpoints) == 0:
                    allStartpoints = oneTimeFrameOfNeuriteStart
                else:
                    allStartpoints = np.vstack((allStartpoints,oneTimeFrameOfNeuriteStart))
                    
            neuronMid = DataframeCleanup.stringToArray(allTimeFramesOfNeurite.loc[rowIndex,'neuronMid'])
            #type of allstartpoints will be list if only one element is present and numpy.ndarray if more than one element is present
            if type(allStartpoints) == np.ndarray:
                
                thisNeuriteClosestStart = DataframeCleanup.get_closest_common_start_point(allStartpoints,neuronMid,radius_of_start_points,min_fraction_of_start_points_closeby)

                #set start point for all neurites of same origin-branch to same point!
                for rowIndex in allTimeFramesOfNeurite.index.values:
                    oneTimeFrameOfNeuriteTime = allTimeFramesOfNeurite.loc[rowIndex,'time']
            
                    Xs = Analyzers.stringToNbArray(str(allTimeFramesOfNeurite.loc[rowIndex,'x']))
                    Ys = Analyzers.stringToNbArray(str(allTimeFramesOfNeurite.loc[rowIndex,'y']))
                    
                    #correct start point by choosing point which is closest to new start point
                    corrected_original_neurite_start = generalTools.getClosestPoint([Xs,Ys],[thisNeuriteClosestStart])
                    
                    for point_nb, x in enumerate(Xs):
                        y = Ys[point_nb]
                        if (x == corrected_original_neurite_start[0]) & (y == corrected_original_neurite_start[1]):
                            break
                    
                    #calculate how much length is lost by correcting start point
                    original_start = [Xs[0],Ys[0]]
                    less_length = distance.cdist([original_start],[corrected_original_neurite_start])[0]
                    
                    Xs = np.delete(Xs,range(0,(point_nb-1)))
                    Ys = np.delete(Ys,range(0,(point_nb-1)))
                    
                    
                    #get distance of original start and new start
                    corrected_original_neurite_start = DataframeCleanup.stringToArray(corrected_original_neurite_start)
                    additionalLength = distance.cdist([corrected_original_neurite_start],[thisNeuriteClosestStart])[0]
                    
                    #add distance to original length of neurite
                    allNeurites.loc[(currentNeuriteNb[0],currentNeuriteNb[1],currentNeuriteNb[2],currentNeuriteNb[3],currentNeuriteNb[4],currentNeuriteNb[5],oneTimeFrameOfNeuriteTime),'length'] += (additionalLength-less_length)
                    
                    #draw line between original start point and new start point
                    connectingLine = line_aa(thisNeuriteClosestStart[0],thisNeuriteClosestStart[1],corrected_original_neurite_start[0],corrected_original_neurite_start[1])
                    
                    #add all points of line to X and Y array in dataframe
                    Xs = np.hstack((connectingLine[0],Xs))
                    Ys = np.hstack((connectingLine[1],Ys))
                    
                    allNeurites.loc[(currentNeuriteNb[0],currentNeuriteNb[1],currentNeuriteNb[2],currentNeuriteNb[3],currentNeuriteNb[4],currentNeuriteNb[5],oneTimeFrameOfNeuriteTime),'start'] = "[{} {}]".format(thisNeuriteClosestStart[0],thisNeuriteClosestStart[1])
                    allNeurites.loc[(currentNeuriteNb[0],currentNeuriteNb[1],currentNeuriteNb[2],currentNeuriteNb[3],currentNeuriteNb[4],currentNeuriteNb[5],oneTimeFrameOfNeuriteTime),'x'] = str(Xs)
                    allNeurites.loc[(currentNeuriteNb[0],currentNeuriteNb[1],currentNeuriteNb[2],currentNeuriteNb[3],currentNeuriteNb[4],currentNeuriteNb[5],oneTimeFrameOfNeuriteTime),'y'] = str(Ys)
                    
        return allNeurites


    @staticmethod
    def get_closest_common_start_point(allStartpoints,neuronMid,radius_of_start_points,min_fraction_of_start_points_closeby):
        nb_of_start_points = len(allStartpoints)
        best_closest_point_found = False
        possible_start_points= []
        possible_start_points_fraction = []
        #go through points closest to mid of soma and check if in radius a large enough fraction of start points is

        while (not best_closest_point_found) & (len(allStartpoints) > 0):
            #reformat startpoints to similar array as output of np.where (prerequisite for "getClosestPoint" function)
            allStartpointsSplit = np.split(allStartpoints,2,axis=1)
            allStartPositisionsSplitX = [i[0] for i in allStartpointsSplit[0]]
            allStartPositisionsSplitY = [i[0] for i in allStartpointsSplit[1]]
            allStartpointsSplit = (allStartPositisionsSplitX,allStartPositisionsSplitY)
            thisNeuriteClosestStart = generalTools.getClosestPoint(allStartpointsSplit,[[neuronMid[0],neuronMid[1]]])
            
            #calculate fraction of start points which are in radius of current start point
            all_distances = distance.cdist([thisNeuriteClosestStart],allStartpoints)
            closeby_start_points = np.where(all_distances < radius_of_start_points)[0]
            nb_closeby_start_points = len(closeby_start_points)
            fraction_of_closeby_start_points = nb_closeby_start_points / nb_of_start_points
            far_start_points = np.where(all_distances > radius_of_start_points)[0]
            #remove start points which where in radius to not check them again in potential next step 
            allStartpoints = allStartpoints[far_start_points]
        
            #save fraction in case no start point shows high enough fraction and therefore best startpoint needs to be choosen afterwards
            possible_start_points.append(thisNeuriteClosestStart)
            possible_start_points_fraction.append(fraction_of_closeby_start_points)
            #if fraction is high enough, current start point is best start point
            if fraction_of_closeby_start_points >= min_fraction_of_start_points_closeby:
                best_closest_point_found = True
        if not best_closest_point_found:
            max_fraction = np.max(possible_start_points_fraction)
            best_start_point_nb = np.where(possible_start_points_fraction == max_fraction)[0]
            thisNeuriteClosestStart = possible_start_points[best_start_point_nb]
        return thisNeuriteClosestStart


    @staticmethod
    def stringToArray(cell):
        if type(cell) == str:
            try:
                cell = cell.replace("[","")
                cell = cell.replace("]","")
                array = []
                for i in cell.split(" "):
                    if i != "":
                        array.append(int(i))
            except:
                array = cell
        else:
            array = cell
        return array