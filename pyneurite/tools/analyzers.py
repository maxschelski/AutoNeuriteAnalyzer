# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:13:28 2019

@author: schelskim
"""
import os
import numpy as np
import pandas as pd
from skimage import io
import copy
import math
from skimage.draw import line_aa
import re


class Analyzers():
    
    @staticmethod
    def analyzeIntensities(date,experiment,neuron,neuronPath,allNeurites,channelUsedForSkel,channelsToAnalyze,allNeuritesColumns,minNeuriteLength,AW_maxIntFac):
        #go through all .csv files available...
        allNeurites.reset_index(inplace=True)
        allNeurites.set_index(['date','neuron','time'],inplace=True)
        allNeurites['maxInt'] = allNeurites.loc[:,'x']
        allNeurites['avIntArray'] = allNeurites.loc[:,'x']
        allNeurites['preAW'] = allNeurites.loc[:,'x']
        #get global median threshold for all timeframes (threshold needs to be similar for analyses of one cell!)
#        print(np.array(allNeurites['threshold']))
#        print(len(allNeurites['threshold']))
        threshold = np.median(np.array(allNeurites['threshold']))
        allTimePoints = allNeurites.reset_index()['time'].drop_duplicates()
        newAllNeurites = pd.DataFrame(columns=allNeuritesColumns)
        if os.path.isdir(neuronPath):
              channelPath = neuronPath+"\\c000"+str(channelUsedForSkel)
              channelNb = channelUsedForSkel
              print("analyze {} - {} - {}".format(date, neuron, channelNb))
              if os.path.isdir(channelPath):
                    timePoint = 0
                    for timeframeImageName in os.listdir(channelPath):
                        timePoint += 1
                        #check if the current timeframe is present in analyzed skeletons
                        if timePoint in np.array(allTimePoints):
    #                                        print(allNeurites)
                            print("{} - {} - {}: TIMEPOINT {}".format(date, neuron, channelNb, timePoint))
    #                                        print(allNeurites.loc[(date,experiment,neuron)])
                            allCurrentNeurites =  allNeurites.loc[(date,neuron,timePoint)].reset_index()                                           
                            allCurrentNeuritesArray = []
                            timeframes = []
                            
                            #first append data for skeletonized channel, then the rest
                            timeframe = io.imread(channelPath+"\\"+timeframeImageName)
                            timeframes.append(timeframe)
                            allCurrentNeuritesThisChannel = copy.copy(allCurrentNeurites)
                            allCurrentNeuritesThisChannel['channel'] = channelNb
                            allCurrentNeuritesArray.append(allCurrentNeuritesThisChannel)
                                    
                            for channel in channelsToAnalyze:
                                if channel != channelUsedForSkel:
                                    timeframeImageNameAdjusted = timeframeImageName.replace("w000"+str(channelNb),"w000"+str(channel))
                                    channelPathAdjusted = neuronPath + "\\c000"+str(channel)
                                    timeframe = io.imread(channelPathAdjusted+"\\"+timeframeImageNameAdjusted)
                                    timeframes.append(timeframe)
                                    allCurrentNeuritesThisChannel = copy.copy(allCurrentNeurites)
                                    allCurrentNeuritesThisChannel['channel'] = channel
                                    allCurrentNeuritesArray.append(allCurrentNeuritesThisChannel)
                                        
                            #temporary fix, ignores all timepoints with only one neurite due to programming difficulties
                            if len(allCurrentNeurites.columns) != 2:
                                
                                for nbOfNeurite in range(0,len(allCurrentNeurites)):
                                    Xs = Analyzers.stringToNbArray(str(allCurrentNeurites.loc[nbOfNeurite,'x']))
                                    Ys = Analyzers.stringToNbArray(str(allCurrentNeurites.loc[nbOfNeurite,'y'])) 
                                    allNeuritePoints = np.c_[Xs,Ys]
                                    if len(allNeuritePoints) > minNeuriteLength:
                                        maxIntArrays = []    
                                        avIntArrays = []    
                                        #initialize all max and average intarraays
                                        n = 0
                                        for channel in channelsToAnalyze:
                                            maxIntArrays.append([])
                                            avIntArrays.append([])
                                            n += 1
                                            
                                        for point in range(0,len(allNeuritePoints)):
                                            maxInts = Analyzers.getNeuriteIntAtPoint(allNeuritePoints,point,timeframes,20,'max',5,threshold)
                                            avInts = Analyzers.getNeuriteIntAtPoint(allNeuritePoints,point,timeframes,8,'mean',5,threshold)
                                            
                                            for n, channel in enumerate(channelsToAnalyze): 
                                                if maxInts[n] > 0:
                                                    maxIntArrays[n].append(maxInts[n])  
                                                if avInts[n] > 0:
                                                    avIntArrays[n].append(avInts[n])
                                        
                                        n = 0
#                                        actinwaveArrays = []
                                        for channel in channelsToAnalyze:
#                                            maxIntArray_copy = np.asarray(maxIntArrays[n])
#                                            neuriteMin = np.mean(maxIntArray_copy[np.argpartition(maxIntArray_copy,10)[:10]])
#                                            actinwaveArrays.append([])
#                                            for pointNb in range(0,len(maxIntArrays[0])):
#                                                if maxIntArrays[0][pointNb] > (neuriteMin*AW_maxIntFac):
#                                                    actinwaveArrays[n].append(1)
#                                                else:
#                                                    actinwaveArrays[n].append(0)
                                         
                                            allCurrentNeuritesArray[n].at[nbOfNeurite,'maxInt'] =maxIntArrays[n]
                                            allCurrentNeuritesArray[n].at[nbOfNeurite,'avIntArray'] =avIntArrays[n]
                                            allCurrentNeuritesArray[n].at[nbOfNeurite,'avInt'] =np.mean(avIntArrays[n])
#                                            allCurrentNeuritesArray[n].at[nbOfNeurite,'preAW'] =actinwaveArrays[n]
                                            allCurrentNeuritesArray[n].at[nbOfNeurite,'avMaxInt'] =np.mean(maxIntArrays)
                                            n += 1
                                        
                            
                                for allCurrentNeurites in allCurrentNeuritesArray:
                                    newAllNeurites = pd.concat([newAllNeurites,allCurrentNeurites])
        return newAllNeurites
    
    
    def get_smooth_growth(row,new_growth_data,av_size,column):
        #calculate smooth growth by averaging the last three growth rates
        return new_growth_data.iloc[int(row.name)-2:int(row.name)+1][column].mean()
    
    def remove_outlier_raw_growth(row,av_size,test):
        growth_difference = abs((row['growth'] - row['growth_raw']) / row['growth'])
        if growth_difference > (av_size*2):
            return row['growth']
        else:
            return row['growth_raw']
    
    @staticmethod
    def analyzeGrowth(date,experiment,neuron,neuronPath,data,allNeuritesColumns,minChangeInNeuriteLength):
        #calculate neurite growth at each time
        
        all_neurite_nbs = data['neurite'].drop_duplicates()
        data['growth_raw'] = np.nan
        data['growth'] = np.nan
        for neurite_nb in all_neurite_nbs.values:
            print("nb neurite: {}".format(neurite_nb))
            oneNeurite = data.loc[data['neurite'] == neurite_nb]
            baseDetails = []
            baseDetails.append(oneNeurite['date'])
            all_lengths = np.array(oneNeurite['length_um'])
            all_growth = all_lengths[1:len(all_lengths)] - all_lengths[0:len(all_lengths)-1]
            all_growth = np.insert(all_growth,0,np.nan)
            data.loc[data['neurite'] == neurite_nb,'growth_raw'] = all_growth
            av_size = 3
            #calculate 1. level smooth growth
            growth_data = data.loc[data['neurite'] == neurite_nb].reset_index()
            data.loc[data['neurite'] == neurite_nb,'growth'] = np.array(growth_data.apply(get_smooth_growth,axis=1,args=(growth_data,av_size,'growth_raw')))
            #check if smoothened growth deviates a lot from raw growth, if so remove this outlier (set raw growth as smooth growth)
            growth_data = data.loc[data['neurite'] == neurite_nb].reset_index()
            data.loc[data['neurite'] == neurite_nb,'growth_corr'] = np.array(growth_data.apply(remove_outlier_raw_growth,axis=1,args=(av_size,1)))
            growth_data = data.loc[data['neurite'] == neurite_nb].reset_index()
            data.loc[data['neurite'] == neurite_nb,'growth'] = np.array(growth_data.apply(get_smooth_growth,axis=1,args=(growth_data,av_size,'growth_corr')))
        #check for directly consecutive growth-retraction events (indicate error in algorithm)
        return data
            
        
        
    @staticmethod
    def stringToNbArray(string):    
            string = string.replace("[","").replace("]","").replace("\r\n","")
            if string.replace(",","") != string:
                array = string.split(',')
            else:
                string = re.sub(r"[\s]+",";",string)
                array = string.split(';')
            array_tmp = []
            for nb in array:
                if nb != "":
                    array_tmp.append(int(nb))
            array = array_tmp
            return array    
        
    @staticmethod
    def getNeuriteIntAtPoint(allNeuritePoints,point,timeframes,diameter,method,smoothening,threshold = 0.1):
            #smoothening is the distance of points from central point taken for perpendicular line
            if (point - smoothening) < 0:
                firstPointNb = 0
            else:
                firstPointNb = point- smoothening            
                
            
            if (point + smoothening) >= len(allNeuritePoints):
                lastPointNb = len(allNeuritePoints) - 1
            else:
                lastPointNb = point + smoothening
            
            firstPointForVector = allNeuritePoints[firstPointNb]
            secondPointForVector = allNeuritePoints[lastPointNb]
            vectorX = secondPointForVector[0] - firstPointForVector[0]
            vectorY = secondPointForVector[1] - firstPointForVector[1]
            intWidth = diameter/2
            vector = vectorX/vectorY #it should be vice versa but formulas down seem to be mixed up!
            dX = intWidth / math.sqrt((vector*vector)+1)
            dY = math.sqrt((intWidth*intWidth) - (intWidth*intWidth)/((vector*vector) + 1))
            
            if ((vectorX > 0) & (vectorY < 0)) | ((vectorX < 0) & (vectorY > 0)):
                dX = round(dX)        
                dY = round(dY)  
                endPosCwX = allNeuritePoints[point][0] - dX
                endPosCwY = allNeuritePoints[point][1] - dY
                endPosCcwX = allNeuritePoints[point][0] + dX
                endPosCcwY = allNeuritePoints[point][1] + dY
            elif ((vectorX > 0) & (vectorY > 0)) | ((vectorX < 0) & (vectorY < 0)):
                dX = round(dX)        
                dY = round(dY)  
                endPosCwX = allNeuritePoints[point][0] - dX
                endPosCwY = allNeuritePoints[point][1] + dY
                endPosCcwX = allNeuritePoints[point][0] + dX
                endPosCcwY = allNeuritePoints[point][1] - dY
    
            if vectorX == 0:
                endPosCwX = allNeuritePoints[point][0]
                endPosCwY = allNeuritePoints[point][1] + intWidth
                endPosCcwX = allNeuritePoints[point][0]
                endPosCcwY = allNeuritePoints[point][1] - intWidth 
                
            if vectorY == 0:
                endPosCwX = allNeuritePoints[point][0] + intWidth
                endPosCwY = allNeuritePoints[point][1]
                endPosCcwX = allNeuritePoints[point][0] - intWidth
                endPosCcwY = allNeuritePoints[point][1] 
            
            endPosCw = np.c_[int(endPosCwX),int(endPosCwY)][0]
            endPosCcw = np.c_[int(endPosCcwX),int(endPosCcwY)][0]
        
        
            intLine = line_aa(endPosCw[0],endPosCw[1],endPosCcw[0],endPosCcw[1])
            pointsOutOfFrame = np.where(((intLine[0]+1) > timeframes[0].shape[0]) | ((intLine[1]+1) > timeframes[0].shape[1]))
            intLineX = list(intLine[0])
            intLineY = list(intLine[1])
            for element in sorted(pointsOutOfFrame[0],reverse=True):
                intLineX.pop(element)
                intLineY.pop(element)
            lineIntensities_tmp = timeframes[0][intLineX,intLineY]
            
            lineIntensities,intLineXcorrected,intLineYcorrected = Analyzers.get_int_of_neurite_above_threshold(lineIntensities_tmp,intLineX,intLineY,threshold)
            
#            print(lineIntensities)
            Ints = []
            n=0
            for timeframe in timeframes:
                if n > 0:
                    lineIntensities = timeframes[n][intLineXcorrected,intLineYcorrected]
                if len(lineIntensities) == 0:
                    Ints.append(0)
                else:
                    if method == 'max':
                        Ints.append(lineIntensities.max())
                    elif (method == 'average') | (method == 'mean'): 
                        Ints.append(lineIntensities.mean())
                    elif (method == 'sum'):
                        Ints.append(np.sum(lineIntensities))
                n += 1
            return Ints
              
    @staticmethod
    def get_int_of_neurite_above_threshold(lineIntensities_tmp,intLineX,intLineY,threshold):
        lineIntensities = []
        intLineXcorrected = []
        intLineYcorrected = []

        nb_of_points = len(lineIntensities_tmp)
        mid_point_np = int(np.round(nb_of_points/2,0))
        #move from mid point to each site and save all int values higher than threshold until one is lower than it, then stop
        #thereby even with higher widths to look at, no two neurites close to each other are measured together (if they are separated in threshold)
        for int_nb in range(mid_point_np,nb_of_points):
            intensity = lineIntensities_tmp[int_nb]
            if intensity >= threshold:
                lineIntensities.append(intensity)
                intLineXcorrected.append(intLineX[int_nb])
                intLineYcorrected.append(intLineY[int_nb])
            else:
                break
            
        for int_nb in range(0,mid_point_np):
            int_nb = mid_point_np-int_nb
            intensity = lineIntensities_tmp[mid_point_np-int_nb]
            if intensity >= threshold:
                lineIntensities.append(intensity)
                intLineXcorrected.append(intLineX[int_nb])
                intLineYcorrected.append(intLineY[int_nb])
            else:
                break
            
        lineIntensities = np.array(lineIntensities)
        return lineIntensities, intLineXcorrected, intLineYcorrected


                                                      
    @staticmethod
    def analyzeActinWaves(allNeurites):       
              
#        allTimepoints = allNeurites.drop_duplicates('time').loc[:,'time']
        
        #---------------------------- ANALYZE ACTIN WAVES (AW) -------------------------------------------
        neurons = allNeurites.drop_duplicates(['date','neuron']).loc[:,['date','neuron']]
        for neuronNb in range(0,len(neurons)):
            neuronDetails = np.asarray(neurons.loc[neuronNb])
            neuron = allNeurites.loc[(allNeurites.date == neuronDetails[0]) & (allNeurites.neuron == neuronDetails[1])]
            origins = neuron.drop_duplicates(['origin']).reset_index().loc[:,['origin']]
            for originNb in range(0,len(origins)):
                currentOrigin = int(origins.loc[originNb])
                #get relative number of frames with this origin - smaller then 0.9, dont use!
                
                neurite = neuron.loc[neuron.origin == currentOrigin]
                timePoints = neurite.drop_duplicates(['time']).reset_index().loc[:,['time']]
                lastTimeframe = neurite.loc[neurite.time == int(timePoints.loc[0])]
                for timePointNb in range(0,len(timePoints)):
                    if(timePointNb >= 0):                
                        currentTimePoint = int(timePoints.loc[timePointNb])
                        timeframe = neurite.loc[neurite.time == currentTimePoint]
                        if len(timeframe) > 1:
                            maxLength = timeframe.loc[:,'length'].max()
                            timeframe = timeframe.loc[timeframe.length == maxLength]
                        lastTimeframe = neurite.loc[neurite.time == (currentTimePoint-1)]
                        if len(lastTimeframe) > 1:
                            maxLength = lastTimeframe.loc[:,'length'].max()
                            lastTimeframe = lastTimeframe.loc[lastTimeframe.length == maxLength]
                        AWarray = Analyzers.stringToNbArray(str(timeframe.reset_index().loc[0,'preAW']))
                        if len(lastTimeframe) > 0:
                            lastAWarray = Analyzers.stringToNbArray(str(lastTimeframe.reset_index().loc[0,'preAW']))
                            checkLastFrame = True
                        else:
                            checkLastFrame = False
                        AWstart = np.nan
                        AWend = np.nan
                        AWdistances = []
                        for pointNb in range(0,len(AWarray)):
                            AWpoint = AWarray[pointNb]
                            AW = False
                            if(np.isnan(AWstart) & AWpoint == 1):
                                AWstart = pointNb
                            if(~np.isnan(AWstart) & AWpoint == 0):
                                AWend = pointNb
                                if (AWend - AWstart) > 3:
                                    isConstantTip = False
                                    writeToArray = False
                                    if checkLastFrame:
                                        #include shifting of start position of neurite in analysis!
                                        if(AWend >= len(AWarray) - 10):
                                            isConstantTip = True
                                            distFromTip = len(AWarray) - AWstart
                                            lastFrameAWstart = len(lastAWarray) - distFromTip - 3
                                            lastFrameAWend = len(lastAWarray) - 1
                                            lastframeAWpoints = np.asarray(lastAWarray[lastFrameAWstart:lastFrameAWend])
                                            if(len(lastframeAWpoints) < 3):
                                                isConstantTip = False
                                        if ~isConstantTip:
                                            if AWstart-3 < 0:
                                                lastFrameAWstart = 0
                                            else:
                                                lastFrameAWstart = AWstart - 3
                                            if AWstart + 3 >= len(lastAWarray):
                                                lastFrameAWend = len(lastAWarray) -1
                                            else:
                                                lastFrameAWend = AWend + 3
                                            
                                            lastframeAWpoints = np.asarray(lastAWarray[lastFrameAWstart:lastFrameAWend])
                                            lastAWpoints = np.where(lastframeAWpoints > 0)[0]
        #                                    print("len")
        #                                    print(lastframeAWpoints)
        #                                    print(len(lastAWpoints))
        #                                    print(AWend)
                                            if len(lastAWpoints) < 3:
                                                writeToArray = True
                                    else:
                                        writeToArray = True
                                    if writeToArray:                        
#                                        print(currentOrigin)
#                                        print(currentTimePoint)
#                                        print(len(AWarray))
                                        AWdistances.append(AWend)
#                                        print(AWdistances)
                                AWstart = np.nan
                                AWend= np.nan
        return allNeurites
    