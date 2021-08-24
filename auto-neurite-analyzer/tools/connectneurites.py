# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:06:13 2019

@author: schelskim
"""

from tools.generaltools import generalTools
from tools.sortpoints import sortPoints

import numpy as np
from skimage import morphology as morph
from scipy import ndimage
from scipy.signal import medfilt2d as median2
import copy
from scipy.spatial import distance
import math
from skimage.draw import line_aa
from skimage.morphology import disk
from matplotlib import pyplot as plt
from skimage.filters import median
from scipy.ndimage.filters import gaussian_filter as gaussian




class connectNeurites():

    @staticmethod
    def start(islands,img,timeframe,neuronMid,somaBorder,maxRelDifferenceOfLengths,minBranchSize,minContrast,maxLocalConnectionContrast,distanceToCheck,backgroundVal):
        
        cX = neuronMid[0]
        cY = neuronMid[1]
        labeledIslands,nbLabels = ndimage.label(islands,structure=[[1,1,1],[1,1,1],[1,1,1]])
        
        
#        plt.figure()
#        plt.imshow(labeledIslands)
        
        #define points of surrounding of neuronal cellBody
        #go through each label and connect
        connections = np.zeros_like(img)
        imgForBorder = gaussian(img,2)
        img = median(img,disk(2))
        while nbLabels > 1:
#            print("nb of labels:{}".format(nbLabels))
            #get first label which is neither background nor main soma
            somaLabel = labeledIslands[cY,cX]
            uniqueLabels = np.unique(labeledIslands)
            for label in uniqueLabels:
                if (label != 0) & (label != somaLabel):
                    break
                
            #create picture of current label (island) and dilate to remove points directly surrounding the island
            thisIsland = np.zeros_like(img)
            thisIsland[labeledIslands == label] = 1
            
            #-----------CROP FOR MORE SPEED---------------------
            thisIsland,borderVals = generalTools.cropImage(thisIsland,[],20)
            
            dilationOfIsland = 6
            thisIsland = ndimage.morphology.binary_dilation(thisIsland,disk(dilationOfIsland))
            
            #remove points around dilated island, to have space in which to create connection
            thisIslandForClearing = ndimage.morphology.binary_dilation(thisIsland,disk(4))
            
            #-----------UNCROP ---------------------
            thisIslandForClearing = generalTools.uncropImage(thisIslandForClearing,borderVals)
            
            islandData = np.where(thisIslandForClearing == 1)
            restTimeframe = copy.copy(islands)
            restTimeframe[islandData[0],islandData[1]] = False
            if(len(np.where(restTimeframe)[0]) == 0):
                nbLabels = 1
            else:
                
                #create border around island to select maximum intensity point in that border to start connection 
                
                islandBorder = ndimage.morphology.binary_dilation(thisIsland,disk(2))
                islandBorder[thisIsland == 1] = 0
                
                #-----------UNCROP ---------------------
                islandBorder = generalTools.uncropImage(islandBorder,borderVals)
                thisIsland = generalTools.uncropImage(thisIsland,borderVals)
                
                islandBorderData = np.where(islandBorder == 1)
                
                #go through up to 6 points, to check if there is one point that can lead to a successfull connection
                targetFound = False
#                print("-----------------NEW ISLAND---------------")
                for a in range(0,6):
                    islandBorderMaxVal = np.max(imgForBorder[islandBorderData[0],islandBorderData[1]])
                    
                    imgForMaxInBorder = copy.copy(imgForBorder)
                    imgForMaxInBorder[islandBorder != 1] = 0
                    islandBorderMaxValPos = np.where(imgForMaxInBorder == islandBorderMaxVal)
                    islandStart = [islandBorderMaxValPos[0][0],islandBorderMaxValPos[1][0]]
                    
                    #find minimal connection length by checking closest point which is not at an unusual angle from island
                    minLength = connectNeurites.getMinimalDistanceOfIsland(restTimeframe,islandStart,thisIsland,somaBorder,islandData)
                    
                    testIslands = np.zeros_like(islands)
                    testImg = copy.copy(img)
                    testImg[thisIsland == 1] = backgroundVal*0.8
                    currentPoint = islandStart
                    connectionFound = False
                    #construct connection by searching for maximum value in distance from currentpoint and going there next, repeat max. 100x
                    for a in range(0,100):
                        
                        newPoint = connectNeurites.getNextMaxPoint(currentPoint,distanceToCheck,testImg)

                        #remove all points surrounding the currentPoint
                        dilation = 2
                        for x in range(-dilation,dilation):
                            for y in range(-dilation,dilation):
                                if ((x != -dilation) | (y == 0)) & ((x != dilation) | (y == 0)) & ((x == 0) | (y != -dilation)) & ((x == 0) | (y != dilation)):
                                    testImg[newPoint[0]+x,newPoint[1]+y] = backgroundVal*0.8
                        
                        testIslands[newPoint[0],newPoint[1]] = 1
                        
                        currentPoint = newPoint
                        
                        #if new point is within resttimeframe, a potentially successfull connection was found, check if it is successfull
                        contrast = 0
                        if restTimeframe[currentPoint[0],currentPoint[1]] == 1:
                            #get maxpoint one more time to get connectedPoint
                            connectionFound = True
                            break
                    
                    if connectionFound:
                        newPoint = connectNeurites.getNextMaxPoint(currentPoint,distanceToCheck,testImg)
                        currentPoint = newPoint
                        
                        #get dilated whole island connection & skeletonized connection
                        testIslands_dil, testIslands_skel = connectNeurites.refineIslandConnection(testIslands,thisIsland,labeledIslands,label,islandStart,islands,img)
                        
                        sortedPoints = [islandStart]
                        allCoords = np.where(testIslands_skel == 1)
                        
                        sortedArray, length, allCoords = sortPoints.startSorting(allCoords,sortedPoints,0,minBranchSize,[],True,0)
                        
                        contrast = connectNeurites.getContrastOfConnection(sortedArray,0,len(sortedArray),img[(islands != 1) & (img != 0)],img)
                        connectPoints = contrast > minContrast
    #                            print("length of connection: {} / min length: {} / contrast: {}".format(length,minLength,contrast))
                        additionalLength = 10
                        if additionalLength > (minLength+dilationOfIsland):
                            additionalLength = minLength+dilationOfIsland
                        
                        maxLength = ((minLength+dilationOfIsland) * maxRelDifferenceOfLengths) + additionalLength
                        #test also if more than 5 points were excluded from the skeletonized image - indicates points going around the island to find target
                        #in that case, middle part of skeleton is deleted due to fusion with island before skeletonization and subtraction afterwards
                        if ((maxLength > length) | (connectPoints)) & (len(allCoords[0]) <= (len(sortedArray) + 5)):
                            #if length first but array is too short, dont reconsider contrast, overwrite connectpoints
                            if len(sortedArray) < 10:
                                connectPoints = True
                            else:
                                start = len(sortedArray) - 8
                                if start > 5:
                                    start = 5
                                img_forContrast = gaussian(img,2)
                                
                                #either the contrast of the whole connection or the local contrast is above set threshold value
                                contrast = connectNeurites.getContrastOfConnection(sortedArray,start,len(sortedArray)-5,img_forContrast[(islands != 1) & (img != 0)],img_forContrast)
                                connectPoints = contrast > minContrast
    #                                    print("contrast: {}".format(contrast))
                                if connectPoints:
                                    img_forContrast = img_forContrast - backgroundVal * 0.9
                                    start = len(sortedArray)-15
                                    if start < 0:
                                        start = 0
                                    localContrast = connectNeurites.getContrastOfConnection(sortedArray,start,len(sortedArray)-5,img[currentPoint[0],currentPoint[1]],img_forContrast)
                                    connectPoints = localContrast > 1/maxLocalConnectionContrast
    #                                        print("contrast: {} / localcontrast: {}".format(contrast,localContrast))
                            if connectPoints:
    #                                    print("CONNECTED!")
                                targetFound = True
                                #islands needed to be dilated before intensity based connection is drawn
                                #therefore, draw connection line from non dilated island to beginning of intensity based connection
                                islands[testIslands_dil == 1] = 1
                                connections[testIslands_dil == 1] = 1
                                
                    if targetFound:
                        break
                    else:
                        #remove current startpoint and surrounding, to iterate through next possibilities
                        islandStartImg = np.zeros_like(img)
                        islandStartImg[islandStart[0],islandStart[1]] = 1
                        islandStartImg = ndimage.morphology.binary_dilation(islandStartImg,disk(4))
                        imgForBorder[islandStartImg == 1] = 0
                    
                if not targetFound:
                    islands[thisIsland == 1] = False
                
                labeledIslands,nbLabels = ndimage.label(islands,structure=[[1,1,1],[1,1,1],[1,1,1]])
        return connections
    
    @staticmethod
    def thresholdOnImgZoom(img_thresholded,value,img,setVal,startCoords):
        newTestImg = copy.copy(img)
        newTestImg[img_thresholded == value] = setVal
        distanceToView = 50
        newImg = newTestImg[np.min(startCoords[0])-distanceToView:np.max(startCoords[0])+distanceToView,np.min(startCoords[1])-distanceToView:np.max(startCoords[1])+distanceToView]
        plt.figure()
        plt.imshow(newImg)
    

    @staticmethod
    def getPointsFromImage(image):
        points = np.where(image == 1)
        points = [points[0],points[1]]
        points = np.transpose(points)
        return points
    
    @staticmethod
    def getNextMaxPoint(currentPoint,distanceToCheck,testImg):
        xStart, xEnd, yStart, yEnd = connectNeurites.getStartAndEndCoords(currentPoint,distanceToCheck,testImg)
        maxPointVal = np.max(testImg[xStart:xEnd,yStart:yEnd])
        newPoints = np.where(testImg[xStart:xEnd,yStart:yEnd] == maxPointVal)
        newPoint = [xStart + newPoints[0][0],yStart + newPoints[1][0]]
        return newPoint
    
    @staticmethod
    def getMinimalDistanceOfIsland(restTimeframe,islandStart,thisIsland,somaBorder,islandData):
        restTimeframe_labeled,nbLabels = ndimage.label(restTimeframe,structure=[[1,1,1],[1,1,1],[1,1,1]])
        islandDataForStart = np.where(morph.skeletonize(thisIsland) == 1)
        somaBorderPoints = connectNeurites.getPointsFromImage(somaBorder)
        islandEnd = generalTools.getFurthestPoint(islandDataForStart,somaBorderPoints)
        baseAngle = connectNeurites.getAngleOfLine(islandEnd,islandStart)

        for a in range(0,4):  
            restTimeframeData = np.where(restTimeframe_labeled > 0)
            closestTimeframePoint = generalTools.getClosestPoint(restTimeframeData,[[islandStart[0],islandStart[1]]])
            closestLabel = restTimeframe_labeled[closestTimeframePoint[0],closestTimeframePoint[1]]
            #remove respective label from rest of timeframe
            restTimeframe_labeled[restTimeframe_labeled == closestLabel] = 0
            if(len(restTimeframeData[0]) == 0):
                break;
            if(len(islandData[0]) < 5):
                break;
            #stop loop if the XY ratio of the new point is not different too much from the island XYratio
            newAngle = connectNeurites.getAngleOfLine(islandStart,closestTimeframePoint)
            angleDiff = abs(newAngle-baseAngle)
            if(angleDiff > 180):
                angleDiff = 360 - angleDiff
            if(angleDiff < 45):
                break;
            break;

        #get distance of current island pipxel from cell body
        minLength = distance.cdist([islandStart],[closestTimeframePoint])[0][0] + 1
        return minLength
    
    @staticmethod
    def getStartAndEndCoords(currentPoint,distanceToCheck,testImg):
        xStart = currentPoint[0]-distanceToCheck
        xEnd = currentPoint[0]+distanceToCheck
        yStart = currentPoint[1]-distanceToCheck
        yEnd = currentPoint[1]+distanceToCheck
        if xStart < 0:
            xStart = 0
        if xEnd >= testImg.shape[0]:
            xEnd = testImg.shape[0] - 1
        if yStart < 0:
            yStart = 0
        if yEnd >= testImg.shape[1]:
            yEnd = testImg.shape[1] - 1
        return xStart, xEnd, yStart, yEnd
    
    @staticmethod
    def getContrastOfConnection(sortedArray,start,end,reference,img):
        #calculate average px intensity 
        allPxVals = []
        allconnectedpoints = np.zeros_like(img)
        for a in range(start,end):
            allPxVals.append(img[sortedArray[a][0],sortedArray[a][1]])
            allconnectedpoints[sortedArray[a][0],sortedArray[a][1]] = 1
        allconnectedpoints = ndimage.morphology.binary_dilation(allconnectedpoints,disk(2))
        averageLastPxVal = np.median(allPxVals)
        
        referenceVal = np.mean(reference)
        contrast = averageLastPxVal / referenceVal
        return contrast
    
    @staticmethod
    def refineIslandConnection(testIslands,_thisIsland,labeledIslands,label,islandStart,_islands,_img):
        
        thisIsland = copy.copy(_thisIsland)
        islands = copy.copy(_islands)
        img = copy.copy(_img)
        
        allIslandCords_pre = np.where(testIslands == 1)
        testIslandsShape = testIslands.shape
        
        #---------------CROP IMAGES FOR SPEED------------------
        testIslands,thisIsland,borderVals = generalTools.crop2Images(testIslands,thisIsland,10)
        labeledIslands,borderVals = generalTools.cropImage(labeledIslands,borderVals)
        islands,borderVals = generalTools.cropImage(islands,borderVals)
        img, borderVals = generalTools.cropImage(img,borderVals)
        
        islandStart = [islandStart[0]-borderVals[0],islandStart[1]-borderVals[2]]
        
        
        if ((np.max(allIslandCords_pre[0]) >= testIslandsShape[0]-2) | (np.min(allIslandCords_pre[0]) <= 2)) | ((np.max(allIslandCords_pre[1]) >= testIslandsShape[1]-2) | (np.min(allIslandCords_pre[1]) <= 2)):
            testIslands_closed = ndimage.morphology.binary_dilation(testIslands,disk(3))
        else:
            testIslands_closed = ndimage.morphology.binary_closing(testIslands,disk(3))
        testIslands_dil = ndimage.morphology.binary_dilation(testIslands_closed,disk(2))

        testIslands_dil[thisIsland == 1] = 1
        
        testIslands_skel = morph.skeletonize(testIslands_dil)
        testIslands_skel[thisIsland == 1] = 0
        testIslands_dil = ndimage.morphology.binary_dilation(testIslands_closed,disk(2))
        
        testIslands_dil[thisIsland == 1] = 0
        connectingLine = connectNeurites.drawLineBridgingDilationGap(labeledIslands,label,islandStart,islands,testIslands_dil,img)
        testIslands_skel[connectingLine == 1] = 1
        testIslands_skel = ndimage.morphology.binary_closing(testIslands_skel,disk(5))
        testIslands_skel = morph.skeletonize(testIslands_skel)
        
        testIslands_dil[connectingLine == 1] = 1
        
        
        #---------------UNCROP IMAGES------------------
        testIslands_dil = generalTools.uncropImage(testIslands_dil,borderVals)
        testIslands_skel = generalTools.uncropImage(testIslands_skel,borderVals)
        
        islandStart = [islandStart[0]+borderVals[0],islandStart[1]+borderVals[2]]
        
        return testIslands_dil, testIslands_skel
    
    @staticmethod
    def drawLineBridgingDilationGap(labeledIslands,label,islandStart,islands,testIslands_dil,img):
        islandData = np.where(labeledIslands == label)
        newIslandStart = generalTools.getClosestPoint(islandData,[[islandStart[0],islandStart[1]]])
#        connectionData = np.where(testIslands_dil == 1)
#        closestConnectionPoint = generalTools.getClosestPoint(connectionData,[[newIslandStart[0],newIslandStart[1]]])
        connectionDistance = distance.cdist([newIslandStart],[islandStart])[0][0]
        connectingLine = connectNeurites.drawConnectingLine(newIslandStart,islandStart,connectionDistance,islands,1)
        return connectingLine


    @staticmethod
    def drawConnectingLine(point1,point2,length,timeframe,lineWidth):
    #    print("------DRAW CONNECTING LINE------")
        distX = point1[0] - point2[0]
        distY = point1[1] - point2[1]
        if distX != 0:
            XYratio = distY/distX
        else:
            XYratio = 0
        dX = length / (math.sqrt(1+(XYratio*XYratio)))
        if XYratio != 0:
            dY = length / math.sqrt(1+(1/(XYratio*XYratio)))
        else:
            dY = length
        dX = int(round(dX))
        dY = int(round(dY))
        if distY < 0:
            endPointY = point1[1]+dY
        else:
            endPointY = point1[1]-dY
            
        if distX < 0: 
            endPointX = point1[0]+dX
        else:
            endPointX = point1[0]-dX
        
        if distY == 0:
            endPointY = point1[1]
            
        if distX == 0:
            endPointX = point1[0]
            
        endPointOfLine = [endPointX,endPointY]
        lineImage = np.zeros_like(timeframe)
        for a, coords in enumerate(point1):
            if coords > timeframe.shape[a]:
                point1[a] = timeframe.shape[a]
        for a, coords in enumerate(endPointOfLine):
            if coords > timeframe.shape[a]:
                endPointOfLine[a] = timeframe.shape[a]
        connectingLine = line_aa(point1[0],point1[1],point2[0],point2[1])
        lineImage[connectingLine[0],connectingLine[1]] = 1
        lineImage = lineImage > 0
        lineImage = ndimage.morphology.binary_dilation(lineImage,morph.square(lineWidth),iterations=1)
        
        return lineImage
    
    @staticmethod
    def getAngleOfLine(startPoint,endPoint):
        lA = endPoint[1]-startPoint[1]
        lH = math.sqrt(lA*lA+((endPoint[0]-startPoint[0])*(endPoint[0]-startPoint[0])))
        angle = math.acos(lA / lH)
        angle = math.degrees(angle)
        if endPoint[1] > startPoint[1]:
            angle = 180 - angle
        if endPoint[0] < startPoint[0]:
            angle = 360 - angle
        return angle
    
    @staticmethod
    def getDistance(point1,point2):
        _dX = point1[0] - point2[0]
        _dY = point1[1] - point2[1]
        _dist = math.sqrt( (_dX*_dX) + (_dY*_dY) )
        return _dist
        