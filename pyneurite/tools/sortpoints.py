# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 16:52:50 2019

@author: schelskim
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:12:33 2019

@author: schelskim
"""
from .generaltools import generalTools

import numpy as np
from matplotlib import pyplot as plt
from skimage import morphology as morph
from skimage.morphology import disk
from skimage.morphology import square
import copy
from skimage.filters import median
from scipy import ndimage
from scipy.signal import argrelextrema
                

from scipy import spatial



class sortPoints():
    
    @staticmethod
    def startSorting(neuriteCoords, sortedPoints, length,minBranchSize,
                     decisivePoints = [], keepLongBranches=False,
                     branchLengthToKeep = -1,
                     distancePossible=True, constructNeurite=False, img=[],
                     groupImage_branches_labeled=[],
                     sumIntensitieGroupImg=[], groupImage=[],
                     maskToExcludeFilopodia=[],
                     minFilopodiaLength=0, maxFilopodiaLength=0,
                     minContrastFilopodia=0,
                     minAllowedContrastForFindingFilopodia=0):

        if branchLengthToKeep == -1:
            branchLengthToKeep = minBranchSize
        continueSorting = True
        navigateThroughBestLabel = False
        branches_image = np.zeros_like(groupImage)
        
        if constructNeurite:
            branches_image = copy.copy(groupImage)
            branches_image[np.where(groupImage_branches_labeled > 0)] = 0
        
        deletedSortedPoints =[]
        loop_points = []
        lengths = np.array([0])
        bestFittingLabel = 0
        perm_deleted_points = []
        while continueSorting:
            (sortedPoints,
             length,
             lengths,
             neuriteCoords,
             deletedSortedPoints,
             navigateThroughBestLabel,
             decisivePoints,
             bestFittingLabel,
             continueSorting,
             loop_points,
             perm_deleted_points) = sortPoints.sortNextPoint(neuriteCoords,
                                                             sortedPoints,
                                                             length,
                                                             lengths,
                                                             navigateThroughBestLabel,
                                                             decisivePoints,
                                                             deletedSortedPoints,
                                                             distancePossible,
                                                             constructNeurite,
                                                             img,
                                                             groupImage_branches_labeled,
                                                             sumIntensitieGroupImg,
                                                             groupImage,
                                                             bestFittingLabel,
                                                             loop_points,
                                                             perm_deleted_points)
             
            if continueSorting:
                continue

            sortedPoints = sortPoints.constructNeuriteFromSortedPoints(sortedPoints,
                                                                       deletedSortedPoints,
                                                                       constructNeurite,
                                                                       minFilopodiaLength,
                                                                       maxFilopodiaLength,
                                                                       minContrastFilopodia,
                                                                       minAllowedContrastForFindingFilopodia,
                                                                       img,
                                                                       maskToExcludeFilopodia,
                                                                       branchLengthToKeep,
                                                                       keepLongBranches,
                                                                       groupImage_branches_labeled,
                                                                       branches_image)
                
        return sortedPoints, length, lengths, neuriteCoords
    
    
    @staticmethod
    def constructNeuriteFromSortedPoints(sortedPoints,deletedSortedPoints,
                                         constructNeurite,minFilopodiaLength,
                                         maxFilopodiaLength,
                                         minContrastFilopodia,
                                         minAllowedContrastForFindingFilopodia,
                                         img,maskToExcludeFilopodia,
                                         branchLengthToKeep,keepLongBranches,
                                         groupImage_branches_labeled,
                                         branches_image):
        
        #median filter disk size for smoothing prior to calculating contrast to remove filopodia
        filterSizeSmoothing = 10
        #add sortedPoints to deleted sorted points to have all branches in one array for further operations
        deletedSortedPoints.append(sortedPoints)
        # if no new px werde added after last trial from last px, compare all branches from single px
        if (len(deletedSortedPoints) > 0):
            (longestBranch,
             deletedSortedPoints) = sortPoints.getLongestBranch(deletedSortedPoints,
                                                                constructNeurite,
                                                                minFilopodiaLength,
                                                                maxFilopodiaLength,
                                                                minContrastFilopodia,
                                                                minAllowedContrastForFindingFilopodia,
                                                                copy.copy(img),
                                                                filterSizeSmoothing,
                                                                branchLengthToKeep,
                                                                groupImage_branches_labeled,
                                                                branches_image)
            
            sortedPoints = longestBranch
            
            if keepLongBranches:
                sortedPoints = sortPoints.checkWhichBranchesToKeep(deletedSortedPoints,
                                                                   branchLengthToKeep,
                                                                   constructNeurite,
                                                                   longestBranch,
                                                                   img)
                
            deletedSortedPoints = []
        elif constructNeurite:
            sortedPoints = [sortedPoints]
        return sortedPoints
    
    
    @staticmethod
    def sortNextPoint(neuriteCoords,sortedPoints, length, lengths,
                      navigateThroughBestLabel,decisivePoints,
                      deletedSortedPoints,distancePossible,constructNeurite,
                      img,groupImage_branches_labeled,sumIntensitieGroupImg,
                      groupImage,bestFittingLabel,loop_points,
                      perm_deleted_points):

        continueSorting = True

        if navigateThroughBestLabel:
            if groupImage_branches_labeled[sortedPoints[-1][0],sortedPoints[-1][1]] != bestFittingLabel:
                bestFittingLabel = 0
                navigateThroughBestLabel = False
        
        (nearestNeighbor,
         distance,
         decisivePoints,
         isDecisivePoint,
         sortedPoints,
         neuriteCoords,
         navigateThroughBestLabel,
         bestFittingLabel,
         loop_found,
         loop_points) = sortPoints.getNextNeighbor(neuriteCoords,
                                                   copy.copy(sortedPoints),
                                                   decisivePoints,
                                                   constructNeurite,
                                                   distancePossible,
                                                   copy.copy(groupImage_branches_labeled),
                                                   sumIntensitieGroupImg,
                                                   groupImage,
                                                   navigateThroughBestLabel,
                                                   loop_points,
                                                   bestFittingLabel)
        if len(nearestNeighbor) > 0:
            #if is decisive point (only true in construct neurite), then more than one point was added already
            if isDecisivePoint:
                for a in range(0,len(distance)):
                    length += distance[a]
                    lengths = np.hstack((lengths, length))
            else:
                sortedPoints = np.vstack((sortedPoints,nearestNeighbor))
                length += distance
                lengths = np.hstack((lengths, length))
            return (sortedPoints, length, lengths, neuriteCoords,
                    deletedSortedPoints,
                    navigateThroughBestLabel, decisivePoints, bestFittingLabel,
                    continueSorting,loop_points,perm_deleted_points)

        # go on to constructing neurite when no decisive points are left
        if (len(decisivePoints) <= 0):
            continueSorting = False
            return (sortedPoints, length, lengths, neuriteCoords,
                    deletedSortedPoints,
                    navigateThroughBestLabel, decisivePoints, bestFittingLabel,
                    continueSorting,loop_points,perm_deleted_points)

        #create copy of sortedPoints to change sortedPoints in for loop without disturbing the loop
        #go through all points in sorted points from last decisive point onwards

        #if reset after loop, first remove last point so that it won't be removed from neuritecoords
        saveDeletedPoints = True
        # if there is a currently active loop point, check if decisive point is the loop point
        reset_after_loop = False
        if (len(loop_points) > 0) & constructNeurite:
            (neuriteCoords,
             reset_after_loop,
             saveDeletedPoints,
             neurite_points) = sortPoints.re_enter_points_from_loop(sortedPoints,
                                                                    decisivePoints,
                                                                    loop_points,
                                                                    neuriteCoords,
                                                                    deletedSortedPoints,
                                                                    perm_deleted_points)

        (sortedPoints,
         neuriteCoords,
         deletedSortedPoints,
         lengths) = sortPoints.deleteNeuriteCoords(neuriteCoords,
                                                               decisivePoints[-1]+1,
                                                               sortedPoints,
                                                               saveDeletedPoints,
                                                               deletedSortedPoints,
                                                   lengths=lengths)
        #add new points to permanently deleted points array (so that they are never added again to neuritecoords)
        if reset_after_loop:
            new_neurite_points = np.transpose(neuriteCoords)
            neurite_points_set = set(tuple(map(tuple,neurite_points)))
            new_neurite_points_set = set(tuple(map(tuple,new_neurite_points)))
            new_perm_deleted_points = np.array(list(neurite_points_set.difference(new_neurite_points_set)))
            for new_perm_deleted_point in new_perm_deleted_points:
                perm_deleted_points.append(new_perm_deleted_point)
#                        print("after - neurite coords: {} - deleted: {}".format(len(neuriteCoords[0]),len(deletedSortedPoints)))
        del decisivePoints[-1]

        return (sortedPoints, length, lengths, neuriteCoords,deletedSortedPoints,
                navigateThroughBestLabel, decisivePoints, bestFittingLabel,
                continueSorting,loop_points,perm_deleted_points)


    @staticmethod
    def re_enter_points_from_loop(sortedPoints,decisivePoints,loop_points,neuriteCoords,deletedSortedPoints,perm_deleted_points):
        reset_after_loop = False
        save_deleted_points = True
        neurite_points = []
        decisive_point_coords = sortedPoints[decisivePoints[-1]]
        for i, loop_point in enumerate(loop_points):
            if (loop_point[0] == decisive_point_coords[0]) & (loop_point[1] == decisive_point_coords[1]):
                if len(neuriteCoords[0]) > 1:
                    neurite_points = list(np.transpose(neuriteCoords))
                else:
                    neurite_points = [neuriteCoords[0],neuriteCoords[1]]
                #add deleted points again to neurite coords until loop point is reached
                for del_point_nb  in range(1,len(deletedSortedPoints)+1):
                    
                    del_point_nb = len(deletedSortedPoints)-del_point_nb
                    one_deleted_sorted_points = deletedSortedPoints[del_point_nb]
                    loop_point_in_array = False
                    for deleted_point in one_deleted_sorted_points:
#                                            print("{} - {}".format(deleted_point,loop_point))
                        if (deleted_point[0] == loop_point[0]) & (deleted_point[1] == loop_point[1]):
                            loop_point_in_array = True
                            break
                    if loop_point_in_array:
                        #add points to neurite points which are not in anymore
                        deleted_points_not_in_coords = generalTools.get_diff_of_two_arrays(one_deleted_sorted_points,neurite_points)
                        if len(deleted_points_not_in_coords) > 0:
                            #get difference between prev difference and perm_deleted points
                            if len(perm_deleted_points) > 0:
                                not_perm_deleted_points = generalTools.get_diff_of_two_arrays(deleted_points_not_in_coords,perm_deleted_points)
                            else:
                                not_perm_deleted_points = deleted_points_not_in_coords
                            if len(not_perm_deleted_points) > 0:
                                #concat the final difference to neurite_points
                                neurite_points = np.concatenate((not_perm_deleted_points,neurite_points))
                    else:
                        break
                neuriteCoords = np.transpose(neurite_points)
                del loop_points[i]
                save_deleted_points = False
                reset_after_loop = True
                break
        return neuriteCoords,reset_after_loop, save_deleted_points,neurite_points


    @staticmethod
    def getAllNeighborsAtMaxDistance(XYPoint,XYCoords,sortedPoints,maxDistance):
        #make first round of rough clean up with distance as max. px distance in each dimension
        neighbors_pre = np.where(((XYCoords[0] <= (XYPoint[0]+maxDistance)) & (XYCoords[0] >= (XYPoint[0]-maxDistance))) & ((XYCoords[1] <= (XYPoint[1]+maxDistance)) & (XYCoords[1] >= (XYPoint[1]-maxDistance))))          
        
        neighbors_pre = np.c_[XYCoords[0][neighbors_pre[0]],XYCoords[1][neighbors_pre[0]]]
        #make precise clean up with distance as real distance of points
        neighbors_dists = spatial.distance.cdist(neighbors_pre,[XYPoint])
        neighbors_dists = np.where(neighbors_dists <= maxDistance)
        neighbors_pre = neighbors_pre[neighbors_dists[0]]
        neighbors = []
        allNeighbors = []
        
        #create new list of neighbors, only containing neighbors not already in sorted points
        for neighbor in neighbors_pre:  
            hit = False
            closebyHit = False
            #check whether point is already in sortedpoints
            for a,point in enumerate(sortedPoints):
                if (neighbor[0] == point[0]) & (neighbor[1] == point[1]):
                    #if the hit is in the last 2 sorted points, make it a close hit
                    if (a >= len(sortedPoints)-4):
                        closebyHit = True
                    hit = True
                    break
            #as long as it is not a closeby hit, add point to allneighbor list    
            if not closebyHit:
                allNeighbors.append(neighbor)
            
            if not hit:
                neighbors.append(neighbor)
        return neighbors,allNeighbors
    
    @staticmethod
    def getNeighborsInBestLabel(neighbors,bestFittingLabel,groupImage_branches_labeled):
        neighborsInBestLabel = []
        for neighbor in neighbors:
            neighborLabel = groupImage_branches_labeled[neighbor[0],neighbor[1]]
            if neighborLabel == bestFittingLabel:
                neighborsInBestLabel.append(neighbor)
        return neighborsInBestLabel

    @staticmethod
    def getNextNeighbors(sortedPoints,neuriteCoords,groupImage_branches_labeled,loop_points):
        currentPoint = sortedPoints[-1]
        minimumDistance = 1.5
        loop_found = False
        for distanceToCheck in [minimumDistance]:
            neighbors, neighbors_inclDistantSortedPoints = sortPoints.getAllNeighborsAtMaxDistance(currentPoint,neuriteCoords,sortedPoints,distanceToCheck)
            
            #if there were distant points which were only included in "neighbors", a loop was formed
            #stop neighbor search, reset sortedPoints
            if len(neighbors) != len(neighbors_inclDistantSortedPoints):
                loop_found = True
                for neighbor in neighbors_inclDistantSortedPoints:
                    neighbor_in_non_sorted_neighbors = False
                    for non_sorted_neighbor in neighbors:
                        if (neighbor[0] == non_sorted_neighbor[0]) & (neighbor[1] == non_sorted_neighbor[1]):
                            neighbor_in_non_sorted_neighbors = True
                    if not neighbor_in_non_sorted_neighbors:
                        loop_points.append(neighbor)
                        break
                neighbors = []
                break
                
            if (len(neighbors) > 0):
                break
                
        return neighbors, loop_found, loop_points

    @staticmethod
    def getNextNeighbor(neuriteCoords,sortedPoints,decisivePoints,
                        constructNeurite,distancePossible,
                        groupImage_branches_labeled,sumIntensitieGroupImg,
                        groupImage,navigateThroughBestLabel,loop_points,
                        bestFittingLabel=0):
            
        neighbors,loop_found,loop_points  = sortPoints.getNextNeighbors(sortedPoints,neuriteCoords,groupImage_branches_labeled,loop_points)        
            
        isDecisivePoint = False        
        nearestNeighbor = []
        distance = 0
        if len(neighbors) > 0:
            minDistance = 10
            nearestNeighbor, distance = sortPoints.getNearestNeighbor(copy.copy(neighbors),constructNeurite,sortedPoints,minDistance,navigateThroughBestLabel,bestFittingLabel,groupImage_branches_labeled)

        if not(((len(neighbors) > 1) & (len(nearestNeighbor) > 0)) | (constructNeurite & (len(nearestNeighbor) > 0))):
                return (nearestNeighbor,distance,decisivePoints,isDecisivePoint,
                        sortedPoints, neuriteCoords,navigateThroughBestLabel,
                        bestFittingLabel,loop_found, loop_points)
             
        if not constructNeurite:
            decisivePoints = sortPoints.appendDecisivePoint(decisivePoints,
                                                            (len(sortedPoints)-1))

            return (nearestNeighbor,distance,decisivePoints,isDecisivePoint,
                    sortedPoints, neuriteCoords,navigateThroughBestLabel,
                    bestFittingLabel,loop_found, loop_points)

        if(groupImage_branches_labeled[nearestNeighbor[0],nearestNeighbor[1]] != 0):
            if (len(neighbors) > 1):
                decisivePoints = sortPoints.appendDecisivePoint(decisivePoints,(len(sortedPoints)-1))
            return (nearestNeighbor,distance,decisivePoints,isDecisivePoint,
                    sortedPoints, neuriteCoords,navigateThroughBestLabel,
                    bestFittingLabel,loop_found, loop_points)

        #set all parts of labeled branches as 0 which are not included in set of points anymore (were deleted from neuritecoords during reset to last decisive point)
        neuriteCoordsImage = np.zeros_like(groupImage_branches_labeled)
        neuriteCoordsImage[neuriteCoords[0],neuriteCoords[1]] = 1
        groupImage_branches_labeled[neuriteCoordsImage != 1] = 0

        groupImage_branches_labeled_tmp = groupImage_branches_labeled.astype(bool)
        groupImage_branches_labeled_tmp = morph.remove_small_objects(groupImage_branches_labeled_tmp,min_size=2,connectivity=2)
        groupImage_branches_labeled[groupImage_branches_labeled_tmp == 0] = 0

        #delete parts in neurite coords image that were deleted in labeled branches
        neuriteCoordsImage[(groupImage_branches_labeled_tmp == 0) & (groupImage_branches_labeled > 0)] = 0
        neuriteCoords = np.where(neuriteCoordsImage == 1)

        groupImage_branches_labeled = groupImage_branches_labeled_tmp

        labelsAroundPoint, branchPointImage = sortPoints.getLabelsAroundPoint(groupImage_branches_labeled,neuriteCoords,copy.copy(nearestNeighbor),groupImage)
        nbOfLabelsAroundPoint = len(labelsAroundPoint)


        if nbOfLabelsAroundPoint <= 3:
            if (len(neighbors) > 1):
                decisivePoints = sortPoints.appendDecisivePoint(decisivePoints,(len(sortedPoints)-1))

            return (nearestNeighbor,distance,decisivePoints,isDecisivePoint,
                    sortedPoints, neuriteCoords,navigateThroughBestLabel,
                    bestFittingLabel,loop_found, loop_points)

        labelsSorted = groupImage_branches_labeled[np.transpose(sortedPoints)[0],np.transpose(sortedPoints)[1]]
        anyLabelNotSorted, nbLabelsSorted, labelsSorted = sortPoints.countLabelsAlreadySorted(labelsAroundPoint,labelsSorted)

        if nbLabelsSorted > 1:

            nearestNeighbor = []
            if len(decisivePoints) > 0:

                (sortedPoints,
                 neuriteCoords,
                 _, _) = sortPoints.deleteNeuriteCoords(neuriteCoords,
                                                        decisivePoints[-1]+1,
                                                        sortedPoints,False,[])
                del decisivePoints[-1]

            return (nearestNeighbor, distance, decisivePoints, isDecisivePoint,
                    sortedPoints, neuriteCoords, navigateThroughBestLabel,
                    bestFittingLabel, loop_found, loop_points)

        if not (anyLabelNotSorted):
            return (nearestNeighbor, distance, decisivePoints, isDecisivePoint,
                    sortedPoints, neuriteCoords, navigateThroughBestLabel,
                    bestFittingLabel, loop_found, loop_points)

        distances = []
        isDecisivePoint = True

        (distances,
         sortedPoints,
         decisivePoints) = sortPoints.appendParamsOfNearestNeighbor(distances,
                                                                    distance,
                                                                    nearestNeighbor,
                                                                    sortedPoints,
                                                                    decisivePoints)
        #for decisivePoint, check which label fits better (better average sum of intensities at each point)
        (bestFittingLabel,
         wrongLabels) = sortPoints.findBestFittingLabel(sortedPoints,
                                                        groupImage_branches_labeled,
                                                        sumIntensitieGroupImg,
                                                        labelsAroundPoint,
                                                        labelsSorted,
                                                        branchPointImage,
                                                        neuriteCoordsImage)

        inTransitToBestLabel = True
        branchPointCoords = np.where(branchPointImage == 1)
        firstDevisivePointIndex = len(decisivePoints)

        foundCorrectPoint = False
        #traverse through branchPoint, finding the route to bestfittinglabel
        while inTransitToBestLabel:
            (distances,
             sortedPoints,
             decisivePoints,
             branchPointCoords,
             inTransitToBestLabel,
             foundCorrectPoint) = sortPoints.navigateThroughBranchPoint(wrongLabels,
                                                                        bestFittingLabel,
                                                                        groupImage_branches_labeled,
                                                                        distances,
                                                                        sortedPoints,
                                                                        decisivePoints,
                                                                        branchPointCoords,
                                                                        firstDevisivePointIndex,
                                                                        foundCorrectPoint)
            #initiate that two more px of bestfitting label will be added (to prevent going back into branchpoint, causing eternal loop)

            if foundCorrectPoint:
                navigateThroughBestLabel = True
                inTransitToBestLabel = False

        distance = distances
        nearestNeighbor = sortedPoints[-1]

        return (nearestNeighbor,distance,decisivePoints,isDecisivePoint,
                sortedPoints, neuriteCoords,navigateThroughBestLabel,
                bestFittingLabel,loop_found, loop_points)

    @staticmethod
    def appendDecisivePoint(decisivePoints,newDecisivePoint):
        if newDecisivePoint not in decisivePoints:
            decisivePoints.append(newDecisivePoint)
        return decisivePoints

    @staticmethod
    def appendParamsOfNearestNeighbor(distances,distance,nearestNeighbor,
                                      sortedPoints,decisivePoints):
        distances.append(distance)
        decisivePoints = sortPoints.appendDecisivePoint(decisivePoints,
                                                        (len(sortedPoints)-1))
        sortedPoints = np.vstack((sortedPoints,nearestNeighbor))
        return distances,  sortedPoints, decisivePoints

    @staticmethod
    def getNextDecisivePointInBranchPoint(branchPointCoords,sortedPoints,
                                          decisivePoints):
            
        #get the last added decisive point which is part of the current branchpoint
        branchPointPoints = np.c_[branchPointCoords[0],branchPointCoords[1]]
        nextDecisivePointNb = np.nan
        for a, decisivePoint in enumerate(decisivePoints):
            decisivePoint = sortedPoints[decisivePoint]
            for branchPointPoint in branchPointPoints:
               if ((decisivePoint[0] == branchPointPoint[0]) &
                       (decisivePoint[1] == branchPointPoint[1])):
                   nextDecisivePointNb = a
                   break
        return nextDecisivePointNb

    @staticmethod
    def navigateThroughBranchPoint(wrongLabels,bestFittingLabel,
                                   groupImage_branches_labeled,distances,
                                   sortedPoints, decisivePoints,
                                   branchPointCoords,firstDevisivePointIndex,
                                   foundCorrectPoint):

        #coords are only branchpoint coords
        #getneighbors -> getnearestneighbor -> if label in best: break / if label in wrong: go back
        currentPoint = sortedPoints[-1]
        
        inTransitToBestLabel = True
        (neighbors,
         neighbors_inclDistantSortedPoints) = sortPoints.getAllNeighborsAtMaxDistance(currentPoint,
                                                                                      branchPointCoords,
                                                                                      sortedPoints,
                                                                                      1.5)
        
        if len(neighbors) == 0:
            
            nextDecisivePointNb = sortPoints.getNextDecisivePointInBranchPoint(branchPointCoords,
                                                                               sortedPoints,
                                                                               decisivePoints)
            
            if not np.isnan(nextDecisivePointNb):
                #reset to last decisive point
                (decisivePoints,
                 distances,
                 sortedPoints,
                 branchPointCoords,
                 deletedSortedPoints) = sortPoints.resetBranchPointNavigationToLastDecisivePoint(decisivePoints,
                                                                                                 nextDecisivePointNb,
                                                                                                 branchPointCoords,
                                                                                                 sortedPoints,
                                                                                                 distances)
                currentPoint = sortedPoints[-1]
                #if branchpoint was entered from the edge of the best label, eventually sorted points will reset to that point on the edge
                if groupImage_branches_labeled[currentPoint[0],currentPoint[1]] == bestFittingLabel:
                    foundCorrectPoint = True
            else:
                inTransitToBestLabel = False
        else:
            nearestNeighbor, distance = sortPoints.getNearestNeighbor(neighbors,True,sortedPoints)
            
            #go back: go to last decisive point, delete all point until then from sorted and coords, start again
            
            distance = spatial.distance.pdist([nearestNeighbor,currentPoint])[0]
            (distances,
             sortedPoints,
             decisivePoints) = sortPoints.appendParamsOfNearestNeighbor(distances,
                                                                        distance,
                                                                        nearestNeighbor,
                                                                        sortedPoints,
                                                                        decisivePoints)
#            print("current label: {}".format(groupImage_branches_labeled[nearestNeighbor[0],nearestNeighbor[1]]))
            if groupImage_branches_labeled[nearestNeighbor[0],nearestNeighbor[1]] == bestFittingLabel:
                #if nearestneighbor is in bestFittinglabel, stop loop
                foundCorrectPoint = True
            elif groupImage_branches_labeled[nearestNeighbor[0],nearestNeighbor[1]] in wrongLabels:
                #if in wrong label, remove wrong label point
                #also clean distances from list!!

                (decisivePoints,distances,
                 sortedPoints,
                 branchPointCoords,
                 deletedSortedPoints)  = sortPoints.resetBranchPointNavigationToLastDecisivePoint(decisivePoints,
                                                                                                  -1,
                                                                                                  branchPointCoords,
                                                                                                  sortedPoints,
                                                                                                  distances)
                
        return distances, sortedPoints, decisivePoints, branchPointCoords, inTransitToBestLabel,foundCorrectPoint

    def resetBranchPointNavigationToLastDecisivePoint(decisivePoints,decisivePointNb,branchPointCoords,sortedPoints,distances):
        firstPointToDelete = decisivePoints[decisivePointNb]+1
        #remove distances of points which will be deleted from distances arrays
        for b in range(firstPointToDelete,len(sortedPoints)):
            if len(distances) > 0:
                del distances[-1]
        #reset to last decisive point
        del decisivePoints[decisivePointNb]
        #check all other decisive points and delete all of them at least as high as current reset point
        for i in range(0,len(decisivePoints)):
            if decisivePoints[i] >= (firstPointToDelete-1):
                del decisivePoints[i]
            
        (sortedPoints,
         branchPointCoords,
         deletedSortedPoints,
         _) = sortPoints.deleteNeuriteCoords(branchPointCoords,
                                             firstPointToDelete, sortedPoints,
                                             saveDeletedPoints=False,
                                             deletedSortedPoints=[])
        return (decisivePoints,distances, sortedPoints, branchPointCoords,
                deletedSortedPoints)

    @staticmethod
    def getNearestNeighbor(neighbors,constructNeurite,sortedPoints,lowestDistance=10,navigateThroughBestLabel=False,bestFittingLabel=0,groupImage_branches_labeled=[]):
        currentPoint = sortedPoints[-1]
        nearestNeighbor = []
        if (navigateThroughBestLabel):
            neighbors_tmp = sortPoints.getNeighborsInBestLabel(neighbors,bestFittingLabel,groupImage_branches_labeled)
            if len(neighbors_tmp) > 0:
                neighbors = neighbors_tmp
        for neighbor in neighbors:     
            currentDistance = spatial.distance.pdist([neighbor,currentPoint])[0]
            #if neurite is constructed, check for neighbor with most similar px intensity
            if currentDistance < lowestDistance:
                lowestDistance = currentDistance
                nearestNeighbor = neighbor
        return nearestNeighbor, lowestDistance


    @staticmethod 
    def countLabelsAlreadySorted(labelsAroundPoint,labelsSorted):
        anyLabelNotSorted = False
        nbLabelsSorted = 0
        labelsSorted_noSmallLabels = []
        for labelAroundPoint in labelsAroundPoint:
            if labelAroundPoint != 0:
                labelIsInSorted = False
                for labelSorted in np.unique(labelsSorted):
                    nbOfPointsWithLabel = len(np.where(labelsSorted == labelSorted)[0])
                    if nbOfPointsWithLabel > 1:
                        labelsSorted_noSmallLabels.append(labelSorted)
                        if labelAroundPoint == labelSorted:
                            labelIsInSorted = True
                            break
                if not labelIsInSorted:
                    anyLabelNotSorted = True
                if labelIsInSorted:
                    nbLabelsSorted += 1
        return anyLabelNotSorted, nbLabelsSorted, labelsSorted_noSmallLabels



    @staticmethod
    def getLabelsAroundPoint(groupImage_branches_labeled,neuriteCoords,nearestNeighbor,groupImage):
        
        neuriteCoordsImg = np.zeros_like(groupImage_branches_labeled)
        neuriteCoordsImg[neuriteCoords[0],neuriteCoords[1]] = 1
        
        
        #CROP FOR SPEED
        neuriteCoordsImg,borderVals = generalTools.cropImage(neuriteCoordsImg,[],10)
        groupImage_branches_labeled, borderVals = generalTools.cropImage(groupImage_branches_labeled,
                                                                         borderVals)
        groupImage,borderVals = generalTools.cropImage(groupImage,borderVals)
        
        nearestNeighbor[0] = nearestNeighbor[0] - (borderVals[0])
        nearestNeighbor[1] = nearestNeighbor[1] - (borderVals[2])
        
        imageOfBranchPoints = np.zeros_like(groupImage_branches_labeled)
        
        imageOfBranchPoints[(groupImage == 1) &
                            (groupImage_branches_labeled == 0) &
                            (neuriteCoordsImg == 1)] = 1
        imageOfBranchPoints_labeled,nbBranchPoints = ndimage.label(imageOfBranchPoints,
                                                                   structure=[[1,1,1],[1,1,1],[1,1,1]])
        
        currentBranchPointLabel = imageOfBranchPoints_labeled[nearestNeighbor[0],nearestNeighbor[1]]
        currentBranchPointImage = np.zeros_like(groupImage_branches_labeled)
        currentBranchPointImage[imageOfBranchPoints_labeled == currentBranchPointLabel] = 1

        currentBranchPointImage = sortPoints.enlargeBranchPointByOnePointEachDirection(currentBranchPointImage,
                                                                                       nearestNeighbor,
                                                                                       groupImage)
            
        labelsAroundPoint = np.unique(groupImage_branches_labeled[currentBranchPointImage == 1])
        
        #UNCROP
        currentBranchPointImage = generalTools.uncropImage(currentBranchPointImage,borderVals)
        
        return labelsAroundPoint, currentBranchPointImage



    @staticmethod
    def enlargeBranchPointByOnePointEachDirection(currentBranchPointImage,nearestNeighbor,groupImage):
        
        #CROP MORE FOR HIGHER SPEED
        currentBranchPointImage,borderVals2 = generalTools.cropImage(currentBranchPointImage,[],3)
        groupImage,borderVals2 = generalTools.cropImage(groupImage,borderVals2)
        
        nearestNeighbor[0] = nearestNeighbor[0] - (borderVals2[0])
        nearestNeighbor[1] = nearestNeighbor[1] - (borderVals2[2])

        currentBranchPointImage_dilated = ndimage.morphology.binary_dilation(currentBranchPointImage,square(3))        
        
        currentBranchPointImage_tmp = np.zeros_like(currentBranchPointImage)
        #exclude parts of branchPoint image which are not included in set of possible points anymore (neuritecoordsImg) (deleted during reset)
#           currentBranchPointImage[(currentBranchPointImage_dilated == 1) & (groupImage == 1) & (neuriteCoordsImg == 1)] = 1
        currentBranchPointImage_tmp[(currentBranchPointImage_dilated == 1) & (groupImage == 1)] = 1
    
        currentBranchPointImage_labeled,nbOfIslandsInBranchPoint = ndimage.label(currentBranchPointImage_tmp,
                                                                                 structure=[[1,1,1],[1,1,1],[1,1,1]])
        currentBranchPointImage_tmp[currentBranchPointImage_labeled != currentBranchPointImage_labeled[nearestNeighbor[0],
                                                                                                       nearestNeighbor[1]]] = 0

        #get area of current branchpoint that was added by dilation        
        currentBranchPointImage_addedArea = np.zeros_like(currentBranchPointImage)
        currentBranchPointImage_addedArea[(currentBranchPointImage == 0) &
                                          (currentBranchPointImage_tmp == 1)] = 1
        currentBranchPointImage_addedArea_labeled,nbLabels = ndimage.label(currentBranchPointImage_addedArea,
                                                                           structure=[[1,1,1],[1,1,1],[1,1,1]])
        
        currentBranchPointImage_final = copy.copy(currentBranchPointImage)
        
        #go through each label of the added area, only add point to branchpointimage that is closest to branchpoint 
        branchPoint_coords = np.where(currentBranchPointImage == 1)
        for label in np.unique(currentBranchPointImage_addedArea_labeled):
            if label != 0:
                oneLabel_coords = np.where(currentBranchPointImage_addedArea_labeled == label)
                closestOneLabelPoint = generalTools.getClosestPoint(oneLabel_coords,np.transpose(branchPoint_coords))
                currentBranchPointImage_final[closestOneLabelPoint[0],closestOneLabelPoint[1]] = 1

        currentBranchPointImage_final = generalTools.uncropImage(currentBranchPointImage_final,borderVals2)
                
        return currentBranchPointImage_final
                    



    @staticmethod
    def findBestFittingLabel(sortedPoints,groupImage_branches_labeled,
                             sumIntensitieGroupImg,labelsAroundPoint,
                             labelsSorted,branchPointImage,neuriteCoordsImage):
        #check several of the last points for the first one to not be background to find starting label
        nbOfConsecutiveLabels = 0
        for a in range(1,21):
            if a > len(sortedPoints):
                break
            else:
                startingLabel = groupImage_branches_labeled[sortedPoints[-a][0],
                                                            sortedPoints[-a][1]]
                if startingLabel != 0:
                    nbOfConsecutiveLabels += 1
                    if nbOfConsecutiveLabels > 1:
                        break
        if len(sortedPoints) < 20:
            start = len(sortedPoints)
        else:
            start = 20
        lastSortedPoints_coords = np.transpose(sortedPoints[-start:-1])
        startingLabelIntensity = np.median(sumIntensitieGroupImg[lastSortedPoints_coords[0],
                                                                 lastSortedPoints_coords[1]])
        
        startingLabelImage = np.zeros_like(groupImage_branches_labeled)
        startingLabelImage[groupImage_branches_labeled == startingLabel] = 1

        bestIntensityDifference = np.nan
        bestFittingLabel = np.nan
        for label in labelsAroundPoint:
            bestLabel = False
            #dont include background label
            if label not in labelsSorted:
                if (label != 0) & (label != startingLabel):
                    neighborLabelImage = np.zeros_like(groupImage_branches_labeled)
                    neighborLabelImage[groupImage_branches_labeled == label] = 1
                    minSizeOfBranch = 20
                    minSize = minSizeOfBranch
                    neighborLabelImage = sortPoints.enlargeBranch(neighborLabelImage,
                                                                  copy.copy(branchPointImage),
                                                                  copy.copy(neuriteCoordsImage),
                                                                  copy.copy(sumIntensitieGroupImg),minSize)
#                    generalTools.showThresholdOnImg(neighborLabelImage,sumIntensitieGroupImg,1)
                    neighborLabelIntensity = np.median(sumIntensitieGroupImg[neighborLabelImage == 1])
                    if neighborLabelIntensity > startingLabelIntensity:
                        intensityDifference = (neighborLabelIntensity/startingLabelIntensity)-1
                    else:
                        intensityDifference = (startingLabelIntensity/neighborLabelIntensity)-1
                    if np.isnan(bestIntensityDifference):
                        bestLabel = True
                    elif intensityDifference < bestIntensityDifference:
                        bestLabel = True

                    if bestLabel:
                        bestIntensityDifference = intensityDifference
                        bestFittingLabel = label
        wrongLabels = []
        for label in labelsAroundPoint:
            if (label != 0) & (label != startingLabel): 
                if (label != bestFittingLabel):
                    wrongLabels.append(label)
        
        return bestFittingLabel, wrongLabels


    @staticmethod
    def enlargeBranch(branchImage,branchPointImage,neuriteCoordsImage,
                      intensityImage,minSize,pointSource=False):
        #if startingLabel length is < 20, increase size by enlarging it backwards
        lengthOfBranch = len(np.where(branchImage == 1)[0])
        if lengthOfBranch < minSize:
            branchImage,borderVals = generalTools.cropImage(branchImage,[],minSize+3)
            branchPointImage,borderVals = generalTools.cropImage(branchPointImage,borderVals)
            neuriteCoordsImage,borderVals = generalTools.cropImage(neuriteCoordsImage,borderVals)
            intensityImage,borderVals = generalTools.cropImage(intensityImage,borderVals)
            if not pointSource:
                branchPointImage = ndimage.morphology.binary_dilation(branchPointImage,disk(2))
            originalBranchImage = copy.copy(branchImage)
            lastBranchImage = copy.copy(branchImage)
            
            for a in range(lengthOfBranch,minSize):
                branchImage_dil = ndimage.morphology.binary_dilation(lastBranchImage,disk(2))
                branchImage_dil[neuriteCoordsImage == 0] = 0
                branchImage_dil[(branchPointImage == 1) & (originalBranchImage == 0)] = 0
                branchImageDifference = np.zeros_like(branchImage)
#                if len(np.where(branchImage_dil==1)[0]) < 10:
#                    generalTools.showThresholdOnImg(branchImage_dil,intensityImage)
#                    generalTools.showThresholdOnImg(branchImage_dil,branchPointImage)
#                    plt.figure()
#                    plt.imshow(branchPointImage)
                
                #only add one point to lastbranchimage - the one with the closest intensity compared to average
                branchImageDifference[(branchImage_dil == 1) & (lastBranchImage == 0)] = 1
                branchImageDifference_coords = np.where(branchImageDifference == 1)
                if len(branchImageDifference_coords[0]) == 0:
                    break
                averageBranchIntensity = np.median(intensityImage[lastBranchImage == 1])
                #select only one px to use, use the px with the intensity level closest to current branch intensity lelel
                bestPointNb = np.nan
                lowestIntensityDifference = np.nan
                for a in range(0,len(branchImageDifference_coords[0])):
                    pointIsBest = False 
                    intensityDifference= abs(intensityImage[branchImageDifference_coords[0][a],
                                                            branchImageDifference_coords[1][a]]-averageBranchIntensity)
                    if np.isnan(lowestIntensityDifference):
                        pointIsBest = True
                    elif intensityDifference < lowestIntensityDifference:
                        pointIsBest = True
                    if pointIsBest:
                        lowestIntensityDifference = intensityDifference
                        bestPointNb = a
                bestPoint = [branchImageDifference_coords[0][bestPointNb],branchImageDifference_coords[1][bestPointNb]]
                branchImage[bestPoint[0],bestPoint[1]] = 1
                
                lastBranchImage = branchImage
                
            branchImage = generalTools.uncropImage(branchImage,borderVals)
        return branchImage


    @staticmethod
    def getLongestBranch(deletedSortedPoints,constructNeurite,
                         minFilopodiaLength,maxFilopodiaLength,
                         minContrastFilopodia,minAllowedContrast,
                         img,filterSizeSmoothing,branchLengthToKeep,
                         groupImage_branches_labeled,branches_image):
        #if already deleted points are more than new points after fallback, delete new points, add old poinds again
        longestBranchSize = 0
        longestBranch = []
        if constructNeurite:
            
            img = median(img,disk(filterSizeSmoothing-6))
        
        #remove all branches which are smaller than the minimum allowed length        
        if constructNeurite:
            deleted_sorted_points_no_short_branches = []

            for oneBranch in deletedSortedPoints:
                if len(oneBranch) > branchLengthToKeep:
                    deleted_sorted_points_no_short_branches.append(oneBranch)
            
            deletedSortedPoints = deleted_sorted_points_no_short_branches
        
        for branchNb, oneBranch in enumerate(deletedSortedPoints):
            
            #if neurite is constructed, check if tip of neurite needs to be removed due to low contrast (filopodia like)
            if constructNeurite:
                
                oneBranch = sortPoints.removeFilopodia(oneBranch,
                                                       minFilopodiaLength,
                                                       maxFilopodiaLength,
                                                       minContrastFilopodia,
                                                       minAllowedContrast,
                                                       copy.copy(img),
                                                       groupImage_branches_labeled,
                                                       branches_image)
                
                #remove wrong start points
#                oneBranch = sortPoints.removeFilopodia(oneBranch,-minFilopodiaLength,-maxFilopodiaLength,minContrastFilopodia,minAllowedContrast,copy.copy(img),maskToExcludeFilopodia,15)
               
                if len(oneBranch) == 2:
                    deletedSortedPoints[branchNb] = oneBranch[0]
                    if len(oneBranch[1]) > branchLengthToKeep:
                        deletedSortedPoints.append(oneBranch[1])
                else:
                    deletedSortedPoints[branchNb] = oneBranch
            if len(oneBranch) > longestBranchSize:
                if len(oneBranch) == 2:
                    oneBranch = oneBranch[0]
                longestBranchSize = len(oneBranch)
                longestBranch = oneBranch
                
        return longestBranch, deletedSortedPoints
        

    @staticmethod
    def removeFilopodia(oneBranch,minFilopodiaLength,maxFilopodiaLength,
                        minContrastFilopodia,minAllowedContrast,img,
                        groupImage_branches_labeled,branches_image,
                        additionalNbOfPxToDelete=0):
        
        oneBranch_tmp = copy.copy(oneBranch)
        start = minFilopodiaLength
        if start == 0:
            start = 1
        stop = maxFilopodiaLength
        nbToCheckBefore = 10
        #max subsequent number of px not recognized as filopodia, after finiding one filopodia
        maxNbOfSubsequentPxBelowThreshBetween = 20
        neuriteTipRemoved = False
        #initialize nb of px lower than abs lowest contrast allowed
        nbSubsequentLowContrast = 0
        nbSubsequentLowerContrastAfterTip = 0
        allContrast = []
        allPointsForContrast = []
        allIntensities = []
        minNbSubsequentLowerContrastOfTip = 3
        nbSubsequentLowerContrastOfTip = 0
        minContrastFilopodiaToUse = minContrastFilopodia
        for a in np.linspace(start,stop,abs(stop-start+1)):
            a = int(a)
            if len(oneBranch) > (abs(a)+nbToCheckBefore):
                contrast = sortPoints.getContrastOfTip(oneBranch,img,a,nbToCheckBefore)
                allContrast.append(contrast)
                allIntensities.append(img[oneBranch[-a][0],oneBranch[-a][1]])
                allPointsForContrast.append(-a)
                
                #increase threshold to identify filopodia (2 fold) 
                #if five iterations in a row, contrast was lower than allowed contrast 
                #(indicates that there is no filopodia at the tip)
                if contrast < minAllowedContrast:
                    nbSubsequentLowContrast += 1
                    if nbSubsequentLowContrast == 5:
                        minContrastFilopodiaToUse = minContrastFilopodia * 2
                else:
                    nbSubsequentLowContrast = 0
                #stop loop once the tip has been removed and the maximum extend of the piece to be removed has been reached
                #meaning once the contrast gets higher again 3 subsequent times, stop the loop
                if neuriteTipRemoved & (contrast <= minContrastFilopodia):
                    nbSubsequentLowerContrastAfterTip += 1
                    if nbSubsequentLowerContrastAfterTip == maxNbOfSubsequentPxBelowThreshBetween:
                        minContrastFilopodiaToUse = minContrastFilopodia * 2
                else:
                    nbSubsequentLowerContrastAfterTip = 0
                        
                if contrast > minContrastFilopodia:
#                    #if contrast is noticeable higher than minimum, only make one more point above contrast threshold necessary
                    if contrast > (minContrastFilopodia*1.4):
                        nbSubsequentLowerContrastOfTip += minNbSubsequentLowerContrastOfTip-1
                    else:
                        nbSubsequentLowerContrastOfTip += 1
                    if nbSubsequentLowerContrastOfTip >= minNbSubsequentLowerContrastOfTip:
                        #if neurite tip is removed (again), enough px with new mincontrast (2x intensity) were removed as well, use new mincontrast now for deleting filopodia
                        minContrastFilopodia = minContrastFilopodiaToUse
                        neuriteTipRemoved = True
                        
                else:
                    nbSubsequentLowerContrastOfTip= 0
        if len(allContrast) > 2:
            oneBranchImg = np.zeros_like(img)
            oneBranchImg[np.transpose(oneBranch)[0],np.transpose(oneBranch)[1]] = 1
            maxContrastNb = sortPoints.getFurthestLocalContrastMaximum(allContrast,minContrastFilopodia,minNbSubsequentLowerContrastOfTip,allPointsForContrast,oneBranch,img,groupImage_branches_labeled,branches_image)    
            
            if not np.isnan(maxContrastNb):
                if minFilopodiaLength >= 0:
                    firstNbToDelete = len(oneBranch)-(maxContrastNb+start+additionalNbOfPxToDelete)
                    lastNbToDelete = len(oneBranch)
                else:
                    firstNbToDelete = 0
                    lastNbToDelete = maxContrastNb+additionalNbOfPxToDelete
                    
                oneBranch_tmp = np.delete(oneBranch,range(firstNbToDelete,lastNbToDelete),axis = 0)

        if len(oneBranch) != len(oneBranch_tmp):
            oneBranch = [oneBranch, oneBranch_tmp]

        return oneBranch


    @staticmethod
    def getFurthestLocalContrastMaximum(allContrast,minContrastFilopodia,minNbSubsequentLowerContrastOfTip,allPointsForContrast,oneBranch,img,groupImage_branches_labeled,branches_image):
        #maximum size of high intensity areas in neurite - can be growth cone or can be in neurite (blebb-like)
        maximumContrastNb = np.nan
        #get all local maxima in all contrasts
        maximaOfContrast = argrelextrema(np.array(allContrast),np.greater)[0]
        #check which of the local maxima are bigger than necessary value
        intensityRatios = []
        
        for maximumOfContrast in maximaOfContrast:
            maximumContrastVal = allContrast[maximumOfContrast]
            if maximumContrastVal > minContrastFilopodia:
                #check if around local max at least minNbSubsequentLowerContrastOfTip-1 other values big enough are
                start = maximumOfContrast-(minNbSubsequentLowerContrastOfTip-1)
                if start < 0:
                    start = 0
                end = maximumOfContrast + minNbSubsequentLowerContrastOfTip
                if end > (len(allContrast)-1):
                    end = len(allContrast)-1
                nbSubsequentLowerContrastOfTip = 1
                for contrastNb in range(start,end):
                    if allContrast[contrastNb] > minContrastFilopodia:
                        nbSubsequentLowerContrastOfTip += 1
                if nbSubsequentLowerContrastOfTip >= minNbSubsequentLowerContrastOfTip:
                    #check whether area around maximum is similarly big on both sites
                    #if yes, indicates that there is no filopodia
                    
                    #before is direction of tip, after is direction of neurite base
                    
                    #nb of px to include in area
                    sizeOfAreas = 10
                    #nb of px after or before high intensity areas to not include 
                    bufferForAreas = 5
                    
                    startBefore = allPointsForContrast[maximumOfContrast] + bufferForAreas + sizeOfAreas
                    sizeBefore = sizeOfAreas
                    sizeAfter = sizeOfAreas
                    
                    if startBefore > -1:
                        sizeBefore = sizeBefore - (startBefore + 1)
                        startBefore = -1
                    
                    endAfter = allPointsForContrast[maximumOfContrast] - (bufferForAreas+sizeOfAreas)
                    
                    if -endAfter > len(oneBranch):
                        sizeAfter = sizeAfter - (-endAfter-len(oneBranch))
                        endAfter = -len(oneBranch)
                    endBefore = startBefore - sizeBefore
                    startAfter = endAfter + sizeAfter
                    
                    if (sizeBefore > 1) & (sizeAfter > 1):
                        pointsBefore = np.transpose(oneBranch[endBefore:startBefore])
                        intensityBefore = np.average(img[pointsBefore[0],pointsBefore[1]])
                        pointsAfter = np.transpose(oneBranch[endAfter:startAfter])
                        intensityAfter = np.average(img[pointsAfter[0],pointsAfter[1]])
                        intensityRatio = intensityAfter/intensityBefore
                        
                        intensityRatios.append(intensityRatio)
                        if intensityRatio > minContrastFilopodia:
                            
                            maximumContrastNb = maximumOfContrast
                    else:
                        maximumContrastNb = maximumOfContrast
             
        if not np.isnan(maximumContrastNb):
            #get furthest contrast that is max of 10% less than maximum contrast and bigger than minimum contrast for filopodia
            maxContrast = allContrast[maximumContrastNb]
            for contrastNb in range(maximumContrastNb,len(allContrast)):
                contrast = allContrast[contrastNb]
                if contrast > minContrastFilopodia:
                    if contrast > (maxContrast*0.9):
                        maximumContrastNb = contrastNb
                else:
                    break
                
        #if no maximum contrast was identified, check if the first px are above contrast threshold, remove until they are not anymore
        if np.isnan(maximumContrastNb):
            subsequentNbOfMinContrast = 0
            for b, contrast in enumerate(allContrast):
                if contrast > minContrastFilopodia:
                    subsequentNbOfMinContrast += 1
                    contrastNb = b
                else:
                    break
            if subsequentNbOfMinContrast >= minNbSubsequentLowerContrastOfTip:
                maximumContrastNb = contrastNb
        return maximumContrastNb

    @staticmethod
    def getContrastOfTip(oneBranch,img,a,nbToCheckBefore):

        if (abs(a)-nbToCheckBefore) < 1:
            nbToCheckBefore = abs(a)-1
        
        lastPoints = oneBranch[-a-nbToCheckBefore:-a]
        lastPoints = np.transpose(lastPoints)
        averageLastPoints = np.mean(img[lastPoints[0],lastPoints[1]])
        
        tipPoints = oneBranch[-a:-a+nbToCheckBefore]
        tipPoints = np.transpose(tipPoints)
        averageTipPoints = np.mean(img[tipPoints[0],tipPoints[1]])
        
        if a < 0:
            contrast = averageTipPoints / averageLastPoints 
        else:
            contrast = averageLastPoints / averageTipPoints
        
        return contrast


    @staticmethod
    def checkWhichBranchesToKeep(deletedSortedPoints,branchLengthToKeep,constructNeurite,longestBranch,img):
        #if not constructNeurite, add points of branches which are long enough to longestBranch (same points are not added twice)
        if constructNeurite:
            #if constructNeurite save all branches with sufficient differences from the longest branch as separate branch
            allLongBranches = []
            allLongBranches.append(longestBranch)

        for oneBranch in deletedSortedPoints:
            if len(oneBranch) > branchLengthToKeep:
                
                #check how many pixels are in oneBranch which are not in longest branch
                if constructNeurite:
                    #for constructneurite check for difference with each branch already added to longest branch array
                    addBranch = True
                    for longestBranch in allLongBranches:
                        addBranch_sub, difference = sortPoints.checkWhetherToAddBranch(oneBranch,longestBranch,branchLengthToKeep,allLongBranches,"two_way")
                        if not addBranch_sub:
                            addBranch = False
                            break
                else:
                    addBranch, difference = sortPoints.checkWhetherToAddBranch(oneBranch,longestBranch,branchLengthToKeep,"one_way")
                    
                if addBranch:
                    if constructNeurite:
                        allLongBranches.append(oneBranch)
                    else:
                        longestBranch = np.vstack((longestBranch,difference))
                
        if constructNeurite:
            longestBranch = allLongBranches
        return longestBranch
    
    @staticmethod
    def checkWhetherToAddBranch(oneBranch,longestBranch,branchLengthToKeep,mode,allLongBranches=[]):
        #does the difference lead to a double addition of last points of neurite?!?!
        addBranch = False
        #if one of the arrays is only a single point, convert them to a point list with one point instead of just one point
        oneBranch = generalTools.convert_points_to_point_list(oneBranch)
        longestBranch = generalTools.convert_points_to_point_list(longestBranch)
        #one way difference looks at difference of longest branch compared to onebranch
        first_difference = generalTools.get_diff_of_two_arrays(oneBranch,longestBranch)
        if mode == "one_way":
            difference = first_difference
        else:
            second_difference = generalTools.get_diff_of_two_arrays(longestBranch,oneBranch)
            if len(second_difference) > len(first_difference):
                difference = second_difference
            else:
                difference = first_difference
        if len(difference) > branchLengthToKeep:
            addBranch = True
        return addBranch, difference
                        

    @staticmethod
    def deleteNeuriteCoords(allCoords,firstPointToDelete,sortedPoints,
                            saveDeletedPoints,deletedSortedPoints,
                            lengths=[],constructNeurite=False):
        #delete all points in sortedPoints from firstPoint on, delete same points in allCoords
        nbsToDelete = range(firstPointToDelete,len(sortedPoints))
        for a in nbsToDelete:
            pointToDelete = sortedPoints[a]
            indexToDelete = np.where((allCoords[0] == pointToDelete[0]) & (allCoords[1] == pointToDelete[1]))
            allCoordsX = np.delete(allCoords[0],indexToDelete[0])
            allCoordsXY = np.delete(allCoords[1],indexToDelete[0])
            allCoords = (allCoordsX,allCoordsXY)
        if saveDeletedPoints == True:
            #test if same points were already saved in deleted sorted points before saving them!
            all_points_present = False
            for one_deleted_sorted_points in deletedSortedPoints:
                all_points_present = True
                if len(one_deleted_sorted_points) == len(sortedPoints):
                    for point_nb, point in enumerate(sortedPoints):
                        if (point[0] != one_deleted_sorted_points[point_nb][0]) & (point[1] != one_deleted_sorted_points[point_nb][1]):
                            all_points_present = False
                            break
                    if all_points_present:
                        break
                else:
                    all_points_present = False
            if not all_points_present:
                deletedSortedPoints.append(sortedPoints)
        sortedPoints = np.delete(sortedPoints,nbsToDelete,axis=0)
        if len(lengths) > 0:
            lengths = np.delete(lengths, nbsToDelete, axis=0)
        return sortedPoints, allCoords, deletedSortedPoints, lengths
        