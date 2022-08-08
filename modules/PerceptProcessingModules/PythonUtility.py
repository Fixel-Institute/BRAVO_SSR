#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 11:06:37 2020

@author: Jackson Cagle
"""

import numpy as np

def listSort(oldList, newIndexes):
    return [oldList[i] for i in newIndexes]

def unwrap(x, cap=None):
    if cap is None:
        cap = np.max(x) + 1
    
    unwrappedX = np.array(range(len(x)))
    unwrappedX[0] = x[0]
    currentRolls = 0
    for n in range(1,len(x)):
        if x[n] < x[n-1]:
            currentRolls += cap
        unwrappedX[n] = x[n] + currentRolls
    
    return unwrappedX

def rangeSelection(array, minMax, type="exclusive"):
    if type=="exclusive":
        return np.bitwise_and(array < minMax[1], array > minMax[0])
    else:
        return np.bitwise_and(array <= minMax[1], array >= minMax[0])

def listSelection(oldList, boolArray):
    newList = list()
    for i in range(len(boolArray)):
        if boolArray[i]:
            newList.append(oldList[i])
    return newList

def ifAllConditions(*conditions):
    baseCondition = np.ones(conditions[0].shape,dtype=bool)
    for i in range(len(conditions)):
        baseCondition = np.bitwise_and(baseCondition, conditions[i].flatten())
    return baseCondition

def ifOrConditions(*conditions):
    baseCondition = np.zeros(conditions[0].shape,dtype=bool)
    for i in range(len(conditions)):
        baseCondition = np.bitwise_or(baseCondition, conditions[i].flatten())
    return baseCondition

def iterativeCompare(listItems, comparedItem, compare="less"):
    result = np.ndarray((len(listItems),1),dtype="bool")
    n = 0
    for item in listItems:
        if compare == "less":
            result[n] = item < comparedItem
        elif compare == "equal":
            result[n] = item == comparedItem
        elif compare == "more":
            result[n] = item > comparedItem
        else:
            raise TypeError("{0} is not a valid compare argument".format(compare))
        n += 1
    return result

def uniqueList(listItems):
    uniqueList = list()
    for item in listItems:
        unique = True
        for existingItem in uniqueList:
            if existingItem == item:
               unique = False 
        
        if unique:
            uniqueList.append(item)
            
    return uniqueList