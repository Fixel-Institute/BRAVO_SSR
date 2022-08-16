# -*- coding: utf-8 -*-
"""
Give a general description of UF AO MPX File here.
@author: Jackson Cagle, University of Florida
@email: jackson.cagle@neurology.ufl.edu
"""

import json
import scipy.io as sio
import sys
import os
import copy
import numpy as np
from datetime import datetime, timezone
import pickle
import dateutil
import pandas as pd
from scipy import optimize
from cryptography.fernet import Fernet

from PythonUtility import *
import SignalProcessingUtility as SPU

def formatLFPTrendTimestamp(dictionary):
    for key in list(dictionary.keys()):
        if key.find("Z") == len(key)-1:
            newKey = "TimeStamp" + str(int(datetime.fromisoformat(key[:-1]).timestamp()))
            dictionary[newKey] = dictionary[key]
            del(dictionary[key])
    return dictionary

def formatKeyName(dictionary):
    for key in list(dictionary.keys()):
        if key.find(" ") > 0 or key.find(".") > 0:
            newKey = key.replace(" ","").replace(".","")
            newKey = newKey
            dictionary[newKey] = dictionary[key]
            del(dictionary[key])
            if type(dictionary[newKey]) is dict:
                dictionary[newKey] = formatKeyName(dictionary[newKey])
        else:
            if type(dictionary[key]) is dict:
                dictionary[key] = formatKeyName(dictionary[key])
        
        if key == "LFPTrendLogs":
            for hemisphere in dictionary[key].keys():
                dictionary[key][hemisphere] = formatLFPTrendTimestamp(dictionary[key][hemisphere])
    return dictionary

def estimateSessionDateTime(JSON):
    sessionDatePools = list()
    if "SessionDate" in JSON.keys() and JSON["SessionEndDate"] == "":
        sessionDatePools.append(datetime.fromisoformat(JSON["SessionDate"]+"+00:00").timestamp())
    if not JSON["SessionEndDate"] == "":
        sessionDatePools.append(datetime.fromisoformat(JSON["SessionEndDate"]+"+00:00").timestamp())
    
    if len(sessionDatePools) == 2:
        if np.abs(sessionDatePools[1]-sessionDatePools[0]) > 24*60*60:
            sessionDatePools = list()
            sessionDatePools.append(datetime.fromisoformat(JSON["SessionEndDate"]+"+00:00").timestamp())
    
    if "EventSummary" in JSON.keys():
        sessionDatePools.append(datetime.fromisoformat(JSON["EventSummary"]["SessionEndDate"]+"+00:00").timestamp())
    if "CalibrationTests" in JSON.keys():
        for i in range(len(JSON["CalibrationTests"])):
            sessionDatePools.append(datetime.fromisoformat(JSON["CalibrationTests"][i]["FirstPacketDateTime"][:-4]+"+00:00").timestamp())
    if "SenseChannelTests" in JSON.keys():
        for i in range(len(JSON["SenseChannelTests"])):
            sessionDatePools.append(datetime.fromisoformat(JSON["SenseChannelTests"][i]["FirstPacketDateTime"][:-4]+"+00:00").timestamp())
    if "BrainSenseTimeDomain" in JSON.keys():
        for i in range(len(JSON["BrainSenseTimeDomain"])):
            sessionDatePools.append(datetime.fromisoformat(JSON["BrainSenseTimeDomain"][i]["FirstPacketDateTime"][:-4]+"+00:00").timestamp())
    if "BrainSenseLfp" in JSON.keys():
        for i in range(len(JSON["BrainSenseLfp"])):
            sessionDatePools.append(datetime.fromisoformat(JSON["BrainSenseLfp"][i]["FirstPacketDateTime"][:-4]+"+00:00").timestamp())
    
    sessionDatePools = np.array(sessionDatePools)
    sessionDatePools = sessionDatePools[sessionDatePools > 1420088400]
    if len(sessionDatePools) == 1:
        #print(f"{JSON['DeviceInformation']['']}: Only One Date: " + datetime.fromtiemstamp(sessionDatePools[0]).isoformat())
        return sessionDatePools[0]
    
    if len(sessionDatePools) == 0:
        return datetime.fromisoformat(JSON["SessionDate"]+"+00:00").timestamp()
    
    if np.std(sessionDatePools) == 0:
        return np.mean(np.array(sessionDatePools))
    zscore = np.abs((np.array(sessionDatePools) - np.mean(sessionDatePools)) / np.std(sessionDatePools)) < 2
    return np.mean(np.array(sessionDatePools)[zscore])

def decodeJSON(inputFilename):
    Data = json.load(open(inputFilename, encoding='utf-8'))
    return Data

def decodeEncryptedJSON(inputFilename, key):
    secureEncoder = Fernet(key)
    with open(inputFilename, "rb") as file:
        Data = json.loads(secureEncoder.decrypt(file.read()))
        return Data

def concatenateJSONs(JSONs):
    # Get the list of fields in all JSONs
    allJsonFields = list()
    for JSON in JSONs:
        allJsonFields.extend(JSON.keys())
    allJsonFields = removeDuplicates(allJsonFields)
    
    CompiledJSON = dict()
    for field in allJsonFields:
        if field == "DiagnosticData":
            CompiledJSON[field] = dict()
            subFields = list()
            for JSON in JSONs:
                if field in JSON.keys():
                    subFields.extend(JSON[field].keys())
            subFields = removeDuplicates(subFields)
            
            for subfield in subFields:
                if subfield == "LFPTrendLogs":
                    CompiledJSON[field][subfield] = dict()
                    
                    for JSON in JSONs:
                        if field in JSON.keys():
                            if subfield in JSON[field].keys():
                                hemisphereDefinitions = list(JSON[field][subfield].keys())
                    
                    for hemisphere in hemisphereDefinitions:
                        CompiledJSON[field][subfield][hemisphere] = dict()
                        
                        dates = list()
                        for JSON in JSONs:
                            if field in JSON.keys():
                                if subfield in JSON[field].keys():
                                    # No check for hemisphere, assuming that hemisphere always exist..... can we?
                                    dates.extend(JSON[field][subfield][hemisphere].keys())
                        dates = removeDuplicates(dates)
                        
                        # Quick Note: Dates here are very confusing. I do not think the Hour/Minutes/Seconds actually matter. 
                        # It is best to concatenate everything and store in one single dictionary in the combined JSON.
                        allLFPTrends = list()
                        for date in dates:
                            allLFPTrends.extend(concatenateLists(JSONs, [field, subfield, hemisphere, date]))
                        allLFPTrends = removeDuplicates(allLFPTrends)
                        Timestamp = [datetime.fromisoformat(allLFPTrends[i]["DateTime"][:-1]+"+00:00").timestamp() for i in range(len(allLFPTrends))]
                        allLFPTrends = listSort(allLFPTrends, np.argsort(Timestamp))
                        CompiledJSON[field][subfield][hemisphere][allLFPTrends[0]["DateTime"]] = allLFPTrends
                
                elif subfield == "EventLogs":
                    CompiledJSON[field][subfield] = concatenateLists(JSONs, [field, subfield])
                    CompiledJSON[field][subfield] = removeDuplicates(CompiledJSON[field][subfield])
                    Timestamp = [datetime.fromisoformat(CompiledJSON[field][subfield][i]["DateTime"][:-1]+"+00:00").timestamp() for i in range(len(CompiledJSON[field][subfield]))]
                    CompiledJSON[field][subfield] = listSort(CompiledJSON[field][subfield], np.argsort(Timestamp))
                    
                # Skipping PSD Logs for now. Test dataset do not contain them.
        
        # Critical Check: Are we looking at the JSONs of the same patient?
        elif field == "PatientInformation" or field == "LeadConfiguration":
            CompiledJSON[field] = concatenateLists(JSONs, [field])
            CompiledJSON[field] = removeDuplicates(CompiledJSON[field])
            if len(CompiledJSON[field]) > 1:
                print(f"Multiple {field}. Are you sure this is the same patient?")
                raise(TypeError)
            CompiledJSON[field] = CompiledJSON[field][0]
        
        # Minor Check. It is possible they are different, we will take only the first occurance
        elif field == "BatteryInformation" or field == "DeviceInformation" or field == "GroupUsagePercentage" or field == "Stimulation" or field == "BatteryReminder":
            CompiledJSON[field] = concatenateLists(JSONs, [field])
            CompiledJSON[field] = removeDuplicates(CompiledJSON[field])
            if len(CompiledJSON[field]) > 1:
                Warning(f"Multiple {field}. Are you sure this is the same patient?")
            CompiledJSON[field] = CompiledJSON[field][0]
        
        # They are list without time. Concat and remove duplicate.
        # Event Summary is a fun one. We should make that into an array even though originally was a dictionary.
        elif field == "Impedance" or field == "MostRecentInSessionSignalCheck" or field == "EventSummary":
            CompiledJSON[field] = concatenateLists(JSONs, [field])
            CompiledJSON[field] = removeDuplicates(CompiledJSON[field])
        
        # GroupHistory is an interesting case. If a setting is changed then the later GroupHistory will be influenced.
        #   This is a very troublesome condition, the initial true history will not be stored for future JSONs. 
        #   Need to consult Medtronic for more information
        elif field == "GroupHistory":
            CompiledJSON[field] = concatenateLists(JSONs, [field])
            CompiledJSON[field] = removeDuplicates(CompiledJSON[field])
            Timestamp = [datetime.fromisoformat(CompiledJSON[field][i]["SessionDate"][:-1]+"+00:00").timestamp() for i in range(len(CompiledJSON[field]))]
            CompiledJSON[field] = listSort(CompiledJSON[field], np.flip(np.argsort(Timestamp)))
        
        # Groups is as confusing as GroupHistry. This is also modifiable in the middle of the recording.
        # Do we care about the change? What do we really need in this structure? If we get all that we want from GroupHistory
        elif field == "Groups":
            CompiledJSON[field] = concatenateLists(JSONs, [field])
            CompiledJSON[field] = removeDuplicates(CompiledJSON[field])
            if len(CompiledJSON[field]) > 1:
                Warning(f"Multiple {field}. Therapy Changed, but only the last JSON is taken")
            CompiledJSON[field] = CompiledJSON[field][-1]

        elif field == "SenseChannelTests" or field == "CalibrationTests" or field == "LfpMontageTimeDomain" or field == "LFPMontage": 
            CompiledJSON[field] = concatenateLists(JSONs, [field])
            CompiledJSON[field] = removeDuplicates(CompiledJSON[field])
            
        elif field == "BrainSenseTimeDomain" or field == "BrainSenseLfp" or field == "IndefiniteStreaming": 
            CompiledJSON[field] = concatenateLists(JSONs, [field])
            CompiledJSON[field] = removeDuplicates(CompiledJSON[field])
        
        else: 
            CompiledJSON[field] = concatenateLists(JSONs, [field])
            CompiledJSON[field] = removeDuplicates(CompiledJSON[field])
            if len(CompiledJSON[field]) > 1:
                Warning(f"Multiple {field}. Are you sure this is the same patient?")
            CompiledJSON[field] = CompiledJSON[field][0]
            
    return CompiledJSON
    
def concatenateLists(array, args):
    if len(args) > 1:
        newArray = [subArray[args[0]] for subArray in array if args[0] in subArray.keys()]
        return concatenateLists(newArray, args[1:])
    
    newList = list()
    for i in range(len(array)):
        if args[0] in array[i].keys(): 
            if type(array[i][args[0]]) == list:
                newList.extend(array[i][args[0]])
            else:
                newList.append(array[i][args[0]])
    return newList

def removeDuplicates(originalList, type="all"):
    finalList = []
    for item in originalList:
        notFound = True
        for finalItem in finalList:
            if type == "Data":
                if "Data" in item.keys() and "Data" in finalItem.keys():
                    if arrayCompare(item["Data"], finalItem["Data"]):
                        notFound = False
                        break
                if "Power" in item.keys() and "Power" in finalItem.keys():
                    if arrayCompare(item["Power"], finalItem["Power"]):
                        notFound = False
                        break
            else:
                if dictionaryCompare(item, finalItem):
                    notFound = False
                    break
        if notFound:
            finalList.append(item)
    return finalList

def arrayCompare(first, second):
    return np.array_equal(first, second)

def dictionaryCompare(first, second):
    if type(first) == list:
        if not len(first) == len(second):
            return False
        
        for i in range(len(first)):
            if not dictionaryCompare(first[i], second[i]):
                return False
    else:
        try:
            return first == second
        except:
            if not second.keys() == first.keys():
                return False
            
            for key in first.keys():
                if not type(first[key]) == type(second[key]):
                    return False
                
                if type(first[key]) == dict:
                    if not dictionaryCompare(first[key], second[key]):
                        return False
                
                elif type(first[key]) == np.ndarray:
                    if not np.all(first[key] == second[key]):
                        return False
                    
                elif type(first[key]) == list:
                    return False
                
                else:
                    if not first[key] == second[key]:
                        return False
    return True
            
def text2num(textList):
    if type(textList) == list:
        Numbers = list()
        for num in textList:
            inputText = num.replace("[","").replace("]","")
            if (len(inputText) > 0):
                Numbers.append(float(inputText))
        return Numbers
    else:
        return float(textList)

def reformatStimulationChannel(channel):
    channelName = ""
    for contact in channel:
        if contact["ElectrodeStateResult"] != "ElectrodeStateDef.None":
            if contact["Electrode"] == "ElectrodeDef.Case":
                electrodeName = "CAN"
            elif contact["Electrode"].find("ElectrodeDef.FourElectrodes_") >= 0:
                electrodeName = "E" + contact["Electrode"].replace("ElectrodeDef.FourElectrodes_","")
            elif contact["Electrode"].find("ElectrodeDef.Sen") >= 0:
                electrodeName = "E" + contact["Electrode"].upper().replace("ELECTRODEDEF.SENSIGHT_","")
                
            if contact["ElectrodeStateResult"] == "ElectrodeStateDef.Negative":
                channelName = "+" + electrodeName + channelName
            else:
                channelName = channelName + "-" + electrodeName
            
    return channelName

def reformatElectrodeDef(electrodeDef):
    electrodeDef = electrodeDef.upper()
    if not electrodeDef.startswith("ELECTRODEDEF."):
        raise ValueError("Incorrect Electrode Definition String.")
        
    if electrodeDef.upper().find("FOURELECTRODES") >= 0:
        channelName = "E" + electrodeDef.replace("ELECTRODEDEF.FOURELECTRODES_","")
        channelID = int(electrodeDef.replace("ELECTRODEDEF.FOURELECTRODES_",""))
    
    elif electrodeDef.upper().find("SENSIGHT") >= 0:
        channelName = "E" + electrodeDef.replace("ELECTRODEDEF.SENSIGHT_","")
        if channelName[-1].isdigit():
            channelID = int(channelName[1:])
            if not channelID % 8 == 0:
                channelID += 4
        elif channelName.endswith("A"):
            channelID = (int(channelName[1:-1]) % 8 - 1) * 3 + 1
        elif channelName.endswith("B"):
            channelID = (int(channelName[1:-1]) % 8 - 1) * 3 + 2
        elif channelName.endswith("C"):
            channelID = (int(channelName[1:-1]) % 8 - 1) * 3 + 3
        
    elif electrodeDef.upper().find("CASE"):
        channelName = "CAN"
        channelID = -1
        
    else:
        raise ValueError("Unknown Electrode Definition String.")
    
    return channelName, channelID

def reformatChannelName(string):
    if string.find(",") >= 0:
        return (reformatChannelName(string[:string.find(",")]),
                reformatChannelName(string[string.find(",")+1:]))
    
    channel = list()
    if string.find("SEGMENT") >= 0:
        if string.find("ONE_A") >= 0:
            channel.append(1.1)
        if string.find("ONE_B") >= 0:
            channel.append(1.2)
        if string.find("ONE_C") >= 0:
            channel.append(1.3)
        if string.find("TWO_A") >= 0:
            channel.append(2.1)
        if string.find("TWO_B") >= 0:
            channel.append(2.2)
        if string.find("TWO_C") >= 0:
            channel.append(2.3)
        if string.find("LEFT") >= 0:
            return (channel, "Left")
        if string.find("RIGHT") >= 0:
            return (channel, "Right")
    else:
        if string.find("ZERO") >= 0:
            channel.append(0)
        if string.find("ONE") >= 0:
            channel.append(1)
        if string.find("TWO") >= 0:
            channel.append(2)
        if string.find("THREE") >= 0:
            channel.append(3)
        if string.find("LEFT") >= 0:
            return (channel, "Left")
        if string.find("RIGHT") >= 0:
            return (channel, "Right")
    
    return channel

def randomDateTimeString(minTS = 0, maxTS = datetime.utcnow().timestamp()):
    return datetime.fromtimestamp(np.random.randint(minTS, maxTS)).strftime("%Y-%m-%dT%H:%M:%SZ")

def getTimestamp(DateTimeString):
    return datetime.fromisoformat(DateTimeString[:-1]+"+00:00").timestamp()

def deIdentification(JSON, patientIdentifier="", saveName=None):
    for key in JSON.keys():
        if key == "PatientInformation":
            for state in JSON[key].keys():
                JSON[key][state]["PatientFirstName"] = ""
                JSON[key][state]["PatientLastName"] = ""
                JSON[key][state]["PatientGender"] = ""
                JSON[key][state]["PatientId"] = patientIdentifier
                JSON[key][state]["PatientDateOfBirth"] = ""
                
        if key == "DeviceInformation":
            for state in JSON[key].keys():
                JSON[key][state]["NeurostimulatorSerialNumber"] = ""
                JSON[key][state]["ImplantDate"] = randomDateTimeString()
                JSON[key][state]["DeviceDateTime"] = randomDateTimeString()
    
        if key == "EventSummary":
            JSON[key]["SessionStartDate"] = randomDateTimeString()
            JSON[key]["SessionEndDate"] = randomDateTimeString()
            
        if key == "GroupHistory":
            for i in range(len(JSON[key])):
                JSON[key][i]["SessionDate"] = randomDateTimeString()
            
        if key == "SenseChannelTests":
            for i in range(len(JSON[key])):
                JSON[key][i]["FirstPacketDateTime"] = randomDateTimeString()
        
        if key == "CalibrationTests":
            for i in range(len(JSON[key])):
                JSON[key][i]["FirstPacketDateTime"] = randomDateTimeString()
        
        if key == "BrainSenseTimeDomain":
            for i in range(len(JSON[key])):
                JSON[key][i]["FirstPacketDateTime"] = randomDateTimeString()
        
        if key == "BrainSenseLfp":
            for i in range(len(JSON[key])):
                JSON[key][i]["FirstPacketDateTime"] = randomDateTimeString()
                
        if key == "LfpMontageTimeDomain":
            for i in range(len(JSON[key])):
                JSON[key][i]["FirstPacketDateTime"] = randomDateTimeString()
                
        if key == "IndefiniteStreaming":
            for i in range(len(JSON[key])):
                JSON[key][i]["FirstPacketDateTime"] = randomDateTimeString()
                
        if key == "DiagnosticData":
            if "EventLogs" in JSON[key].keys():
                i = 0
                while i < len(JSON[key]["EventLogs"]):
                    if "SessionType" in JSON[key]["EventLogs"][i].keys():
                        if JSON[key]["EventLogs"][i]["SessionType"].find("SessionStateDef") >= 0:
                            del(JSON[key]["EventLogs"][i])
                        else:
                            i += 1
                    elif "ParameterTrendId" in JSON[key]["EventLogs"][i].keys():
                        if JSON[key]["EventLogs"][i]["ParameterTrendId"].find("LeadIntegrityPerformed") >= 0:
                            del(JSON[key]["EventLogs"][i])
                        else:
                            i += 1
                    else:
                        i += 1
        
    JSON["SessionDate"] = randomDateTimeString()
    JSON["SessionEndDate"] = randomDateTimeString()
    
    if saveName is not None:
        json.dump(JSON, open(saveName, "w+"))
    return JSON

def checkMissingPackage(Data):
    if "IndefiniteStream" in Data.keys():
        for nStream in range(len(Data["IndefiniteStream"])):
            TDSequences = unwrap(Data["IndefiniteStream"][nStream]["Sequences"], cap=256)
            
            missingSequence = list()
            for n in range(1,len(TDSequences)):
                jumppedSequence = TDSequences[n]-TDSequences[n-1]
                if jumppedSequence > 1:
                    missingIndexes = np.array(range(1, jumppedSequence)) + TDSequences[n-1]
                    missingSequence.extend(missingIndexes)
                    
            Data["IndefiniteStream"][nStream]["Missing"] = np.zeros(Data["IndefiniteStream"][nStream]["Data"].shape)
            PacketSize = int(np.mean(Data["IndefiniteStream"][nStream]["PacketSizes"]))
            if len(missingSequence) > 0:
                for nMissing in missingSequence:
                    insertionIndex = np.where(TDSequences < nMissing)[0][-1] + 1
                    startIndex = int(np.sum(Data["IndefiniteStream"][nStream]["PacketSizes"][:insertionIndex]))
                    TDSequences = np.concatenate((TDSequences[:insertionIndex], [nMissing], TDSequences[insertionIndex:]))
                    Data["IndefiniteStream"][nStream]["PacketSizes"] = np.concatenate((Data["IndefiniteStream"][nStream]["PacketSizes"][:insertionIndex], [PacketSize], TDSequences[insertionIndex:]))
                    Data["IndefiniteStream"][nStream]["Data"] = np.concatenate((Data["IndefiniteStream"][nStream]["Data"][:startIndex],np.zeros((PacketSize)),Data["IndefiniteStream"][nStream]["Data"][startIndex:]))
                    Data["IndefiniteStream"][nStream]["Missing"] = np.concatenate((Data["IndefiniteStream"][nStream]["Missing"][:startIndex],np.ones((PacketSize)),Data["IndefiniteStream"][nStream]["Missing"][startIndex:]))
                Data["IndefiniteStream"][nStream]["Time"] = np.array(range(len(Data["IndefiniteStream"][nStream]["Data"]))) / Data["IndefiniteStream"][nStream]["SamplingRate"]
                #print(f"Warning: Missing sequence occured for Stream #{nStream}, Data insertion complete. Check ['Missing'] field.")
            
    if "StreamingTD" in Data.keys() and "StreamingPower" in Data.keys():
        if len(Data["StreamingTD"]) != len(Data["StreamingPower"]):
            raise Warning("Number of stream in Time-domain does not match number of stream in Power channel")
            
        for nStream in range(len(Data["StreamingTD"])):
            Data["StreamingTD"][nStream]["Missing"] = np.zeros(Data["StreamingTD"][nStream]["Data"].shape)
            Data["StreamingPower"][nStream]["Missing"] = np.zeros((Data["StreamingPower"][nStream]["Power"].shape[0]))
            
            # TicksInMs
            ChangesInMs = np.diff(Data["StreamingTD"][nStream]["Ticks"])
            TimePerPackage = np.median(ChangesInMs)
            if len(np.where(ChangesInMs < 0)[0]) > 0:
                print("TicksInMs Revamped")
                raise Exception("Bad Format in TicksInMs")
            
            TimePerPacket = np.median(ChangesInMs)
            MissingPacket = np.where(ChangesInMs > TimePerPacket)[0] + 1
            TDSequences = np.arange(len(Data["StreamingTD"][nStream]["Ticks"]))
            
            # is all missing sequence accounted for?
            PacketSize = int(np.mean(Data["StreamingTD"][nStream]["PacketSizes"]))
            if len(MissingPacket) > 0:
                for missingIndex in MissingPacket:
                    if not ChangesInMs[missingIndex-1] % TimePerPacket == 0:
                        print("TicksInMs Reversed")
                        raise Exception("Time Skip is not full package drop")
                        
                    numMissingPacket = int(ChangesInMs[missingIndex-1] / TimePerPacket - 1)
                    insertionIndex = np.where(TDSequences < missingIndex)[0][-1] + 1
                    startIndex = int(np.sum(Data["StreamingTD"][nStream]["PacketSizes"][:insertionIndex]))
                    TDSequences = np.concatenate((TDSequences[:insertionIndex], np.zeros(numMissingPacket), TDSequences[insertionIndex:]))
                    Data["StreamingTD"][nStream]["PacketSizes"] = np.concatenate((Data["StreamingTD"][nStream]["PacketSizes"][:insertionIndex], PacketSize*np.ones(numMissingPacket), Data["StreamingTD"][nStream]["PacketSizes"][insertionIndex:]))
                    Data["StreamingTD"][nStream]["Data"] = np.concatenate((Data["StreamingTD"][nStream]["Data"][:startIndex],np.zeros((PacketSize*numMissingPacket)),Data["StreamingTD"][nStream]["Data"][startIndex:]))
                    Data["StreamingTD"][nStream]["Missing"] = np.concatenate((Data["StreamingTD"][nStream]["Missing"][:startIndex],np.ones((PacketSize*numMissingPacket)),Data["StreamingTD"][nStream]["Missing"][startIndex:]))
                Data["StreamingTD"][nStream]["Time"] = np.array(range(len(Data["StreamingTD"][nStream]["Data"]))) / Data["StreamingTD"][nStream]["SamplingRate"]
                #print(f"Warning: Missing sequence occured for Stream #{nStream}, Data insertion complete. Check ['Missing'] field.")
            
            # Check Power Channel Now
            ChangesInMs = np.around(np.diff(Data["StreamingPower"][nStream]["Time"]),3)
            if len(np.where(ChangesInMs < 0)[0]) > 0:
                print("TicksInMs Reversed")
                raise Exception("Bad Format in TicksInMs for Power Channel")
            
            TimePerPacket = np.percentile(ChangesInMs,5)
            MissingPacket = np.where(ChangesInMs > TimePerPacket)[0] + 1
            
            if len(MissingPacket) > 0:
                newTimestamp = np.arange(Data["StreamingPower"][nStream]["Time"][0], Data["StreamingPower"][nStream]["Time"][-1]+TimePerPacket, TimePerPacket)
                processedPower = np.zeros((len(newTimestamp),2))
                processedStimulation = np.zeros((len(newTimestamp),2))
                Data["StreamingPower"][nStream]["Missing"] = np.zeros((len(newTimestamp)))
                for i in range(2):
                    processedPower[:,i] = np.interp(newTimestamp, Data["StreamingPower"][nStream]["Time"], Data["StreamingPower"][nStream]["Power"][:,i])
                    processedStimulation[:,i] = np.interp(newTimestamp, Data["StreamingPower"][nStream]["Time"], Data["StreamingPower"][nStream]["Stimulation"][:,i])
                
                for t in range(len(newTimestamp)):
                    if not newTimestamp[t] in Data["StreamingPower"][nStream]["Time"]:
                        Data["StreamingPower"][nStream]["Missing"][t] = 1
                        
                Data["StreamingPower"][nStream]["Power"] = processedPower
                Data["StreamingPower"][nStream]["Stimulation"] = processedStimulation
                Data["StreamingPower"][nStream]["Time"] = newTimestamp
                
            # Compensate for shifting less than 500ms, which will be indicated by the TickInMs
            TimeShift = Data["StreamingPower"][nStream]["InitialTickInMs"] - Data["StreamingTD"][nStream]["Ticks"][0] / 1000
            if TimeShift < -1000:
                TimeShift = np.round(TimeShift+3276.8,2)
            elif TimeShift > 1000:
                TimeShift = np.round(TimeShift-3276.8,2)
            else:
                TimeShift = np.round(TimeShift,2)
            Data["StreamingPower"][nStream]["Time"] += TimeShift
            #print(f"Warning: Missing sequence occured for Stream #{nStream}, Data insertion complete. Check ['Missing'] field.")
            
    return Data

def processBreakingTimeDomain(Data):
    if "StreamingTD" in Data.keys() and "StreamingPower" in Data.keys():
        StreamTimestamp = np.array([datetime.fromisoformat(Data["StreamingTD"][i]["FirstPacketDateTime"][:-1]+"+00:00").timestamp() for i in range(len(Data["StreamingTD"]))])
        UniqueStreamTimestamp = np.unique(StreamTimestamp)
        StreamLabel = np.zeros(StreamTimestamp.shape)
        for i in range(len(UniqueStreamTimestamp)):
            StreamLabel[StreamTimestamp == UniqueStreamTimestamp[i]] = i
        
        i = 0
        while i < len(UniqueStreamTimestamp):
            numStream = np.where(StreamLabel==i)[0]
            numNextStream = np.where(StreamLabel==i+1)[0]
            if len(numStream) != len(numNextStream):
                i += 1
                continue
            
            orderedNextStream = list()
            for nIndex in range(len(numStream)):
                for iIndex in range(len(numNextStream)):
                    if Data["StreamingTD"][numStream[nIndex]]["Channel"] == Data["StreamingTD"][numNextStream[iIndex]]["Channel"]:
                        orderedNextStream.append(numNextStream[iIndex])
            
            # This occur if not both channels matches
            if len(orderedNextStream) != len(numStream):
                i += 1
                continue 
            
            # Now that if both TD Channels match, we need to verify that both of their therapy are the same.
            TherapyMatching = True
            for nIndex in range(len(numStream)):
                channels, hemisphere = reformatChannelName(Data["StreamingTD"][numStream[nIndex]]["Channel"])
                
                # First, we will remove LowerLimitInMilliAmps and UpperLimitInMilliAmps from comparison.
                # Because adjustment of amplitude could change such limit. 
                Data["StreamingTD"][numStream[nIndex]]["PowerDomain"]["TherapySnapshot"][hemisphere]["LowerLimitInMilliAmps"] = 0
                Data["StreamingTD"][numStream[nIndex]]["PowerDomain"]["TherapySnapshot"][hemisphere]["UpperLimitInMilliAmps"] = 0
                Data["StreamingTD"][orderedNextStream[nIndex]]["PowerDomain"]["TherapySnapshot"][hemisphere]["LowerLimitInMilliAmps"] = 0
                Data["StreamingTD"][orderedNextStream[nIndex]]["PowerDomain"]["TherapySnapshot"][hemisphere]["UpperLimitInMilliAmps"] = 0 
                
                if Data["StreamingTD"][numStream[nIndex]]["PowerDomain"]["TherapySnapshot"][hemisphere] != Data["StreamingTD"][orderedNextStream[nIndex]]["PowerDomain"]["TherapySnapshot"][hemisphere]: 
                    TherapyMatching = False
            
            # This occur if the therapy settings are different
            if not TherapyMatching:
                i += 1
                continue 
            
            
            # Now that we know they are supposed to be identical. There is still the concern that "Segmented Stimulation" is not stored in SenSight Leads.
            # Our Criteria should be the following:
            #       1) 2nd Stream start stimulation amplitude = 1st Stream
            #       2) If both end/start are 0, the 1st Stream does not contain high stimulation amplitude
            StimulationContinuation = True
            nIndex = 0
            StimIndex = np.where(Data["StreamingTD"][numStream[nIndex]]["PowerDomain"]["Missing"]==0)[0][-1]
            EndStimulationAmplitude = Data["StreamingTD"][numStream[nIndex]]["PowerDomain"]["Stimulation"][StimIndex,:]
            StimIndex = np.where(Data["StreamingTD"][orderedNextStream[nIndex]]["PowerDomain"]["Missing"]==0)[0][0]
            StartStimulationAmplitude = Data["StreamingTD"][orderedNextStream[nIndex]]["PowerDomain"]["Stimulation"][StimIndex,:]
            
            StimulationAmplitudes = Data["StreamingTD"][numStream[nIndex]]["PowerDomain"]["Stimulation"][Data["StreamingTD"][numStream[nIndex]]["PowerDomain"]["Missing"]==0,:]
            
            if not np.all(EndStimulationAmplitude == StartStimulationAmplitude):
                StimulationContinuation = False
            
            elif np.all(EndStimulationAmplitude == 0):
                if len(np.unique(StimulationAmplitudes,axis=0)) > 1:
                    StimulationContinuation = False
            
            # This is another scenario: If the other hemisphere is constantly on high stimulation.
            elif np.any(EndStimulationAmplitude == 0):
                channel = np.where(EndStimulationAmplitude == 0)[0][0]
                ExistingAmplitudes = list(np.unique(StimulationAmplitudes[:,channel]))
                if 0 in ExistingAmplitudes: ExistingAmplitudes.remove(0);
                if len(ExistingAmplitudes) > 1:
                    StimulationContinuation = False
            
            # How do we concatenate? Using the typical function but additionally we need to delete the 2nd Stream
            if StimulationContinuation:
                for nIndex in range(len(numStream)):
                    FinalStream = mergeBrainSenseStreams(Data, [numStream[nIndex],orderedNextStream[nIndex]])
                    Data["StreamingTD"][numStream[nIndex]] = FinalStream["StreamingTD"]
                    Data["StreamingPower"][numStream[nIndex]] = FinalStream["StreamingPower"]
                    Data["StreamingTD"][numStream[nIndex]]["PowerDomain"] = FinalStream["StreamingPower"]
                    Data["StreamingTD"][numStream[nIndex]]["PowerDomain"]["Channel"] = FinalStream["StreamingTD"]["PowerDomain"]["Channel"]
                
                for nIndex in sorted(orderedNextStream, reverse=True):
                    Data["StreamingTD"].pop(nIndex)
                    Data["StreamingPower"].pop(nIndex)
                
                StreamLabel = np.delete(StreamLabel, numStream)
            
            i += 1
    
    return Data

def mergeBrainSenseStreams(Data, StreamToConcatenate):
    if not ("StreamingTD" in Data.keys() and "StreamingPower" in Data.keys()):
        raise Exception("BrainSense Streams are not present in the data")
        
    FinalStream = dict()
    FinalStream["StreamingTD"] = copy.deepcopy(Data["StreamingTD"][StreamToConcatenate[0]])
    FinalStream["StreamingPower"] = copy.deepcopy(Data["StreamingPower"][StreamToConcatenate[0]])
    
    for nStream in StreamToConcatenate[1:]:
        LastStreamDateTime = datetime.fromisoformat(FinalStream["StreamingTD"]["FirstPacketDateTime"][:-1]+"+00:00")
        NewStreamDateTime = datetime.fromisoformat(Data["StreamingTD"][nStream]["FirstPacketDateTime"][:-1]+"+00:00")
        TimeElapsed = NewStreamDateTime - LastStreamDateTime
        nSampleSkipped = int(TimeElapsed.seconds * Data["StreamingTD"][nStream]["SamplingRate"] - len(FinalStream["StreamingTD"]["Data"]))
        FinalStream["StreamingTD"]["Data"] = np.concatenate((FinalStream["StreamingTD"]["Data"], np.zeros((nSampleSkipped)), Data["StreamingTD"][nStream]["Data"]))
        FinalStream["StreamingTD"]["Missing"] = np.concatenate((FinalStream["StreamingTD"]["Missing"], np.ones((nSampleSkipped)), Data["StreamingTD"][nStream]["Missing"]))

        nSampleSkipped = int(TimeElapsed.seconds * Data["StreamingPower"][nStream]["SamplingRate"] - len(FinalStream["StreamingPower"]["Power"]))
        FinalStream["StreamingPower"]["Power"] = np.concatenate((FinalStream["StreamingPower"]["Power"], Data["StreamingPower"][nStream]["Power"]), axis=0)
        FinalStream["StreamingPower"]["Missing"] = np.concatenate((FinalStream["StreamingPower"]["Missing"], Data["StreamingPower"][nStream]["Missing"]))
        FinalStream["StreamingPower"]["Stimulation"] = np.concatenate((FinalStream["StreamingPower"]["Stimulation"], Data["StreamingPower"][nStream]["Stimulation"]), axis=0)
        Data["StreamingPower"][nStream]["Time"] += TimeElapsed.seconds
        FinalStream["StreamingPower"]["Time"] = np.concatenate((FinalStream["StreamingPower"]["Time"], Data["StreamingPower"][nStream]["Time"]))
        
    FinalStream["StreamingTD"]["Time"] = np.array(range(len(FinalStream["StreamingTD"]["Data"]))) / FinalStream["StreamingTD"]["SamplingRate"]
    return FinalStream

def concatenateStreams(Data, Type, StreamToConcatenate):
    if Type == "StreamingTD":
        FinalStream = copy.deepcopy(Data["StreamingTD"][StreamToConcatenate[0]])
        
        for nStream in StreamToConcatenate:
            if nStream is StreamToConcatenate[0]:
                continue
            
            LastStreamDateTime = datetime.fromisoformat(FinalStream["FirstPacketDateTime"][:-1]+"+00:00")
            NewStreamDateTime = datetime.fromisoformat(Data["StreamingTD"][nStream]["FirstPacketDateTime"][:-1]+"+00:00")
            TimeElapsed = NewStreamDateTime - LastStreamDateTime
            nSampleSkipped = int(TimeElapsed.seconds * Data["StreamingTD"][nStream]["SamplingRate"] - len(FinalStream["Data"]))
            FinalStream["Data"] = np.concatenate((FinalStream["Data"], np.zeros((nSampleSkipped)), Data["StreamingTD"][nStream]["Data"]))
            FinalStream["Missing"] = np.concatenate((FinalStream["Missing"], np.ones((nSampleSkipped)), Data["StreamingTD"][nStream]["Missing"]))
            
        FinalStream["Time"] = np.array(range(len(FinalStream["Data"]))) / Data["StreamingTD"][nStream]["SamplingRate"]
        return FinalStream
    
    elif Type == "StreamingPower":
        FinalStream = copy.deepcopy(Data["StreamingPower"][StreamToConcatenate[0]])
        
        for nStream in StreamToConcatenate:
            if nStream is StreamToConcatenate[0]:
                continue
            
            LastStreamDateTime = datetime.fromisoformat(FinalStream["FirstPacketDateTime"][:-1]+"+00:00")
            NewStreamDateTime = datetime.fromisoformat(Data["StreamingPower"][nStream]["FirstPacketDateTime"][:-1]+"+00:00")
            TimeElapsed = NewStreamDateTime - LastStreamDateTime
            nSampleSkipped = int(TimeElapsed.seconds * Data["StreamingPower"][nStream]["SamplingRate"] - len(FinalStream["Power"]))
            FinalStream["Power"] = np.concatenate((FinalStream["Power"], np.zeros((nSampleSkipped,2)), Data["StreamingPower"][nStream]["Power"]))
            FinalStream["Stimulation"] = np.concatenate((FinalStream["Stimulation"], np.ones((nSampleSkipped,2))*-1, Data["StreamingPower"][nStream]["Stimulation"]))
            FinalStream["Missing"] = np.concatenate((FinalStream["Missing"], np.ones((nSampleSkipped)), Data["StreamingPower"][nStream]["Missing"]))
            
        FinalStream["Time"] = np.array(range(len(FinalStream["Power"]))) / Data["StreamingPower"][nStream]["SamplingRate"]
        return FinalStream
    
    else:
        FinalStream = copy.deepcopy(Data[Type][StreamToConcatenate[0]])
        
        for nStream in StreamToConcatenate:
            if nStream is StreamToConcatenate[0]:
                continue
            
            LastStreamDateTime = datetime.fromisoformat(FinalStream["FirstPacketDateTime"][:-1]+"+00:00")
            NewStreamDateTime = datetime.fromisoformat(Data[Type][nStream]["FirstPacketDateTime"][:-1]+"+00:00")
            TimeElapsed = NewStreamDateTime - LastStreamDateTime
            nSampleSkipped = int(TimeElapsed.seconds * Data[Type][nStream]["SamplingRate"] - len(FinalStream["Data"]))
            FinalStream["Data"] = np.concatenate((FinalStream["Data"], np.zeros((nSampleSkipped)), Data[Type][nStream]["Data"]))
            FinalStream["Missing"] = np.concatenate((FinalStream["Missing"], np.ones((nSampleSkipped)), Data[Type][nStream]["Missing"]))
            
        FinalStream["Time"] = np.array(range(len(FinalStream["Data"]))) / Data[Type][nStream]["SamplingRate"]
        return FinalStream

def extractPerceptJSON(JSON):
    FunctionsToProcess = [extractPatientInformation, extractTherapySettings, extractStreamingData, 
                          extractIndefiniteStreaming, extractBrainSenseSurvey, extractSignalCalibration,
                          extractChronicLFP]
    
    Data = dict()
    for func in FunctionsToProcess:
        try:
            func(JSON,Data)
        except:
            pass
    
    Data = checkMissingPackage(Data)
    
    if "StreamingPower" in Data.keys() and "StreamingTD" in Data.keys():
        for i in range(len(Data["StreamingPower"])):
            Data["StreamingTD"][i]["PowerDomain"] = copy.deepcopy(Data["StreamingPower"][i])
            Data["StreamingTD"][i]["PowerDomain"]["Channel"] = Data["StreamingTD"][i]["PowerDomain"]["Channel"].replace("_Duplicate","")
    
    Data = processBreakingTimeDomain(Data)
    
    return Data

def extractPatientInformation(JSON, sourceData):
    Data = dict()
    if "PatientInformation" in JSON.keys():
        Data["PatientInformation"] = copy.deepcopy(JSON["PatientInformation"]["Final"])
        
    if "DeviceInformation" in JSON.keys():
        Data["DeviceInformation"] = copy.deepcopy(JSON["DeviceInformation"]["Final"])
        
    if "LeadConfiguration" in JSON.keys():
        Data["LeadConfiguration"] = copy.deepcopy(JSON["LeadConfiguration"]["Final"])

    if "LeadConfiguration" in JSON.keys():
        Data["BatteryInformation"] = copy.deepcopy(JSON["BatteryInformation"])

    if "Impedance" in JSON.keys():
        key = "Impedance"
        if len(JSON[key]) > 0:
            if "ImpedanceStatus" in JSON[key][-1].keys():
                Data["Impedance"] = dict()
                Data["Impedance"]["Status"] = JSON[key][-1]["ImpedanceStatus"].replace("ImpedanceStateDef.","")
                if "TestCurrentMA" in JSON[key][-1].keys():
                    Data["Impedance"]["Amplitude"] = JSON[key][-1]["TestCurrentMA"] + "mA"
                else:
                    Data["Impedance"]["Amplitude"] = JSON[key][-1]["TestVoltage"] + "V"
                    
                for impedanceData in JSON[key][-1]["Hemisphere"]:
                    hemisphere = impedanceData["Hemisphere"].replace("HemisphereLocationDef.","")
                    Data["Impedance"][hemisphere] = dict()
                    
                    numContacts = -1
                    for leadInfo in Data["LeadConfiguration"]:
                        if leadInfo["Hemisphere"].find(hemisphere) >= 0:
                            if leadInfo["Model"] == "LeadModelDef.LEAD_B33015":
                                Data["Impedance"][hemisphere]["LeadModel"] = "LEAD_B33015"
                                numContacts = 8
                            elif leadInfo["Model"] == "LeadModelDef.LEAD_B33005":
                                Data["Impedance"][hemisphere]["LeadModel"] = "LEAD_B33005"
                                numContacts = 8
                            elif leadInfo["Model"] == "LeadModelDef.LEAD_3387":
                                Data["Impedance"][hemisphere]["LeadModel"] = "LEAD_3387"
                                numContacts = 4
                            elif leadInfo["Model"] == "LeadModelDef.LEAD_3389":
                                Data["Impedance"][hemisphere]["LeadModel"] = "LEAD_3389"
                                numContacts = 4
                            else:
                                Data["Impedance"][hemisphere]["LeadModel"] = "Unknown"
                                numContacts = 4
                                
                    Data["Impedance"][hemisphere]["Monopolar"] = np.zeros((numContacts))
                    Data["Impedance"][hemisphere]["Bipolar"] = np.zeros((numContacts, numContacts))
                    
                    if Data["Impedance"][hemisphere]["LeadModel"] == "LEAD_B33015" or Data["Impedance"][hemisphere]["LeadModel"] == "LEAD_B33005":
                        for measurement in impedanceData["SessionImpedance"]["Monopolar"]:
                            _, electrodeID = reformatElectrodeDef(measurement["Electrode2"])
                            if measurement["ResultValue"] == "HIGH":
                                measurement["ResultValue"] = 999999
                            elif measurement["ResultValue"] == ">5K":
                                measurement["ResultValue"] = 9999
                            elif measurement["ResultValue"] == ">10K":
                                measurement["ResultValue"] = 99999
                            Data["Impedance"][hemisphere]["Monopolar"][electrodeID % 8] = measurement["ResultValue"]
                    
                        for measurement in impedanceData["SessionImpedance"]["Bipolar"]:
                            _, electrodeID1 = reformatElectrodeDef(measurement["Electrode1"])
                            _, electrodeID2 = reformatElectrodeDef(measurement["Electrode2"]) 
                            if measurement["ResultValue"] == "HIGH":
                                measurement["ResultValue"] = 999999
                            elif measurement["ResultValue"] == ">5K":
                                measurement["ResultValue"] = 9999
                            elif measurement["ResultValue"] == ">10K":
                                measurement["ResultValue"] = 99999
                            Data["Impedance"][hemisphere]["Bipolar"][electrodeID1 % 8][electrodeID2 % 8] = measurement["ResultValue"]
                            
                    elif Data["Impedance"][hemisphere]["LeadModel"] == "LEAD_3387" or Data["Impedance"][hemisphere]["LeadModel"] == "LEAD_3389":
                        for measurement in impedanceData["SessionImpedance"]["Monopolar"]:
                            _, electrodeID = reformatElectrodeDef(measurement["Electrode2"])
                            if measurement["ResultValue"] == "HIGH":
                                measurement["ResultValue"] = 999999
                            elif measurement["ResultValue"] == ">5K":
                                measurement["ResultValue"] = 9999
                            elif measurement["ResultValue"] == ">10K":
                                measurement["ResultValue"] = 99999
                            Data["Impedance"][hemisphere]["Monopolar"][electrodeID % 4] = measurement["ResultValue"]
                            
                        for measurement in impedanceData["SessionImpedance"]["Bipolar"]:
                            _, electrodeID1 = reformatElectrodeDef(measurement["Electrode1"])
                            _, electrodeID2 = reformatElectrodeDef(measurement["Electrode2"])
                            if measurement["ResultValue"] == "HIGH":
                                measurement["ResultValue"] = 999999
                            elif measurement["ResultValue"] == ">5K":
                                measurement["ResultValue"] = 9999
                            elif measurement["ResultValue"] == ">10K":
                                measurement["ResultValue"] = 99999
                            Data["Impedance"][hemisphere]["Bipolar"][electrodeID1 % 4][electrodeID2 % 4] = measurement["ResultValue"]
                  
    for key in Data.keys():
        sourceData[key] = Data[key]
    return Data

def processTherapySettings(TherapyGroup):
    Therapy = dict()
    Therapy["GroupId"] = TherapyGroup["GroupId"]
    if "GroupName" in TherapyGroup.keys():
        Therapy["GroupName"] = TherapyGroup["GroupName"]
    else:
        Therapy["GroupName"] = ""
    
    if "ActiveGroup" in TherapyGroup.keys():
        Therapy["ActiveGroup"] = TherapyGroup["ActiveGroup"]
    else:
        Therapy["ActiveGroup"] = False
    
    if not "ProgramSettings" in TherapyGroup.keys():
        return Therapy
    
    if "SensingChannel" in TherapyGroup["ProgramSettings"].keys():
        for side in range(len(TherapyGroup["ProgramSettings"]["SensingChannel"])):
            if TherapyGroup["ProgramSettings"]["SensingChannel"][side]["HemisphereLocation"] == "HemisphereLocationDef.Left":
                hemisphere = "LeftHemisphere"
            else:
                hemisphere = "RightHemisphere"
                
            Therapy[hemisphere] = dict()
            Therapy[hemisphere]["Mode"] = "BrainSense"
            if "RateInHertz" in TherapyGroup["ProgramSettings"].keys():
                Therapy[hemisphere]["Frequency"] = TherapyGroup["ProgramSettings"]["RateInHertz"]
            else:
                Therapy[hemisphere]["Frequency"] = TherapyGroup["ProgramSettings"]["SensingChannel"][side]["RateInHertz"]
                
            Therapy[hemisphere]["PulseWidth"] = TherapyGroup["ProgramSettings"]["SensingChannel"][side]["PulseWidthInMicroSecond"]
            Therapy[hemisphere]["Unit"] = "mA"
            Therapy[hemisphere]["Amplitude"] = TherapyGroup["ProgramSettings"]["SensingChannel"][side]["SuspendAmplitudeInMilliAmps"]
            Therapy[hemisphere]["Channel"] = TherapyGroup["ProgramSettings"]["SensingChannel"][side]["ElectrodeState"]
            Therapy[hemisphere]["LFPThresholds"] = [TherapyGroup["ProgramSettings"]["SensingChannel"][side]["LowerLfpThreshold"],
                                                    TherapyGroup["ProgramSettings"]["SensingChannel"][side]["UpperLfpThreshold"]]
            Therapy[hemisphere]["CaptureAmplitudes"] = [TherapyGroup["ProgramSettings"]["SensingChannel"][side]["LowerCaptureAmplitudeInMilliAmps"],
                                                        TherapyGroup["ProgramSettings"]["SensingChannel"][side]["UpperCaptureAmplitudeInMilliAmps"]]
            Therapy[hemisphere]["MeasuredLFP"] = [TherapyGroup["ProgramSettings"]["SensingChannel"][side]["MeasuredLowerLfp"],
                                                  TherapyGroup["ProgramSettings"]["SensingChannel"][side]["MeasuredUpperLfp"]]
            
            if "Mode" in TherapyGroup.keys():
                if TherapyGroup["Mode"] == "LimitModeDef.AdvanceEdit" and "LowerLimitInMilliAmps" in TherapyGroup["ProgramSettings"]["SensingChannel"][side].keys():
                    Therapy[hemisphere]["AmplitudeThreshold"] = [TherapyGroup["ProgramSettings"]["SensingChannel"][side]["LowerLimitInMilliAmps"],
                                                                 TherapyGroup["ProgramSettings"]["SensingChannel"][side]["UpperLimitInMilliAmps"]]
                
                elif hemisphere in TherapyGroup["ProgramSettings"].keys():
                    if TherapyGroup["Mode"] == "LimitModeDef.AdvanceEdit" and "LowerLimitInMilliAmps" in TherapyGroup["ProgramSettings"][hemisphere].keys():
                       Therapy[hemisphere]["AmplitudeThreshold"] = [TherapyGroup["ProgramSettings"][hemisphere]["LowerLimitInMilliAmps"],
                                                                    TherapyGroup["ProgramSettings"][hemisphere]["UpperLimitInMilliAmps"]]
            else:
                Therapy[hemisphere]["AmplitudeThreshold"] = [0,0]
                
            Therapy[hemisphere]["SensingSetup"] = TherapyGroup["ProgramSettings"]["SensingChannel"][side]["SensingSetup"]
            if "ChannelSignalResult" in Therapy[hemisphere]["SensingSetup"].keys():
                del(Therapy[hemisphere]["SensingSetup"]["ChannelSignalResult"])
            Therapy[hemisphere]["SensingSetup"]["Status"] = TherapyGroup["ProgramSettings"]["SensingChannel"][side]["BrainSensingStatus"]

    if "LeftHemisphere" in TherapyGroup["ProgramSettings"]:
        Therapy["LeftHemisphere"] = dict()
        if len(TherapyGroup["ProgramSettings"]["LeftHemisphere"]["Programs"]) > 1:
            Therapy["LeftHemisphere"]["Mode"] = "Interleaving"
        else:
            Therapy["LeftHemisphere"]["Mode"] = "Standard"
        
        if Therapy["LeftHemisphere"]["Mode"] == "Interleaving":
            if "RateInHertz" in TherapyGroup["ProgramSettings"].keys():
                Therapy["LeftHemisphere"]["Frequency"] = [TherapyGroup["ProgramSettings"]["RateInHertz"],TherapyGroup["ProgramSettings"]["RateInHertz"]]
            else:
                Therapy["LeftHemisphere"]["Frequency"] = [TherapyGroup["ProgramSettings"]["LeftHemisphere"]["Programs"][0]["RateInHertz"],
                                                          TherapyGroup["ProgramSettings"]["LeftHemisphere"]["Programs"][1]["RateInHertz"]]
            
            Therapy["LeftHemisphere"]["ProgramId"] = list()
            Therapy["LeftHemisphere"]["PulseWidth"] = list()
            Therapy["LeftHemisphere"]["Amplitude"] = list()
            Therapy["LeftHemisphere"]["Channel"] = list()
            Therapy["LeftHemisphere"]["Unit"] = list()
            for i in range(len(TherapyGroup["ProgramSettings"]["LeftHemisphere"]["Programs"])):
                Therapy["LeftHemisphere"]["ProgramId"].append(TherapyGroup["ProgramSettings"]["LeftHemisphere"]["Programs"][i]["ProgramId"])
                Therapy["LeftHemisphere"]["PulseWidth"].append(TherapyGroup["ProgramSettings"]["LeftHemisphere"]["Programs"][i]["PulseWidthInMicroSecond"])
                if "AmplitudeInMilliAmps" in TherapyGroup["ProgramSettings"]["LeftHemisphere"]["Programs"][i].keys():
                    Therapy["LeftHemisphere"]["Amplitude"].append(TherapyGroup["ProgramSettings"]["LeftHemisphere"]["Programs"][i]["AmplitudeInMilliAmps"])
                    Therapy["LeftHemisphere"]["Unit"].append("mA")
                elif "AmplitudeInVolts" in TherapyGroup["ProgramSettings"]["LeftHemisphere"]["Programs"][i].keys():
                    Therapy["LeftHemisphere"]["Amplitude"].append(TherapyGroup["ProgramSettings"]["LeftHemisphere"]["Programs"][i]["AmplitudeInVolts"])
                    Therapy["LeftHemisphere"]["Unit"].append("V")
                Therapy["LeftHemisphere"]["Channel"].append(TherapyGroup["ProgramSettings"]["LeftHemisphere"]["Programs"][i]["ElectrodeState"])
                
        else:
            if "RateInHertz" in TherapyGroup["ProgramSettings"].keys():
                Therapy["LeftHemisphere"]["Frequency"] = TherapyGroup["ProgramSettings"]["RateInHertz"]
            else:
                Therapy["LeftHemisphere"]["Frequency"] = TherapyGroup["ProgramSettings"]["LeftHemisphere"]["Programs"][0]["RateInHertz"]
            Therapy["LeftHemisphere"]["PulseWidth"] = TherapyGroup["ProgramSettings"]["LeftHemisphere"]["Programs"][0]["PulseWidthInMicroSecond"]
            if "AmplitudeInMilliAmps" in TherapyGroup["ProgramSettings"]["LeftHemisphere"]["Programs"][0].keys():
                Therapy["LeftHemisphere"]["Amplitude"] = TherapyGroup["ProgramSettings"]["LeftHemisphere"]["Programs"][0]["AmplitudeInMilliAmps"]
                Therapy["LeftHemisphere"]["Unit"] = "mA"
            elif "AmplitudeInVolts" in TherapyGroup["ProgramSettings"]["LeftHemisphere"]["Programs"][0].keys():
                Therapy["LeftHemisphere"]["Amplitude"] = TherapyGroup["ProgramSettings"]["LeftHemisphere"]["Programs"][0]["AmplitudeInVolts"]
                Therapy["LeftHemisphere"]["Unit"] = "V"
            Therapy["LeftHemisphere"]["Channel"] = TherapyGroup["ProgramSettings"]["LeftHemisphere"]["Programs"][0]["ElectrodeState"]

    if "RightHemisphere" in TherapyGroup["ProgramSettings"]:
        Therapy["RightHemisphere"] = dict()
        if len(TherapyGroup["ProgramSettings"]["RightHemisphere"]["Programs"]) > 1:
            Therapy["RightHemisphere"]["Mode"] = "Interleaving"
        else:
            Therapy["RightHemisphere"]["Mode"] = "Standard"
        
        if Therapy["RightHemisphere"]["Mode"] == "Interleaving":
            if "RateInHertz" in TherapyGroup["ProgramSettings"].keys():
                Therapy["RightHemisphere"]["Frequency"] = [TherapyGroup["ProgramSettings"]["RateInHertz"],TherapyGroup["ProgramSettings"]["RateInHertz"]]
            else:
                Therapy["RightHemisphere"]["Frequency"] = [TherapyGroup["ProgramSettings"]["RightHemisphere"]["Programs"][0]["RateInHertz"],
                                                          TherapyGroup["ProgramSettings"]["RightHemisphere"]["Programs"][1]["RateInHertz"]]
            
            Therapy["RightHemisphere"]["ProgramId"] = list()
            Therapy["RightHemisphere"]["PulseWidth"] = list()
            Therapy["RightHemisphere"]["Amplitude"] = list()
            Therapy["RightHemisphere"]["Channel"] = list()
            Therapy["RightHemisphere"]["Unit"] = list()
            for i in range(len(TherapyGroup["ProgramSettings"]["RightHemisphere"]["Programs"])):
                Therapy["RightHemisphere"]["ProgramId"].append(TherapyGroup["ProgramSettings"]["RightHemisphere"]["Programs"][i]["ProgramId"])
                Therapy["RightHemisphere"]["PulseWidth"].append(TherapyGroup["ProgramSettings"]["RightHemisphere"]["Programs"][i]["PulseWidthInMicroSecond"])
                if "AmplitudeInMilliAmps" in TherapyGroup["ProgramSettings"]["RightHemisphere"]["Programs"][i].keys():
                    Therapy["RightHemisphere"]["Amplitude"].append(TherapyGroup["ProgramSettings"]["RightHemisphere"]["Programs"][i]["AmplitudeInMilliAmps"])
                    Therapy["RightHemisphere"]["Unit"].append("mA")
                elif "AmplitudeInVolts" in TherapyGroup["ProgramSettings"]["RightHemisphere"]["Programs"][i].keys():
                    Therapy["RightHemisphere"]["Amplitude"].append(TherapyGroup["ProgramSettings"]["RightHemisphere"]["Programs"][i]["AmplitudeInVolts"])
                    Therapy["RightHemisphere"]["Unit"].append("V")
                Therapy["RightHemisphere"]["Channel"].append(TherapyGroup["ProgramSettings"]["RightHemisphere"]["Programs"][i]["ElectrodeState"])
                 
        else:
            if "RateInHertz" in TherapyGroup["ProgramSettings"].keys():
                Therapy["RightHemisphere"]["Frequency"] = TherapyGroup["ProgramSettings"]["RateInHertz"]
            else:
                Therapy["RightHemisphere"]["Frequency"] = TherapyGroup["ProgramSettings"]["RightHemisphere"]["Programs"][0]["RateInHertz"]
            Therapy["RightHemisphere"]["PulseWidth"] = TherapyGroup["ProgramSettings"]["RightHemisphere"]["Programs"][0]["PulseWidthInMicroSecond"]
            if "AmplitudeInMilliAmps" in TherapyGroup["ProgramSettings"]["RightHemisphere"]["Programs"][0].keys():
               Therapy["RightHemisphere"]["Amplitude"] = TherapyGroup["ProgramSettings"]["RightHemisphere"]["Programs"][0]["AmplitudeInMilliAmps"]
               Therapy["RightHemisphere"]["Unit"] = "mA"
            elif "AmplitudeInVolts" in TherapyGroup["ProgramSettings"]["RightHemisphere"]["Programs"][0].keys():
                Therapy["RightHemisphere"]["Amplitude"] = TherapyGroup["ProgramSettings"]["RightHemisphere"]["Programs"][0]["AmplitudeInVolts"]
                Therapy["RightHemisphere"]["Unit"] = "V"
            Therapy["RightHemisphere"]["Channel"] = TherapyGroup["ProgramSettings"]["RightHemisphere"]["Programs"][0]["ElectrodeState"]
    
    return Therapy

def extractTherapySettings(JSON, sourceData=dict()):
    Data = dict()
    if "Groups" in JSON.keys():
        key = "Groups"
        Data["PreviousGroups"] = list()
        for groupID in range(len(JSON[key]["Initial"])):
            Data["PreviousGroups"].append(processTherapySettings(JSON[key]["Initial"][groupID]))
            
    if "Groups" in JSON.keys():
        key = "Groups"
        Data["StimulationGroups"] = list()
        for groupID in range(len(JSON[key]["Final"])):
            Data["StimulationGroups"].append(processTherapySettings(JSON[key]["Final"][groupID]))
            
    if "GroupHistory" in JSON.keys():
        key = "GroupHistory"
        Data["TherapyHistory"] = list()
        for historyID in range(len(JSON[key])):
            therapySettings = list()
            CurrentDateSetting = datetime.fromisoformat(JSON["SessionDate"][:-1]+"+00:00").date() == datetime.fromisoformat(JSON["GroupHistory"][historyID]["SessionDate"][:-1]+"+00:00").date()
            for groupID in range(len(JSON[key][historyID]["Groups"])):
                therapySettings.append(processTherapySettings(JSON[key][historyID]["Groups"][groupID]))
            Data["TherapyHistory"].append({"DateTime": JSON[key][historyID]["SessionDate"], "Therapy": therapySettings})
    
    if "DiagnosticData" in JSON.keys():
        key = "DiagnosticData"
        if "EventLogs" in JSON[key].keys():
            Data["TherapyChangeHistory"] = list()
            for event in JSON[key]["EventLogs"]:
                if "ParameterTrendId" in event.keys() and "NewGroupId" in event.keys():
                    switchTime = datetime.fromisoformat(event["DateTime"][:-1]+"+00:00").astimezone(dateutil.tz.tzlocal())
                    
                    if not "OldGroupId" in event.keys():
                        event["OldGroupId"] = "GroupIdDef.GROUP_UNKNOWN"
                    
                    if not "NewGroupId" in event.keys():
                        event["NewGroupId"] = "GroupIdDef.GROUP_UNKNOWN"
                    
                    if event["OldGroupId"] == "GroupIdDef.GROUP_A":
                        OldGroupId = 0
                    elif event["OldGroupId"] == "GroupIdDef.GROUP_B":
                        OldGroupId = 1
                    elif event["OldGroupId"] == "GroupIdDef.GROUP_C":
                        OldGroupId = 2
                    elif event["OldGroupId"] == "GroupIdDef.GROUP_D":
                        OldGroupId = 3
                    else:
                        OldGroupId = -1
                    
                    if event["NewGroupId"] == "GroupIdDef.GROUP_A":
                        NewGroupId = 0
                    elif event["NewGroupId"] == "GroupIdDef.GROUP_B":
                        NewGroupId = 1
                    elif event["NewGroupId"] == "GroupIdDef.GROUP_C":
                        NewGroupId = 2
                    elif event["NewGroupId"] == "GroupIdDef.GROUP_D":
                        NewGroupId = 3
                    else:
                        NewGroupId = -1
                    
                    Data["TherapyChangeHistory"].append({"DateTime": switchTime, "OldGroup": OldGroupId, "NewGroup": NewGroupId, "OldGroupId": event["OldGroupId"], "NewGroupId": event["NewGroupId"]})
                        
    for key in Data.keys():
        sourceData[key] = Data[key]
    return Data

def extractStreamingData(JSON, sourceData=dict()):
    Data = dict()
    if "BrainSenseTimeDomain" in JSON.keys():
        key = "BrainSenseTimeDomain"
        Data["StreamingTD"] = copy.deepcopy(JSON[key])
        for Stream in Data["StreamingTD"]:
            Stream["Sequences"] = np.array(text2num(Stream["GlobalSequences"].split(",")))
            Stream["PacketSizes"] = np.array(text2num(Stream["GlobalPacketSizes"].split(",")))
            Stream["Ticks"] = np.array(text2num(Stream["TicksInMses"].split(",")))
            Stream["Data"] = np.array(Stream["TimeDomainData"])
            Stream["SamplingRate"] = text2num(Stream["SampleRateInHz"])
            Stream["Time"] = np.array(range(len(Stream["Data"]))) / Stream["SamplingRate"]
            del(Stream["GlobalSequences"])
            del(Stream["TicksInMses"])
            del(Stream["TimeDomainData"])
            del(Stream["SampleRateInHz"])
            
            if len(Stream["Sequences"]) > 1:
                # There is situation when StreamingTD first packet is outlier sequence packet
                if Stream["Sequences"][0] > Stream["Sequences"][1]:
                    Stream["Sequences"] = Stream["Sequences"][1:]
                    Stream["PacketSizes"] = Stream["PacketSizes"][1:]
                    Stream["Ticks"] = Stream["Ticks"][1:]
                    Stream["Data"] = Stream["Data"][125:]
                    Stream["Time"] = np.array(range(len(Stream["Data"]))) / Stream["SamplingRate"]
    
        i = 0
        while i < len(Data["StreamingTD"]):
            if len(Data["StreamingTD"][i]["Sequences"]) == 1:
                del(Data["StreamingTD"][i])
            else:
                i += 1
            
    if "BrainSenseLfp" in JSON.keys():
        key = "BrainSenseLfp"
        Data["StreamingPower"] = copy.deepcopy(JSON[key])
        for Stream in Data["StreamingPower"]:
            Stream["Power"] = np.ndarray((len(Stream["LfpData"]), 2))
            Stream["Stimulation"] = np.ndarray((len(Stream["LfpData"]), 2))
            Stream["Time"] = np.ndarray((len(Stream["LfpData"]), 1))
            Stream["Sequences"] = np.ndarray((len(Stream["LfpData"]), 1))
            Stream["TimeSinceStimulationChange"] = np.zeros((len(Stream["LfpData"]), 2))
            for PackageID in range(len(Stream["LfpData"])):
                Stream["Power"][PackageID,0] = Stream["LfpData"][PackageID]["Left"]["LFP"]
                Stream["Stimulation"][PackageID,0] = Stream["LfpData"][PackageID]["Left"]["mA"]
                Stream["Power"][PackageID,1] = Stream["LfpData"][PackageID]["Right"]["LFP"]
                Stream["Stimulation"][PackageID,1] = Stream["LfpData"][PackageID]["Right"]["mA"]
                Stream["Sequences"][PackageID] = Stream["LfpData"][PackageID]["Seq"]
                Stream["Time"][PackageID] = Stream["LfpData"][PackageID]["TicksInMs"] / 1000.0
                
            Stream["InitialTickInMs"] = Stream["LfpData"][0]["TicksInMs"] / 1000.0
            Stream["Time"] -= Stream["Time"][0]
            Stream["Time"] = Stream["Time"].flatten()
            Stream["SamplingRate"] = text2num(Stream["SampleRateInHz"])
            Stream["Sequences"] = Stream["Sequences"].flatten()
            
            for PackageID in range(len(Stream["LfpData"])):
                if PackageID == 0:        
                    lastStimulationChanged = [0,0]
                else:
                    for chan in range(2):
                        if Stream["Stimulation"][PackageID,chan] != Stream["Stimulation"][PackageID-1,chan]:
                            lastStimulationChanged[chan] = Stream["Time"][PackageID-1]
                        Stream["TimeSinceStimulationChange"][PackageID,chan] = Stream["Time"][PackageID] - lastStimulationChanged[chan]
            
            Stream["TimeSinceStimulationChange"] = np.round(Stream["TimeSinceStimulationChange"], 1)
            
            del(Stream["LfpData"])
            del(Stream["SampleRateInHz"])
        
    if "StreamingPower" in Data.keys():
        if len(Data["StreamingPower"]) < len(Data["StreamingTD"]):
            for i in range(1,len(Data["StreamingTD"])):
                if Data["StreamingTD"][i]["FirstPacketDateTime"] == Data["StreamingTD"][i-1]["FirstPacketDateTime"]:
                    Data["StreamingPower"].insert(i,copy.deepcopy(Data["StreamingPower"][i-1]))
                    Data["StreamingPower"][i]["Channel"] += "_Duplicate"
        
        # This is a process to delete extremely short time-domain data without corresponding power domain data.
        if len(Data["StreamingPower"]) < len(Data["StreamingTD"]):
            StreamToInclude = np.zeros(len(Data["StreamingTD"]),dtype=bool)
            Timestamps = np.array([datetime.fromisoformat(Data["StreamingTD"][i]["FirstPacketDateTime"][:-1]+"+00:00").timestamp() for i in range(len(Data["StreamingTD"]))])
            for i in range(len(Data["StreamingPower"])):
                TargetTimestamp = datetime.fromisoformat(Data["StreamingPower"][i]["FirstPacketDateTime"][:-1]+"+00:00").timestamp()
                StreamToInclude[np.argmin(abs(TargetTimestamp - Timestamps))] = True
            
            if np.sum(StreamToInclude) == len(Data["StreamingPower"]):
                for i in range(len(StreamToInclude)):
                    if not StreamToInclude[i]:
                        if Data["StreamingTD"][i]["Time"][-1] >= 2:
                            raise Exception(f"Attempting to delete {Data['StreamingTD'][i]['Time'][-1]} seconds of data. Check data content to confirm")
                        else:
                            del(Data["StreamingTD"][i])
        
        # This will be executed only if last procedure still didn't handle the problem of mismatch
        # One condition is that StreamingTD got divided but not StreamingPower.
        #   NOTE: After emailing with Medtronic, it is confirmed that this scenario is a very very rare error that they didn't realize either. 
        #         It is not worth the time to develop a whole new pipeline for it
        """
        if len(Data["StreamingPower"]) < len(Data["StreamingTD"]): 
            for i in range(len(Data["StreamingPower"])):
                if Data["StreamingPower"][i]["Time"][-1] - Data["StreamingTD"][i]["Time"][-1] > Data["StreamingTD"][i+1]["Time"][-1]
                    pass 
        """
    
    for key in Data.keys():
        sourceData[key] = Data[key]
        
    return Data

def extractIndefiniteStreaming(JSON, sourceData=dict()):
    Data = dict()
    if "IndefiniteStreaming" in JSON.keys():
        key = "IndefiniteStreaming"
        Data["IndefiniteStream"] = copy.deepcopy(JSON[key])
        for Stream in Data["IndefiniteStream"]:
            Stream["Sequences"] = np.array(text2num(Stream["GlobalSequences"].split(",")))
            Stream["PacketSizes"] = np.array(text2num(Stream["GlobalPacketSizes"].split(",")))
            Stream["Ticks"] = np.array(text2num(Stream["TicksInMses"].split(",")))
            Stream["Data"] = np.array(Stream["TimeDomainData"])
            Stream["SamplingRate"] = text2num(Stream["SampleRateInHz"])
            Stream["Time"] = np.array(range(len(Stream["Data"]))) / Stream["SamplingRate"]
            del(Stream["GlobalSequences"])
            del(Stream["TicksInMses"])
            del(Stream["TimeDomainData"])
            del(Stream["SampleRateInHz"])
        
    for key in Data.keys():
        sourceData[key] = Data[key]
    return Data

def extractBrainSenseSurvey(JSON, sourceData=dict()):
    Data = dict()
    if "LFPMontage" in JSON.keys():
        key = "LFPMontage"
        Data["MontagesPSD"] = copy.deepcopy(JSON[key])
    
    if "LfpMontageTimeDomain" in JSON.keys():
        key = "LfpMontageTimeDomain"
        Data["MontagesTD"] = copy.deepcopy(JSON[key])
        for Stream in Data["MontagesTD"]:
            Stream["Sequences"] = np.array(text2num(Stream["GlobalSequences"].split(",")))
            Stream["Ticks"] = np.array(text2num(Stream["TicksInMses"].split(",")))
            Stream["Data"] = np.array(Stream["TimeDomainData"])
            Stream["SamplingRate"] = text2num(Stream["SampleRateInHz"])
            Stream["Time"] = np.array(range(len(Stream["Data"]))) / Stream["SamplingRate"]
            del(Stream["GlobalSequences"])
            del(Stream["TicksInMses"])
            del(Stream["TimeDomainData"])
            del(Stream["SampleRateInHz"])
        
    for key in Data.keys():
        sourceData[key] = Data[key]
    return Data
    
def extractSignalCalibration(JSON, sourceData=dict()):
    Data = dict()
    if "SenseChannelTests" in JSON.keys():
        key = "SenseChannelTests"
        Data["BaselineTD"] = copy.deepcopy(JSON[key])
        for Stream in Data["BaselineTD"]:
            Stream["Sequences"] = np.array(text2num(Stream["GlobalSequences"].split(",")))
            Stream["Ticks"] = np.array(text2num(Stream["TicksInMses"].split(",")))
            Stream["Data"] = np.array(Stream["TimeDomainData"])
            Stream["SamplingRate"] = text2num(Stream["SampleRateInHz"])
            Stream["Time"] = np.array(range(len(Stream["Data"]))) / Stream["SamplingRate"]
            del(Stream["GlobalSequences"])
            del(Stream["TicksInMses"])
            del(Stream["TimeDomainData"])
            del(Stream["SampleRateInHz"])
    
    if "CalibrationTests" in JSON.keys():
        key = "CalibrationTests"
        Data["StimulationTD"] = copy.deepcopy(JSON[key])
        for Stream in Data["StimulationTD"]:
            Stream["Sequences"] = np.array(text2num(Stream["GlobalSequences"].split(",")))
            Stream["Ticks"] = np.array(text2num(Stream["TicksInMses"].split(",")))
            Stream["Data"] = np.array(Stream["TimeDomainData"])
            Stream["SamplingRate"] = text2num(Stream["SampleRateInHz"])
            Stream["Time"] = np.array(range(len(Stream["Data"]))) / Stream["SamplingRate"]
            del(Stream["GlobalSequences"])
            del(Stream["TicksInMses"])
            del(Stream["TimeDomainData"])
            del(Stream["SampleRateInHz"])
    
    for key in Data.keys():
        sourceData[key] = Data[key]
    return Data
    

def extractChronicLFP(JSON, sourceData=dict()):
    Data = dict()
    if "PatientEvents" in JSON.keys():
        key = "PatientEvents"
        if "Initial" in JSON[key].keys():
            Data["PatientEvents"] = copy.deepcopy(JSON[key]["Initial"])
    
    if "DiagnosticData" in JSON.keys():
        key = "DiagnosticData"
        for subkey in JSON[key].keys():
            if subkey == "LFPTrendLogs":
                Data["LFPTrends"] = dict()
                for hemisphere in JSON[key][subkey].keys():
                    Data["LFPTrends"][hemisphere] = list()
                    for date in JSON[key][subkey][hemisphere].keys():
                        Data["LFPTrends"][hemisphere].extend(JSON[key][subkey][hemisphere][date])

                # Further processing to arrange the data
                RealtimeLFP = dict()
                for hemisphere in Data["LFPTrends"].keys():
                    RealtimeLFP[hemisphere] = dict()
                    RealtimeLFP[hemisphere]["LFP"] = np.ndarray((len(Data["LFPTrends"][hemisphere]),1))
                    RealtimeLFP[hemisphere]["Amplitude"] = np.ndarray((len(Data["LFPTrends"][hemisphere]),1))
                    RealtimeLFP[hemisphere]["DateTime"] = list()
                    for packet in range(len(Data["LFPTrends"][hemisphere])):
                        RealtimeLFP[hemisphere]["LFP"][packet] = Data["LFPTrends"][hemisphere][packet]["LFP"]
                        RealtimeLFP[hemisphere]["Amplitude"][packet] = Data["LFPTrends"][hemisphere][packet]["AmplitudeInMilliAmps"]
                        RealtimeLFP[hemisphere]["DateTime"].append(datetime.fromisoformat(Data["LFPTrends"][hemisphere][packet]["DateTime"][:-1]+"+00:00").astimezone(dateutil.tz.tzlocal()))
                    sortedIndex = np.argsort(RealtimeLFP[hemisphere]["DateTime"],axis=0).flatten()
                    RealtimeLFP[hemisphere]["LFP"] = RealtimeLFP[hemisphere]["LFP"][sortedIndex].flatten()
                    RealtimeLFP[hemisphere]["Amplitude"] = RealtimeLFP[hemisphere]["Amplitude"][sortedIndex].flatten()
                    RealtimeLFP[hemisphere]["DateTime"] = np.array(listSort(RealtimeLFP[hemisphere]["DateTime"], sortedIndex)).flatten()
                
                    Data["LFPTrends"][hemisphere] = copy.deepcopy(RealtimeLFP[hemisphere])
                
        if subkey == "LfpFrequencySnapshotEvents":
            Data["PatientEventLogs"] = copy.deepcopy(JSON[key][subkey])
            for event in Data["PatientEventLogs"]:
                event["DateTime"] = datetime.fromisoformat(event["DateTime"][:-1]+"+00:00").astimezone(dateutil.tz.tzlocal())

            # Extract Events from Event Logs
            Data["PSDEvents"] = dict()
            Data["LFPEvents"] = dict()
            PSDArrays = dict({"ZERO_TWO_LEFT": {"Power": list(), "Time": list()},
                              "ONE_THREE_LEFT": {"Power": list(), "Time": list()},
                              "ZERO_TWO_RIGHT": {"Power": list(), "Time": list()},
                              "ONE_THREE_RIGHT": {"Power": list(), "Time": list()}})
            
            for event in JSON[key][subkey]:
                if event["EventName"] not in Data["LFPEvents"].keys():
                    Data["LFPEvents"][event["EventName"]] = list()
                    Data["PSDEvents"][event["EventName"]] = copy.deepcopy(PSDArrays)
                    
                Data["LFPEvents"][event["EventName"]].append(datetime.fromisoformat(event["DateTime"][:-1]+"+00:00").astimezone(dateutil.tz.tzlocal()))
                
                if "LfpFrequencySnapshotEvents" in event.keys():
                    for hemisphere in ["HemisphereLocationDef.Left","HemisphereLocationDef.Right"]:
                        if hemisphere in event["LfpFrequencySnapshotEvents"].keys():
                            channel = reformatChannelName(event["LfpFrequencySnapshotEvents"][hemisphere]["SenseID"])
                            if channel == [0,2] and hemisphere == "HemisphereLocationDef.Left":
                                Data["PSDEvents"][event["EventName"]]["ZERO_TWO_LEFT"]["Power"].append(event["LfpFrequencySnapshotEvents"][hemisphere]["FFTBinData"])
                                Data["PSDEvents"][event["EventName"]]["ZERO_TWO_LEFT"]["Time"].append(datetime.fromisoformat(event["DateTime"][:-1]+"+00:00").astimezone(dateutil.tz.tzlocal()))
                            elif channel == [1,3] and hemisphere == "HemisphereLocationDef.Left":
                                Data["PSDEvents"][event["EventName"]]["ONE_THREE_LEFT"]["Power"].append(event["LfpFrequencySnapshotEvents"][hemisphere]["FFTBinData"])
                                Data["PSDEvents"][event["EventName"]]["ONE_THREE_LEFT"]["Time"].append(datetime.fromisoformat(event["DateTime"][:-1]+"+00:00").astimezone(dateutil.tz.tzlocal()))
                            elif channel == [0,2] and hemisphere == "HemisphereLocationDef.Right":
                                Data["PSDEvents"][event["EventName"]]["ZERO_TWO_RIGHT"]["Power"].append(event["LfpFrequencySnapshotEvents"][hemisphere]["FFTBinData"])
                                Data["PSDEvents"][event["EventName"]]["ZERO_TWO_RIGHT"]["Time"].append(datetime.fromisoformat(event["DateTime"][:-1]+"+00:00").astimezone(dateutil.tz.tzlocal()))
                            elif channel == [1,3] and hemisphere == "HemisphereLocationDef.Right":
                                Data["PSDEvents"][event["EventName"]]["ONE_THREE_RIGHT"]["Power"].append(event["LfpFrequencySnapshotEvents"][hemisphere]["FFTBinData"])
                                Data["PSDEvents"][event["EventName"]]["ONE_THREE_RIGHT"]["Time"].append(datetime.fromisoformat(event["DateTime"][:-1]+"+00:00").astimezone(dateutil.tz.tzlocal()))

    for key in Data.keys():
        sourceData[key] = Data[key]
    return Data

def processTherapyHistory(JSON, Data, cacheFilename=None):
    # Saving Therapy History
    TherapyData = dict()
    TherapyData["TherapyHistory"] = []
    TherapyHistory = []
    TherapyData["TherapyChangeHistory"] = []
    TherapyData["TherapyHistoryEndDate"] = datetime.fromisoformat("1960-01-01T00:00:00+00:00")

    if not cacheFilename == None:
        if os.path.exists(cacheFilename):
            with open(cacheFilename, "rb") as filehandler:
                TherapyData = pickle.load(filehandler)

    TherapyData["TherapyHistory"] = concatenateLists([Data, TherapyData], ["TherapyHistory"])
    Timestamp = [datetime.fromisoformat(TherapyData["TherapyHistory"][i]["DateTime"][:-1]+"+00:00").timestamp() for i in range(len(TherapyData["TherapyHistory"]))]
    UniqueTimestamp = np.unique(np.array(Timestamp))
    for i in range(len(UniqueTimestamp)):
        for j in range(len(Timestamp)):
            if Timestamp[j] == UniqueTimestamp[i]:
                TherapyHistory.append(TherapyData["TherapyHistory"][j])
                break
    TherapyData["TherapyHistory"] = TherapyHistory
    Timestamp = [datetime.fromisoformat(TherapyData["TherapyHistory"][i]["DateTime"][:-1]+"+00:00").timestamp() for i in range(len(TherapyData["TherapyHistory"]))]
    TherapyData["TherapyHistory"] = listSort(TherapyData["TherapyHistory"], np.flip(np.argsort(Timestamp)))
    
    TherapyData["TherapyChangeHistory"] = concatenateLists([Data, TherapyData], ["TherapyChangeHistory"])
    TherapyData["TherapyChangeHistory"] = removeDuplicates(TherapyData["TherapyChangeHistory"])
    Timestamp = [TherapyData["TherapyChangeHistory"][i]["DateTime"].timestamp() for i in range(len(TherapyData["TherapyChangeHistory"]))]
    TherapyData["TherapyChangeHistory"] = listSort(TherapyData["TherapyChangeHistory"], np.argsort(Timestamp))

    if not JSON["SessionEndDate"] == "":
        TherapyEndDate = datetime.fromisoformat(JSON["SessionEndDate"]+"+00:00")
    elif "EventSummary" in JSON.keys():
        TherapyEndDate = datetime.fromisoformat(JSON["EventSummary"]["SessionEndDate"]+"+00:00")
    else:
        TherapyEndDate = datetime.fromisoformat(JSON["SessionDate"]+"+00:00")

    if TherapyData["TherapyHistoryEndDate"] < TherapyEndDate:
        TherapyData["TherapyHistoryEndDate"] = TherapyEndDate
    TherapyData["TherapyHistoryStartDate"] = datetime.fromisoformat(JSON["DeviceInformation"]["Final"]["ImplantDate"][:-1]+"+00:00")

    if not cacheFilename == None:
        with open(cacheFilename, "wb+") as fileHandler:
            pickle.dump(TherapyData, fileHandler)
            
    return TherapyData

def processBaselinePSDs(JSON, Data, cacheFilename=None):
    # Save Baseline PSD
    RestingPSD = dict()
    RestingPSD["SurveyPSD"] = list()
    
    if not cacheFilename == None:
        if os.path.exists(cacheFilename):
            with open(cacheFilename, "rb") as filehandler:
                RestingPSD = pickle.load(filehandler)

    if "BaselineTD" in Data.keys():
        Data["SurveyPSD"] = Data["BaselineTD"]
        RestingPSD["SurveyPSD"] = concatenateLists([Data, RestingPSD], ["SurveyPSD"])
        RestingPSD["SurveyPSD"] = removeDuplicates(RestingPSD["SurveyPSD"])
        Timestamp = [datetime.fromisoformat(RestingPSD["SurveyPSD"][i]["FirstPacketDateTime"][:-1]+"+00:00").timestamp() for i in range(len(RestingPSD["SurveyPSD"]))]
        RestingPSD["SurveyPSD"] = listSort(RestingPSD["SurveyPSD"], np.flip(np.argsort(Timestamp)))

    if "MontagesTD" in Data.keys():
        Data["SurveyPSD"] = Data["MontagesTD"]
        RestingPSD["SurveyPSD"] = concatenateLists([Data, RestingPSD], ["SurveyPSD"])
        RestingPSD["SurveyPSD"] = removeDuplicates(RestingPSD["SurveyPSD"])
        Timestamp = [datetime.fromisoformat(RestingPSD["SurveyPSD"][i]["FirstPacketDateTime"][:-1]+"+00:00").timestamp() for i in range(len(RestingPSD["SurveyPSD"]))]
        RestingPSD["SurveyPSD"] = listSort(RestingPSD["SurveyPSD"], np.flip(np.argsort(Timestamp)))

    if not cacheFilename == None:
        with open(cacheFilename, "wb+") as fileHandler:
            pickle.dump(RestingPSD, fileHandler)
    
    return RestingPSD

def processChronicLFPs(JSON, Data, cacheFilename=None):
    # Chronic Events
    ChronicLFPs = dict()
    ChronicLFPs["PowerRecords"] = list()
    ChronicLFPs["Events"] = list()

    if not cacheFilename == None:
        if os.path.exists(cacheFilename):
            with open(cacheFilename, "rb") as filehandler:
                ChronicLFPs = pickle.load(filehandler)

    if "LFPTrends" in Data.keys() and "TherapyChangeHistory" in Data.keys():
        LFPRecord = dict()
        LFPRecord["LFP"] = dict()
        LFPRecord["Amplitude"] = dict()
        LFPRecord["Time"] = dict()
        for hemisphere in Data["LFPTrends"].keys():
            LFPSelection = np.array(Data["LFPTrends"][hemisphere]["DateTime"]) < Data["TherapyChangeHistory"][0]["DateTime"]
            if np.any(LFPSelection):
                LFPRecord["LFP"][hemisphere] = Data["LFPTrends"][hemisphere]["LFP"].flatten()[LFPSelection]
                OutlierIndex = np.where(np.bitwise_or(LFPRecord["LFP"][hemisphere] > 1e8, LFPRecord["LFP"][hemisphere] < 0))[0]
                if not len(OutlierIndex) == np.sum(LFPSelection):
                    while len(OutlierIndex > 0):
                        LFPRecord["LFP"][hemisphere][OutlierIndex] = LFPRecord["LFP"][hemisphere][OutlierIndex-1]
                        OutlierIndex = np.where(np.bitwise_or(LFPRecord["LFP"][hemisphere] > 1e8, LFPRecord["LFP"][hemisphere] < 0))[0]
                LFPRecord["Amplitude"][hemisphere] = Data["LFPTrends"][hemisphere]["Amplitude"].flatten()[LFPSelection]
                LFPRecord["Time"][hemisphere] = np.array(Data["LFPTrends"][hemisphere]["DateTime"])[LFPSelection]
                LFPRecord["DateTime"] = LFPRecord["Time"][hemisphere][0]
        if len(LFPRecord["LFP"]) > 0:
            GroupChangeTimestamp = Data["TherapyChangeHistory"][0]["DateTime"].timestamp()
            TherapyHistoryTimePoints = [datetime.fromisoformat(Data["TherapyHistory"][i]["DateTime"][:-1]+"+00:00").timestamp() for i in range(len(Data["TherapyHistory"]))]
            GroupIndex = np.where(GroupChangeTimestamp - np.array(TherapyHistoryTimePoints) > 0)[0][0]
            LFPRecord["Therapy"] = Data["TherapyHistory"][GroupIndex]["Therapy"][Data["TherapyChangeHistory"][0]["OldGroup"]]
            ChronicLFPs["PowerRecords"].append(LFPRecord)

        for historyIndex in range(len(Data["TherapyChangeHistory"])-1):
            LFPRecord = dict()
            LFPRecord["LFP"] = dict()
            LFPRecord["Amplitude"] = dict()
            LFPRecord["Time"] = dict()
            for hemisphere in Data["LFPTrends"].keys():
                LFPSelection = rangeSelection(np.array(Data["LFPTrends"][hemisphere]["DateTime"]), [Data["TherapyChangeHistory"][historyIndex]["DateTime"], Data["TherapyChangeHistory"][historyIndex+1]["DateTime"]])
                if np.any(LFPSelection):
                    LFPRecord["LFP"][hemisphere] = Data["LFPTrends"][hemisphere]["LFP"].flatten()[LFPSelection]
                    OutlierIndex = np.where(np.bitwise_or(LFPRecord["LFP"][hemisphere] > 1e8, LFPRecord["LFP"][hemisphere] < 0))[0]
                    if not len(OutlierIndex) == np.sum(LFPSelection):
                        while len(OutlierIndex > 0):
                            LFPRecord["LFP"][hemisphere][OutlierIndex] = LFPRecord["LFP"][hemisphere][OutlierIndex-1]
                            OutlierIndex = np.where(np.bitwise_or(LFPRecord["LFP"][hemisphere] > 1e8, LFPRecord["LFP"][hemisphere] < 0))[0]
                    LFPRecord["Amplitude"][hemisphere] = Data["LFPTrends"][hemisphere]["Amplitude"].flatten()[LFPSelection]
                    LFPRecord["Time"][hemisphere] = np.array(Data["LFPTrends"][hemisphere]["DateTime"])[LFPSelection]
                    LFPRecord["DateTime"] = LFPRecord["Time"][hemisphere][0]
            if len(LFPRecord["LFP"]) > 0:
                GroupChangeTimestamp = Data["TherapyChangeHistory"][historyIndex+1]["DateTime"].timestamp()
                TherapyHistoryTimePoints = [datetime.fromisoformat(Data["TherapyHistory"][i]["DateTime"][:-1]+"+00:00").timestamp() for i in range(len(Data["TherapyHistory"]))]
                if np.any(GroupChangeTimestamp - np.array(TherapyHistoryTimePoints) < 0):
                    GroupIndex = np.where(GroupChangeTimestamp - np.array(TherapyHistoryTimePoints) < 0)[0][-1]
                    LFPRecord["Therapy"] = Data["TherapyHistory"][GroupIndex]["Therapy"][Data["TherapyChangeHistory"][historyIndex+1]["OldGroup"]]
                    ChronicLFPs["PowerRecords"].append(LFPRecord)

        LFPRecord = dict()
        LFPRecord["LFP"] = dict()
        LFPRecord["Amplitude"] = dict()
        LFPRecord["Time"] = dict()
        for hemisphere in Data["LFPTrends"].keys():
            LFPSelection = np.array(Data["LFPTrends"][hemisphere]["DateTime"]) > Data["TherapyChangeHistory"][-1]["DateTime"]
            if np.any(LFPSelection):
                LFPRecord["LFP"][hemisphere] = Data["LFPTrends"][hemisphere]["LFP"].flatten()[LFPSelection]
                OutlierIndex = np.where(np.bitwise_or(LFPRecord["LFP"][hemisphere] > 1e8, LFPRecord["LFP"][hemisphere] < 0))[0]
                if not len(OutlierIndex) == np.sum(LFPSelection):
                    while len(OutlierIndex > 0):
                        LFPRecord["LFP"][hemisphere][OutlierIndex] = LFPRecord["LFP"][hemisphere][OutlierIndex-1]
                        OutlierIndex = np.where(np.bitwise_or(LFPRecord["LFP"][hemisphere] > 1e8, LFPRecord["LFP"][hemisphere] < 0))[0]
                LFPRecord["Amplitude"][hemisphere] = Data["LFPTrends"][hemisphere]["Amplitude"].flatten()[LFPSelection]
                LFPRecord["Time"][hemisphere] = np.array(Data["LFPTrends"][hemisphere]["DateTime"])[LFPSelection]
                LFPRecord["DateTime"] = LFPRecord["Time"][hemisphere][0]
        if len(LFPRecord["LFP"]) > 0:
            GroupChangeTimestamp = Data["TherapyChangeHistory"][-1]["DateTime"].timestamp()
            TherapyHistoryTimePoints = [datetime.fromisoformat(Data["TherapyHistory"][i]["DateTime"][:-1]+"+00:00").timestamp() for i in range(len(Data["TherapyHistory"]))]
            GroupIndex = np.where(GroupChangeTimestamp - np.array(TherapyHistoryTimePoints) < 0)[0][-1]
            LFPRecord["Therapy"] = Data["TherapyHistory"][GroupIndex]["Therapy"][Data["TherapyChangeHistory"][-1]["NewGroup"]]
            ChronicLFPs["PowerRecords"].append(LFPRecord)

        ChronicLFPs["PowerRecords"] = removeDuplicates(ChronicLFPs["PowerRecords"])
        Timestamp = [ChronicLFPs["PowerRecords"][i]["DateTime"].timestamp() for i in range(len(ChronicLFPs["PowerRecords"]))]
        ChronicLFPs["PowerRecords"] = listSort(ChronicLFPs["PowerRecords"], np.flip(np.argsort(Timestamp)))
    
        i = 0
        while i < len(ChronicLFPs["PowerRecords"]):
            MatchingSession = [i]
            for j in range(i+1, len(ChronicLFPs["PowerRecords"])):
                if dictionaryCompare(ChronicLFPs["PowerRecords"][i]["Time"], ChronicLFPs["PowerRecords"][j]["Time"]):
                    MatchingSession.append(j)
            
            if len(MatchingSession) > 1:
                ToSave = -1
                for j in sorted(MatchingSession):
                    for hemisphereDef in ChronicLFPs["PowerRecords"][j]["Time"].keys():
                        SideDef = hemisphereDef.replace("HemisphereLocationDef.","")+"Hemisphere"
                        if SideDef in ChronicLFPs["PowerRecords"][j]["Therapy"].keys():
                            if "SensingSetup" in ChronicLFPs["PowerRecords"][j]["Therapy"][SideDef].keys():
                                if ChronicLFPs["PowerRecords"][j]["Therapy"][SideDef]["SensingSetup"]["FrequencyInHertz"] != 0:
                                    ToSave = j
                
                if ToSave > 0:
                    for j in np.flip(sorted(MatchingSession)):
                        if ToSave != j:
                            del(ChronicLFPs["PowerRecords"][j])
                else:
                    for j in np.flip(sorted(MatchingSession))[:-1]:
                        if ToSave != j:
                            del(ChronicLFPs["PowerRecords"][j])
            else:
                i += 1
    
    if "DiagnosticData" in JSON.keys():
        if "LfpFrequencySnapshotEvents" in JSON["DiagnosticData"].keys():
            for event in JSON["DiagnosticData"]["LfpFrequencySnapshotEvents"]:
                if "LfpFrequencySnapshotEvents" in event.keys():
                    for hemisphere in event["LfpFrequencySnapshotEvents"].keys():
                        ChronicLFPs["Events"].append(dict())
                        channel = reformatChannelName(event["LfpFrequencySnapshotEvents"][hemisphere]["SenseID"])
                        if channel == []:
                            # LEAD DEFINITION NOT SET
                            channel = [-1,-1]
                        for therapySetting in Data["PreviousGroups"]:
                            if therapySetting["GroupId"] == event["LfpFrequencySnapshotEvents"][hemisphere]["GroupId"]:
                                SideOfInterest = hemisphere.replace("HemisphereLocationDef.","")+"Hemisphere"
                                #TODO: TEMP FIX. Check with Medtronic about Event Sensing
                                if SideOfInterest not in therapySetting.keys():
                                    ChronicLFPs["Events"][-1]["Therapy"] = {"Frequency": -1,
                                                                          "PulseWidth": -1,
                                                                          "Channel": channel,
                                                                          "Hemisphere": hemisphere}
                                else:
                                    ChronicLFPs["Events"][-1]["Therapy"] = {"Frequency": therapySetting[SideOfInterest]["Frequency"],
                                                                          "PulseWidth": therapySetting[SideOfInterest]["PulseWidth"],
                                                                          "Channel": channel,
                                                                          "Hemisphere": hemisphere}
                        ChronicLFPs["Events"][-1]["Power"] = event["LfpFrequencySnapshotEvents"][hemisphere]["FFTBinData"]
                        ChronicLFPs["Events"][-1]["EventName"] = event["EventName"]
                        ChronicLFPs["Events"][-1]["Time"] = datetime.fromisoformat(event["DateTime"][:-1]+"+00:00").astimezone(dateutil.tz.tzlocal())
    
            ChronicLFPs["Events"] = removeDuplicates(ChronicLFPs["Events"])
            Timestamp = [ChronicLFPs["Events"][i]["Time"].timestamp() for i in range(len(ChronicLFPs["Events"]))]
            ChronicLFPs["Events"] = listSort(ChronicLFPs["Events"], np.flip(np.argsort(Timestamp)))

    if not cacheFilename == None:
        with open(cacheFilename, "wb+") as fileHandler:
            pickle.dump(ChronicLFPs, fileHandler)

    return ChronicLFPs

def processBrainSenseStream(JSON, Data, cacheFilename=None):
    # Real-time Streams
    RealtimeStream = dict()
    RealtimeStream["StreamingTD"] = list()
    RealtimeStream["StreamingPower"] = list()

    if not cacheFilename == None:
        if os.path.exists(cacheFilename):
            with open(cacheFilename, "rb") as filehandler:
                RealtimeStream = pickle.load(filehandler)

    if "StreamingTD" in Data.keys() and "StreamingPower" in Data.keys():
        RealtimeStream["StreamingTD"] = concatenateLists([Data, RealtimeStream], ["StreamingTD"])
        RealtimeStream["StreamingTD"] = removeDuplicates(RealtimeStream["StreamingTD"])
        Timestamp = [datetime.fromisoformat(RealtimeStream["StreamingTD"][i]["FirstPacketDateTime"][:-1]+"+00:00").timestamp() for i in range(len(RealtimeStream["StreamingTD"]))]
        RealtimeStream["StreamingTD"] = listSort(RealtimeStream["StreamingTD"], np.flip(np.argsort(Timestamp)))

        RealtimeStream["StreamingPower"] = concatenateLists([Data, RealtimeStream], ["StreamingPower"])
        RealtimeStream["StreamingPower"] = removeDuplicates(RealtimeStream["StreamingPower"])
        Timestamp = [datetime.fromisoformat(RealtimeStream["StreamingPower"][i]["FirstPacketDateTime"][:-1]+"+00:00").timestamp() for i in range(len(RealtimeStream["StreamingPower"]))]
        RealtimeStream["StreamingPower"] = listSort(RealtimeStream["StreamingPower"], np.flip(np.argsort(Timestamp)))

        if not cacheFilename == None:
            with open(cacheFilename, "wb+") as fileHandler:
                pickle.dump(RealtimeStream, fileHandler)

    return RealtimeStream
    
def processIndefiniteStream(JSON, Data, cacheFilename=None):
    # Montage Stream
    MontageStream = dict()
    MontageStream["IndefiniteStream"] = list()

    if not cacheFilename == None:
        if os.path.exists(cacheFilename):
            with open(cacheFilename, "rb") as filehandler:
                MontageStream = pickle.load(filehandler)

    if "IndefiniteStream" in Data.keys():
        Timestamp = list()
        for Stream in Data["IndefiniteStream"]:
            Timestamp.append(datetime.fromisoformat(Stream["FirstPacketDateTime"][:-1]+"+00:00").timestamp())
        Timestamp = np.array(Timestamp)

        UniqueSessions = list()
        for session in Timestamp:
            if np.sum(np.bitwise_and(np.array(UniqueSessions) > session - 5, np.array(UniqueSessions) < session + 5)) == 0:
                UniqueSessions.append(session)

        for sessionID in range(len(UniqueSessions)):
            MontageStream["IndefiniteStream"].append(list())
            for Stream in Data["IndefiniteStream"]:
                StreamTimestamp = datetime.fromisoformat(Stream["FirstPacketDateTime"][:-1]+"+00:00").timestamp()
                if StreamTimestamp < UniqueSessions[sessionID] + 5 and StreamTimestamp > UniqueSessions[sessionID] - 5:
                    MontageStream["IndefiniteStream"][-1].append(Stream)

        MontageStream["IndefiniteStream"] = removeDuplicates(MontageStream["IndefiniteStream"])

        if not cacheFilename == None:
            with open(cacheFilename, "wb+") as fileHandler:
                pickle.dump(MontageStream, fileHandler)

    return MontageStream

def extractPredictionModel(Data, nStream, CenterFrequency=-1, Normalized=False):
    PredictionModel = dict()
    
    if CenterFrequency < 0:
        CenterFrequency = Data["StreamingTD"][nStream]["Spectrum"]["PredictedCenterFrequency"]
    PowerSelection = np.bitwise_and(Data["StreamingTD"][nStream]["Spectrum"]["Frequency"] > CenterFrequency - 3, Data["StreamingTD"][nStream]["Spectrum"]["Frequency"] <= CenterFrequency + 3)

    if Normalized:
        SpectralFeature = np.mean(Data["StreamingTD"][nStream]["Spectrum"]["NormalizedPower"][PowerSelection,:],axis=0)
    else:
        SpectralFeature = np.mean(Data["StreamingTD"][nStream]["Spectrum"]["SegmentedPower"][PowerSelection,:],axis=0)

    xdata = np.zeros(0)
    ydata = np.zeros(0)
    for level in range(len(Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"])):
        if len(SpectralFeature[Data["StreamingTD"][nStream]["Spectrum"]["StimulationAmplitude"] == Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"][level]]) > 15:
            FeaturePower = SpectralFeature[Data["StreamingTD"][nStream]["Spectrum"]["StimulationAmplitude"] == Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"][level]]
            FeaturePower = SPU.removeOutlier(FeaturePower, method="zscore")
            #FeaturePower = SPU.removeOutlier(SPU.smooth(FeaturePower,5)[::5], method="zscore")
            #FeaturePower = np.mean(FeaturePower)
            ydata = np.append(ydata, FeaturePower)
            xdata = np.append(xdata, np.ones(FeaturePower.shape)*Data["StreamingTD"][nStream]["Spectrum"]["StimulationLevels"][level])

    if len(ydata) == 0:
        return PredictionModel

    PowerDecayFitFound = False
    try:
        scale = np.percentile(ydata,95)
        #ModelBounds = (0,[np.inf,20,np.inf,np.inf])
        ModelBounds = ([-np.inf,np.inf])
        parameters, fit_covariance = optimize.curve_fit(SPU.PowerDecayFunc, xdata, ydata/scale, bounds=ModelBounds, method="trf")
        ConfidenceError = np.sqrt(np.diag(fit_covariance))
        FitError = np.mean(np.power(ydata/scale - SPU.PowerDecayFunc(xdata, *parameters),2))
        fitted_xdata = np.linspace(np.min(xdata),np.max(xdata),int(np.max(xdata)-np.min(xdata))*10)
        fitted_line = SPU.PowerDecayFunc(fitted_xdata, *parameters)*scale
        PowerDecayFitFound = True
    except:
        pass

    InverseSigmoidFound = False
    try:
        scale = np.percentile(ydata,95)
        #ModelBounds = (0,[np.inf,100,np.max(xdata),np.inf])
        ModelBounds = ([-np.inf,np.inf])
        parameters, fit_covariance = optimize.curve_fit(SPU.InverseSigmoidFunc, xdata, ydata/scale, bounds=ModelBounds, method="trf")
        ConfidenceError = np.sqrt(np.diag(fit_covariance))
        ModelError = np.mean(np.power(ydata/scale - SPU.InverseSigmoidFunc(xdata, *parameters),2))
        if PowerDecayFitFound:
            if ModelError <= FitError:
                fitted_xdata = np.linspace(np.min(xdata),np.max(xdata),int(np.max(xdata)-np.min(xdata))*10)
                fitted_line = SPU.InverseSigmoidFunc(fitted_xdata, *parameters)*scale
                FitError = ModelError
                InverseSigmoidFound = True
        else:
            fitted_xdata = np.linspace(np.min(xdata),np.max(xdata),int(np.max(xdata)-np.min(xdata))*10)
            fitted_line = SPU.InverseSigmoidFunc(fitted_xdata, *parameters)*scale
            FitError = ModelError
            InverseSigmoidFound = True
    except:
        pass
    
    if PowerDecayFitFound or InverseSigmoidFound:
        if fitted_line[-1] > fitted_line[0]:
            PredictionModel["StimulationArtifacts"] = True
        else:
            PredictionModel["StimulationArtifacts"] = False
        PredictionModel["MaximumChanges"] = np.max(fitted_line) - np.min(fitted_line)
        PredictionModel["FinalLevel"] = np.min(fitted_line)
        PredictionModel["TransitionalPoint"] = fitted_xdata[np.argmax(np.abs(np.diff(fitted_line)))]
        PredictionModel["SuggestionAmplitude"] = fitted_xdata[np.argmin(np.abs(fitted_line - PredictionModel["MaximumChanges"]*0.05 - PredictionModel["FinalLevel"]))]
        PredictionModel["AmplitudeError"] = ConfidenceError[0]/parameters[0]
        PredictionModel["SlopeError"] = ConfidenceError[1]/parameters[1]
        PredictionModel["ShiftError"] = ConfidenceError[2]/parameters[2]
        PredictionModel["OffsetError"] = ConfidenceError[3]/parameters[3]
        PredictionModel["FitError"] = FitError
        PredictionModel["ModelError"] = np.divide(ConfidenceError,parameters)

    return PredictionModel

def extractPredictionForStimulation(xdata, ydata):
    uniqueAmplitude = np.unique(xdata)
    simplifiedYData = []
    for i in range(len(uniqueAmplitude)):
        simplifiedYData.append(np.median(ydata[xdata==uniqueAmplitude[i]]))
    
    PredictionModel = dict()
    PredictionData = dict()
    
    PowerDecayFitFound = False
    try:
        if len(simplifiedYData) > 4:
            scale = np.percentile(ydata,95)
            #ModelBounds = (0,[np.inf,20,np.inf,np.inf])
            ModelBounds = ([-np.inf,np.inf])
            parameters, fit_covariance = optimize.curve_fit(SPU.PowerDecayFunc, uniqueAmplitude+0.1, simplifiedYData, p0=[1,1,1,15], bounds=ModelBounds, method="trf")
            ConfidenceError = np.sqrt(np.diag(fit_covariance))
            FitError = np.mean(np.power(ydata/scale - SPU.PowerDecayFunc(xdata, *parameters),2))
            fitted_xdata = np.linspace(np.min(xdata),np.max(xdata),int(np.max(xdata)-np.min(xdata))*10)
            fitted_line = SPU.PowerDecayFunc(fitted_xdata, *parameters)*scale
            PowerDecayFitFound = True
        else:
            scale = np.percentile(ydata,95)
            #ModelBounds = (0,[np.inf,20,np.inf,np.inf])
            ModelBounds = ([-np.inf,np.inf])
            parameters, fit_covariance = optimize.curve_fit(SPU.PowerDecayFunc, xdata, ydata/scale, bounds=ModelBounds, method="trf")
            ConfidenceError = np.sqrt(np.diag(fit_covariance))
            FitError = np.mean(np.power(ydata/scale - SPU.PowerDecayFunc(xdata, *parameters),2))
            fitted_xdata = np.linspace(np.min(xdata),np.max(xdata),int(np.max(xdata)-np.min(xdata))*10)
            fitted_line = SPU.PowerDecayFunc(fitted_xdata, *parameters)*scale
            PowerDecayFitFound = True
    except:
        pass

    InverseSigmoidFound = False
    try:
        scale = np.percentile(ydata,95)
        #ModelBounds = (0,[np.inf,100,np.max(xdata),np.inf])
        ModelBounds = ([-np.inf,np.inf])
        parameters, fit_covariance = optimize.curve_fit(SPU.InverseSigmoidFunc, xdata, ydata/scale, bounds=ModelBounds, method="trf")
        ConfidenceError = np.sqrt(np.diag(fit_covariance))
        ModelError = np.mean(np.power(ydata/scale - SPU.InverseSigmoidFunc(xdata, *parameters),2))
        if PowerDecayFitFound:
            if ModelError <= FitError:
                fitted_xdata = np.linspace(np.min(xdata),np.max(xdata),int(np.max(xdata)-np.min(xdata))*10)
                fitted_line = SPU.InverseSigmoidFunc(fitted_xdata, *parameters)*scale
                FitError = ModelError
                InverseSigmoidFound = True
        else:
            fitted_xdata = np.linspace(np.min(xdata),np.max(xdata),int(np.max(xdata)-np.min(xdata))*10)
            fitted_line = SPU.InverseSigmoidFunc(fitted_xdata, *parameters)*scale
            FitError = ModelError
            InverseSigmoidFound = True
    except:
        pass
    
    if PowerDecayFitFound or InverseSigmoidFound:
        if len(fitted_line) == 0:
            return PredictionModel, PredictionData
    
        if fitted_line[-1] > fitted_line[0]:
            PredictionModel["StimulationArtifacts"] = True
        else:
            PredictionModel["StimulationArtifacts"] = False
            
        PredictionData["FittedLine"] = fitted_line
        PredictionData["StimulationAmplitude"] = fitted_xdata
        
        PredictionModel["MaximumChanges"] = np.max(fitted_line) - np.min(fitted_line)
        PredictionModel["FinalLevel"] = np.min(fitted_line)
        PredictionModel["TransitionalPoint"] = fitted_xdata[np.argmax(np.abs(np.diff(fitted_line)))]
        PredictionModel["SuggestionAmplitude"] = fitted_xdata[np.argmin(np.abs(fitted_line - PredictionModel["MaximumChanges"]*0.05 - PredictionModel["FinalLevel"]))]
        PredictionModel["AmplitudeError"] = ConfidenceError[0]/parameters[0]
        PredictionModel["SlopeError"] = ConfidenceError[1]/parameters[1]
        PredictionModel["ShiftError"] = ConfidenceError[2]/parameters[2]
        PredictionModel["OffsetError"] = ConfidenceError[3]/parameters[3]
        PredictionModel["FitError"] = FitError
        PredictionModel["ModelError"] = np.divide(ConfidenceError,parameters)

    return PredictionModel, PredictionData


if __name__ == "__main__":
    if len(sys.argv) == 3:
        Data = decodeJSON(sys.argv[1])
        Data = formatKeyName(Data)
        sio.savemat(sys.argv[2], Data, long_field_names=True)