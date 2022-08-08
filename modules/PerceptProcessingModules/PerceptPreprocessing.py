#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jackson Cagle, University of Florida, ©2021
@email: jackson.cagle@neurology.ufl.edu
@date: Wed Sep 15 11:56:13 2021
"""

import json
import sys, os
import numpy as np
import copy
from datetime import datetime, date
import dateutil
import pickle
import pandas as pd

from scipy import signal, stats
import Percept
import SignalProcessingUtility as SPU
from PythonUtility import *

def preprocessStreamingFiles(RealtimeStream, iStream=-1, cacheDirectory=None):
    for nStream in range(len(RealtimeStream["StreamingTD"])):
        if iStream < 0 or nStream == iStream:
            
            [b,a] = signal.butter(5, np.array([1,100])*2/RealtimeStream["StreamingTD"][nStream]["SamplingRate"], 'bp', output='ba')
            filtered = signal.filtfilt(b, a, RealtimeStream["StreamingTD"][nStream]["Data"])
            
            (channels, hemisphere) = Percept.reformatChannelName(RealtimeStream["StreamingTD"][nStream]["Channel"])
            if hemisphere == "Left":
                StimulationSide = 0
            else:
                StimulationSide = 1
            
            Timestamp = datetime.fromisoformat(RealtimeStream["StreamingTD"][nStream]["FirstPacketDateTime"][:-1]+"+00:00").timestamp()
            
            # Wavelet Computation
            RealtimeStream["StreamingTD"][nStream]["Wavelet"] = SPU.waveletTimeFrequency(filtered,
                                                              freq=np.array(range(1,200))/2, ma=int(RealtimeStream["StreamingTD"][nStream]["SamplingRate"]/2),
                                                              fs=RealtimeStream["StreamingTD"][nStream]["SamplingRate"])
            RealtimeStream["StreamingTD"][nStream]["Wavelet"]["Power"] = RealtimeStream["StreamingTD"][nStream]["Wavelet"]["Power"][:,::int(RealtimeStream["StreamingTD"][nStream]["SamplingRate"]/2)]
            RealtimeStream["StreamingTD"][nStream]["Wavelet"]["Time"] = RealtimeStream["StreamingTD"][nStream]["Wavelet"]["Time"][::int(RealtimeStream["StreamingTD"][nStream]["SamplingRate"]/2)]
            RealtimeStream["StreamingTD"][nStream]["Wavelet"]["Type"] = "Wavelet"
            del(RealtimeStream["StreamingTD"][nStream]["Wavelet"]["logPower"])
            
            # SFFT Computation
            RealtimeStream["StreamingTD"][nStream]["Spectrogram"] = SPU.defaultSpectrogram(filtered,
                                                              window=1.0, overlap=0.5,
                                                              frequency_resolution=0.5,
                                                              fs=RealtimeStream["StreamingTD"][nStream]["SamplingRate"])
            RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["Type"] = "Spectrogram"
            RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["Time"] += RealtimeStream["StreamingTD"][nStream]["Time"][0]/1000
            del(RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["logPower"])

            # Stimulation in Time-domain
            RealtimeStream["StreamingTD"][nStream]["Stimulation"] = np.interp(RealtimeStream["StreamingTD"][nStream]["Time"], RealtimeStream["StreamingTD"][nStream]["PowerDomain"]["Time"], RealtimeStream["StreamingTD"][nStream]["PowerDomain"]["Stimulation"][:,StimulationSide])
            ConstantStimulation = np.append(np.append(0, np.where(np.diff(RealtimeStream["StreamingTD"][nStream]["Stimulation"]) != 0)),len(RealtimeStream["StreamingTD"][nStream]["Stimulation"]))
            StimulationDuration = np.diff(ConstantStimulation) / RealtimeStream["StreamingTD"][nStream]["SamplingRate"]
            
            AltSideStimulation = np.interp(RealtimeStream["StreamingTD"][nStream]["Time"], RealtimeStream["StreamingTD"][nStream]["PowerDomain"]["Time"], RealtimeStream["StreamingTD"][nStream]["PowerDomain"]["Stimulation"][:,StimulationSide-1])
            AltSideConstantStimulation = np.append(np.append(0, np.where(np.diff(AltSideStimulation) != 0)),len(AltSideStimulation))
            AltSideStimulationDuration = np.diff(AltSideConstantStimulation) / RealtimeStream["StreamingTD"][nStream]["SamplingRate"]
            if len(AltSideStimulationDuration) > len(StimulationDuration):
                RealtimeStream["StreamingTD"][nStream]["Stimulation"] = AltSideStimulation
                ConstantStimulation = AltSideConstantStimulation
                StimulationDuration = AltSideStimulationDuration
                
                RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["Stimulation"] = np.interp(RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["Time"], RealtimeStream["StreamingTD"][nStream]["PowerDomain"]["Time"].flatten(), RealtimeStream["StreamingTD"][nStream]["PowerDomain"]["Stimulation"][:,StimulationSide-1].flatten())
                RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["TimeSinceStimulationChange"] = copy.deepcopy(RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["Time"])

            else:
                # Stimulation Sequence
                RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["Stimulation"] = np.interp(RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["Time"], RealtimeStream["StreamingTD"][nStream]["PowerDomain"]["Time"].flatten(), RealtimeStream["StreamingTD"][nStream]["PowerDomain"]["Stimulation"][:,StimulationSide].flatten())
                RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["TimeSinceStimulationChange"] = copy.deepcopy(RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["Time"])

            # Calculate time since stimulation changes for spectrogram method
            lastStimulationTime = RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["TimeSinceStimulationChange"][0]
            RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["TimeSinceStimulationChange"][0] = 0
            for i in range(1, len(RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["Time"])):
                if RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["Stimulation"][i] != RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["Stimulation"][i-1]:
                    lastStimulationTime = RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["Time"][i-1]
                    RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["TimeSinceStimulationChange"][i-1] = RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["Time"][i-1] - lastStimulationTime
                RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["TimeSinceStimulationChange"][i] = RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["Time"][i] - lastStimulationTime

 
            RealtimeStream["StreamingTD"][nStream]["StimulationEpochs"] = list()
            StimulationSortIndex = list()
            for i in range(len(StimulationDuration)):
                if StimulationDuration[i] > 10:
                    StimulationAmplitude = RealtimeStream["StreamingTD"][nStream]["Stimulation"][ConstantStimulation[i]+1]
                    StimulationEpoch = RealtimeStream["StreamingTD"][nStream]["Data"][ConstantStimulation[i]+500:ConstantStimulation[i+1]-500]

                    TimeSelection = np.bitwise_and(RealtimeStream["StreamingTD"][nStream]["PowerDomain"]["Time"] > RealtimeStream["StreamingTD"][nStream]["Time"][ConstantStimulation[i]+1] + 3,
                                                   RealtimeStream["StreamingTD"][nStream]["PowerDomain"]["Time"] < RealtimeStream["StreamingTD"][nStream]["Time"][ConstantStimulation[i+1]-1] - 2)
                    PowerDomainSignal = RealtimeStream["StreamingTD"][nStream]["PowerDomain"]["Power"][TimeSelection,StimulationSide]

                    freq, PSD = signal.welch(StimulationEpoch, fs = RealtimeStream["StreamingTD"][nStream]["SamplingRate"],
                                       nperseg=RealtimeStream["StreamingTD"][nStream]["SamplingRate"] * 1,
                                       noverlap=RealtimeStream["StreamingTD"][nStream]["SamplingRate"] * 0.5,
                                       nfft=RealtimeStream["StreamingTD"][nStream]["SamplingRate"] * 2, scaling="density")

                    StimulationSortIndex.append(StimulationAmplitude)

                    RealtimeStream["StreamingTD"][nStream]["StimulationEpochs"].append({"Stimulation": StimulationAmplitude,
                                                                              "Epoch": StimulationEpoch,
                                                                              "Frequency": freq,
                                                                              "PSD": PSD,
                                                                              "PowerDomain": PowerDomainSignal})
            RealtimeStream["StreamingTD"][nStream]["StimulationEpochs"] = listSort(RealtimeStream["StreamingTD"][nStream]["StimulationEpochs"], np.argsort(StimulationSortIndex))

            if len(RealtimeStream["StreamingTD"][nStream]["StimulationEpochs"]) > 0:
                frequency = RealtimeStream["StreamingTD"][nStream]["StimulationEpochs"][0]["Frequency"]
                fitted_label = np.ndarray(frequency.shape, dtype=bool)
                fitted_label[:] = True
                fitted_label[frequency < 2] = False
                fitted_label[frequency >= 10] = False
                fitted_label[frequency > 50] = True
                fitted_label[frequency > 90] = False

                for epoch in RealtimeStream["StreamingTD"][nStream]["StimulationEpochs"]:
                    fittedBaseline = SPU.fittedNormalization(np.log10(epoch["PSD"]), frequency, fitted_label, order=5)
                    epoch["NormalizedPSD"] = np.log10(epoch["PSD"]) - fittedBaseline
                    epoch["NormalizedPSD"] = np.power(10,epoch["NormalizedPSD"])

                TimePeriodSelection = RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["TimeSinceStimulationChange"] > 3
                RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["StimulationAmplitude"] = RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["Stimulation"][TimePeriodSelection]
                RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["StimulationLevels"] = np.unique(RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["StimulationAmplitude"])

                frequency = RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["Frequency"]
                fitted_label = np.ndarray(frequency.shape, dtype=bool)
                fitted_label[:] = True
                fitted_label[frequency < 2] = False
                fitted_label[frequency >= 10] = False
                fitted_label[frequency > 50] = True
                fitted_label[frequency > 90] = False

                RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["SegmentedPower"] = RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["Power"][:,TimePeriodSelection]
                RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["NormalizedPower"] = RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["Power"][:,TimePeriodSelection]
                for level in RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["StimulationLevels"]:
                    AverageSpectrum = np.mean(RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["NormalizedPower"][:, RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["StimulationAmplitude"] == level], axis=1)
                    fittedBaseline = SPU.fittedNormalization(np.log10(AverageSpectrum), frequency, fitted_label, order=3)
                    RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["NormalizedPower"][:, RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["StimulationAmplitude"] == level] = np.divide(RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["NormalizedPower"][:, RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["StimulationAmplitude"] == level], np.repeat(np.power(10, fittedBaseline).reshape((len(frequency),1)), sum(RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["StimulationAmplitude"] == level), axis=1))

                StimPSD = np.mean(RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["NormalizedPower"][:, RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["StimulationAmplitude"] == np.max(RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["StimulationAmplitude"])], axis=1) - np.mean(RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["NormalizedPower"][:, RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["StimulationAmplitude"] == np.min(RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["StimulationAmplitude"])], axis=1)
                BetaSelection = np.bitwise_and(RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["Frequency"] > 10, RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["Frequency"] < 55)
                CenterFrequency = RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["Frequency"][BetaSelection][np.where(StimPSD[BetaSelection].min() == StimPSD[BetaSelection])[0]]
                if len(CenterFrequency) > 1:
                    RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["PredictedCenterFrequency"] = 22
                else:
                    RealtimeStream["StreamingTD"][nStream]["Spectrogram"]["PredictedCenterFrequency"] = CenterFrequency[0]
            
            if not cacheDirectory == None:
                cacheFilename = f"StreamingTD_{RealtimeStream['StreamingTD'][nStream]['Channel']}_{int(Timestamp)}.preprocess"
                with open(cacheDirectory + os.path.sep + cacheFilename, "wb+") as fileHandler:
                    pickle.dump(RealtimeStream["StreamingTD"][nStream], fileHandler)
        
    return RealtimeStream

def preprocessIndefiniteStreamFiles(MontageStream, iStream=-1, cacheDirectory=None):
    [b,a] = signal.butter(5, np.array([1,100])*2/250, 'bp', output='ba')
    for i in range(len(MontageStream[iStream])):
        filtered = signal.filtfilt(b, a, MontageStream[iStream][i]["Data"])
        MontageStream[iStream][i]["Spectrum"] = SPU.defaultSpectrogram(filtered,
                                                          window=1.0, overlap=0.5,
                                                          frequency_resolution=0.5,
                                                          fs=250)
    if not cacheDirectory == None:
        Timestamp = datetime.fromisoformat(MontageStream[iStream][i]["FirstPacketDateTime"][:-1]+"+00:00").timestamp()
        cacheFilename = f"MontageStream_{int(Timestamp)}.preprocess"
        with open(cacheDirectory + os.path.sep + cacheFilename, "wb+") as fileHandler:
            pickle.dump(MontageStream[iStream], fileHandler)
