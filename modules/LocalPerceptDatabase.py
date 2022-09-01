import os, sys, pathlib
RESOURCES = str(pathlib.Path(__file__).parent.parent.resolve())
PERCEPT_DIR = os.getenv('PERCEPT_DIR')
sys.path.append(PERCEPT_DIR)

import json
import uuid
import numpy as np
import copy
from shutil import copyfile, rmtree
from datetime import datetime, date, timedelta
import dateutil, pytz
import pickle, joblib
import pandas as pd
from cryptography.fernet import Fernet
import hashlib

from scipy import signal, stats, optimize, interpolate
import matplotlib.pyplot as plt

import Percept
from PerceptPreprocessing import preprocessStreamingFiles, preprocessIndefiniteStreamFiles
import SignalProcessingUtility as SPU
from PythonUtility import *

import PerceptDashboard.models as models

DATABASE_PATH = os.environ.get('DATASERVER_PATH')
key = os.environ.get('ENCRYPTION_KEY')

def extractUserInfo(user):
    userInfo = dict()
    userInfo["Name"] = [user.first_name, user.last_name]
    userInfo["Email"] = user.email
    userInfo["Institute"] = user.institute
    userInfo["Clinician"] = user.is_clinician
    userInfo["Admin"] = user.is_admin
    userInfo["Demo"] = user.email == "Demo@bravo.edu"
    return userInfo

def retrieveProcessingSettings(config=dict()):
    options = {
        "RealtimeStream": {
            "SpectrogramMethod": {
                "name": "Time-Frequency Analysis Algorithm",
                "description": "",
                "options": ["Welch","Spectrogram","Wavelet"],
                "value": "Spectrogram"
            },
            "PSDMethod": {
                "name": "Stimulation Epoch Power Spectrum Algorithm",
                "description": "",
                "options": ["Welch","Time-Frequency Analysis"],
                "value": "Welch"
            },
            "NormalizedPSD": {
                "name": "Normalize Stimulation Epoch Power Spectrum",
                "description": "",
                "options": ["true", "false"],
                "value": "false"
            },
        }
    }

    for key in config.keys():
        if type(config[key]) == dict:
            for subkey in config[key].keys():
                if type(config[key][subkey]) == dict and subkey in options[key].keys():
                    if config[key][subkey]["name"] == options[key][subkey]["name"] and config[key][subkey]["description"] == options[key][subkey]["description"] and config[key][subkey]["options"] == options[key][subkey]["options"]:
                        options[key][subkey]["value"] = config[key][subkey]["value"]
    return options

def colorTextFromCmap(color):
    if type(color) == str:
        colorInfoString = color.split(",")
        colorInfoString = [string.replace("rgb(","").replace(")","") for string in colorInfoString]
        colorInfo = [int(i) for i in colorInfoString]
    else:
        colorInfo = np.array(color[:-1]) * 255
    colorText = f"#{hex(int(colorInfo[0])).replace('0x',''):0>2}{hex(int(colorInfo[1])).replace('0x',''):0>2}{hex(int(colorInfo[2])).replace('0x',''):0>2}"
    return colorText

def retrievePatientInformation(PatientInformation, Institute, encoder=None):
    if not encoder:
        encoder = secureEncoder = Fernet(key)
        
    FirstName = encoder.encrypt(PatientInformation["PatientFirstName"].capitalize().encode('utf_8')).decode("utf-8")
    LastName = encoder.encrypt(PatientInformation["PatientLastName"].capitalize().encode('utf_8')).decode("utf-8")
    Diagnosis = PatientInformation["Diagnosis"].replace("DiagnosisTypeDef.","")
    MRN = encoder.encrypt(PatientInformation["PatientId"].encode('utf_8')).decode("utf-8")

    hashfield = hashlib.sha256((PatientInformation["PatientFirstName"].capitalize() + " " + PatientInformation["PatientLastName"].capitalize()).encode("utf-8")).hexdigest()

    try:
        PatientDateOfBirth = datetime.fromisoformat(PatientInformation["PatientDateOfBirth"][:-1]+"+00:00")
    except:
        PatientDateOfBirth = datetime.fromtimestamp(0)

    newPatient = False
    try:
        patient = models.Patient.objects.get(patient_identifier_hashfield=hashfield, birth_date=PatientDateOfBirth, diagnosis=Diagnosis, institute=Institute)
    except:
        patient = models.Patient(first_name=FirstName, last_name=LastName, patient_identifier_hashfield=hashfield, birth_date=PatientDateOfBirth, diagnosis=Diagnosis, medical_record_number=MRN, institute=Institute)
        patient.save()
        newPatient = True

    return patient, newPatient

def saveTherapySettings(deviceID, therapyList, sessionDate, type, sourceFile):
    NewTherapyFound = False
    TheraySavingList = list()
    for therapy in therapyList:
        TherapyObject = models.TherapyHistory.objects.filter(device_deidentified_id=deviceID, therapy_date=sessionDate, therapy_type=type, group_id=therapy["GroupId"]).all()
        TherapyFound = False

        for pastTherapy in TherapyObject:
            if pastTherapy.extractTherapy() == therapy:
                TherapyFound = True
                break
        if not TherapyFound:
            TheraySavingList.append(models.TherapyHistory(device_deidentified_id=deviceID, therapy_date=sessionDate, source_file=sourceFile,
                                  group_name=therapy["GroupName"], group_id=therapy["GroupId"], active_group=therapy["ActiveGroup"],
                                  therapy_type=type, therapy_details=therapy))
            NewTherapyFound = True

    if len(TheraySavingList) > 0:
        models.TherapyHistory.objects.bulk_create(TheraySavingList,ignore_conflicts=True)

    return NewTherapyFound

def saveBrainSenseSurvey(deviceID, surveyList, sourceFile):
    NewRecordingFound = False
    for survey in surveyList:
        SurveyDate = datetime.fromtimestamp(Percept.getTimestamp(survey["FirstPacketDateTime"]), tz=pytz.utc)
        recording_info = {"Channel": survey["Channel"]}
        if not models.BrainSenseRecording.objects.filter(device_deidentified_id=deviceID, recording_type="BrainSenseSurvey", recording_date=SurveyDate, recording_info=recording_info).exists():
            recording = models.BrainSenseRecording(device_deidentified_id=deviceID, recording_date=SurveyDate, recording_type="BrainSenseSurvey", recording_info=recording_info, source_file=sourceFile)
            filename = saveSourceFiles(survey, "BrainSenseSurvey", survey["Channel"], recording.recording_id)
            recording.recording_datapointer = filename
            recording.save()
            NewRecordingFound = True
    return NewRecordingFound

def saveMontageStreams(deviceID, streamList, sourceFile):
    NewRecordingFound = False
    StreamDates = list()
    for stream in streamList:
        StreamDates.append(datetime.fromtimestamp(Percept.getTimestamp(stream["FirstPacketDateTime"]), tz=pytz.utc))
    UniqueSessionDates = np.unique(StreamDates)

    for date in UniqueSessionDates:
        recording_data = dict()
        for stream in streamList:
            if datetime.fromtimestamp(Percept.getTimestamp(stream["FirstPacketDateTime"]), tz=pytz.utc) == date:
                if len(recording_data.keys()) == 0:
                    recording_data["Time"] = stream["Time"]
                    recording_data["Missing"] = stream["Missing"]
                    recording_data["Channels"] = list()
                recording_data["Channels"].append(stream["Channel"])
                recording_data[stream["Channel"]] = stream["Data"]

        recording_info = {"Channel": recording_data["Channels"]}
        if not models.BrainSenseRecording.objects.filter(device_deidentified_id=deviceID, recording_type="IndefiniteStream", recording_date=date, recording_info=recording_info).exists():
            recording = models.BrainSenseRecording(device_deidentified_id=deviceID, recording_date=date, source_file=sourceFile,
                                  recording_type="IndefiniteStream", recording_info=recording_info)
            filename = saveSourceFiles(recording_data, "IndefiniteStream", "Combined", recording.recording_id)
            recording.recording_datapointer = filename
            recording.recording_duration = recording_data["Time"][-1]
            recording.save()
            NewRecordingFound = True
    return NewRecordingFound

def saveRealtimeStreams(deviceID, StreamingTD, StreamingPower, sourceFile):
    NewRecordingFound = False
    StreamDates = list()
    for stream in StreamingTD:
        StreamDates.append(datetime.fromtimestamp(Percept.getTimestamp(stream["FirstPacketDateTime"]), tz=pytz.utc))
    UniqueSessionDates = np.unique(StreamDates)

    for date in UniqueSessionDates:
        selectedIndex = np.where(iterativeCompare(StreamDates, date, "equal").flatten())[0]
        recording_data = dict()
        recording_data["Missing"] = dict()
        recording_data["Channels"] = list()
        for index in selectedIndex:
            recording_data["Channels"].append(StreamingTD[index]["Channel"])
            recording_data[StreamingTD[index]["Channel"]] = StreamingTD[index]["Data"]
            recording_data["Time"] = StreamingTD[index]["Time"]
            recording_data["Missing"][StreamingTD[index]["Channel"]] = StreamingTD[index]["Missing"]
            recording_data["Stimulation"] = np.zeros((len(recording_data["Time"]),2))
            recording_data["PowerBand"] = np.zeros((len(recording_data["Time"]),2))
            for i in range(2):
                recording_data["Stimulation"][:,i] = np.interp(StreamingTD[index]["Time"], StreamingPower[index]["Time"], StreamingPower[index]["Stimulation"][:,i])
                recording_data["PowerBand"][:,i] = np.interp(StreamingTD[index]["Time"], StreamingPower[index]["Time"], StreamingPower[index]["Power"][:,i])
            recording_data["Therapy"] = StreamingPower[index]["TherapySnapshot"]

        recording_info = {"Channel": recording_data["Channels"], "Therapy": recording_data["Therapy"]}
        if not models.BrainSenseRecording.objects.filter(device_deidentified_id=deviceID, recording_type="BrainSenseStream", recording_date=date, recording_info__Channel=recording_data["Channels"]).exists():
            recording = models.BrainSenseRecording(device_deidentified_id=deviceID, recording_date=date, source_file=sourceFile, recording_type="BrainSenseStream", recording_info=recording_info)
            if len(selectedIndex) == 2:
                info = "Bilateral"
            else:
                info = "Unilateral"
            filename = saveSourceFiles(recording_data, "BrainSenseStream", info, recording.recording_id)
            recording.recording_datapointer = filename
            recording.recording_duration = recording_data["Time"][-1]
            recording.save()
            NewRecordingFound = True
        else:
            recording = models.BrainSenseRecording.objects.filter(device_deidentified_id=deviceID, recording_type="BrainSenseStream", recording_date=date, recording_info__Channel=recording_data["Channels"]).first()
            if len(selectedIndex) == 2:
                info = "Bilateral"
            else:
                info = "Unilateral"
            filename = saveSourceFiles(recording_data, "BrainSenseStream", info, recording.recording_id)

    return NewRecordingFound

def saveChronicLFP(deviceID, ChronicLFPs, sourceFile):
    NewRecordingFound = False
    for key in ChronicLFPs.keys():
        recording_info = {"Hemisphere": key}
        if not models.BrainSenseRecording.objects.filter(device_deidentified_id=deviceID, recording_type="ChronicLFPs", recording_info__Hemisphere=recording_info["Hemisphere"]).exists():
            recording = models.BrainSenseRecording(device_deidentified_id=deviceID, recording_type="ChronicLFPs", recording_info=recording_info)
            filename = saveSourceFiles(ChronicLFPs[key], "ChronicLFPs", key.replace("HemisphereLocationDef.",""), recording.recording_id)
            recording.recording_datapointer = filename
            recording.save()
            NewRecordingFound = True
        else:
            recording = models.BrainSenseRecording.objects.filter(device_deidentified_id=deviceID, recording_type="ChronicLFPs", recording_info__Hemisphere=recording_info["Hemisphere"]).first()
            pastChronicLFPs = loadSourceFiles("ChronicLFPs", key.replace("HemisphereLocationDef.",""), recording.recording_id)

            Common = set(ChronicLFPs[key]["DateTime"]) & set(pastChronicLFPs["DateTime"])

            toInclude = np.zeros(len(ChronicLFPs[key]["DateTime"]), dtype=bool)
            IndexToInclude = list()
            for i in range(len(ChronicLFPs[key]["DateTime"])):
                if not ChronicLFPs[key]["DateTime"][i] in Common:
                    toInclude[i] = True

            if np.any(toInclude):
                pastChronicLFPs["DateTime"] = np.concatenate((pastChronicLFPs["DateTime"], ChronicLFPs[key]["DateTime"][toInclude]),axis=0)
                pastChronicLFPs["Amplitude"] = np.concatenate((pastChronicLFPs["Amplitude"], ChronicLFPs[key]["Amplitude"][toInclude]),axis=0)
                pastChronicLFPs["LFP"] = np.concatenate((pastChronicLFPs["LFP"], ChronicLFPs[key]["LFP"][toInclude]),axis=0)

                sortedIndex = np.argsort(pastChronicLFPs["DateTime"],axis=0).flatten()
                pastChronicLFPs["DateTime"] = pastChronicLFPs["DateTime"][sortedIndex]
                pastChronicLFPs["Amplitude"] = pastChronicLFPs["Amplitude"][sortedIndex]
                pastChronicLFPs["LFP"] = pastChronicLFPs["LFP"][sortedIndex]
                filename = saveSourceFiles(pastChronicLFPs, "ChronicLFPs", key.replace("HemisphereLocationDef.",""), recording.recording_id)
                NewRecordingFound = True

    return NewRecordingFound

def saveBrainSenseEvents(deviceID, LfpFrequencySnapshotEvents, sourceFile):
    NewRecordingFound = False
    batchStorage = list()
    for event in LfpFrequencySnapshotEvents:
        EventTime = datetime.fromtimestamp(Percept.getTimestamp(event["DateTime"]),tz=pytz.utc)
        SensingExist = False
        if "LfpFrequencySnapshotEvents" in event.keys():
            SensingExist = True
            EventData = event["LfpFrequencySnapshotEvents"]

        if not models.PatientCustomEvents.objects.filter(device_deidentified_id=deviceID, event_name=event["EventName"], event_time=EventTime, sensing_exist=SensingExist).exists():
            customEvent = models.PatientCustomEvents(device_deidentified_id=deviceID, event_name=event["EventName"], event_time=EventTime, sensing_exist=SensingExist)
            if SensingExist:
                customEvent.brainsense_psd = EventData
            batchStorage.append(customEvent)

    if len(batchStorage) > 0:
        NewRecordingFound = True
        models.PatientCustomEvents.objects.bulk_create(batchStorage,ignore_conflicts=True)

    return NewRecordingFound

def processBrainSenseSurvey(survey, method="spectrogram"):
    [b,a] = signal.butter(5, np.array([1,100])*2/250, 'bp', output='ba')
    filtered = signal.filtfilt(b, a, survey["Data"])
    survey["Spectrum"] = SPU.defaultSpectrogram(filtered, window=1.0, overlap=0.5,frequency_resolution=0.5, fs=250)
    return survey

def processMontageStreams(stream, method="spectrogram"):
    [b,a] = signal.butter(5, np.array([1,100])*2/250, 'bp', output='ba')
    stream["Spectrums"] = dict()
    for channel in stream["Channels"]:
        filtered = signal.filtfilt(b, a, stream[channel])
        stream["Spectrums"][channel] = SPU.defaultSpectrogram(filtered, window=1.0, overlap=0.5,frequency_resolution=1, fs=250)
    return stream

def processRealtimeStreams(stream, cardiacFilter=False):
    stream["Wavelet"] = dict()
    stream["Spectrogram"] = dict()
    stream["Filtered"] = dict()

    for channel in stream["Channels"]:
        [b,a] = signal.butter(5, np.array([1,100])*2/250, 'bp', output='ba')
        stream["Filtered"][channel] = signal.filtfilt(b, a, stream[channel])
        (channels, hemisphere) = Percept.reformatChannelName(channel)
        if hemisphere == "Left":
            StimulationSide = 0
        else:
            StimulationSide = 1

        if cardiacFilter:
            # Cardiac Filter
            posPeaks,_ = signal.find_peaks(stream["Filtered"][channel], prominence=[10,200], distance=250*0.5)
            PosCardiacVariability = np.std(np.diff(posPeaks))
            negPeaks,_ = signal.find_peaks(-stream["Filtered"][channel], prominence=[10,200], distance=250*0.5)
            NegCardiacVariability = np.std(np.diff(negPeaks))

            if PosCardiacVariability < NegCardiacVariability:
                peaks = posPeaks
            else:
                peaks = negPeaks
            CardiacRate = int(np.mean(np.diff(peaks)))

            PrePeak = int(CardiacRate*0.25)
            PostPeak = int(CardiacRate*0.65)
            EKGMatrix = np.zeros((len(peaks)-2,PrePeak+PostPeak))
            for i in range(1,len(peaks)-1):
                EKGMatrix[i-1,:] = stream["Filtered"][channel][peaks[i]-PrePeak:peaks[i]+PostPeak]

            EKGTemplate = np.mean(EKGMatrix,axis=0)
            EKGTemplate = EKGTemplate / (np.max(EKGTemplate)-np.min(EKGTemplate))

            def EKGTemplateFunc(xdata, amplitude, offset):
                return EKGTemplate * amplitude + offset

            for i in range(len(peaks)):
                if peaks[i]-PrePeak < 0:
                    pass
                elif peaks[i]+PostPeak >= len(stream["Filtered"][channel]) :
                    pass
                else:
                    sliceSelection = np.arange(peaks[i]-PrePeak,peaks[i]+PostPeak)
                    params, covmat = optimize.curve_fit(EKGTemplateFunc, sliceSelection, stream["Filtered"][channel][sliceSelection])
                    stream["Filtered"][channel][sliceSelection] = stream["Filtered"][channel][sliceSelection] - EKGTemplateFunc(sliceSelection, *params)

        # Wavelet Computation
        stream["Wavelet"][channel] = SPU.waveletTimeFrequency(stream["Filtered"][channel], freq=np.array(range(1,200))/2, ma=125, fs=250)
        stream["Wavelet"][channel]["Power"] = stream["Wavelet"][channel]["Power"][:,::int(250/2)]
        stream["Wavelet"][channel]["Time"] = stream["Wavelet"][channel]["Time"][::int(250/2)]
        stream["Wavelet"][channel]["Type"] = "Wavelet"
        del(stream["Wavelet"][channel]["logPower"])

        # SFFT Computation
        stream["Spectrogram"][channel] = SPU.defaultSpectrogram(stream["Filtered"][channel], window=1.0, overlap=0.5, frequency_resolution=0.5, fs=250)
        stream["Spectrogram"][channel]["Type"] = "Spectrogram"
        stream["Spectrogram"][channel]["Time"] += stream["Time"][0]
        del(stream["Spectrogram"][channel]["logPower"])

    return stream

def saveSourceFiles(datastruct, datatype, info, id):
    filename = datatype + "_" + info + "_" + str(id) + ".pkl"
    with open(DATABASE_PATH + "recordings" + os.path.sep + filename, "wb+") as file:
        pickle.dump(datastruct, file)
    return filename

def loadSourceFiles(datatype, info, id):
    datastruct = dict()
    filename = datatype + "_" + info + "_" + str(id) + ".pkl"
    with open(DATABASE_PATH + "recordings" + os.path.sep + filename, "rb") as file:
        datastruct = pickle.load(file)
    return datastruct

def loadSourceDataPointer(filename):
    with open(DATABASE_PATH + "recordings" + os.path.sep + filename, "rb") as file:
        datastruct = pickle.load(file)
    return datastruct

def getPerceptDevices(user, patientUniqueID, authority):
    availableDevices = None
    if authority["Level"] == 1:
        if user.is_clinician or user.is_admin:
            availableDevices = models.PerceptDevice.objects.filter(patient_deidentified_id=patientUniqueID, authority_level="Clinic", authority_user=user.institute).all()
        else:
            availableDevices = models.PerceptDevice.objects.filter(patient_deidentified_id=patientUniqueID, authority_level="Research", authority_user=user.email).all()
    elif authority["Level"] == 2:
        availableDevices = models.PerceptDevice.objects.filter(patient_deidentified_id=patientUniqueID).all()
        for device in availableDevices:
            device.serial_number = str(device.deidentified_id)

    return availableDevices

def processPerceptJSON(user, filename, rawBytes, device_deidentified_id="", process=True):
    secureEncoder = Fernet(key)
    with open(DATABASE_PATH + "cache" + os.path.sep + filename, "wb+") as file:
        file.write(secureEncoder.encrypt(rawBytes))

    try:
        JSON = Percept.decodeEncryptedJSON(DATABASE_PATH + "cache" + os.path.sep + filename, key)
    except:
        return "JSON Format Error: " + filename, None, None

    if not process:
        os.remove(DATABASE_PATH + "cache" + os.path.sep + filename)
        return "Success", None, JSON

    if JSON["DeviceInformation"]["Final"]["NeurostimulatorSerialNumber"] != "":
        DeviceSerialNumber = secureEncoder.encrypt(JSON["DeviceInformation"]["Final"]["NeurostimulatorSerialNumber"].encode("utf-8")).decode("utf-8")
    else:
        DeviceSerialNumber = "Unknown"
    deviceHashfield = hashlib.sha256(JSON["DeviceInformation"]["Final"]["NeurostimulatorSerialNumber"].encode("utf-8")).hexdigest()

    try:
        Data = Percept.extractPerceptJSON(JSON)
    except:
        return "Decoding Error: " + filename, None, None

    SessionDate = datetime.fromtimestamp(Percept.estimateSessionDateTime(JSON),tz=pytz.utc)
    if user.is_clinician or user.is_admin:
        deviceID = models.PerceptDevice.objects.filter(device_identifier_hashfield=deviceHashfield, authority_level="Clinic", authority_user=user.institute).first()
    else:
        deviceID = models.PerceptDevice.objects.filter(deidentified_id=device_deidentified_id, authority_level="Research", authority_user=user.email).first()
        if deviceID == None:
            return "Device ID Error: " + filename, None, None

    newPatient = None
    if deviceID == None:
        PatientInformation = JSON["PatientInformation"]["Final"]
        patient, isNewPatient = retrievePatientInformation(PatientInformation, user.institute, secureEncoder)
        if isNewPatient:
            newPatient = patient
            patient.institute = user.institute
            patient.save()

        DeviceInformation = JSON["DeviceInformation"]["Final"]
        DeviceType = DeviceInformation["Neurostimulator"]
        NeurostimulatorLocation = DeviceInformation["NeurostimulatorLocation"].replace("InsLocation.","")
        ImplantDate = datetime.fromisoformat(DeviceInformation["ImplantDate"][:-1]+"+00:00")
        LeadConfigurations = list()

        LeadInformation = JSON["LeadConfiguration"]["Final"]
        for lead in LeadInformation:
            LeadConfiguration = dict()
            LeadConfiguration["TargetLocation"] = lead["Hemisphere"].replace("HemisphereLocationDef.","") + " "
            if lead["LeadLocation"] == "LeadLocationDef.Vim":
                LeadConfiguration["TargetLocation"] += "VIM"
            elif lead["LeadLocation"] == "LeadLocationDef.Stn":
                LeadConfiguration["TargetLocation"] += "STN"
            elif lead["LeadLocation"] == "LeadLocationDef.Gpi":
                LeadConfiguration["TargetLocation"] += "GPi"
            else:
                LeadConfiguration["TargetLocation"] += lead["LeadLocation"].replace("LeadLocationDef.","")

            if lead["ElectrodeNumber"] == "InsPort.ZERO_THREE":
                LeadConfiguration["ElectrodeNumber"] = "E00-E03"
            elif lead["ElectrodeNumber"] == "InsPort.ZERO_SEVEN":
                LeadConfiguration["ElectrodeNumber"] = "E00-E07"
            elif lead["ElectrodeNumber"] == "InsPort.EIGHT_ELEVEN":
                LeadConfiguration["ElectrodeNumber"] = "E08-E11"
            elif lead["ElectrodeNumber"] == "InsPort.EIGHT_FIFTEEN":
                LeadConfiguration["ElectrodeNumber"] = "E08-E15"
            if lead["Model"] == "LeadModelDef.LEAD_B33015":
                LeadConfiguration["ElectrodeType"] = "SenSight B33015"
            elif lead["Model"] == "LeadModelDef.LEAD_B33005":
                LeadConfiguration["ElectrodeType"] = "SenSight B33005"
            elif lead["Model"] == "LeadModelDef.LEAD_3387":
                LeadConfiguration["ElectrodeType"] = "Medtronic 3387"
            elif lead["Model"] == "LeadModelDef.LEAD_3389":
                LeadConfiguration["ElectrodeType"] = "Medtronic 3389"
            elif lead["Model"] == "LeadModelDef.LEAD_OTHER":
                LeadConfiguration["ElectrodeType"] = "Unknown Lead"
            else:
                LeadConfiguration["ElectrodeType"] = lead["Model"]

            LeadConfigurations.append(LeadConfiguration)

        if user.is_clinician or user.is_admin:
            deviceID = models.PerceptDevice(patient_deidentified_id=patient.deidentified_id, authority_level="Clinic", authority_user=user.institute, serial_number=DeviceSerialNumber, device_type=DeviceType, implant_date=ImplantDate, device_location=NeurostimulatorLocation, device_lead_configurations=LeadConfigurations, device_last_seen=datetime.fromtimestamp(0, tz=pytz.utc))
        else:
            deviceID = models.PerceptDevice(patient_deidentified_id=patient.deidentified_id, authority_level="Research", authority_user=user.email, serial_number=DeviceSerialNumber, device_type=DeviceType, implant_date=ImplantDate, device_location=NeurostimulatorLocation, device_lead_configurations=LeadConfigurations, device_last_seen=datetime.fromtimestamp(0, tz=pytz.utc))

        SessionDate = datetime.fromtimestamp(Percept.estimateSessionDateTime(JSON),tz=pytz.utc)
        if "EstimatedBatteryLifeMonths" in JSON["BatteryInformation"].keys():
            deviceID.device_eol_date = SessionDate + timedelta(days=30*JSON["BatteryInformation"]["EstimatedBatteryLifeMonths"])
        else:
            deviceID.device_eol_date = datetime.fromtimestamp(0, tz=pytz.utc)
        deviceID.device_identifier_hashfield=deviceHashfield
        deviceID.save()

        patient.addDevice(str(deviceID.deidentified_id))
    else:
        patient = models.Patient.objects.filter(deidentified_id=deviceID.patient_deidentified_id).first()

    DeviceInformation = JSON["DeviceInformation"]["Final"]
    NeurostimulatorLocation = DeviceInformation["NeurostimulatorLocation"].replace("InsLocation.","")
    ImplantDate = datetime.fromisoformat(DeviceInformation["ImplantDate"][:-1]+"+00:00")
    LeadConfigurations = list()

    LeadInformation = JSON["LeadConfiguration"]["Final"]
    for lead in LeadInformation:
        LeadConfiguration = dict()
        LeadConfiguration["TargetLocation"] = lead["Hemisphere"].replace("HemisphereLocationDef.","") + " "
        if lead["LeadLocation"] == "LeadLocationDef.Vim":
            LeadConfiguration["TargetLocation"] += "VIM"
        elif lead["LeadLocation"] == "LeadLocationDef.Stn":
            LeadConfiguration["TargetLocation"] += "STN"
        elif lead["LeadLocation"] == "LeadLocationDef.Gpi":
            LeadConfiguration["TargetLocation"] += "GPI"
        else:
            LeadConfiguration["TargetLocation"] += lead["LeadLocation"].replace("LeadLocationDef.","")

        if lead["ElectrodeNumber"] == "InsPort.ZERO_THREE":
            LeadConfiguration["ElectrodeNumber"] = "E00-E03"
        elif lead["ElectrodeNumber"] == "InsPort.ZERO_SEVEN":
            LeadConfiguration["ElectrodeNumber"] = "E00-E07"
        elif lead["ElectrodeNumber"] == "InsPort.EIGHT_ELEVEN":
            LeadConfiguration["ElectrodeNumber"] = "E08-E11"
        elif lead["ElectrodeNumber"] == "InsPort.EIGHT_FIFTEEN":
            LeadConfiguration["ElectrodeNumber"] = "E08-E15"
        if lead["Model"] == "LeadModelDef.LEAD_B33015":
            LeadConfiguration["ElectrodeType"] = "SenSight B33015"
        elif lead["Model"] == "LeadModelDef.LEAD_B33005":
            LeadConfiguration["ElectrodeType"] = "SenSight B33005"
        elif lead["Model"] == "LeadModelDef.LEAD_3387":
            LeadConfiguration["ElectrodeType"] = "Medtronic 3387"
        elif lead["Model"] == "LeadModelDef.LEAD_3389":
            LeadConfiguration["ElectrodeType"] = "Medtronic 3389"
        else:
            LeadConfiguration["ElectrodeType"] = lead["Model"]

        LeadConfigurations.append(LeadConfiguration)

    deviceID.implant_date = ImplantDate
    deviceID.device_location = NeurostimulatorLocation
    deviceID.device_lead_configurations = LeadConfigurations
    if "EstimatedBatteryLifeMonths" in JSON["BatteryInformation"].keys():
        deviceID.device_eol_date = SessionDate + timedelta(days=30*JSON["BatteryInformation"]["EstimatedBatteryLifeMonths"])
    else:
        deviceID.device_eol_date = datetime.fromtimestamp(0, tz=pytz.utc)
    deviceID.save()

    if SessionDate >= deviceID.device_last_seen:
        deviceID.device_last_seen = SessionDate
        if "EstimatedBatteryLifeMonths" in JSON["BatteryInformation"].keys():
            deviceID.device_eol_date = SessionDate + timedelta(days=30*JSON["BatteryInformation"]["EstimatedBatteryLifeMonths"])

        LeadConfigurations = list()
        LeadInformation = JSON["LeadConfiguration"]["Final"]
        for lead in LeadInformation:
            LeadConfiguration = dict()
            LeadConfiguration["TargetLocation"] = lead["Hemisphere"].replace("HemisphereLocationDef.","") + " "
            if lead["LeadLocation"] == "LeadLocationDef.Vim":
                LeadConfiguration["TargetLocation"] += "VIM"
            elif lead["LeadLocation"] == "LeadLocationDef.Stn":
                LeadConfiguration["TargetLocation"] += "STN"
            elif lead["LeadLocation"] == "LeadLocationDef.Gpi":
                LeadConfiguration["TargetLocation"] += "GPi"
            else:
                LeadConfiguration["TargetLocation"] += lead["LeadLocation"].replace("LeadLocationDef.","")

            if lead["ElectrodeNumber"] == "InsPort.ZERO_THREE":
                LeadConfiguration["ElectrodeNumber"] = "E00-E03"
            elif lead["ElectrodeNumber"] == "InsPort.ZERO_SEVEN":
                LeadConfiguration["ElectrodeNumber"] = "E00-E07"
            elif lead["ElectrodeNumber"] == "InsPort.EIGHT_ELEVEN":
                LeadConfiguration["ElectrodeNumber"] = "E08-E11"
            elif lead["ElectrodeNumber"] == "InsPort.EIGHT_FIFTEEN":
                LeadConfiguration["ElectrodeNumber"] = "E08-E15"
            if lead["Model"] == "LeadModelDef.LEAD_B33015":
                LeadConfiguration["ElectrodeType"] = "SenSight B33015"
            elif lead["Model"] == "LeadModelDef.LEAD_B33005":
                LeadConfiguration["ElectrodeType"] = "SenSight B33005"
            elif lead["Model"] == "LeadModelDef.LEAD_3387":
                LeadConfiguration["ElectrodeType"] = "Medtronic 3387"
            elif lead["Model"] == "LeadModelDef.LEAD_3389":
                LeadConfiguration["ElectrodeType"] = "Medtronic 3389"
            else:
                LeadConfiguration["ElectrodeType"] = lead["Model"]

            LeadConfigurations.append(LeadConfiguration)
        deviceID.device_lead_configurations = LeadConfigurations
        deviceID.save()

    session = models.PerceptSession(device_deidentified_id=deviceID.deidentified_id, session_source_filename=filename)
    session.session_file_path = DATABASE_PATH + "sessions" + os.path.sep + str(session.device_deidentified_id)+"_"+str(session.deidentified_id)+".json"
    sessionUUID = str(session.deidentified_id)

    # Process Therapy History
    NewDataFound = False
    if saveTherapySettings(deviceID.deidentified_id, Data["StimulationGroups"], SessionDate, "Post-visit Therapy", sessionUUID):
        NewDataFound = True

    if saveTherapySettings(deviceID.deidentified_id, Data["PreviousGroups"], SessionDate, "Pre-visit Therapy", sessionUUID):
        NewDataFound = True

    if "TherapyHistory" in Data.keys():
        for i in range(len(Data["TherapyHistory"])):
            SessionDate = datetime.fromtimestamp(Percept.getTimestamp(Data["TherapyHistory"][i]["DateTime"]),tz=pytz.utc)
            if saveTherapySettings(deviceID.deidentified_id, Data["TherapyHistory"][i]["Therapy"], SessionDate, "Past Therapy", sessionUUID):
                NewDataFound = True

    if "TherapyChangeHistory" in Data.keys():
        logToSave = list()
        for therapyChange in Data["TherapyChangeHistory"]:
            TherapyObject = models.TherapyChangeLog.objects.filter(device_deidentified_id=deviceID.deidentified_id, date_of_change=therapyChange["DateTime"].astimezone(pytz.utc)).first()
            if TherapyObject == None:
                logToSave.append(models.TherapyChangeLog(device_deidentified_id=deviceID.deidentified_id, date_of_change=therapyChange["DateTime"].astimezone(pytz.utc),
                                    previous_group=therapyChange["OldGroupId"], new_group=therapyChange["NewGroupId"], source_file=sessionUUID))
                NewDataFound = True
        if len(logToSave) > 0:
            models.TherapyChangeLog.objects.bulk_create(logToSave, ignore_conflicts=True)

    # Process BrainSense Survey
    if "MontagesTD" in Data.keys():
        if saveBrainSenseSurvey(deviceID.deidentified_id, Data["MontagesTD"], sessionUUID):
            NewDataFound = True
    if "BaselineTD" in Data.keys():
        if saveBrainSenseSurvey(deviceID.deidentified_id, Data["BaselineTD"], sessionUUID):
            NewDataFound = True

    # Process Montage Streams
    if "IndefiniteStream" in Data.keys():
        if saveMontageStreams(deviceID.deidentified_id, Data["IndefiniteStream"], sessionUUID):
            NewDataFound = True

    # Process Realtime Streams
    if "StreamingTD" in Data.keys() and "StreamingPower" in Data.keys():
        if saveRealtimeStreams(deviceID.deidentified_id, Data["StreamingTD"], Data["StreamingPower"], sessionUUID):
            NewDataFound = True

    # Stiore Chronic LFPs
    if "LFPTrends" in Data.keys():
        if saveChronicLFP(deviceID.deidentified_id, Data["LFPTrends"], sessionUUID):
            NewDataFound = True

    if "LFPEvents" in Data.keys():
        if saveBrainSenseEvents(deviceID.deidentified_id, JSON["DiagnosticData"]["LfpFrequencySnapshotEvents"], sessionUUID):
            NewDataFound = True

    if NewDataFound:
        os.rename(DATABASE_PATH + "cache" + os.path.sep + filename, session.session_file_path)
        session.save()
    else:
        os.remove(DATABASE_PATH + "cache" + os.path.sep + filename)

    JSON["PatientID"] = str(patient.deidentified_id)
    return "Success", newPatient, JSON

def getAllResearchUsers():
    ResearchUserList = list()
    users = models.PlatformUser.objects.filter(is_clinician=0, is_admin=0).all()
    for user in users:
        ResearchUserList.append({"Username": user.email, "FirstName": user.first_name, "LastName": user.last_name, "ID": user.uniqueUserID})
    return ResearchUserList

def extractPatientList(user):
    PatientInfo = list()
    if user.is_admin or user.is_clinician:
        availableDevices = models.PerceptDevice.objects.filter(authority_level="Clinic", authority_user=user.institute).all()
        if len(availableDevices) > 0:
            df = pd.DataFrame.from_records(availableDevices.values("patient_deidentified_id"))
            PatientIDs = df["patient_deidentified_id"].unique()

            for i in range(len(PatientIDs)):
                patient = models.Patient.objects.get(deidentified_id=PatientIDs[i])
                info = extractPatientTableRow(user, patient)
                PatientInfo.append(info)
    else:
        patients = models.Patient.objects.filter(institute=user.email).all()
        for patient in patients:
            info = extractPatientTableRow(user, patient)
            PatientInfo.append(info)

        DeidentifiedPatientID = models.DeidentifiedPatientID.objects.filter(researcher_id=user.uniqueUserID).all()
        for deidentified_patient in DeidentifiedPatientID:
            patient = models.Patient.objects.filter(deidentified_id=deidentified_patient.deidentified_id, institute=user.uniqueUserID).first()
            info = extractPatientTableRow(user, patient)
            PatientInfo.append(info)

    PatientInfo = sorted(PatientInfo, key=lambda patient: patient["LastName"]+", "+patient["FirstName"])
    return PatientInfo

def extractAuthorizedAccessList(researcher_id):
    AuthorizedList = list()
    DeidentifiedPatientID = models.DeidentifiedPatientID.objects.filter(researcher_id=researcher_id).all()
    for patient in DeidentifiedPatientID:
        AuthorizedList.append({"ID": patient.authorized_patient_id})
    return AuthorizedList

def extractAvailableRecordingList(user, researcher_id, patient_id):
    RecordingList = dict()
    try:
        DeidentifiedPatientID = models.DeidentifiedPatientID.objects.get(researcher_id=researcher_id, authorized_patient_id=patient_id)
        RecordingList["TimeRange"] = DeidentifiedPatientID.authorized_time_range
    except:
        RecordingList["TimeRange"] = models.PerceptRecordingDefaultAuthorization()

    RecordingList["Recordings"] = list()
    RecordingInfo = {"Device": "", "ID": "", "Type": "TherapyHistory",
                    "Date": RecordingList["TimeRange"]["TherapyHistory"][1],
                    "Authorized": models.ResearchAuthorizedAccess.objects.filter(researcher_id=researcher_id, authorized_patient_id=patient_id, authorized_recording_type="TherapyHistory").exists()}
    RecordingList["Recordings"].append(RecordingInfo)

    RecordingInfo = {"Device": "", "ID": "", "Type": "ChronicLFPs",
                    "Date": RecordingList["TimeRange"]["ChronicLFPs"][1],
                    "Authorized": models.ResearchAuthorizedAccess.objects.filter(researcher_id=researcher_id, authorized_patient_id=patient_id, authorized_recording_type="ChronicLFPs").exists()}
    RecordingList["Recordings"].append(RecordingInfo)

    AvailableDevices = models.PerceptDevice.objects.filter(patient_deidentified_id=patient_id, authority_level="Clinic", authority_user=user.institute).all()
    for device in AvailableDevices:
        AvailableRecordings = models.BrainSenseRecording.objects.filter(device_deidentified_id=device.deidentified_id).order_by("-recording_date").all()
        for recording in AvailableRecordings:
            if recording.recording_type == "ChronicLFPs":
                continue
            RecordingInfo = {"Device": device.getDeviceSerialNumber(key), "ID": recording.recording_id, "Type": recording.recording_type,
                            "Date": recording.recording_date.timestamp(),
                            "Authorized": models.ResearchAuthorizedAccess.objects.filter(researcher_id=researcher_id, authorized_patient_id=patient_id, authorized_recording_id=recording.recording_id).exists()}

            if not RecordingInfo in RecordingList["Recordings"]:
                RecordingList["Recordings"].append(RecordingInfo)

    return RecordingList

def verifyAccess(user, patient_id):
    if not (user.is_clinician or user.is_admin):
        if models.DeidentifiedPatientID.objects.filter(researcher_id=user.uniqueUserID, deidentified_id=patient_id):
            return 2
        elif models.Patient.objects.filter(deidentified_id=patient_id).exists():
            return 1
        else:
            return 0
    else:
        patient = models.Patient.objects.filter(deidentified_id=patient_id).first()
        return 1

def verifyPermission(user, patient_id, authority, access_type):
    if authority["Level"] == 1:
        return [0, 0]

    if access_type == "TherapyHistory" and models.ResearchAuthorizedAccess.objects.filter(researcher_id=user.uniqueUserID, authorized_patient_id=patient_id, authorized_recording_type="TherapyHistory").exists():
        DeidentifiedPatientID = models.DeidentifiedPatientID.objects.get(researcher_id=user.uniqueUserID, authorized_patient_id=patient_id)
        TimeRange = DeidentifiedPatientID.authorized_time_range
        return TimeRange[access_type]

    elif access_type == "BrainSenseSurvey" and models.ResearchAuthorizedAccess.objects.filter(researcher_id=user.uniqueUserID, authorized_patient_id=patient_id, authorized_recording_type=access_type).exists():
        recording_ids = models.ResearchAuthorizedAccess.objects.filter(researcher_id=user.uniqueUserID, authorized_patient_id=patient_id, authorized_recording_type=access_type).all().values("authorized_recording_id")
        recording_ids = [id["authorized_recording_id"] for id in recording_ids]
        return recording_ids

    elif access_type == "BrainSenseStream" and models.ResearchAuthorizedAccess.objects.filter(researcher_id=user.uniqueUserID, authorized_patient_id=patient_id, authorized_recording_type=access_type).exists():
        recording_ids = models.ResearchAuthorizedAccess.objects.filter(researcher_id=user.uniqueUserID, authorized_patient_id=patient_id, authorized_recording_type=access_type).all().values("authorized_recording_id")
        recording_ids = [id["authorized_recording_id"] for id in recording_ids]
        return recording_ids

    elif access_type == "IndefiniteStream" and models.ResearchAuthorizedAccess.objects.filter(researcher_id=user.uniqueUserID, authorized_patient_id=patient_id, authorized_recording_type=access_type).exists():
        recording_ids = models.ResearchAuthorizedAccess.objects.filter(researcher_id=user.uniqueUserID, authorized_patient_id=patient_id, authorized_recording_type=access_type).all().values("authorized_recording_id")
        recording_ids = [id["authorized_recording_id"] for id in recording_ids]
        return recording_ids

    elif access_type == "ChronicLFPs" and models.ResearchAuthorizedAccess.objects.filter(researcher_id=user.uniqueUserID, authorized_patient_id=patient_id, authorized_recording_type="ChronicLFPs").exists():
        DeidentifiedPatientID = models.DeidentifiedPatientID.objects.get(researcher_id=user.uniqueUserID, authorized_patient_id=patient_id)
        TimeRange = DeidentifiedPatientID.authorized_time_range
        return TimeRange[access_type]

    return None

def extractAccess(user, patient_id):
    if not (user.is_clinician or user.is_admin):
        if models.DeidentifiedPatientID.objects.filter(researcher_id=user.uniqueUserID, deidentified_id=patient_id):
            deidentified = models.DeidentifiedPatientID.objects.filter(researcher_id=user.uniqueUserID, deidentified_id=patient_id).first()
            return deidentified
            #return models.Patient.objects.filter(deidentified_id=deidentified.authorized_patient_id).first()
        elif models.Patient.objects.filter(deidentified_id=patient_id).exists():
            return models.Patient.objects.filter(deidentified_id=patient_id).first()
        else:
            return 0
    else:
        return models.Patient.objects.filter(deidentified_id=patient_id).first()

def AuthorizeResearchAccess(user, researcher_id, patient_id, permission):
    if permission:
        if not models.DeidentifiedPatientID.objects.filter(researcher_id=researcher_id, authorized_patient_id=patient_id).exists():
            identified_patient = models.Patient.objects.get(deidentified_id=patient_id)
            patient = models.Patient(first_name="Deidentified", last_name=patient_id, birth_date=datetime.now(tz=pytz.utc), diagnosis="", medical_record_number="", institute=researcher_id)
            patient.save()
            models.DeidentifiedPatientID(researcher_id=researcher_id, authorized_patient_id=patient_id, deidentified_id=patient.deidentified_id).save()
    else:
        if models.DeidentifiedPatientID.objects.filter(researcher_id=researcher_id, authorized_patient_id=patient_id).exists():
            DeidentifiedID = models.DeidentifiedPatientID.objects.get(researcher_id=researcher_id, authorized_patient_id=patient_id)
            if models.ResearchAuthorizedAccess.objects.filter(researcher_id=researcher_id, authorized_patient_id=patient_id).exists():
                models.ResearchAuthorizedAccess.objects.filter(researcher_id=researcher_id, authorized_patient_id=patient_id).all().delete()
            models.Patient.objects.filter(institute=researcher_id, deidentified_id=DeidentifiedID.deidentified_id).all().delete()
            models.DeidentifiedPatientID.objects.filter(researcher_id=researcher_id, authorized_patient_id=patient_id).all().delete()

def AuthorizeRecordingAccess(user, researcher_id, patient_id, recording_id="", recording_type="", permission=True):
    if recording_id == "":
        if permission:
            if recording_type == "TherapyHistory":
                if not models.ResearchAuthorizedAccess.objects.filter(researcher_id=researcher_id, authorized_patient_id=patient_id, authorized_recording_type=recording_type).exists():
                    models.ResearchAuthorizedAccess(researcher_id=researcher_id, authorized_patient_id=patient_id, authorized_recording_type=recording_type).save()
            elif recording_type == "ChronicLFPs":
                if not models.ResearchAuthorizedAccess.objects.filter(researcher_id=researcher_id, authorized_patient_id=patient_id, authorized_recording_type=recording_type).exists():
                    models.ResearchAuthorizedAccess(researcher_id=researcher_id, authorized_patient_id=patient_id, authorized_recording_type=recording_type).save()
            else:
                DeidentifiedPatientID = models.DeidentifiedPatientID.objects.get(researcher_id=researcher_id, authorized_patient_id=patient_id)
                TimeRange = [datetime.fromtimestamp(timestamp) for timestamp in DeidentifiedPatientID.authorized_time_range[recording_type]]
                AvailableDevices = models.PerceptDevice.objects.filter(patient_deidentified_id=patient_id, authority_level="Clinic", authority_user=user.institute).all()
                for device in AvailableDevices:
                    AvailableRecordings = models.BrainSenseRecording.objects.filter(device_deidentified_id=device.deidentified_id, recording_type=recording_type, recording_date__gte=TimeRange[0], recording_date__lte=TimeRange[1]).order_by("-recording_date").all()
                    for recording in AvailableRecordings:
                        if not models.ResearchAuthorizedAccess.objects.filter(researcher_id=researcher_id, authorized_patient_id=patient_id, authorized_recording_type=recording_type, authorized_recording_id=recording.recording_id).exists():
                            models.ResearchAuthorizedAccess(researcher_id=researcher_id, authorized_patient_id=patient_id, authorized_recording_type=recording_type, authorized_recording_id=recording.recording_id).save()
        else:
            if models.ResearchAuthorizedAccess.objects.filter(researcher_id=researcher_id, authorized_patient_id=patient_id, authorized_recording_type=recording_type).exists():
                models.ResearchAuthorizedAccess.objects.filter(researcher_id=researcher_id, authorized_patient_id=patient_id, authorized_recording_type=recording_type).all().delete()
    else:
        if permission:
            if not models.ResearchAuthorizedAccess.objects.filter(researcher_id=researcher_id, authorized_patient_id=patient_id, authorized_recording_id=recording_id).exists():
                models.ResearchAuthorizedAccess(researcher_id=researcher_id, authorized_patient_id=patient_id, authorized_recording_id=recording_id).save()
        else:
            if models.ResearchAuthorizedAccess.objects.filter(researcher_id=researcher_id, authorized_patient_id=patient_id, authorized_recording_id=recording_id).exists():
                models.ResearchAuthorizedAccess.objects.filter(researcher_id=researcher_id, authorized_patient_id=patient_id, authorized_recording_id=recording_id).all().delete()

def extractPatientTableRow(user, patient):
    info = dict()
    info["FirstName"] = patient.getPatientFirstName(key)
    info["LastName"] = patient.getPatientLastName(key)
    if patient.diagnosis == "ParkinsonsDisease":
        info["Diagnosis"] = "Parkinson\'s Disease"
    else:
        info["Diagnosis"] = patient.diagnosis
    info["MRN"] = patient.medical_record_number
    #info["DOB"] = patient.birth_date
    #info["Institute"] = patient.institute
    info["DaysSinceImplant"] = ""
    lastTimestamp = datetime.fromtimestamp(0, tz=pytz.utc)

    deviceIDs = patient.device_deidentified_id
    for id in deviceIDs:
        device = models.PerceptDevice.objects.filter(deidentified_id=id).first()
        if device == None:
            patient.device_deidentified_id.remove(id)
            patient.save()
            continue

        if not (user.is_admin or user.is_clinician):
            device = models.PerceptDevice.objects.filter(deidentified_id=id).first()
            device.serial_number = id

        if not (user.is_admin or user.is_clinician):
            daysSinceImplant = 0
        else:
            daysSinceImplant = np.round((datetime.now(tz=pytz.utc) - device.implant_date).total_seconds() / (3600*24))
        if device.device_name == "":
            deviceName = device.getDeviceSerialNumber(key)
        else:
            deviceName = device.device_name
        info["DaysSinceImplant"] += f"{deviceName} ({device.device_type})"
        if daysSinceImplant > 1:
            info["DaysSinceImplant"] += " <br>"
        else:
            info["DaysSinceImplant"] += " <br>"

        if device.device_last_seen > lastTimestamp:
            lastTimestamp = device.device_last_seen

    info["LastSeen"] = lastTimestamp.strftime("%Y/%m/%d")
    info["ID"] = str(patient.deidentified_id)
    return info

def extractPatientInfo(user, patientUniqueID):
    patient = models.Patient.objects.get(deidentified_id=patientUniqueID)
    info = dict()
    info["FirstName"] = patient.getPatientFirstName(key)
    info["LastName"] = patient.getPatientLastName(key)
    if user.is_clinician:
        info["Name"] = info["FirstName"] + " " + info["LastName"]
    else:
        info["Name"] = info["FirstName"] + " (" + info["LastName"] + ")"

    info["Diagnosis"] = patient.diagnosis
    info["MRN"] = patient.medical_record_number
    info["DOB"] = patient.birth_date
    info["Institute"] = patient.institute

    info["Devices"] = list()
    deviceIDs = patient.device_deidentified_id

    sessions_to_remove = []
    for id in deviceIDs:
        deviceInfo = dict()
        device = models.PerceptDevice.objects.get(deidentified_id=id)
        if not (user.is_admin or user.is_clinician):
            device.serial_number = id

        deviceInfo["ID"] = id
        deviceInfo["Location"] = device.device_location
        if device.device_name == "":
            deviceInfo["DeviceName"] = device.getDeviceSerialNumber(key)
        else:
            deviceInfo["DeviceName"] = device.device_name

        deviceInfo["DeviceType"] = device.device_type
        deviceInfo["ImplantDate"] = device.implant_date.strftime("%Y-%m-%d")
        deviceInfo["LastSeenDate"] = device.device_last_seen.strftime("%Y-%m-%d")
        deviceInfo["EOLDate"] = device.device_eol_date.strftime("%B, %Y")
        deviceInfo["Leads"] = device.device_lead_configurations
        info["Devices"].append(deviceInfo)

    return info

def queryTherapyHistory(user, patientUniqueID, authority):
    TherapyHistoryContext = list()
    if not authority["Permission"]:
        return TherapyHistoryContext

    availableDevices = getPerceptDevices(user, patientUniqueID, authority)
    for device in availableDevices:
        TherapyChangeData = dict()
        TherapyChangeData["device"] = device.deidentified_id
        if device.device_name == "":
            TherapyChangeData["device_name"] = device.getDeviceSerialNumber(key)
        else:
            TherapyChangeData["device_name"] = device.device_name
        TherapyChangeHistory = models.TherapyChangeLog.objects.filter(device_deidentified_id=device.deidentified_id).order_by("date_of_change").all()
        if len(TherapyChangeHistory) > 0:
            TherapyChangeHistory = pd.DataFrame.from_records(TherapyChangeHistory.values("date_of_change", "previous_group", "new_group"))
            TherayHistoryObjs = models.TherapyHistory.objects.filter(device_deidentified_id=device.deidentified_id).order_by("therapy_date").all()
            TherapyHistory = pd.DataFrame.from_records(TherayHistoryObjs.values("therapy_date", "group_id", "therapy_type","therapy_details"))
            DateSelection = pd.to_datetime(TherapyChangeHistory["date_of_change"]).view(np.int64) > authority["Permission"][0]*1000000000
            if authority["Permission"][1] > 0:
                DateSelection = np.bitwise_and(DateSelection, pd.to_datetime(TherapyChangeHistory["date_of_change"]).view(np.int64) < authority["Permission"][1]*1000000000)
            TherapyChangeData["date_of_change"] = TherapyChangeHistory["date_of_change"].values[DateSelection].tolist()
            TherapyChangeData["previous_group"] = TherapyChangeHistory["previous_group"].values[DateSelection].tolist()
            TherapyChangeData["new_group"] = TherapyChangeHistory["new_group"].values[DateSelection].tolist()
            TherapyChangeData["therapy"] = list()
            for i in range(len(TherapyChangeData["date_of_change"])):
                DetailTherapy, DetailTherapy_date = getTherapyDetails(TherapyHistory, TherapyChangeData["date_of_change"][i]/1000000000, TherapyChangeData["new_group"][i], "Pre-visit Therapy")
                BriefTherapy, BriefTherapy_date = getTherapyDetails(TherapyHistory, TherapyChangeData["date_of_change"][i]/1000000000, TherapyChangeData["new_group"][i], "Past Therapy")
                PostVisitTherapy, PostVisitTherapy_date = getTherapyDetails(TherapyHistory, TherapyChangeData["date_of_change"][i]/1000000000, TherapyChangeData["new_group"][i], "Post-visit Therapy")
                if DetailTherapy == None:
                    if not BriefTherapy == None:
                        TherapyChangeData["therapy"].append(BriefTherapy)
                    elif not PostVisitTherapy == None:
                        TherapyChangeData["therapy"].append(PostVisitTherapy)
                else:
                    if BriefTherapy == None:
                        TherapyChangeData["therapy"].append(DetailTherapy)
                    elif datetime.fromtimestamp(BriefTherapy_date).date() < datetime.fromtimestamp(DetailTherapy_date).date():
                        TherapyChangeData["therapy"].append(BriefTherapy)
                    else:
                        TherapyChangeData["therapy"].append(DetailTherapy)

            for i in range(len(TherapyHistory["therapy_date"])):
                if TherapyHistory["therapy_date"][i].timestamp() > TherapyChangeData["date_of_change"][-1]/1000000000 and (TherapyHistory["therapy_date"][i].timestamp() < authority["Permission"][1] or authority["Permission"][1] == 0):
                    TherapyChangeData["date_of_change"].append(TherapyHistory["therapy_date"][i].timestamp()*1000000000)
                    TherapyChangeData["previous_group"].append(TherapyChangeData["new_group"][-1])
                    TherapyChangeData["new_group"].append(TherapyChangeData["new_group"][-1])

                    DetailTherapy, DetailTherapy_date = getTherapyDetails(TherapyHistory, TherapyHistory["therapy_date"][i].timestamp(), TherapyChangeData["new_group"][-1], "Pre-visit Therapy")
                    BriefTherapy, BriefTherapy_date = getTherapyDetails(TherapyHistory, TherapyHistory["therapy_date"][i].timestamp(), TherapyChangeData["new_group"][-1], "Past Therapy")
                    PostVisitTherapy, PostVisitTherapy_date = getTherapyDetails(TherapyHistory, TherapyHistory["therapy_date"][i].timestamp(), TherapyChangeData["new_group"][-1], "Post-visit Therapy")
                    if DetailTherapy == None:
                        if not BriefTherapy == None:
                            TherapyChangeData["therapy"].append(BriefTherapy)
                        elif not PostVisitTherapy == None:
                            TherapyChangeData["therapy"].append(PostVisitTherapy)
                    else:
                        if BriefTherapy == None:
                            TherapyChangeData["therapy"].append(DetailTherapy)
                        elif datetime.fromtimestamp(BriefTherapy_date).date() < datetime.fromtimestamp(DetailTherapy_date).date():
                            TherapyChangeData["therapy"].append(BriefTherapy)
                        else:
                            TherapyChangeData["therapy"].append(DetailTherapy)

            TherapyHistoryContext.append(TherapyChangeData)

    return TherapyHistoryContext

def queryTherapyConfigurations(user, patientUniqueID, authority, therapy_type="Past Therapy"):
    TherapyHistory = list()
    if not authority["Permission"]:
        return TherapyHistory

    availableDevices = getPerceptDevices(user, patientUniqueID, authority)
    for device in availableDevices:
        if therapy_type == "":
            TherapyHistoryObjs = models.TherapyHistory.objects.filter(device_deidentified_id=device.deidentified_id).order_by("therapy_date").all()
        else:
            TherapyHistoryObjs = models.TherapyHistory.objects.filter(device_deidentified_id=device.deidentified_id, therapy_type=therapy_type).order_by("therapy_date").all()

        for therapy in TherapyHistoryObjs:
            TherapyInfo = {"DeviceID": str(device.deidentified_id), "Device": device.getDeviceSerialNumber(key), "DeviceLocation": device.device_location}
            TherapyInfo["TherapyDate"] = therapy.therapy_date.timestamp()
            TherapyInfo["TherapyGroup"] = therapy.group_id
            TherapyInfo["TherapyType"] = therapy.therapy_type
            TherapyInfo["LogID"] = str(therapy.history_log_id)
            TherapyInfo["Therapy"] = therapy.therapy_details

            if TherapyInfo["TherapyDate"] > authority["Permission"][0]:
                if authority["Permission"][1] > 0 and TherapyInfo["TherapyDate"] < authority["Permission"][1]:
                    TherapyHistory.append(TherapyInfo)
                elif authority["Permission"][1] == 0:
                    TherapyHistory.append(TherapyInfo)

    return TherapyHistory

def getTherapyDetails(TherapyHistory, timestamp, group_id, typeID):
    for j in range(len(TherapyHistory["therapy_date"])):
        if TherapyHistory["therapy_date"][j].timestamp() > timestamp and TherapyHistory["group_id"][j] == group_id and TherapyHistory["therapy_type"][j] == typeID:
            therapy_details = copy.deepcopy(TherapyHistory["therapy_details"][j])
            if "LeftHemisphere" in therapy_details.keys():
                if type(therapy_details["LeftHemisphere"]["Channel"][0]) == list:
                    for i in range(len(therapy_details["LeftHemisphere"]["Channel"])):
                        therapy_details["LeftHemisphere"]["Channel"][i] = Percept.reformatStimulationChannel(therapy_details["LeftHemisphere"]["Channel"][i])
                else:
                    therapy_details["LeftHemisphere"]["Channel"] = Percept.reformatStimulationChannel(therapy_details["LeftHemisphere"]["Channel"])
            if "RightHemisphere" in therapy_details.keys():
                if type(therapy_details["RightHemisphere"]["Channel"][0]) == list:
                    for i in range(len(therapy_details["RightHemisphere"]["Channel"])):
                        therapy_details["RightHemisphere"]["Channel"][i] = Percept.reformatStimulationChannel(therapy_details["RightHemisphere"]["Channel"][i])
                else:
                    therapy_details["RightHemisphere"]["Channel"] = Percept.reformatStimulationChannel(therapy_details["RightHemisphere"]["Channel"])
            return therapy_details, TherapyHistory["therapy_date"][j].timestamp()
    return None, None

def processTherapyDetails(TherapyConfigurations, TherapyChangeLog=[], resolveConflicts=True):
    TherapyData = dict()

    # Normalize Timestamp
    DeviceTimestamp = dict()
    for i in range(0,len(TherapyConfigurations)):
        if not TherapyConfigurations[i]["DeviceID"] in DeviceTimestamp.keys():
            DeviceTimestamp[TherapyConfigurations[i]["DeviceID"]] = list()

    ExistingTimestamp = list()
    for deviceID in DeviceTimestamp.keys():
        for i in range(len(TherapyConfigurations)):
            if TherapyConfigurations[i]["DeviceID"] == deviceID:
                TimeDifferences = np.array([np.abs(TherapyConfigurations[i]["TherapyDate"] - time) for time in ExistingTimestamp])
                Indexes = np.where(TimeDifferences < 3600*24)[0]
                if len(Indexes > 0):
                    TherapyConfigurations[i]["TherapyDate"] = ExistingTimestamp[Indexes[0]]
                else:
                    ExistingTimestamp.append(TherapyConfigurations[i]["TherapyDate"])

                if not TherapyConfigurations[i]["TherapyDate"] in DeviceTimestamp[deviceID]:
                    DeviceTimestamp[deviceID].append(TherapyConfigurations[i]["TherapyDate"])

    for deviceID in DeviceTimestamp.keys():
        for nConfig in range(len(DeviceTimestamp[deviceID])):
            if nConfig == 0:
                lastMeasuredTimestamp = 0
            else:
                lastMeasuredTimestamp = DeviceTimestamp[deviceID][nConfig-1]

            TherapyDutyPercent = dict()
            for i in range(len(TherapyChangeLog)):
                if str(TherapyChangeLog[i]["device"]) == deviceID:
                    for k in range(len(TherapyChangeLog[i]["date_of_change"])):
                        if lastMeasuredTimestamp < DeviceTimestamp[deviceID][nConfig]:
                            if k > 0:
                                if TherapyChangeLog[i]["previous_group"][k] == TherapyChangeLog[i]["new_group"][k-1] or TherapyChangeLog[i]["previous_group"][k] == -1:
                                    if TherapyChangeLog[i]["date_of_change"][k]/1000000000 > lastMeasuredTimestamp:
                                        if not TherapyChangeLog[i]["previous_group"][k] in TherapyDutyPercent.keys():
                                            TherapyDutyPercent[TherapyChangeLog[i]["previous_group"][k]] = 0
                                        if TherapyChangeLog[i]["date_of_change"][k]/1000000000 > DeviceTimestamp[deviceID][nConfig]:
                                            TherapyDutyPercent[TherapyChangeLog[i]["previous_group"][k]] += (DeviceTimestamp[deviceID][nConfig]-lastMeasuredTimestamp)
                                            lastMeasuredTimestamp = DeviceTimestamp[deviceID][nConfig]
                                        else:
                                            TherapyDutyPercent[TherapyChangeLog[i]["previous_group"][k]] += (TherapyChangeLog[i]["date_of_change"][k]/1000000000-lastMeasuredTimestamp)
                                            lastMeasuredTimestamp = TherapyChangeLog[i]["date_of_change"][k]/1000000000
                            else:
                                if TherapyChangeLog[i]["date_of_change"][k]/1000000000 > lastMeasuredTimestamp:
                                    if not TherapyChangeLog[i]["previous_group"][k] in TherapyDutyPercent.keys():
                                        TherapyDutyPercent[TherapyChangeLog[i]["previous_group"][k]] = 0
                                    if TherapyChangeLog[i]["date_of_change"][k]/1000000000 > DeviceTimestamp[deviceID][nConfig]:
                                        TherapyDutyPercent[TherapyChangeLog[i]["previous_group"][k]] += (DeviceTimestamp[deviceID][nConfig]-lastMeasuredTimestamp)
                                        lastMeasuredTimestamp = DeviceTimestamp[deviceID][nConfig]
                                    else:
                                        TherapyDutyPercent[TherapyChangeLog[i]["previous_group"][k]] += (TherapyChangeLog[i]["date_of_change"][k]/1000000000-lastMeasuredTimestamp)
                                        lastMeasuredTimestamp = TherapyChangeLog[i]["date_of_change"][k]/1000000000

            for i in range(len(TherapyConfigurations)):
                if TherapyConfigurations[i]["DeviceID"] == deviceID and TherapyConfigurations[i]["TherapyDate"] == DeviceTimestamp[deviceID][nConfig]:
                    TherapyConfigurations[i]["TherapyDutyPercent"] = TherapyDutyPercent
            print(TherapyDutyPercent)
    for nConfig in range(len(TherapyConfigurations)):
        therapy = TherapyConfigurations[nConfig]
        if not int(therapy["TherapyDate"]) in TherapyData.keys():
            TherapyData[int(therapy["TherapyDate"])] = list()
        therapy["Overview"] = dict()
        totalHours = np.sum([therapy["TherapyDutyPercent"][key] for key in therapy["TherapyDutyPercent"].keys()])
        if not therapy["Therapy"]["GroupId"] in therapy["TherapyDutyPercent"].keys():
            therapy["Overview"]["DutyPercent"] = "(0%)"
        else:
            therapy["Overview"]["DutyPercent"] = f"({therapy['TherapyDutyPercent'][therapy['Therapy']['GroupId']]/totalHours*100:.2f}%)"

        therapy["Overview"]["GroupName"] = therapy["Therapy"]["GroupId"].replace("GroupIdDef.GROUP_","Group ")
        therapy["Overview"]["TherapyLogID"] = therapy["LogID"]
        therapy["Overview"]["TherapyType"] = therapy["TherapyType"]
        therapy["Overview"]["TherapyUsage"] = 0

        therapy["Overview"]["Frequency"] = ""
        therapy["Overview"]["Amplitude"] = ""
        therapy["Overview"]["PulseWidth"] = ""
        therapy["Overview"]["Contacts"] = ""
        therapy["Overview"]["BrainSense"] = ""
        for hemisphere in ["LeftHemisphere","RightHemisphere"]:
            if hemisphere in therapy["Therapy"].keys():
                if hemisphere == "LeftHemisphere":
                    symbol = '<div class="d-flex align-items-center"><span class="badge badge-primary">L</span>'
                else:
                    symbol = '<div class="d-flex align-items-center"><span class="badge badge-warning">R</span>'

                if therapy["Therapy"][hemisphere]["Mode"] == "Interleaving":
                    for i in range(len(therapy['Therapy'][hemisphere]['Frequency'])):
                        therapy["Overview"]["Frequency"] += symbol + '<h6 class="text-sm mb-0">' + f"{therapy['Therapy'][hemisphere]['Frequency'][i]} Hz" + '</h6></div>'
                    for i in range(len(therapy['Therapy'][hemisphere]['Amplitude'])):
                        therapy["Overview"]["Amplitude"] += symbol + '<h6 class="text-sm mb-0">' + f"{therapy['Therapy'][hemisphere]['Amplitude'][i]} mA" + '</h6></div>'
                    for i in range(len(therapy['Therapy'][hemisphere]['PulseWidth'])):
                        therapy["Overview"]["PulseWidth"] += symbol + '<h6 class="text-sm mb-0">' + f"{therapy['Therapy'][hemisphere]['PulseWidth'][i]} μS" + '</h6></div>'
                    for i in range(len(therapy['Therapy'][hemisphere]['Channel'])):
                        therapy["Overview"]["Contacts"] += '<div class="d-flex align-items-center">'
                        for contact in therapy['Therapy'][hemisphere]['Channel'][i]:
                            if contact["ElectrodeStateResult"] == "ElectrodeStateDef.Negative":
                                ContactPolarity = '<span class="text-md badge badge-info">'
                                ContactSign = "-"
                            elif contact["ElectrodeStateResult"] == "ElectrodeStateDef.Positive":
                                ContactPolarity = '<span class="text-md badge badge-danger">'
                                ContactSign = "+"
                            else:
                                ContactPolarity = '<span class="text-md badge badge-success">'
                                ContactSign = ""
                            ContactName, ContactID = Percept.reformatElectrodeDef(contact["Electrode"])
                            if not ContactName == "CAN" or len(therapy['Therapy'][hemisphere]['Channel']) == 2:
                                therapy["Overview"]["Contacts"] += ContactPolarity + ContactName.replace("E","") + ContactSign + '</span>'
                        therapy["Overview"]["Contacts"] += '</div>'
                else:
                    if therapy["Therapy"][hemisphere]["Mode"] == "BrainSense":
                        therapy["Overview"]["BrainSense"] += symbol + '<h6 class="text-sm mb-0">' + f"{therapy['Therapy'][hemisphere]['SensingSetup']['FrequencyInHertz']} Hz" + '</h6></div>'
                    therapy["Overview"]["Frequency"] += symbol + '<h6 class="text-sm mb-0">' + f"{therapy['Therapy'][hemisphere]['Frequency']} Hz" + '</h6></div>'
                    therapy["Overview"]["Amplitude"] += symbol + '<h6 class="text-sm mb-0">' + f"{therapy['Therapy'][hemisphere]['Amplitude']} mA" + '</h6></div>'
                    therapy["Overview"]["PulseWidth"] += symbol + '<h6 class="text-sm mb-0">' + f"{therapy['Therapy'][hemisphere]['PulseWidth']} μS" + '</h6></div>'
                    therapy["Overview"]["Contacts"] += '<div class="d-flex align-items-center">'
                    for contact in therapy['Therapy'][hemisphere]['Channel']:
                        if contact["ElectrodeStateResult"] == "ElectrodeStateDef.Negative":
                            ContactPolarity = '<span class="text-md badge badge-info">'
                            ContactSign = "-"
                        elif contact["ElectrodeStateResult"] == "ElectrodeStateDef.Positive":
                            ContactPolarity = '<span class="text-md badge badge-danger">'
                            ContactSign = "+"
                        else:
                            ContactPolarity = '<span class="text-md badge badge-success">'
                            ContactSign = ""
                        ContactName, ContactID = Percept.reformatElectrodeDef(contact["Electrode"])
                        if not ContactName == "CAN" or len(therapy['Therapy'][hemisphere]['Channel']) == 2:
                            therapy["Overview"]["Contacts"] += ContactPolarity + ContactName.replace("E","") + ContactSign + '</span>'
                    therapy["Overview"]["Contacts"] += '</div>'

        TherapyData[int(therapy["TherapyDate"])].append(therapy)

    for key in TherapyData.keys():
        TherapyData[key] = sorted(TherapyData[key], key=lambda therapy: therapy["Overview"]["GroupName"])
        if resolveConflicts:
            existList = list(); i = 0;
            while i < len(TherapyData[key]):
                if not (TherapyData[key][i]["Overview"]["GroupName"] + TherapyData[key][i]["Device"]) in existList and not TherapyData[key][i]["Overview"]["Frequency"] == "":
                    existList.append(TherapyData[key][i]["Overview"]["GroupName"] + TherapyData[key][i]["Device"])
                    i += 1
                else:
                    del(TherapyData[key][i])

    return TherapyData

def querySurveyResults(user, patientUniqueID, authority):
    BrainSenseData = list()
    if not authority["Permission"]:
        return BrainSenseData

    availableDevices = getPerceptDevices(user, patientUniqueID, authority)
    for device in availableDevices:
        allSurveys = models.BrainSenseRecording.objects.filter(device_deidentified_id=device.deidentified_id, recording_type="BrainSenseSurvey").order_by("-recording_date").all()
        if len(allSurveys) > 0:
            leads = device.device_lead_configurations

        for recording in allSurveys:
            if not recording.recording_id in authority["Permission"] and authority["Level"] == 2:
                continue

            survey = loadSourceFiles(recording.recording_type,recording.recording_info["Channel"],recording.recording_id)
            if not "Spectrum" in survey.keys():
                survey = processBrainSenseSurvey(survey)
                saveSourceFiles(survey, "BrainSenseSurvey", survey["Channel"], recording.recording_id)
            data = dict()
            if device.device_name == "":
                data["DeviceName"] = device.getDeviceSerialNumber(key)
            else:
                data["DeviceName"] = device.device_name
            data["Timestamp"] = recording.recording_date.timestamp()
            data["Channel"], data["Hemisphere"] = Percept.reformatChannelName(recording.recording_info["Channel"])
            for lead in leads:
                if lead["TargetLocation"].startswith(data["Hemisphere"]):
                    data["Hemisphere"] = lead["TargetLocation"]
                    break
            data["Frequency"] = survey["Spectrum"]["Frequency"]
            data["MeanPower"] = np.mean(survey["Spectrum"]["Power"],axis=1).tolist()
            data["StdPower"] = SPU.stderr(survey["Spectrum"]["Power"],axis=1).tolist()
            BrainSenseData.append(data)
    return BrainSenseData

def queryMontageDataOverview(user, patientUniqueID, authority):
    BrainSenseData = list()
    if not authority["Permission"]:
        return BrainSenseData

    availableDevices = getPerceptDevices(user, patientUniqueID, authority)
    for device in availableDevices:
        allSurveys = models.BrainSenseRecording.objects.filter(device_deidentified_id=device.deidentified_id, recording_type="IndefiniteStream").order_by("recording_date").all()
        for recording in allSurveys:
            if not recording.recording_id in authority["Permission"] and authority["Level"] == 2:
                continue

            data = dict()
            if device.device_name == "":
                data["DeviceName"] = device.getDeviceSerialNumber(key)
            else:
                data["DeviceName"] = device.device_name
            data["Timestamp"] = recording.recording_date.timestamp()
            data["Duration"] = recording.recording_duration
            data["DeviceID"] = device.deidentified_id
            data["DeviceLocation"] = device.device_location
            BrainSenseData.append(data)
    return BrainSenseData

def queryPatientEventPSDsByTime(user, patientUniqueID, timeRange, authority):
    PatientEventPSDs = list()
    availableDevices = getPerceptDevices(user, patientUniqueID, authority)

    for device in availableDevices:
        EventPSDs = models.PatientCustomEvents.objects.filter(device_deidentified_id=device.deidentified_id, sensing_exist=True, event_time__gt=timeRange[0], event_time__lt=timeRange[1]).all()
        if len(EventPSDs) > 0:
            leads = device.device_lead_configurations
            for hemisphere in ["HemisphereLocationDef.Left","HemisphereLocationDef.Right"]:
                if device.device_name == "":
                    PatientEventPSDs.append({"Device": device.getDeviceSerialNumber(key), "DeviceLocation": device.device_location, "PSDs": list(), "EventName": list(), "EventTime": list(), "Therapy": list()})
                else:
                    PatientEventPSDs.append({"Device": device.device_name, "DeviceLocation": device.device_location, "PSDs": list(), "EventName": list(), "EventTime": list(), "Therapy": list()})

                for lead in leads:
                    if lead["TargetLocation"].startswith(hemisphere.replace("HemisphereLocationDef.","")):
                        PatientEventPSDs[-1]["Hemisphere"] = lead["TargetLocation"]

                TherapyKey = hemisphere.replace("HemisphereLocationDef.","") + "Hemisphere"
                for eventPSD in EventPSDs:
                    if hemisphere in eventPSD.brainsense_psd.keys():
                        EventTimestamp = Percept.getTimestamp(eventPSD.brainsense_psd[hemisphere]["DateTime"])
                        if EventTimestamp > authority["Permission"][0]:
                            if authority["Permission"][1] > 0 and EventTimestamp < authority["Permission"][1]:
                                PatientEventPSDs[-1]["Therapy"].append("Generic")
                                PatientEventPSDs[-1]["PSDs"].append(eventPSD.brainsense_psd[hemisphere]["FFTBinData"])
                                PatientEventPSDs[-1]["EventName"].append(eventPSD.event_name)
                                PatientEventPSDs[-1]["EventTime"].append(EventTimestamp)
                                break
                            elif authority["Permission"][1] == 0:
                                PatientEventPSDs[-1]["Therapy"].append("Generic")
                                PatientEventPSDs[-1]["PSDs"].append(eventPSD.brainsense_psd[hemisphere]["FFTBinData"])
                                PatientEventPSDs[-1]["EventName"].append(eventPSD.event_name)
                                PatientEventPSDs[-1]["EventTime"].append(EventTimestamp)
                                break

    i = 0;
    while i < len(PatientEventPSDs):
        if not "Hemisphere" in PatientEventPSDs[i]:
            del(PatientEventPSDs[i])
        else:
            i += 1

    return PatientEventPSDs

def queryPatientEventPSDs(user, patientUniqueID, TherapyHistory, authority):
    PatientEventPSDs = list()
    if not authority["Permission"]:
        return PatientEventPSDs

    availableDevices = getPerceptDevices(user, patientUniqueID, authority)
    TherapyConfigurations = queryTherapyConfigurations(user, patientUniqueID, authority, therapy_type="Past Therapy")
    for device in availableDevices:
        EventPSDs = models.PatientCustomEvents.objects.filter(device_deidentified_id=device.deidentified_id, sensing_exist=True).all()
        if len(EventPSDs) > 0:
            leads = device.device_lead_configurations
            for hemisphere in ["HemisphereLocationDef.Left","HemisphereLocationDef.Right"]:
                if device.device_name == "":
                    PatientEventPSDs.append({"Device": device.getDeviceSerialNumber(key), "DeviceLocation": device.device_location, "PSDs": list(), "EventName": list(), "Therapy": list()})
                else:
                    PatientEventPSDs.append({"Device": device.device_name, "DeviceLocation": device.device_location, "PSDs": list(), "EventName": list(), "Therapy": list()})

                for lead in leads:
                    if lead["TargetLocation"].startswith(hemisphere.replace("HemisphereLocationDef.","")):
                        PatientEventPSDs[-1]["Hemisphere"] = lead["TargetLocation"]

                TherapyKey = hemisphere.replace("HemisphereLocationDef.","") + "Hemisphere"
                for eventPSD in EventPSDs:
                    if hemisphere in eventPSD.brainsense_psd.keys():
                        EventTimestamp = Percept.getTimestamp(eventPSD.brainsense_psd[hemisphere]["DateTime"])
                        if EventTimestamp > authority["Permission"][0]:
                            if authority["Permission"][1] > 0 and EventTimestamp < authority["Permission"][1]:
                                for therapy in TherapyConfigurations:
                                    if therapy["DeviceID"] == str(device.deidentified_id) and therapy["TherapyGroup"] == eventPSD.brainsense_psd[hemisphere]["GroupId"] and therapy["TherapyDate"] > EventTimestamp and TherapyKey in therapy["Therapy"].keys():
                                        PatientEventPSDs[-1]["Therapy"].append(therapy["Therapy"][TherapyKey])
                                        PatientEventPSDs[-1]["PSDs"].append(eventPSD.brainsense_psd[hemisphere]["FFTBinData"])
                                        PatientEventPSDs[-1]["EventName"].append(eventPSD.event_name)
                                        break
                            elif authority["Permission"][1] == 0:
                                for therapy in TherapyConfigurations:
                                    if therapy["DeviceID"] == str(device.deidentified_id) and therapy["TherapyGroup"] == eventPSD.brainsense_psd[hemisphere]["GroupId"] and therapy["TherapyDate"] > EventTimestamp and TherapyKey in therapy["Therapy"].keys():
                                        PatientEventPSDs[-1]["Therapy"].append(therapy["Therapy"][TherapyKey])
                                        PatientEventPSDs[-1]["PSDs"].append(eventPSD.brainsense_psd[hemisphere]["FFTBinData"])
                                        PatientEventPSDs[-1]["EventName"].append(eventPSD.event_name)
                                        break

    i = 0;
    while i < len(PatientEventPSDs):
        if not "Hemisphere" in PatientEventPSDs[i]:
            del(PatientEventPSDs[i])
        else:
            i += 1

    return PatientEventPSDs

def formatTherapyString(Therapy):
    return f"Stimulation {Percept.reformatStimulationChannel(Therapy['Channel'])} {Therapy['Frequency']}Hz {Therapy['PulseWidth']}uS"

def processEventPSDs(PatientEventPSDs, EventNames):
    EventColor = dict()
    cmap = plt.get_cmap("Set1", 9)
    for cIndex in range(len(EventNames)):
        EventColor[EventNames[cIndex]] = colorTextFromCmap(cmap(cIndex))

    for i in range(len(PatientEventPSDs)):
        PatientEventPSDs[i]["Render"] = list()
        PatientEventPSDs[i]["PSDs"] = np.array(PatientEventPSDs[i]["PSDs"])
        StimulationConfigurations = [PatientEventPSDs[i]["Therapy"][j] if PatientEventPSDs[i]["Therapy"][j] == "Generic" else formatTherapyString(PatientEventPSDs[i]["Therapy"][j]) for j in range(len(PatientEventPSDs[i]["Therapy"]))]
        uniqueConfiguration = uniqueList(StimulationConfigurations)
        for config in uniqueConfiguration:
            PatientEventPSDs[i]["Render"].append({"Therapy": config, "Hemisphere": PatientEventPSDs[i]["Device"] + " " + PatientEventPSDs[i]["Hemisphere"], "Events": list()})

            selectedPSDs = iterativeCompare(StimulationConfigurations, config, "equal").flatten()
            Events = listSelection(PatientEventPSDs[i]["EventName"], selectedPSDs)
            for eventName in uniqueList(Events):
                eventSelection = iterativeCompare(PatientEventPSDs[i]["EventName"], eventName, "equal").flatten()
                PatientEventPSDs[i]["Render"][-1]["Events"].append({
                    "EventName": eventName,
                    "Count": f"(n={np.sum(np.bitwise_and(selectedPSDs, eventSelection))})",
                    "EventColor": EventColor[eventName],
                    "MeanPSD": np.mean(PatientEventPSDs[i]["PSDs"][np.bitwise_and(selectedPSDs, eventSelection),:],axis=0).flatten(),
                    "StdPSD": SPU.stderr(PatientEventPSDs[i]["PSDs"][np.bitwise_and(selectedPSDs, eventSelection),:],axis=0).flatten()
                })

    return PatientEventPSDs

def queryChronicLFPsByTime(user, patientUniqueID, timeRange, EventNames, authority):
    EventColor = dict()
    cmap = plt.get_cmap("Set1", 9)
    for cIndex in range(len(EventNames)):
        EventColor[EventNames[cIndex]] = colorTextFromCmap(cmap(cIndex))

    LFPTrends = list()
    availableDevices = getPerceptDevices(user, patientUniqueID, authority)
    for device in availableDevices:
        leads = device.device_lead_configurations
        for hemisphere in ["HemisphereLocationDef.Left","HemisphereLocationDef.Right"]:
            recording = models.BrainSenseRecording.objects.filter(device_deidentified_id=device.deidentified_id, recording_type="ChronicLFPs", recording_info__Hemisphere=hemisphere).first()
            if not recording == None:
                ChronicLFPs = loadSourceFiles(recording.recording_type,hemisphere.replace("HemisphereLocationDef.",""),recording.recording_id)
                if device.device_name == "":
                    LFPTrends.append({"Device": device.getDeviceSerialNumber(key), "DeviceLocation": device.device_location})
                else:
                    LFPTrends.append({"Device": device.device_name, "DeviceLocation": device.device_location})

                for lead in leads:
                    if lead["TargetLocation"].startswith(hemisphere.replace("HemisphereLocationDef.","")):
                        LFPTrends[-1]["Hemisphere"] = lead["TargetLocation"]

                LFPTimestamps = ChronicLFPs["DateTime"]
                LFPPowers = ChronicLFPs["LFP"]
                StimulationAmplitude = ChronicLFPs["Amplitude"]
                LFPTimestamps = np.array([time.timestamp() for time in LFPTimestamps])
                LFPTrends[-1]["Timestamp"] = list()
                LFPTrends[-1]["Power"] = list()
                LFPTrends[-1]["Amplitude"] = list()
                LFPTrends[-1]["EventName"] = list()
                LFPTrends[-1]["EventTime"] = list()

                # Remove Outliers
                LFPSelection = LFPPowers < np.median(LFPPowers) + np.std(LFPPowers)*6
                LFPTimestamps = LFPTimestamps[LFPSelection]
                LFPPowers = LFPPowers[LFPSelection]
                StimulationAmplitude = StimulationAmplitude[LFPSelection]

                LFPTrends[-1]["PowerRange"] = [0,0]
                LFPTrends[-1]["Timestamp"].append(LFPTimestamps)
                FiltPower = np.array(LFPPowers).tolist()
                LFPTrends[-1]["Power"].append(FiltPower)
                LFPTrends[-1]["Amplitude"].append(np.array(StimulationAmplitude).tolist())
                if np.percentile(FiltPower,5) < LFPTrends[-1]["PowerRange"][0]:
                    LFPTrends[-1]["PowerRange"][0] = np.percentile(FiltPower,5)
                if np.percentile(FiltPower,95) > LFPTrends[-1]["PowerRange"][1]:
                    LFPTrends[-1]["PowerRange"][1] = np.percentile(FiltPower,95)

                ChronicEvents = models.PatientCustomEvents.objects.filter(device_deidentified_id=device.deidentified_id,
                                    event_time__gt=timeRange[0], event_time__lt=timeRange[1]).all()
                ChronicEvents = pd.DataFrame.from_records(ChronicEvents.values("event_name", "event_time"))
                if "event_name" in ChronicEvents.keys():
                    LFPTrends[-1]["EventName"] = ChronicEvents["event_name"]
                    LFPTrends[-1]["EventTime"] = [time.timestamp() for time in ChronicEvents["event_time"]]
                else:
                    LFPTrends[-1]["EventName"] = []
                    LFPTrends[-1]["EventTime"] = []

                LFPTrends[-1]["Power"] = np.array(LFPTrends[-1]["Power"])
                LFPTrends[-1]["Timestamp"] = np.array(LFPTrends[-1]["Timestamp"])

                # Event Locked Power
                EventToInclude = list()
                LFPTrends[-1]["EventLockedPower"] = dict()
                LFPTrends[-1]["EventLockedPower"]["TimeArray"] = np.arange(37)*600 - 180*60
                EventLockedPower = np.zeros((len(LFPTrends[-1]["EventName"]),len(LFPTrends[-1]["EventLockedPower"]["TimeArray"])))
                for iEvent in range(len(LFPTrends[-1]["EventName"])):
                    dataSelected = rangeSelection(LFPTrends[-1]["Timestamp"], [LFPTrends[-1]["EventTime"][iEvent]+LFPTrends[-1]["EventLockedPower"]["TimeArray"][0], LFPTrends[-1]["EventTime"][iEvent]+LFPTrends[-1]["EventLockedPower"]["TimeArray"][-1]])
                    PowerTrend = LFPTrends[-1]["Power"][dataSelected]
                    Timestamp = LFPTrends[-1]["Timestamp"][dataSelected]
                    if len(Timestamp) > 35:
                        index = np.argsort(Timestamp)
                        EventLockedPower[iEvent,:] = np.interp(LFPTrends[-1]["EventLockedPower"]["TimeArray"]+LFPTrends[-1]["EventTime"][iEvent], Timestamp[index], PowerTrend[index])
                        EventToInclude.append(iEvent)
                EventToInclude = np.array(EventToInclude)

                if not len(EventToInclude) == 0:
                    LFPTrends[-1]["EventLockedPower"]["PowerChart"] = list()
                    LFPTrends[-1]["EventLockedPower"]["EventName"] = np.array(LFPTrends[-1]["EventName"])[EventToInclude]
                    EventLockedPower = EventLockedPower[EventToInclude,:]
                    for name in np.unique(LFPTrends[-1]["EventLockedPower"]["EventName"]):
                        SelectedEvent = LFPTrends[-1]["EventLockedPower"]["EventName"] == name
                        LFPTrends[-1]["EventLockedPower"]["PowerChart"].append({"EventName": name + f" (n={np.sum(SelectedEvent)})", "EventColor": EventColor[name],
                                                        "Line": np.mean(EventLockedPower[SelectedEvent,:], axis=0),
                                                        "Shade": SPU.stderr(EventLockedPower[SelectedEvent,:],axis=0)})

                    LFPTrends[-1]["EventLockedPower"]["PowerRange"] = [np.percentile(EventLockedPower.flatten(),1),np.percentile(EventLockedPower.flatten(),99)]
                    LFPTrends[-1]["EventLockedPower"]["TimeArray"] = LFPTrends[-1]["EventLockedPower"]["TimeArray"] / 60

    return LFPTrends

def queryChronicLFPs(user, patientUniqueID, TherapyHistory, authority):
    LFPTrends = list()
    if not authority["Permission"]:
        return LFPTrends

    availableDevices = getPerceptDevices(user, patientUniqueID, authority)
    for device in availableDevices:
        leads = device.device_lead_configurations
        for hemisphere in ["HemisphereLocationDef.Left","HemisphereLocationDef.Right"]:
            recording = models.BrainSenseRecording.objects.filter(device_deidentified_id=device.deidentified_id, recording_type="ChronicLFPs", recording_info__Hemisphere=hemisphere).first()
            if not recording == None:
                ChronicLFPs = loadSourceFiles(recording.recording_type,hemisphere.replace("HemisphereLocationDef.",""),recording.recording_id)
                if device.device_name == "":
                    LFPTrends.append({"Device": device.getDeviceSerialNumber(key), "DeviceLocation": device.device_location})
                else:
                    LFPTrends.append({"Device": device.device_name, "DeviceLocation": device.device_location})

                for lead in leads:
                    if lead["TargetLocation"].startswith(hemisphere.replace("HemisphereLocationDef.","")):
                        LFPTrends[-1]["Hemisphere"] = lead["TargetLocation"]

                LFPTimestamps = ChronicLFPs["DateTime"]
                LFPPowers = ChronicLFPs["LFP"]
                StimulationAmplitude = ChronicLFPs["Amplitude"]
                LFPTimestamps = np.array([time.timestamp() for time in LFPTimestamps])
                LFPTrends[-1]["Timestamp"] = list()
                LFPTrends[-1]["Power"] = list()
                LFPTrends[-1]["Amplitude"] = list()
                LFPTrends[-1]["Therapy"] = list()
                LFPTrends[-1]["EventName"] = list()
                LFPTrends[-1]["EventTime"] = list()

                # Remove Outliers
                LFPSelection = LFPPowers < np.median(LFPPowers) + np.std(LFPPowers)*6
                LFPTimestamps = LFPTimestamps[LFPSelection]
                LFPPowers = LFPPowers[LFPSelection]
                StimulationAmplitude = StimulationAmplitude[LFPSelection]

                LFPTrends[-1]["PowerRange"] = [0,0]

                # Remove Outliers
                LFPSelection = LFPPowers < np.median(LFPPowers) + np.std(LFPPowers)*6
                LFPTimestamps = LFPTimestamps[LFPSelection]
                LFPPowers = LFPPowers[LFPSelection]
                StimulationAmplitude = StimulationAmplitude[LFPSelection]

                LFPTrends[-1]["PowerRange"] = [0,0]

                #[b,a] = signal.butter(5, 0.00003*2*600, 'high', output='ba')
                for therapy in TherapyHistory:
                    if therapy["device"] == device.deidentified_id:
                        for i in range(len(therapy["date_of_change"])-1):
                            rangeSelected = rangeSelection(LFPTimestamps,[therapy["date_of_change"][i]/1000000000,therapy["date_of_change"][i+1]/1000000000])
                            if np.any(rangeSelected):
                                LFPTrends[-1]["Timestamp"].append(LFPTimestamps[rangeSelected])
                                #FiltPower = signal.filtfilt(b,a,LFPPowers[rangeSelected])
                                FiltPower = np.array(LFPPowers[rangeSelected]).tolist()
                                LFPTrends[-1]["Power"].append(FiltPower)
                                LFPTrends[-1]["Amplitude"].append(np.array(StimulationAmplitude[rangeSelected]).tolist())
                                LFPTrends[-1]["Therapy"].append(copy.deepcopy(therapy["therapy"][i]))
                                if np.percentile(FiltPower,5) < LFPTrends[-1]["PowerRange"][0]:
                                    LFPTrends[-1]["PowerRange"][0] = np.percentile(FiltPower,5)
                                if np.percentile(FiltPower,95) > LFPTrends[-1]["PowerRange"][1]:
                                    LFPTrends[-1]["PowerRange"][1] = np.percentile(FiltPower,95)

                                ChronicEvents = models.PatientCustomEvents.objects.filter(device_deidentified_id=device.deidentified_id,
                                                    event_time__gt=datetime.fromtimestamp(therapy["date_of_change"][i]/1000000000,tz=pytz.utc), event_time__lt=datetime.fromtimestamp(therapy["date_of_change"][i+1]/1000000000,tz=pytz.utc)).all()
                                ChronicEvents = pd.DataFrame.from_records(ChronicEvents.values("event_name", "event_time"))
                                if "event_name" in ChronicEvents.keys():
                                    LFPTrends[-1]["EventName"].append(ChronicEvents["event_name"])
                                    LFPTrends[-1]["EventTime"].append([time.timestamp() for time in ChronicEvents["event_time"]])
                                else:
                                    LFPTrends[-1]["EventName"].append([])
                                    LFPTrends[-1]["EventTime"].append([])

    return LFPTrends

def processEventMarkers(LFPTrends, EventNames):
    EventColor = dict()
    cmap = plt.get_cmap("Set1", 9)
    for cIndex in range(len(EventNames)):
        EventColor[EventNames[cIndex]] = colorTextFromCmap(cmap(cIndex))

    EventMarker = list()
    for i in range(len(LFPTrends)):
        EventMarker.append({"EventPower": list(), "EventTime": list(), "EventName": list(), "EventColor": list()})
        for cIndex in range(len(EventNames)):
            EventMarker[i]["EventName"].append(EventNames[cIndex])
            EventMarker[i]["EventColor"].append(EventColor[EventNames[cIndex]])
            EventMarker[i]["EventPower"].append(list())
            EventMarker[i]["EventTime"].append(list())
            for j in range(len(LFPTrends[i]["EventTime"])):
                for k in range(len(LFPTrends[i]["EventTime"][j])):
                    if LFPTrends[i]["EventName"][j][k] == EventNames[cIndex]:
                        index = np.argmin(np.abs(LFPTrends[i]["EventTime"][j][k] - LFPTrends[i]["Timestamp"][j]))
                        EventMarker[i]["EventPower"][cIndex].append(LFPTrends[i]["Power"][j][index])
                        EventMarker[i]["EventTime"][cIndex].append(LFPTrends[i]["EventTime"][j][k]*1000)
    return EventMarker

def processChronicLFPs(LFPTrends, EventNames, timezoneOffset=0):
    EventColor = dict()
    cmap = plt.get_cmap("Set1", 9)
    for cIndex in range(len(EventNames)):
        EventColor[EventNames[cIndex]] = colorTextFromCmap(cmap(cIndex))

    for i in range(len(LFPTrends)):
        if LFPTrends[i]["Hemisphere"].startswith("Left"):
            Hemisphere = "LeftHemisphere"
        else:
            Hemisphere = "RightHemisphere"

        TherapyList = list()
        for j in range(len(LFPTrends[i]["Therapy"])):
            if not Hemisphere in LFPTrends[i]["Therapy"][j].keys():
                continue
            Therapy = LFPTrends[i]["Therapy"][j][Hemisphere]
            if "SensingSetup" in Therapy.keys():
                TherapyOverview = f"{Therapy['Frequency']}Hz {Therapy['PulseWidth']}uS @ {Therapy['SensingSetup']['FrequencyInHertz']}Hz"
            else:
                TherapyOverview = f"{Therapy['Frequency']}Hz {Therapy['PulseWidth']}uS @ {0}Hz"
            LFPTrends[i]["Therapy"][j]["TherapyOverview"] = TherapyOverview

            TherapyList.append(TherapyOverview)
        UniqueTherapyList = uniqueList(TherapyList)

        LFPTrends[i]["EventLockedPower"] = list()
        LFPTrends[i]["TherapyAmplitudes"] = list()
        LFPTrends[i]["CircadianPowers"] = list()
        for therapy in UniqueTherapyList:
            LFPTrends[i]["EventLockedPower"].append({"EventName": list(), "Timestamp": list(), "Therapy": therapy})
            LFPTrends[i]["CircadianPowers"].append({"Power": list(), "Timestamp": list(), "Therapy": therapy})
            LFPTrends[i]["TherapyAmplitudes"].append({"Power": list(), "Amplitude": list(), "Therapy": therapy})

            for j in range(len(LFPTrends[i]["Therapy"])):
                if not Hemisphere in LFPTrends[i]["Therapy"][j].keys():
                    continue
                Therapy = LFPTrends[i]["Therapy"][j][Hemisphere]
                if "SensingSetup" in Therapy.keys():
                    TherapyOverview = f"{Therapy['Frequency']}Hz {Therapy['PulseWidth']}uS @ {Therapy['SensingSetup']['FrequencyInHertz']}Hz"
                else:
                    TherapyOverview = f"{Therapy['Frequency']}Hz {Therapy['PulseWidth']}uS @ {0}Hz"

                if TherapyOverview == therapy:
                    LFPTrends[i]["CircadianPowers"][-1]["Power"].extend(LFPTrends[i]["Power"][j])
                    LFPTrends[i]["CircadianPowers"][-1]["Timestamp"].extend(LFPTrends[i]["Timestamp"][j])

                    LFPTrends[i]["TherapyAmplitudes"][-1]["Power"].extend(LFPTrends[i]["Power"][j])
                    LFPTrends[i]["TherapyAmplitudes"][-1]["Amplitude"].extend(LFPTrends[i]["Amplitude"][j])

                    LFPTrends[i]["EventLockedPower"][-1]["EventName"].extend(LFPTrends[i]["EventName"][j])
                    LFPTrends[i]["EventLockedPower"][-1]["Timestamp"].extend(LFPTrends[i]["EventTime"][j])

            LFPTrends[i]["CircadianPowers"][-1]["Power"] = np.array(LFPTrends[i]["CircadianPowers"][-1]["Power"])
            LFPTrends[i]["CircadianPowers"][-1]["Timestamp"] = np.array(LFPTrends[i]["CircadianPowers"][-1]["Timestamp"])

            # Event Locked Power
            EventToInclude = list()
            LFPTrends[i]["EventLockedPower"][-1]["TimeArray"] = np.arange(37)*600 - 180*60
            EventLockedPower = np.zeros((len(LFPTrends[i]["EventLockedPower"][-1]["EventName"]),len(LFPTrends[i]["EventLockedPower"][-1]["TimeArray"])))
            for iEvent in range(len(LFPTrends[i]["EventLockedPower"][-1]["EventName"])):
                dataSelected = rangeSelection(LFPTrends[i]["CircadianPowers"][-1]["Timestamp"], [LFPTrends[i]["EventLockedPower"][-1]["Timestamp"][iEvent]+LFPTrends[i]["EventLockedPower"][-1]["TimeArray"][0], LFPTrends[i]["EventLockedPower"][-1]["Timestamp"][iEvent]+LFPTrends[i]["EventLockedPower"][-1]["TimeArray"][-1]])
                PowerTrend = LFPTrends[i]["CircadianPowers"][-1]["Power"][dataSelected]
                Timestamp = LFPTrends[i]["CircadianPowers"][-1]["Timestamp"][dataSelected]
                if len(Timestamp) > 35:
                    index = np.argsort(Timestamp)
                    EventLockedPower[iEvent,:] = np.interp(LFPTrends[i]["EventLockedPower"][-1]["TimeArray"]+LFPTrends[i]["EventLockedPower"][-1]["Timestamp"][iEvent], Timestamp[index], PowerTrend[index])
                    EventToInclude.append(iEvent)
            EventToInclude = np.array(EventToInclude)

            if not len(EventToInclude) == 0:
                LFPTrends[i]["EventLockedPower"][-1]["PowerChart"] = list()
                LFPTrends[i]["EventLockedPower"][-1]["EventName"] = np.array(LFPTrends[i]["EventLockedPower"][-1]["EventName"])[EventToInclude]
                EventLockedPower = EventLockedPower[EventToInclude,:]
                for name in np.unique(LFPTrends[i]["EventLockedPower"][-1]["EventName"]):
                    SelectedEvent = LFPTrends[i]["EventLockedPower"][-1]["EventName"] == name
                    LFPTrends[i]["EventLockedPower"][-1]["PowerChart"].append({"EventName": name + f" (n={np.sum(SelectedEvent)})", "EventColor": EventColor[name],
                                                    "Line": np.mean(EventLockedPower[SelectedEvent,:], axis=0),
                                                    "Shade": SPU.stderr(EventLockedPower[SelectedEvent,:],axis=0)})

                LFPTrends[i]["EventLockedPower"][-1]["PowerRange"] = [np.percentile(EventLockedPower.flatten(),1),np.percentile(EventLockedPower.flatten(),99)]
                del(LFPTrends[i]["EventLockedPower"][-1]["EventName"])
                del(LFPTrends[i]["EventLockedPower"][-1]["Timestamp"])
                LFPTrends[i]["EventLockedPower"][-1]["TimeArray"] = LFPTrends[i]["EventLockedPower"][-1]["TimeArray"] / 60

            LFPTrends[i]["CircadianPowers"][-1]["Timestamp"] = (np.array(LFPTrends[i]["CircadianPowers"][-1]["Timestamp"])-timezoneOffset) % (24*60*60)

            # Calculate Average Power/Std Power
            LFPTrends[i]["CircadianPowers"][-1]["AverageTimestamp"] = np.arange(24*12)*300
            LFPTrends[i]["CircadianPowers"][-1]["AveragePower"] = np.zeros(LFPTrends[i]["CircadianPowers"][-1]["AverageTimestamp"].shape)
            LFPTrends[i]["CircadianPowers"][-1]["StdErrPower"] = np.zeros(LFPTrends[i]["CircadianPowers"][-1]["AverageTimestamp"].shape)
            for t in range(len(LFPTrends[i]["CircadianPowers"][-1]["AverageTimestamp"])):
                timeSelection = rangeSelection(LFPTrends[i]["CircadianPowers"][-1]["Timestamp"],[LFPTrends[i]["CircadianPowers"][-1]["AverageTimestamp"][t]-20*60, LFPTrends[i]["CircadianPowers"][-1]["AverageTimestamp"][t]+20*60])
                if np.any(timeSelection):
                    LFPTrends[i]["CircadianPowers"][-1]["AveragePower"][t] = np.median(LFPTrends[i]["CircadianPowers"][-1]["Power"][timeSelection])
                    LFPTrends[i]["CircadianPowers"][-1]["StdErrPower"][t] = SPU.stderr(LFPTrends[i]["CircadianPowers"][-1]["Power"][timeSelection])*2
                else:
                    LFPTrends[i]["CircadianPowers"][-1]["AveragePower"][t] = 0
                    LFPTrends[i]["CircadianPowers"][-1]["StdErrPower"][t] = 0

            LFPTrends[i]["CircadianPowers"][-1]["Power"] = LFPTrends[i]["CircadianPowers"][-1]["Power"][::1].tolist()
            LFPTrends[i]["CircadianPowers"][-1]["Timestamp"] = (LFPTrends[i]["CircadianPowers"][-1]["Timestamp"] + timezoneOffset).tolist()
            LFPTrends[i]["CircadianPowers"][-1]["AveragePower"] = LFPTrends[i]["CircadianPowers"][-1]["AveragePower"].tolist()
            LFPTrends[i]["CircadianPowers"][-1]["StdErrPower"] = LFPTrends[i]["CircadianPowers"][-1]["StdErrPower"].tolist()
            LFPTrends[i]["CircadianPowers"][-1]["AverageTimestamp"] = (LFPTrends[i]["CircadianPowers"][-1]["AverageTimestamp"] + timezoneOffset).tolist()

            LFPTrends[i]["CircadianPowers"][-1]["PowerRange"] = [np.percentile(LFPTrends[i]["CircadianPowers"][-1]["Power"],5),np.percentile(LFPTrends[i]["CircadianPowers"][-1]["Power"],95)]
    return LFPTrends

def queryEventPSDs(user, patientUniqueID, therapyHistory, authority):
    EventPSDs = list()
    if not authority["Permission"]:
        return EventPSDs

    availableDevices = getPerceptDevices(user, patientUniqueID, authority)
    for device in availableDevices:
        allEvents = models.PatientCustomEvents.objects.filter(device_deidentified_id=device.deidentified_id).order_by("-event_time").all()
        if len(allEvents) > 0:
            allEvents = pd.DataFrame.from_records(allEvents.values("event_time", "sensing_exist", "event_name","brainsense_psd"))
            TherapyParameters = list()
            for i in range(len(allEvents["event_time"])):
                allEvents["event_time"][i]
                TherapyParameters.append()
            EventTime = allEvents["event_time"][allEvents["sensing_exist"]]
            EventName = allEvents["event_name"][allEvents["sensing_exist"]]
            PSDs = allEvents["brainsense_psd"][allEvents["sensing_exist"]]

            leads = device.device_lead_configurations

            EventPSDs.append({"Device": device.getDeviceSerialNumber(key), "DeviceLocation": device.device_location})
            for therapy in TherapyHistory:
                if therapy["device"] == device.deidentified_id:
                    for i in range(len(therapy["date_of_change"])-1):
                        rangeSelected = rangeSelection(EventTime,[therapy["date_of_change"][i]/1000000000,therapy["date_of_change"][i+1]/1000000000])
                        if np.any(rangeSelected):
                            LFPTrends[-1]["Timestamp"].append(LFPTimestamps[rangeSelected])
                            #LFPTrends[-1]["Power"].append(LFPPowers[rangeSelected])
                            LFPTrends[-1]["Therapy"].append(therapy["therapy"][i])


            for hemisphere in ["HemisphereLocationDef.Left","HemisphereLocationDef.Right"]:
                if np.any(ChronicLFPs["hemisphere"] == hemisphere):
                    LFPTrends.append({"Device": device.getDeviceSerialNumber(key), "DeviceLocation": device.device_location})
                    for lead in leads:
                        if lead["TargetLocation"].startswith(hemisphere.replace("HemisphereLocationDef.","")):
                            LFPTrends[-1]["Hemisphere"] = lead["TargetLocation"]
                    LFPTimestamps = ChronicLFPs["timestamp"][ChronicLFPs["hemisphere"] == hemisphere]
                    LFPPowers = ChronicLFPs["power"][ChronicLFPs["hemisphere"] == hemisphere]
                    LFPTimestamps = np.array([time.timestamp() for time in LFPTimestamps])
                    LFPTrends[-1]["Timestamp"] = list()
                    LFPTrends[-1]["Power"] = list()
                    LFPTrends[-1]["Therapy"] = list()

                    # Remove Outliers
                    LFPSelection = LFPPowers < np.median(LFPPowers) + np.std(LFPPowers)*6
                    LFPTimestamps = LFPTimestamps[LFPSelection]
                    LFPPowers = LFPPowers[LFPSelection]

                    LFPTrends[-1]["PowerRange"] = [np.percentile(LFPPowers,5),np.percentile(LFPPowers,95)]

                    for therapy in TherapyHistory:
                        if therapy["device"] == device.deidentified_id:
                            for i in range(len(therapy["date_of_change"])-1):
                                rangeSelected = rangeSelection(LFPTimestamps,[therapy["date_of_change"][i]/1000000000,therapy["date_of_change"][i+1]/1000000000])
                                if np.any(rangeSelected):
                                    LFPTrends[-1]["Timestamp"].append(LFPTimestamps[rangeSelected])
                                    LFPTrends[-1]["Power"].append(LFPPowers[rangeSelected])
                                    LFPTrends[-1]["Therapy"].append(therapy["therapy"][i])
    return LFPTrends

def queryMontageData(user, devices, timestamps, authority):
    BrainSenseData = list()
    for i in range(len(devices)):
        device = models.PerceptDevice.objects.filter(deidentified_id=devices[i]).first()

        if not device == None:
            leads = device.device_lead_configurations
            recording = models.BrainSenseRecording.objects.filter(device_deidentified_id=devices[i], recording_date=datetime.fromtimestamp(timestamps[i],tz=pytz.utc), recording_type="IndefiniteStream").first()
            if not recording == None:
                if not recording.recording_id in authority["Permission"] and authority["Level"] == 2:
                    continue

                stream = loadSourceFiles(recording.recording_type,"Combined",recording.recording_id)
                if not "Spectrums" in stream.keys():
                    stream = processMontageStreams(stream)
                    saveSourceFiles(stream,recording.recording_type,"Combined",recording.recording_id)
                data = dict()
                data["Timestamp"] = recording.recording_date.timestamp()
                data["DeviceID"] = devices[i]
                data["Channels"] = stream["Channels"]
                data["ChannelNames"] = list()
                for channel in stream["Channels"]:
                    contacts, hemisphere = Percept.reformatChannelName(channel)
                    for lead in leads:
                        if lead["TargetLocation"].startswith(hemisphere):
                            data["ChannelNames"].append(lead["TargetLocation"] + f" E{contacts[0]:02}-E{contacts[1]:02}")
                for channel in stream["Channels"]:
                    data[channel] = stream[channel]
                data["Spectrums"] = stream["Spectrums"]
                BrainSenseData.append(data)
    return BrainSenseData

def queryRealtimeStreamOverview(user, patientUniqueID, authority):
    BrainSenseData = list()
    if not authority["Permission"]:
        return BrainSenseData

    includedRecording = list()
    availableDevices = getPerceptDevices(user, patientUniqueID, authority)
    for device in availableDevices:
        allSurveys = models.BrainSenseRecording.objects.filter(device_deidentified_id=device.deidentified_id, recording_type="BrainSenseStream").order_by("-recording_date").all()
        if len(allSurveys) > 0:
            leads = device.device_lead_configurations

        for recording in allSurveys:
            if not recording.recording_id in authority["Permission"] and authority["Level"] == 2:
                continue

            data = dict()
            data["Timestamp"] = recording.recording_date.timestamp()
            data["Duration"] = recording.recording_duration

            if data["Timestamp"] in includedRecording:
                continue

            if data["Duration"] < 30:
                continue

            data["RecordingID"] = recording.recording_id
            if device.device_name == "":
                data["DeviceName"] = device.getDeviceSerialNumber(key)
            else:
                data["DeviceName"] = device.device_name
            data["DeviceID"] = device.deidentified_id
            data["DeviceLocation"] = device.device_location
            data["Channels"] = list()
            data["ContactTypes"] = list()

            if not "Therapy" in recording.recording_info:
                if len(recording.recording_info["Channel"]) == 2:
                    info = "Bilateral"
                else:
                    info = "Unilateral"
                RawData = loadSourceFiles(recording.recording_type,info,recording.recording_id)
                recording.recording_info["Therapy"] = RawData["Therapy"]
                recording.save()
            data["Therapy"] = recording.recording_info["Therapy"]

            Channels = recording.recording_info["Channel"]

            if not "ContactType" in recording.recording_info:
                recording.recording_info["ContactType"] = ["Ring" for channel in Channels]
                recording.save()
            if not len(recording.recording_info["ContactType"]) == len(Channels):
                recording.recording_info["ContactType"] = ["Ring" for channel in Channels]
                recording.save()

            data["ContactType"] = recording.recording_info["ContactType"]
            for channel in Channels:
                contacts, hemisphere = Percept.reformatChannelName(channel)
                for lead in leads:
                    if lead["TargetLocation"].startswith(hemisphere):
                        data["Channels"].append({"Hemisphere": lead["TargetLocation"], "Contacts": contacts, "Type": lead["ElectrodeType"]})
                        if lead["ElectrodeType"].startswith("SenSight"):
                            data["ContactTypes"].append(["Ring","Segment A","Segment B","Segment C","Segment AB","Segment BC","Segment AC"])
                        else:
                            data["ContactTypes"].append(["Ring"])

            BrainSenseData.append(data)
            includedRecording.append(data["Timestamp"])
    return BrainSenseData

def queryRealtimeStreamData(user, device, timestamp, authority, cardiacFilter=False, refresh=False):
    BrainSenseData = None
    RecordingID = None
    if authority["Level"] == 0:
        return BrainSenseData, RecordingID

    if not authority["Permission"]:
        return BrainSenseData, RecordingID

    recording_info = {"CardiacFilter": cardiacFilter}
    if models.BrainSenseRecording.objects.filter(device_deidentified_id=device, recording_date=datetime.fromtimestamp(timestamp,tz=pytz.utc), recording_type="BrainSenseStream", recording_info__contains=recording_info).exists():
        recording = models.BrainSenseRecording.objects.get(device_deidentified_id=device, recording_date=datetime.fromtimestamp(timestamp,tz=pytz.utc), recording_type="BrainSenseStream", recording_info__contains=recording_info)
    else:
        recording = models.BrainSenseRecording.objects.filter(device_deidentified_id=device, recording_date=datetime.fromtimestamp(timestamp,tz=pytz.utc), recording_type="BrainSenseStream").first()

    if not recording == None:
        if authority["Level"] == 2:
            if not recording.recording_id in authority["Permission"]:
                return BrainSenseData, RecordingID

        if len(recording.recording_info["Channel"]) == 2:
            info = "Bilateral"
        else:
            info = "Unilateral"

        BrainSenseData = loadSourceFiles(recording.recording_type,info,recording.recording_id)
        BrainSenseData["Info"] = recording.recording_info;

        if not "CardiacFilter" in recording.recording_info:
            recording.recording_info["CardiacFilter"] = cardiacFilter
            recording.save()

        if not "Spectrogram" in BrainSenseData.keys() or (refresh and (not recording.recording_info["CardiacFilter"] == cardiacFilter)):
            recording.recording_info["CardiacFilter"] = cardiacFilter
            BrainSenseData = processRealtimeStreams(BrainSenseData, cardiacFilter=cardiacFilter)
            saveSourceFiles(BrainSenseData,recording.recording_type,info,recording.recording_id)
            recording.save()

        BrainSenseData["Info"] = recording.recording_info;
        RecordingID = recording.recording_id
    return BrainSenseData, RecordingID

def processRealtimeStreamRenderingData(stream, options=dict()):
    stream["Stimulation"] = processRealtimeStreamStimulationAmplitude(stream)
    stream["PowerBand"] = processRealtimeStreamPowerBand(stream)
    data = dict()
    data["Channels"] = stream["Channels"]
    data["Stimulation"] = stream["Stimulation"]
    data["PowerBand"] = stream["PowerBand"]
    data["Info"] = stream["Info"]
    for channel in stream["Channels"]:
        data[channel] = dict()
        data[channel]["Time"] = stream["Time"]
        data[channel]["RawData"] = stream["Filtered"][channel]
        if options["SpectrogramMethod"]["value"] == "Spectrogram":
            data[channel]["Spectrogram"] = copy.deepcopy(stream["Spectrogram"][channel])
            data[channel]["Spectrogram"]["Power"][data[channel]["Spectrogram"]["Power"] == 0] = 1e-10
            data[channel]["Spectrogram"]["Power"] = np.log10(data[channel]["Spectrogram"]["Power"])*10
            data[channel]["Spectrogram"]["ColorRange"] = [-20,20]
        elif options["SpectrogramMethod"]["value"]  == "Wavelet":
            data[channel]["Spectrogram"] = copy.deepcopy(stream["Wavelet"][channel])
            data[channel]["Spectrogram"]["Power"][data[channel]["Spectrogram"]["Power"] == 0] = 1e-10
            data[channel]["Spectrogram"]["Power"] = np.log10(data[channel]["Spectrogram"]["Power"])*10
            data[channel]["Spectrogram"]["ColorRange"] = [-10,20]

        if options["PSDMethod"]["value"] == "Time-Frequency Analysis":
            data[channel]["StimPSD"] = processRealtimeStreamStimulationPSD(stream, channel, method=options["SpectrogramMethod"]["value"], stim_label="Ipsilateral")
        else:
            data[channel]["StimPSD"] = processRealtimeStreamStimulationPSD(stream, channel, method=options["PSDMethod"]["value"], stim_label="Ipsilateral")

    return data

def processRealtimeStreamStimulationAmplitude(stream):
    StimulationSeries = list()
    Hemisphere = ["Left","Right"]
    for StimulationSide in range(2):
        if Hemisphere[StimulationSide] in stream["Therapy"].keys():
            Stimulation = np.around(stream["Stimulation"][:,StimulationSide],2)
            indexOfChanges = np.where(np.abs(np.diff(Stimulation)) > 0)[0]-1
            if len(indexOfChanges) == 0:
                indexOfChanges = np.insert(indexOfChanges,0,0)
            elif indexOfChanges[0] < 0:
                indexOfChanges[0] = 0
            else:
                indexOfChanges = np.insert(indexOfChanges,0,0)
            indexOfChanges = np.insert(indexOfChanges,len(indexOfChanges),len(Stimulation)-1)
            for channelName in stream["Channels"]:
                channels, hemisphere = Percept.reformatChannelName(channelName)
                if hemisphere == Hemisphere[StimulationSide]:
                    StimulationSeries.append({"Name": channelName, "Time": stream["Time"][indexOfChanges], "Amplitude": np.around(stream["Stimulation"][indexOfChanges,StimulationSide],2)})
    return StimulationSeries

def processRealtimeStreamPowerBand(stream):
    PowerSensing = list()
    Hemisphere = ["Left","Right"]
    for StimulationSide in range(2):
        if Hemisphere[StimulationSide] in stream["Therapy"].keys():
            Power = stream["PowerBand"][:,StimulationSide]
            selectedData = np.abs(stats.zscore(Power)) < 3
            for channelName in stream["Channels"]:
                channels, hemisphere = Percept.reformatChannelName(channelName)
                if hemisphere == Hemisphere[StimulationSide]:
                    PowerSensing.append({"Name": channelName, "Time": stream["Time"][selectedData], "Power": stream["PowerBand"][selectedData,StimulationSide]})
    return PowerSensing

def processRealtimeStreamStimulationPSD(stream, channel, method="Spectrogram", stim_label="Ipsilateral", centerFrequency=0):
    if stim_label == "Ipsilateral":
        for Stimulation in stream["Stimulation"]:
            if Stimulation["Name"] == channel:
                StimulationSeries = Stimulation
    else:
        for Stimulation in stream["Stimulation"]:
            if not Stimulation["Name"] == channel:
                StimulationSeries = Stimulation

    if not "StimulationSeries" in locals():
        raise Exception("Data not available")

    cIndex = 0;
    StimulationEpochs = list()
    for i in range(1,len(StimulationSeries["Time"])):
        StimulationDuration = StimulationSeries["Time"][i] - StimulationSeries["Time"][i-1]
        if StimulationDuration < 7:
            continue
        cIndex += 1

        if method == "Welch":
            timeSelection = rangeSelection(stream["Time"],[StimulationSeries["Time"][i-1]+2,StimulationSeries["Time"][i]-2])
            StimulationEpoch = stream["Filtered"][channel][timeSelection]
            fxx, pxx = signal.welch(StimulationEpoch, fs=250, nperseg=250 * 1, noverlap=250 * 0.5, nfft=250 * 2, scaling="density")
            StimulationEpochs.append({"Stimulation": StimulationSeries["Amplitude"][i], "Frequency": fxx, "PSD": pxx})
            timeSelection = rangeSelection(stream["Spectrogram"][channel]["Time"],[StimulationSeries["Time"][i-1]+2,StimulationSeries["Time"][i]-2])

        elif method == "Spectrogram":
            timeSelection = rangeSelection(stream["Spectrogram"][channel]["Time"],[StimulationSeries["Time"][i-1]+2,StimulationSeries["Time"][i]-2])
            StimulationEpochs.append({"Stimulation": StimulationSeries["Amplitude"][i], "Frequency": stream["Spectrogram"][channel]["Frequency"], "PSD": np.mean(stream["Spectrogram"][channel]["Power"][:,timeSelection],axis=1)})

        elif method == "Wavelet":
            timeSelection = rangeSelection(stream["Wavelet"][channel]["Time"],[StimulationSeries["Time"][i-1]+2,StimulationSeries["Time"][i]-2])
            StimulationEpochs.append({"Stimulation": StimulationSeries["Amplitude"][i], "Frequency": stream["Wavelet"][channel]["Frequency"], "PSD": np.mean(stream["Wavelet"][channel]["Power"][:,timeSelection],axis=1)})

        StimulationEpochs[-1]["TimeSelection"] = timeSelection

    if len(StimulationEpochs) == 0:
        return StimulationEpochs

    StimulationEpochs = sorted(StimulationEpochs, key=lambda epoch: epoch["Stimulation"])
    if centerFrequency == 0:
        centerFrequency = predictCenterFrequency(StimulationEpochs)

    frequencySelection = rangeSelection(StimulationEpochs[0]["Frequency"], [centerFrequency-2,centerFrequency+2])

    cmap = plt.get_cmap("jet", cIndex);
    for i in range(len(StimulationEpochs)):
        StimulationEpochs[i]["StimulationColor"] = colorTextFromCmap(cmap(i))
        StimulationEpochs[i]["CenterFrequency"] = centerFrequency
        timeSelection = StimulationEpochs[i]["TimeSelection"]

        if method == "Wavelet":
            StimulationEpochs[i]["SpectralFeatures"] = stream["Wavelet"][channel]["Power"][:,timeSelection]
        else:
            StimulationEpochs[i]["SpectralFeatures"] = stream["Spectrogram"][channel]["Power"][:,timeSelection]

        StimulationEpochs[i]["SpectralFeatures"] = np.mean(StimulationEpochs[i]["SpectralFeatures"][frequencySelection,:],axis=0)
        del(StimulationEpochs[i]["TimeSelection"])

    return StimulationEpochs

def predictCenterFrequency(StimulationEpochs, method="Max Beta Desync"):
    if len(StimulationEpochs) >= 3:
        scores = list()
        features = np.zeros((len(StimulationEpochs[0]["Frequency"]),len(StimulationEpochs)))
        for i in range(len(StimulationEpochs)):
            features[:,i] = np.log10(StimulationEpochs[i]["PSD"])
            scores.append(StimulationEpochs[i]["Stimulation"])

        if method == "Max Beta Desync":
            highStim = np.max(scores) - (np.max(scores)-np.min(scores))*0.2
            lowStim = np.min(scores) + (np.max(scores)-np.min(scores))*0.2
            BetaSelection = rangeSelection(StimulationEpochs[0]["Frequency"], [10,50])
            Frequency = StimulationEpochs[0]["Frequency"][BetaSelection]

            minIndex = np.argmin(np.mean(features[BetaSelection,:][:,scores >= highStim],axis=1) - np.mean(features[BetaSelection,:][:,scores <= lowStim],axis=1))
            return Frequency[minIndex]
        else:
            statistc, pvalue = SPU.regressionStatistic(features, scores, 5)
            targetFrequencies = rangeSelection(StimulationEpochs[0]["Frequency"], [5,50])
            index = np.argmax(np.abs(statistc[targetFrequencies])) + np.where(targetFrequencies > 0)[0][0]
            return StimulationEpochs[0]["Frequency"][index]
    else:
        return 0

def queryAvailableSessionFiles(user, patient_id, authority):
    availableDevices = getPerceptDevices(user, patient_id, authority)

    sessions = list()
    for device in availableDevices:
        availableSessions = models.PerceptSession.objects.filter(device_deidentified_id=device.deidentified_id).all()
        for session in availableSessions:
            sessionInfo = dict()
            if not device.device_name == "":
                sessionInfo["DeviceName"] = device.device_name
            else:
                sessionInfo["DeviceName"] = device.getDeviceSerialNumber(key)
            sessionInfo["SessionFilename"] = session.session_source_filename
            sessionInfo["SessionID"] = session.deidentified_id

            sessionInfo["AvailableRecording"] = ""
            sessionInfo["AvailableRecording"] += "Realtime Streaming Data: " + str(models.BrainSenseRecording.objects.filter(source_file=session.deidentified_id, recording_type="BrainSenseStream").count()) + "<br>"
            sessionInfo["AvailableRecording"] += "Indefinite Streaming Data: " + str(models.BrainSenseRecording.objects.filter(source_file=session.deidentified_id, recording_type="IndefiniteStream").count()) + "<br>"
            sessionInfo["AvailableRecording"] += "BrainSense Survey Count: " + str(models.BrainSenseRecording.objects.filter(source_file=session.deidentified_id, recording_type="BrainSenseSurvey").count()) + "<br>"
            sessionInfo["AvailableRecording"] += "Therapy History Info: " + str(models.TherapyHistory.objects.filter(source_file=session.deidentified_id).count()) + "<br>"
            sessionInfo["AvailableRecording"] += "# of Therapy History Changed: " + str(models.TherapyChangeLog.objects.filter(source_file=session.deidentified_id).count()) + "<br>"
            sessionInfo["AvailableRecording"] += "# of Chronic Recording Info: " + str(models.ChronicSensingLFP.objects.filter(source_file=session.deidentified_id).count()) + "<br>"
            sessions.append(sessionInfo)
    return sessions

def deleteDevice(user, patient_id, device_id):
    recordings = models.BrainSenseRecording.objects.filter(device_deidentified_id=device_id).all()
    for recording in recordings:
        try:
            os.remove(DATABASE_PATH + "recordings" + os.path.sep + recording.recording_datapointer)
        except:
            pass
    recordings.delete()

    Sessions = models.PerceptSession.objects.filter(device_deidentified_id=device_id).all()
    for session in Sessions:
        models.TherapyHistory.objects.filter(source_file=str(session.deidentified_id)).delete()
        models.TherapyChangeLog.objects.filter(source_file=str(session.deidentified_id)).delete()
        models.ChronicSensingLFP.objects.filter(source_file=str(session.deidentified_id)).delete()
        try:
            os.remove(session.session_file_path)
        except:
            pass
    Sessions.delete()

def deleteSessions(user, patient_id, session_ids, authority):
    availableDevices = getPerceptDevices(user, patient_id, authority)

    for i in range(len(session_ids)):
        for device in availableDevices:
            if models.PerceptSession.objects.filter(device_deidentified_id=device.deidentified_id, deidentified_id=str(session_ids[i])).exists():
                models.TherapyHistory.objects.filter(source_file=str(session_ids[i])).delete()
                models.TherapyChangeLog.objects.filter(source_file=str(session_ids[i])).delete()
                models.ChronicSensingLFP.objects.filter(source_file=str(session_ids[i])).delete()
                recordings = models.BrainSenseRecording.objects.filter(source_file=str(session_ids[i])).all()
                for recording in recordings:
                    try:
                        os.remove(DATABASE_PATH + "recordings" + os.path.sep + recording.recording_datapointer)
                    except:
                        pass
                recordings.delete()
                session = models.PerceptSession.objects.filter(deidentified_id=session_ids[i]).first()
                try:
                    os.remove(session.session_file_path)
                except:
                    pass
                session.delete()

def queryImpedanceMeasurement(user, patient_id, authority):
    availableDevices = getPerceptDevices(user, patient_id, authority)

    Impedances = list()
    for device in availableDevices:
        Sessions = models.PerceptSession.objects.filter(device_deidentified_id=device.deidentified_id).all()
        for session in Sessions:
            JSON = Percept.decodeJSON(session.session_file_path)
            Overview = Percept.extractPatientInformation(JSON, {})
            if "Impedance" in Overview.keys():
                Impedances.append(Overview["Impedance"])
    return Impedances

def processSessionFile(JSON):
    SessionDate = datetime.fromtimestamp(Percept.estimateSessionDateTime(JSON),tz=pytz.utc).timestamp()

    Overview = {}
    Overview["Overall"] = Percept.extractPatientInformation(JSON, {})
    Overview["Therapy"] = Percept.extractTherapySettings(JSON, {})
    Overview["Overall"]["SessionDate"] = datetime.fromtimestamp(SessionDate).strftime("%Y/%m/%d")

    lastMeasuredTimestamp = 0
    if "TherapyHistory" in Overview["Therapy"].keys():
        for i in range(len(Overview["Therapy"]["TherapyHistory"])):
            HistoryDate = datetime.fromisoformat(Overview["Therapy"]["TherapyHistory"][i]["DateTime"][:-1]+"+00:00").timestamp()
            if SessionDate-HistoryDate > 24*3600:
                lastMeasuredTimestamp = HistoryDate
                break

    if lastMeasuredTimestamp == 0:
        lastMeasuredTimestamp = datetime.fromisoformat(Overview["Overall"]["DeviceInformation"]["ImplantDate"][:-1]+"+00:00").timestamp()

    if "TherapyChangeHistory" in Overview["Therapy"].keys():
        if SessionDate > Overview["Therapy"]["TherapyChangeHistory"][-1]["DateTime"].timestamp():
            Overview["Therapy"]["TherapyChangeHistory"].append({"DateTime": datetime.fromtimestamp(SessionDate), "OldGroupId": Overview["Therapy"]["TherapyChangeHistory"][-1]["NewGroupId"], "NewGroupId": Overview["Therapy"]["TherapyChangeHistory"][-1]["NewGroupId"]})
    else:
        Overview["Therapy"]["TherapyChangeHistory"] = [{"DateTime": datetime.fromtimestamp(SessionDate), "OldGroupId": "GroupIdDef.GROUP_A", "NewGroupId": "GroupIdDef.GROUP_A"}]

    TherapyDutyPercent = dict()
    for i in range(len(Overview["Therapy"]["TherapyChangeHistory"])):
        if lastMeasuredTimestamp < SessionDate:
            if i > 0:
                if Overview["Therapy"]["TherapyChangeHistory"][i]["OldGroupId"] == Overview["Therapy"]["TherapyChangeHistory"][i-1]["NewGroupId"]:
                    Overview["Therapy"]["TherapyChangeHistory"][i]["DateTime"] = Overview["Therapy"]["TherapyChangeHistory"][i]["DateTime"].timestamp()
                    DateOfChange = Overview["Therapy"]["TherapyChangeHistory"][i]["DateTime"]
                    if DateOfChange > lastMeasuredTimestamp:
                        if not Overview["Therapy"]["TherapyChangeHistory"][i]["OldGroupId"] in TherapyDutyPercent.keys():
                            TherapyDutyPercent[Overview["Therapy"]["TherapyChangeHistory"][i]["OldGroupId"]] = 0
                        if DateOfChange > SessionDate:
                            TherapyDutyPercent[Overview["Therapy"]["TherapyChangeHistory"][i]["OldGroupId"]] += (SessionDate-lastMeasuredTimestamp)
                            lastMeasuredTimestamp = SessionDate
                        else:
                            TherapyDutyPercent[Overview["Therapy"]["TherapyChangeHistory"][i]["OldGroupId"]] += (DateOfChange-lastMeasuredTimestamp)
                            lastMeasuredTimestamp = DateOfChange
                else:
                    Overview["Therapy"]["TherapyChangeHistory"][i]["DateTime"] = Overview["Therapy"]["TherapyChangeHistory"][i]["DateTime"].timestamp()
                    DateOfChange = Overview["Therapy"]["TherapyChangeHistory"][i]["DateTime"]
                    if DateOfChange > lastMeasuredTimestamp:
                        if not Overview["Therapy"]["TherapyChangeHistory"][i]["OldGroupId"] in TherapyDutyPercent.keys():
                            TherapyDutyPercent[Overview["Therapy"]["TherapyChangeHistory"][i]["OldGroupId"]] = 0
                        if DateOfChange > SessionDate:
                            TherapyDutyPercent[Overview["Therapy"]["TherapyChangeHistory"][i]["OldGroupId"]] += (SessionDate-lastMeasuredTimestamp)
                            lastMeasuredTimestamp = SessionDate
                        else:
                            TherapyDutyPercent[Overview["Therapy"]["TherapyChangeHistory"][i]["OldGroupId"]] += (DateOfChange-lastMeasuredTimestamp)
                            lastMeasuredTimestamp = DateOfChange
            else:
                Overview["Therapy"]["TherapyChangeHistory"][i]["DateTime"] = Overview["Therapy"]["TherapyChangeHistory"][i]["DateTime"].timestamp()
                DateOfChange = Overview["Therapy"]["TherapyChangeHistory"][i]["DateTime"]
                if DateOfChange > lastMeasuredTimestamp:
                    if not Overview["Therapy"]["TherapyChangeHistory"][i]["OldGroupId"] in TherapyDutyPercent.keys():
                        TherapyDutyPercent[Overview["Therapy"]["TherapyChangeHistory"][i]["OldGroupId"]] = 0
                    if DateOfChange > SessionDate:
                        TherapyDutyPercent[Overview["Therapy"]["TherapyChangeHistory"][i]["OldGroupId"]] += (SessionDate-lastMeasuredTimestamp)
                        lastMeasuredTimestamp = SessionDate
                    else:
                        TherapyDutyPercent[Overview["Therapy"]["TherapyChangeHistory"][i]["OldGroupId"]] += (DateOfChange-lastMeasuredTimestamp)
                        lastMeasuredTimestamp = DateOfChange

    totalHours = np.sum([TherapyDutyPercent[key] for key in TherapyDutyPercent.keys()])
    for i in range(len(Overview["Therapy"]["PreviousGroups"])):
        if Overview["Therapy"]["PreviousGroups"][i]["GroupId"] in TherapyDutyPercent.keys():
            DutyPercent = TherapyDutyPercent[Overview["Therapy"]["PreviousGroups"][i]["GroupId"]] / totalHours
            Overview["Therapy"]["PreviousGroups"][i]["DutyPercent"] = f"({DutyPercent*100:.2f}%)"
        else:
            Overview["Therapy"]["PreviousGroups"][i]["DutyPercent"] = "(0%)"

    if "Impedance" in Overview["Overall"].keys():
        del(Overview["Overall"]["Impedance"])

    return Overview

def viewSession(user, patient_id, session_id, authority):
    availableDevices = getPerceptDevices(user, patient_id, authority)
    for device in availableDevices:
        if models.PerceptSession.objects.filter(device_deidentified_id=device.deidentified_id, deidentified_id=str(session_id)).exists():
            session = models.PerceptSession.objects.filter(deidentified_id=session_id).first()
            JSON = Percept.decodeEncryptedJSON(session.session_file_path, key)
            Overview = processSessionFile(JSON)

            if not user.is_clinician:
                Overview["Overall"]["PatientInformation"]["PatientFirstName"] = "Deidentified FirstName"
                Overview["Overall"]["PatientInformation"]["PatientLastName"] = "Deidentified LastName"
                Overview["Overall"]["PatientInformation"]["Diagnosis"] = "Unknown"
                Overview["Overall"]["PatientInformation"]["PatientId"] = "Unknown"
                Overview["Overall"]["PatientInformation"]["PatientDateOfBirth"] = "Unknown"
                Overview["Overall"]["DeviceInformation"]["NeurostimulatorSerialNumber"] = "Unknown"
            return Overview
    return {}
