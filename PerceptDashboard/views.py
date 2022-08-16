from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseNotFound

import rest_framework.views as RestViews
import rest_framework.parsers as RestParsers
from rest_framework.response import Response

import modules.LocalPerceptDatabase as database
import json
import datetime, pytz
import Percept
import copy

from PythonUtility import *

from . import models

# Create your views here.
def index(request):
    return redirect("patients")

    if not request.user.is_authenticated:
        return redirect("signin")

    if not "ProcessingSettings" in request.session:
        request.session["ProcessingSettings"] = database.retrieveProcessingSettings(request.user)
        request.session.modified = True

    context = dict()
    context["User"] = database.extractUserInfo(request.user)
    context["PageView"] = {"ExpandProcessed": False}
    return render(request, "dashboard.html", context=context)

def patientList(request):
    if not request.user.is_authenticated:
        return redirect("signin")

    if not "ProcessingSettings" in request.session:
        request.session["ProcessingSettings"] = database.retrieveProcessingSettings()
        request.session.modified = True

    context = dict()
    context["User"] = database.extractUserInfo(request.user)
    context["PageView"] = {"ExpandProcessed": False}

    context["Patients"] = database.extractPatientList(request.user)

    if request.user.is_clinician:
        context["Authorization"] = True

    return render(request, "patients.html", context=context)

def patientOverview(request):
    if not request.user.is_authenticated:
        return redirect("signin")

    if not "patient_deidentified_id" in request.session:
        return redirect("patients")

    if request.session["patient_deidentified_id"] == "":
        return redirect("patients")

    Authority = {}
    Authority["Level"] = database.verifyAccess(request.user, request.session["patient_deidentified_id"])
    if Authority["Level"] == 0:
        return redirect("patients")

    context = dict()
    context["User"] = database.extractUserInfo(request.user)
    context["PageView"] = {"ExpandProcessed": True}

    if Authority["Level"] == 1:
        context["Patient"] = database.extractPatientInfo(request.user, request.session["patient_deidentified_id"])
        context["PatientID"] = request.session["patient_deidentified_id"]
    elif Authority["Level"] == 2:
        PatientInfo = database.extractAccess(request.user, request.session["patient_deidentified_id"])
        deidentification = database.extractPatientInfo(request.user, PatientInfo.authorized_patient_id)
        context["Patient"] = database.extractPatientInfo(request.user, PatientInfo.deidentified_id)
        context["Patient"]["Devices"] = deidentification["Devices"]
        context["PatientID"] = request.session["patient_deidentified_id"]
    context["User"]["Permission"] = Authority["Level"]
    return render(request, "patient_overview.html", context=context)

class ResearchAccessView(RestViews.APIView):
    parser_classes = [RestParsers.MultiPartParser, RestParsers.FormParser]

    def get(self, request):
        if not request.user.is_authenticated:
            return redirect("signin")

        if not request.user.is_admin:
            return HttpResponseNotFound("Page Not Found")

        context = dict()
        context["User"] = database.extractUserInfo(request.user)
        context["PageView"] = {"ExpandProcessed": False}
        context["ResearchUserList"] = database.getAllResearchUsers()
        context["PatientList"] = database.extractPatientList(request.user)

        return render(request, "admin_authorizeaccess.html", context=context)

    def post(self, request):
        if not request.user.is_authenticated:
            return Response(status=404)

        if not request.user.is_admin:
            return Response(status=404)

        if "Request" in request.data:
            if request.data["Request"] == "AuthorizedPatientList":
                data = database.extractAuthorizedAccessList(request.data["ResearchAccount"])
                return Response(status=200, data=data)
            elif request.data["Request"] == "AuthorizedRecordingList":
                data = database.extractAvailableRecordingList(request.user, request.data["ResearchAccount"], request.data["PatientID"])
                return Response(status=200, data=data)
            elif request.data["Request"] == "TogglePatientPermission":
                database.AuthorizeResearchAccess(request.user, request.data["ResearchAccount"], request.data["PatientID"], request.data["Permission"] == "Allow")
                return Response(status=200)
            elif request.data["Request"] == "ToggleRecordingPermission":
                database.AuthorizeRecordingAccess(request.user, request.data["ResearchAccount"], request.data["PatientID"], recording_type=request.data["RecordingType"], permission=request.data["Permission"] == "Allow")
                data = database.extractAvailableRecordingList(request.user, request.data["ResearchAccount"], request.data["PatientID"])
                return Response(status=200, data=data)
            elif request.data["Request"] == "ToggleIndividualPermission":
                database.AuthorizeRecordingAccess(request.user, request.data["ResearchAccount"], request.data["PatientID"], recording_id=request.data["RecordingID"], permission=request.data["Permission"] == "Allow")
                return Response(status=200)

        return Response(status=400)

class PatientSessionFiles(RestViews.APIView):
    parser_classes = [RestParsers.MultiPartParser, RestParsers.FormParser]

    def get(self, request):
        if not request.user.is_authenticated:
            return redirect("signin")

        if not "patient_deidentified_id" in request.session:
            return redirect("patients")

        if request.session["patient_deidentified_id"] == "":
            return redirect("patients")

        Authority = {}
        Authority["Level"] = database.verifyAccess(request.user, request.session["patient_deidentified_id"])
        if Authority["Level"] == 0:
            return redirect("patients")

        context = dict()
        context["User"] = database.extractUserInfo(request.user)
        context["PageView"] = {"ExpandProcessed": True}

        if Authority["Level"] == 1:
            context["SessionFiles"] = database.queryAvailableSessionFiles(request.user, request.session["patient_deidentified_id"], Authority)
            context["PatientID"] = request.session["patient_deidentified_id"]
        else:
            return redirect("patient overview")

        return render(request, "report_sessionmanagements.html", context=context)

    def post(self, request):
        if not request.user.is_authenticated:
            return Response(status=404)

        if "session_id" in request.data and "deleteSession" in request.data:
            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.data["deleteSession"])
            if Authority["Level"] == 0:
                return Response(status=404)

            database.deleteSessions(request.user, request.data["deleteSession"], [request.data["session_id"]], Authority)
            return Response(status=200)

        if "session_id" in request.data and "viewSession" in request.data:
            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.data["viewSession"])
            if Authority["Level"] == 0:
                return Response(status=404)

            request.session["SessionJsonId"] = request.data["session_id"]
            return Response(status=200)

        return Response(status=404)

class PatientSessionReport(RestViews.APIView):
    parser_classes = [RestParsers.MultiPartParser, RestParsers.FormParser]

    def get(self, request):
        if not request.user.is_authenticated:
            return redirect("signin")

        if not "patient_deidentified_id" in request.session:
            return redirect("patients")

        if request.session["patient_deidentified_id"] == "":
            return redirect("patients")

        if not "SessionJsonId" in request.session and not "SessionOverview" in request.session:
            return redirect("patients")

        if request.session["patient_deidentified_id"] == "TemporarySession" or request.user.is_clinician:
            context = dict()
            context["User"] = database.extractUserInfo(request.user)
            context["PageView"] = {"ExpandProcessed": True}
            context["SessionOverview"] = copy.deepcopy(request.session["SessionOverview"])
            request.session["SessionOverview"] = None
            request.session["patient_deidentified_id"] = ""
            request.session.modified = True
            context["PatientID"] = ""
            context["Patient"] = {"Name": "TemporarySession"}
        else:
            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.session["patient_deidentified_id"])
            if Authority["Level"] == 0:
                return redirect("patients")
            
            context = dict()
            context["User"] = database.extractUserInfo(request.user)
            context["PageView"] = {"ExpandProcessed": True}
            context["SessionOverview"] = database.viewSession(request.user, request.session["patient_deidentified_id"], session_id=request.session["SessionJsonId"], authority=Authority)
            request.session["SessionJsonId"] = None
            request.session.modified = True
            context["PatientID"] = request.session["patient_deidentified_id"]
            context["Patient"] = database.extractPatientInfo(request.user, context["PatientID"])
        return render(request, "report_sessionreport.html", context=context)

class PatientInformationUpdate(RestViews.APIView):
    parser_classes = [RestParsers.MultiPartParser, RestParsers.FormParser]

    def post(self, request):
        if not request.user.is_authenticated:
            return Response(status=404)

        if "createNewPatientInfo" in request.data:
            if "StudyID" in request.data and "StudyName" in request.data and "Diagnosis" in request.data:
                if request.data["StudyName"] == "" or request.data["StudyID"] == "":
                    return Response(status=400)
                data = dict()
                patient = models.Patient(first_name=request.data["StudyID"], last_name=request.data["StudyName"], diagnosis=request.data["Diagnosis"], institute=request.user.email)

                if "saveDeviceID" in request.data and not request.user.is_clinician:
                    device = models.PerceptDevice(patient_deidentified_id=patient.deidentified_id, serial_number=request.data["saveDeviceID"], device_name=request.data["newDeviceName"], device_location=request.data["newDeviceLocation"])
                    device.device_eol_date = datetime.datetime.fromtimestamp(0, tz=pytz.utc)
                    device.authority_level = "Research"
                    device.authority_user = request.user.email
                    device.save()
                    patient.addDevice(str(device.deidentified_id))
                    request.session["patient_deidentified_id"] = str(patient.deidentified_id)
                    request.session.modified = True
                    data["deviceID"] = str(device.deidentified_id)

                patient.save()
                models.ResearchAuthorizedAccess(researcher_id=request.user.uniqueUserID, authorized_patient_id=patient.deidentified_id, can_edit=True).save()
                data["newPatient"] = database.extractPatientTableRow(request.user, patient)
                return Response(status=200, data=data)
            else:
                return Response(status=404)

        elif "updatePatientInfo" in request.data:
            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.data["updatePatientInfo"])
            if Authority["Level"] == 0:
                return Response(status=404)

            patient = models.Patient.objects.get(deidentified_id=request.data["updatePatientInfo"])
            if "saveDeviceID" in request.data and not request.user.is_clinician:
                device = models.PerceptDevice(patient_deidentified_id=patient.deidentified_id, serial_number=request.data["saveDeviceID"], device_name=request.data["newDeviceName"], device_location=request.data["newDeviceLocation"])
                device.device_eol_date = datetime.datetime.fromtimestamp(0, tz=pytz.utc)
                device.authority_level = "Research"
                device.authority_user = request.user.email
                device.save()
                patient.addDevice(str(device.deidentified_id))
                return Response(status=200)

            elif "deletePatientID" in request.data and Authority["Level"] == 1:
                deidentification = database.extractPatientInfo(request.user, request.data["updatePatientInfo"])
                DeviceIDs = [deidentification["Devices"][i]["ID"] for i in range(len(deidentification["Devices"]))]

                for device in DeviceIDs:
                    device = models.PerceptDevice.objects.filter(deidentified_id=device).first()
                    database.deleteDevice(request.user, request.data["updatePatientInfo"], device.deidentified_id)
                    patient.removeDevice(device)
                    device.delete()

                patient.delete()
                return Response(status=200)

            elif "deleteDeviceID" in request.data and Authority["Level"] == 1:
                deidentification = database.extractPatientInfo(request.user, request.data["updatePatientInfo"])
                DeviceIDs = [deidentification["Devices"][i]["ID"] for i in range(len(deidentification["Devices"]))]
                if not request.data["deleteDeviceID"] in DeviceIDs:
                    return Response(status=404)

                if not models.PerceptDevice.objects.filter(deidentified_id=request.data["deleteDeviceID"]).exists():
                    return Response(status=404)

                device = models.PerceptDevice.objects.filter(deidentified_id=request.data["deleteDeviceID"]).first()
                database.deleteDevice(request.user, request.data["deleteDeviceID"], device.deidentified_id)
                patient.removeDevice(str(device.deidentified_id))
                device.delete()
                return Response(status=200)

            elif "updateDeviceID" in request.data:
                if request.data["updateDeviceID"] in patient.device_deidentified_id:
                    device = models.PerceptDevice.objects.get(deidentified_id=request.data["updateDeviceID"])
                    device.device_name = request.data["newDeviceName"]
                    device.save()
                    return Response(status=200)

            elif "FirstName" in request.data:
                patient.first_name = request.data["FirstName"]
                patient.last_name = request.data["LastName"]
                patient.diagnosis = request.data["Diagnosis"]
                patient.medical_record_number = request.data["MRN"]
                patient.save()
                return Response(status=200)

        return Response(status=400)

class TherapyHistoryView(RestViews.APIView):
    parser_classes = [RestParsers.MultiPartParser, RestParsers.FormParser]

    def get(self, request):
        if not request.user.is_authenticated:
            return redirect("signin")

        if not "patient_deidentified_id" in request.session:
            return redirect("patients")

        if request.session["patient_deidentified_id"] == "":
            return redirect("patients")

        Authority = {}
        Authority["Level"] = database.verifyAccess(request.user, request.session["patient_deidentified_id"])
        if Authority["Level"] == 0:
            return redirect("patients")

        request.session["ProcessingSettings"] = database.retrieveProcessingSettings(request.session["ProcessingSettings"])
        request.session.modified = True

        context = dict()
        context["User"] = database.extractUserInfo(request.user)
        context["PageView"] = {"ExpandProcessed": True}

        context["Patient"] = database.extractPatientInfo(request.user, request.session["patient_deidentified_id"])
        context["PatientID"] = request.session["patient_deidentified_id"]
        return render(request, "report_therapyhistory.html", context=context)

    def post(self, request):
        if not request.user.is_authenticated:
            return Response(status=404)

        if "requestData" in request.data:
            data = dict()

            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.data["requestData"])
            if Authority["Level"] == 0:
                return Response(status=403, data=data)

            if Authority["Level"] == 1:
                Authority["Permission"] = database.verifyPermission(request.user, request.data["requestData"], Authority, "TherapyHistory")
                data["TherapyChangeLogs"] = database.queryTherapyHistory(request.user, request.data["requestData"], Authority)
                TherapyConfigurations = database.queryTherapyConfigurations(request.user, request.data["requestData"], Authority)
                data["TherapyConfigurations"] = database.processTherapyDetails(TherapyConfigurations, TherapyChangeLog=data["TherapyChangeLogs"])
            elif Authority["Level"] == 2:
                PatientInfo = database.extractAccess(request.user, request.session["patient_deidentified_id"])
                Authority["Permission"] = database.verifyPermission(request.user, PatientInfo.authorized_patient_id, Authority, "TherapyHistory")
                data["TherapyChangeLogs"] = database.queryTherapyHistory(request.user, PatientInfo.authorized_patient_id, Authority)
                TherapyConfigurations = database.queryTherapyConfigurations(request.user, PatientInfo.authorized_patient_id, Authority)
                data["TherapyConfigurations"] = database.processTherapyDetails(TherapyConfigurations, TherapyChangeLog=data["TherapyChangeLogs"])

            return Response(status=200, data=data)

class ResolveTherapyHistoryConflicts(RestViews.APIView):
    parser_classes = [RestParsers.MultiPartParser, RestParsers.FormParser]

    def get(self, request):
        if not request.user.is_authenticated:
            return redirect("signin")

        if not "patient_deidentified_id" in request.session:
            return redirect("patients")

        if request.session["patient_deidentified_id"] == "":
            return redirect("patients")

        request.session["ProcessingSettings"] = database.retrieveProcessingSettings(request.session["ProcessingSettings"])
        request.session.modified = True

        Authority = {}
        Authority["Level"] = database.verifyAccess(request.user, request.session["patient_deidentified_id"])
        if Authority["Level"] == 0:
            return redirect("patients")

        PatientID = request.session["patient_deidentified_id"]
        if Authority["Level"] == 1:
            Authority["Permission"] = database.verifyPermission(request.user, PatientID, Authority, "TherapyHistory")
        elif Authority["Level"] == 2:
            PatientInfo = database.extractAccess(request.user, PatientID)
            Authority["Permission"] = database.verifyPermission(request.user, PatientInfo.authorized_patient_id, Authority, "TherapyHistory")
            PatientID = PatientInfo.authorized_patient_id

        context = dict()
        context["User"] = database.extractUserInfo(request.user)
        context["PageView"] = {"ExpandProcessed": True}

        context["Patient"] = database.extractPatientInfo(request.user, request.session["patient_deidentified_id"])
        context["PatientID"] = request.session["patient_deidentified_id"]
        TherapyConfigurations = database.queryTherapyConfigurations(request.user, PatientID, Authority, therapy_type="")
        context["TherapyConfigurations"] = database.processTherapyDetails(TherapyConfigurations, resolveConflicts=False)

        return render(request, "report_therapyhistory_resolveconflicts.html", context=context)

    def post(self, request):
        if not request.user.is_authenticated:
            return Response(status=404)

        if "removeTherapyLog" in request.data:
            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.session["patient_deidentified_id"])
            if models.TherapyHistory.objects.filter(history_log_id=request.data["removeTherapyLog"]).exists() and Authority["Level"] > 0:
                log = models.TherapyHistory.objects.filter(history_log_id=request.data["removeTherapyLog"]).first()
                if request.user.is_admin:
                    if models.PerceptDevice.objects.filter(deidentified_id=log.device_deidentified_id, authority_level="Clinic", authority_user=request.user.institute).exists():
                        log.delete()
                        return Response(status=200)

                elif Authority["Level"] == 1:
                    deidentification = database.extractPatientInfo(request.user, request.session["patient_deidentified_id"])
                    DeviceIDs = [deidentification["Devices"][i]["ID"] for i in range(len(deidentification["Devices"]))]
                    if str(log.device_deidentified_id) in DeviceIDs:
                        log.delete()
                        return Response(status=200)

        return Response(status=400)

class BrainSenseSurveyView(RestViews.APIView):
    parser_classes = [RestParsers.MultiPartParser, RestParsers.FormParser]

    def get(self, request):
        if not request.user.is_authenticated:
            return redirect("signin")

        if not "patient_deidentified_id" in request.session:
            return redirect("patients")

        if request.session["patient_deidentified_id"] == "":
            return redirect("patients")

        request.session["ProcessingSettings"] = database.retrieveProcessingSettings(request.session["ProcessingSettings"])
        request.session.modified = True

        Authority = {}
        Authority["Level"] = database.verifyAccess(request.user, request.session["patient_deidentified_id"])
        if Authority["Level"] == 0:
            return redirect("patients")

        context = dict()
        context["User"] = database.extractUserInfo(request.user)
        context["PageView"] = {"ExpandProcessed": True}

        context["Patient"] = database.extractPatientInfo(request.user, request.session["patient_deidentified_id"])
        context["PatientID"] = request.session["patient_deidentified_id"]
        return render(request, "report_baselinesurvey.html", context=context)

    def post(self, request):
        if not request.user.is_authenticated:
            return redirect("signin")

        if "requestData" in request.data:
            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.data["requestData"])
            if Authority["Level"] == 0:
                return Response(status=403)

            if Authority["Level"] == 1:
                Authority["Permission"] = database.verifyPermission(request.user, request.data["requestData"], Authority, "BrainSenseSurvey")
                data = database.querySurveyResults(request.user, request.data["requestData"], Authority)
                return Response(status=200, data=data)
            elif Authority["Level"] == 2:
                PatientInfo = database.extractAccess(request.user, request.data["requestData"])
                Authority["Permission"] = database.verifyPermission(request.user, PatientInfo.authorized_patient_id, Authority, "BrainSenseSurvey")
                data = database.querySurveyResults(request.user, PatientInfo.authorized_patient_id, Authority)
                return Response(status=200, data=data)


        return Response(status=404)

class RealtimeStreamView(RestViews.APIView):
    parser_classes = [RestParsers.MultiPartParser, RestParsers.FormParser]

    def get(self, request):
        if not request.user.is_authenticated:
            return redirect("signin")

        if not "patient_deidentified_id" in request.session:
            return redirect("patients")

        if request.session["patient_deidentified_id"] == "":
            return redirect("patients")

        request.session["ProcessingSettings"] = database.retrieveProcessingSettings(request.session["ProcessingSettings"])
        request.session.modified = True

        Authority = {}
        Authority["Level"] = database.verifyAccess(request.user, request.session["patient_deidentified_id"])
        if Authority["Level"] == 0:
            return redirect("patients")

        context = dict()
        context["User"] = database.extractUserInfo(request.user)
        context["PageView"] = {"ExpandProcessed": True}

        context["Patient"] = database.extractPatientInfo(request.user, request.session["patient_deidentified_id"])
        context["PatientID"] = request.session["patient_deidentified_id"]
        context["Config"] = request.session["ProcessingSettings"]["RealtimeStream"]
        return render(request, "report_realtimestreamlist.html", context=context)

    def post(self, request):
        if not request.user.is_authenticated:
            return redirect("signin")

        if "requestOverview" in request.data:
            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.data["requestOverview"])

            if Authority["Level"] == 1:
                Authority["Permission"] = database.verifyPermission(request.user, request.data["requestOverview"], Authority, "BrainSenseStream")
                data = database.queryRealtimeStreamOverview(request.user, request.data["requestOverview"], Authority)
                return Response(status=200, data=data)
            elif Authority["Level"] == 2:
                PatientInfo = database.extractAccess(request.user, request.data["requestOverview"])
                Authority["Permission"] = database.verifyPermission(request.user, PatientInfo.authorized_patient_id, Authority, "BrainSenseStream")
                data = database.queryRealtimeStreamOverview(request.user, PatientInfo.authorized_patient_id, Authority)
                return Response(status=200, data=data)

        if "updateRecordingContactType" in request.data:
            if request.user.is_admin or request.user.is_clinician:
                if not models.PerceptDevice.objects.filter(deidentified_id=request.data["requestData"], authority_level="Clinic", authority_user=request.user.institute).exists():
                    return Response(status=404)
            else:
                if not models.PerceptDevice.objects.filter(deidentified_id=request.data["requestData"], authority_level="Research", authority_user=request.user.email).exists():
                    return Response(status=404)

            recording = models.BrainSenseRecording.objects.filter(device_deidentified_id=request.data["requestData"], recording_date=datetime.datetime.fromtimestamp(int(request.data["requestTimestamp"]),tz=pytz.utc), recording_type="BrainSenseStream").first()
            for i in range(len(recording.recording_info["Channel"])):
                if recording.recording_info["Channel"][i].find(request.data["channelID"]) >= 0:
                    recording.recording_info["ContactType"][i] = request.data["updateRecordingContactType"]
            recording.save()
            return Response(status=200)

        if "updateCardiacFilter" in request.data and "requestData" in request.data and "requestTimestamp" in request.data:
            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.session["patient_deidentified_id"])
            if Authority["Level"] == 0 or Authority["Level"] == 2:
                return Response(status=403)

            elif Authority["Level"] == 1:
                deidentification = database.extractPatientInfo(request.user, request.session["patient_deidentified_id"])
                DeviceIDs = [deidentification["Devices"][i]["ID"] for i in range(len(deidentification["Devices"]))]
                if not request.data["requestData"] in DeviceIDs:
                    return Response(status=403)
                Authority["Permission"] = database.verifyPermission(request.user, request.session["patient_deidentified_id"], Authority, "BrainSenseStream")

            else:
                return Response(status=403)

            BrainSenseData, _ = database.queryRealtimeStreamData(request.user, request.data["requestData"], int(request.data["requestTimestamp"]), Authority, refresh=True, cardiacFilter=request.data["updateCardiacFilter"] == "true")
            if BrainSenseData == None:
                return Response(status=400)
            data = database.processRealtimeStreamRenderingData(BrainSenseData, request.session["ProcessingSettings"]["RealtimeStream"])

            return Response(status=200, data=data)

        if "updateConfigurations" in request.data and "device" in request.data and "timestamp" in request.data:
            request.session["ProcessingSettings"] = database.retrieveProcessingSettings(request.session["ProcessingSettings"])
            for key in request.data.keys():
                if key in request.session["ProcessingSettings"]["RealtimeStream"].keys():
                    if request.data[key] in request.session["ProcessingSettings"]["RealtimeStream"][key]["options"]:
                        request.session["ProcessingSettings"]["RealtimeStream"][key]["value"] = request.data[key]
            request.session.modified = True

            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.session["patient_deidentified_id"])
            if Authority["Level"] == 0 or Authority["Level"] == 2:
                return Response(status=403)

            elif Authority["Level"] == 1:
                deidentification = database.extractPatientInfo(request.user, request.session["patient_deidentified_id"])
                DeviceIDs = [deidentification["Devices"][i]["ID"] for i in range(len(deidentification["Devices"]))]
                if not request.data["device"] in DeviceIDs:
                    return Response(status=403)

            BrainSenseData, _ = database.queryRealtimeStreamData(request.user, request.data["device"], int(request.data["timestamp"]), Authority, refresh=False)
            if BrainSenseData == None:
                return Response(status=400)
            data = database.processRealtimeStreamRenderingData(BrainSenseData, request.session["ProcessingSettings"]["RealtimeStream"])
            return Response(status=200, data=data)

        if "requestData" in request.data and "requestTimestamp" in request.data and not "updateFigure" in request.data:
            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.session["patient_deidentified_id"])
            if Authority["Level"] == 0:
                return Response(status=403)

            if Authority["Level"] == 1:
                Authority["Permission"] = database.verifyPermission(request.user, request.data["requestData"], Authority, "BrainSenseStream")
                BrainSenseData, _ = database.queryRealtimeStreamData(request.user, request.data["requestData"], int(request.data["requestTimestamp"]), Authority, refresh=False)
                if BrainSenseData == None:
                    return Response(status=400)
                data = database.processRealtimeStreamRenderingData(BrainSenseData, request.session["ProcessingSettings"]["RealtimeStream"])
                return Response(status=200, data=data)

            elif Authority["Level"] == 2:
                PatientInfo = database.extractAccess(request.user, request.session["patient_deidentified_id"])
                deidentification = database.extractPatientInfo(request.user, PatientInfo.authorized_patient_id)
                DeviceIDs = [deidentification["Devices"][i]["ID"] for i in range(len(deidentification["Devices"]))]
                if not request.data["requestData"] in DeviceIDs:
                    return Response(status=403)

                Authority["Permission"] = database.verifyPermission(request.user, PatientInfo.authorized_patient_id, Authority, "BrainSenseStream")
                BrainSenseData, _ = database.queryRealtimeStreamData(request.user, request.data["requestData"], int(request.data["requestTimestamp"]), Authority, refresh=False)
                if BrainSenseData == None:
                    return Response(status=400)
                data = database.processRealtimeStreamRenderingData(BrainSenseData, request.session["ProcessingSettings"]["RealtimeStream"])
                return Response(status=200, data=data)

        if "requestData" in request.data and "requestTimestamp" in request.data and "updateFigure" in request.data:
            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.session["patient_deidentified_id"])
            if Authority["Level"] == 0:
                return Response(status=403)

            if Authority["Level"] == 1:
                Authority["Permission"] = database.verifyPermission(request.user, request.session["patient_deidentified_id"], Authority, "BrainSenseStream")
            elif Authority["Level"] == 2:
                PatientInfo = database.extractAccess(request.user, request.session["patient_deidentified_id"])
                deidentification = database.extractPatientInfo(request.user, PatientInfo.authorized_patient_id)
                DeviceIDs = [deidentification["Devices"][i]["ID"] for i in range(len(deidentification["Devices"]))]
                if not request.data["requestData"] in DeviceIDs:
                    return Response(status=403)
                Authority["Permission"] = database.verifyPermission(request.user, PatientInfo.authorized_patient_id, Authority, "BrainSenseStream")

            if request.data["updateFigure"] == "Update Stimulation PSD Reference":
                BrainSenseData, _ = database.queryRealtimeStreamData(request.user, request.data["requestData"], int(request.data["requestTimestamp"]), Authority)
                if BrainSenseData == None:
                    return Response(status=400)

                BrainSenseData["Stimulation"] = database.processRealtimeStreamStimulationAmplitude(BrainSenseData)
                if request.session["ProcessingSettings"]["RealtimeStream"]["PSDMethod"]["value"] == "Time-Frequency Analysis":
                    StimPSD = database.processRealtimeStreamStimulationPSD(BrainSenseData, request.data["ChannelID"], method=request.session["ProcessingSettings"]["RealtimeStream"]["SpectrogramMethod"]["value"], stim_label=request.data["StimReference"])
                else:
                    StimPSD = database.processRealtimeStreamStimulationPSD(BrainSenseData, request.data["ChannelID"], method=request.session["ProcessingSettings"]["RealtimeStream"]["PSDMethod"]["value"], stim_label=request.data["StimReference"])

                return Response(status=200, data=StimPSD)

            if request.data["updateFigure"] == "Update Stimulation Boxplot":
                BrainSenseData, _ = database.queryRealtimeStreamData(request.user, request.data["requestData"], int(request.data["requestTimestamp"]), Authority)
                if BrainSenseData == None:
                    return Response(status=400)

                data = dict()
                BrainSenseData["Stimulation"] = database.processRealtimeStreamStimulationAmplitude(BrainSenseData)
                if request.session["ProcessingSettings"]["RealtimeStream"]["PSDMethod"]["value"] == "Time-Frequency Analysis":
                    data["StimPSD"] = database.processRealtimeStreamStimulationPSD(BrainSenseData, request.data["ChannelID"], method=request.session["ProcessingSettings"]["RealtimeStream"]["SpectrogramMethod"]["value"], stim_label=request.data["StimReference"], centerFrequency=float(request.data["CenterFrequency"]))
                else:
                    data["StimPSD"] = database.processRealtimeStreamStimulationPSD(BrainSenseData, request.data["ChannelID"], method=request.session["ProcessingSettings"]["RealtimeStream"]["PSDMethod"]["value"], stim_label=request.data["StimReference"], centerFrequency=float(request.data["CenterFrequency"]))
                return Response(status=200, data=data)

        return Response(status=404)

class IndefiniteStreamView(RestViews.APIView):
    parser_classes = [RestParsers.MultiPartParser, RestParsers.FormParser]

    def get(self, request):
        if not request.user.is_authenticated:
            return redirect("signin")

        if not "patient_deidentified_id" in request.session:
            return redirect("patients")

        if request.session["patient_deidentified_id"] == "":
            return redirect("patients")

        request.session["ProcessingSettings"] = database.retrieveProcessingSettings(request.session["ProcessingSettings"])
        request.session.modified = True

        Authority = {}
        Authority["Level"] = database.verifyAccess(request.user, request.session["patient_deidentified_id"])
        if Authority["Level"] == 0:
            return redirect("patients")

        context = dict()
        context["User"] = database.extractUserInfo(request.user)
        context["PageView"] = {"ExpandProcessed": True}

        context["Patient"] = database.extractPatientInfo(request.user, request.session["patient_deidentified_id"])
        context["PatientID"] = request.session["patient_deidentified_id"]
        #context["Config"] = request.session["ProcessingSettings"]
        return render(request, "report_indefinitestreams.html", context=context)

    def post(self, request):
        if not request.user.is_authenticated:
            return Response(status=404)

        if "requestOverview" in request.data:
            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.data["requestOverview"])

            if Authority["Level"] == 1:
                Authority["Permission"] = database.verifyPermission(request.user, request.data["requestOverview"], Authority, "IndefiniteStream")
                data = database.queryMontageDataOverview(request.user, request.data["requestOverview"], Authority)
                return Response(status=200, data=data)
            elif Authority["Level"] == 2:
                PatientInfo = database.extractAccess(request.user, request.data["requestOverview"])
                Authority["Permission"] = database.verifyPermission(request.user, PatientInfo.authorized_patient_id, Authority, "IndefiniteStream")
                data = database.queryMontageDataOverview(request.user, PatientInfo.authorized_patient_id, Authority)
                return Response(status=200, data=data)

            return Response(status=404)

        if "requestData" in request.data and "requestDevice" in request.data:
            timestamps = request.data["requestData"].split(",")
            timestamps = [int(timestamp) for timestamp in timestamps]
            devices = request.data["requestDevice"].split(",")

            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.session["patient_deidentified_id"])
            if Authority["Level"] == 0:
                return Response(status=404)

            PatientID = request.session["patient_deidentified_id"]
            if Authority["Level"] == 2:
                PatientInfo = database.extractAccess(request.user, request.session["patient_deidentified_id"])
                deidentification = database.extractPatientInfo(request.user, PatientInfo.authorized_patient_id)
                DeviceIDs = [deidentification["Devices"][i]["ID"] for i in range(len(deidentification["Devices"]))]
                for device in devices:
                    if not device in DeviceIDs:
                        return Response(status=403)
                PatientID = PatientInfo.authorized_patient_id

            Authority["Permission"] = database.verifyPermission(request.user, PatientID, Authority, "IndefiniteStream")
            data = database.queryMontageData(request.user, devices, timestamps, Authority)
            return Response(status=200, data=data)

        return Response(status=404)

class ChronicLFPView(RestViews.APIView):
    parser_classes = [RestParsers.MultiPartParser, RestParsers.FormParser]

    def get(self, request):
        if not request.user.is_authenticated:
            return redirect("signin")

        if not "patient_deidentified_id" in request.session:
            return redirect("patients")

        if request.session["patient_deidentified_id"] == "":
            return redirect("patients")

        request.session["ProcessingSettings"] = database.retrieveProcessingSettings(request.session["ProcessingSettings"])
        request.session.modified = True

        Authority = {}
        Authority["Level"] = database.verifyAccess(request.user, request.session["patient_deidentified_id"])
        if Authority["Level"] == 0:
            return redirect("patients")

        context = dict()
        context["User"] = database.extractUserInfo(request.user)
        context["PageView"] = {"ExpandProcessed": True}

        context["Patient"] = database.extractPatientInfo(request.user, request.session["patient_deidentified_id"])
        context["PatientID"] = request.session["patient_deidentified_id"]
        #context["Config"] = request.session["ProcessingSettings"]
        return render(request, "report_chronicLFP.html", context=context)

    def post(self, request):
        if not request.user.is_authenticated:
            return Response(status=404)

        if "requestData" in request.data and "timezoneOffset" in request.data:
            data = dict()

            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.data["requestData"])
            if Authority["Level"] == 0:
                return Response(status=404)

            elif Authority["Level"] == 1:
                Authority["Permission"] = database.verifyPermission(request.user, request.data["requestData"], Authority, "ChronicLFPs")
                PatientID = request.data["requestData"]

            elif Authority["Level"] == 2:
                PatientInfo = database.extractAccess(request.user, request.data["requestData"])
                deidentification = database.extractPatientInfo(request.user, PatientInfo.authorized_patient_id)
                Authority["Permission"] = database.verifyPermission(request.user, PatientInfo.authorized_patient_id, Authority, "ChronicLFPs")
                PatientID = PatientInfo.authorized_patient_id

            data = dict()
            TherapyHistory = database.queryTherapyHistory(request.user, PatientID, Authority)
            data["ChronicData"] = database.queryChronicLFPs(request.user, PatientID, TherapyHistory, Authority)
            data["EventPSDs"] = database.queryPatientEventPSDs(request.user, PatientID, TherapyHistory, Authority)

            Events = list()
            EventColor = dict()
            for i in range(len(data["ChronicData"])):
                if "EventName" in data["ChronicData"][i].keys():
                    for k in range(len(data["ChronicData"][i]["EventName"])):
                        if not type(data["ChronicData"][i]["EventName"][k]) == list:
                            Events.extend(data["ChronicData"][i]["EventName"][k].tolist())
            for i in range(len(data["EventPSDs"])):
                Events.extend(data["EventPSDs"][i]["EventName"])
            EventNames = uniqueList(Events)

            data["ChronicData"] = database.processChronicLFPs(data["ChronicData"], EventNames, int(request.data["timezoneOffset"]))
            #data["EventPSDs"] = database.processEventPSDs(data["EventPSDs"], EventNames)
            data["EventMarker"] = database.processEventMarkers(data["ChronicData"], EventNames)
            #data["CustomEvent"] = database.queryEventPSDs(request.user, request.data["requestData"], TherapyHistory)
            return Response(status=200, data=data)

        return Response(status=404)

class CustomAnalysisTable(RestViews.APIView):
    parser_classes = [RestParsers.MultiPartParser, RestParsers.FormParser]

    def get(self, request):
        if not request.user.is_authenticated:
            return redirect("signin")

        if not "patient_deidentified_id" in request.session:
            return redirect("patients")

        if request.session["patient_deidentified_id"] == "":
            return redirect("patients")

        Authority = {}
        Authority["Level"] = database.verifyAccess(request.user, request.session["patient_deidentified_id"])
        if Authority["Level"] == 0:
            return redirect("patients")

        context = dict()
        context["User"] = database.extractUserInfo(request.user)
        context["PageView"] = {"ExpandProcessed": True}

        context["Patient"] = database.extractPatientInfo(request.user, request.session["patient_deidentified_id"])
        context["PatientID"] = request.session["patient_deidentified_id"]
        return render(request, "report_pipelineselection.html", context=context)

class MedicationSchedule(RestViews.APIView):
    parser_classes = [RestParsers.MultiPartParser, RestParsers.FormParser]

    def get(self, request):
        if not request.user.is_authenticated:
            return redirect("signin")

        if not "patient_deidentified_id" in request.session:
            return redirect("patients")

        if request.session["patient_deidentified_id"] == "":
            return redirect("patients")

        Authority = {}
        Authority["Level"] = database.verifyAccess(request.user, request.session["patient_deidentified_id"])
        if Authority["Level"] == 0:
            return redirect("patients")

        context = dict()
        context["User"] = database.extractUserInfo(request.user)
        context["PageView"] = {"ExpandProcessed": True}

        context["Patient"] = database.extractPatientInfo(request.user, request.session["patient_deidentified_id"])
        context["PatientID"] = request.session["patient_deidentified_id"]
        return render(request, "pipeline_medicationschedule.html", context=context)

    def post(self, request):
        if not request.user.is_authenticated:
            return redirect("signin")

        if "requestData" in request.data:
            data = dict()

            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.data["requestData"])
            if Authority["Level"] == 0:
                return Response(status=403, data=data)

            if Authority["Level"] == 1:
                Authority["Permission"] = database.verifyPermission(request.user, request.data["requestData"], Authority, "TherapyHistory")
                PatientID = request.data["requestData"]
            elif Authority["Level"] == 2:
                PatientInfo = database.extractAccess(request.user, request.session["patient_deidentified_id"])
                Authority["Permission"] = database.verifyPermission(request.user, PatientInfo.authorized_patient_id, Authority, "TherapyHistory")
                PatientID = PatientInfo.authorized_patient_id

            TherapyHistory = database.queryTherapyHistory(request.user, PatientID, Authority)
            data["ChronicData"] = database.queryChronicLFPs(request.user, PatientID, TherapyHistory, Authority)
            return Response(status=200, data=data)

        return Response(status=404)

class EventPSDAnalysis(RestViews.APIView):
    parser_classes = [RestParsers.MultiPartParser, RestParsers.FormParser]

    def get(self, request):
        if not request.user.is_authenticated:
            return redirect("signin")

        if not "patient_deidentified_id" in request.session:
            return redirect("patients")

        if request.session["patient_deidentified_id"] == "":
            return redirect("patients")

        Authority = {}
        Authority["Level"] = database.verifyAccess(request.user, request.session["patient_deidentified_id"])
        if Authority["Level"] == 0:
            return redirect("patients")

        context = dict()
        context["User"] = database.extractUserInfo(request.user)
        context["PageView"] = {"ExpandProcessed": True}

        context["Patient"] = database.extractPatientInfo(request.user, request.session["patient_deidentified_id"])
        context["PatientID"] = request.session["patient_deidentified_id"]
        return render(request, "pipeline_eventpsd.html", context=context)

    def post(self, request):
        if not request.user.is_authenticated:
            return Response(status=404)

        if "requestData" in request.data:
            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.data["requestData"])
            if Authority["Level"] == 0:
                return Response(status=403, data=data)

            if Authority["Level"] == 1:
                Authority["Permission"] = database.verifyPermission(request.user, request.data["requestData"], Authority, "ChronicLFPs")
                PatientID = request.data["requestData"]
            elif Authority["Level"] == 2:
                PatientInfo = database.extractAccess(request.user, request.session["patient_deidentified_id"])
                Authority["Permission"] = database.verifyPermission(request.user, PatientInfo.authorized_patient_id, Authority, "ChronicLFPs")
                PatientID = PatientInfo.authorized_patient_id

            data = dict()
            data["EventPSDs"] = database.queryPatientEventPSDsByTime(request.user, PatientID, [datetime.datetime.fromisoformat(request.data["startTimestamp"]),datetime.datetime.fromisoformat(request.data["endTimestamp"])], Authority)
            Events = list()
            for i in range(len(data["EventPSDs"])):
                Events.extend(data["EventPSDs"][i]["EventName"])
            EventNames = uniqueList(Events)

            data["EventPSDs"] = database.processEventPSDs(data["EventPSDs"], EventNames)
            data["ChronicData"] = database.queryChronicLFPsByTime(request.user, request.data["requestData"], [datetime.datetime.fromisoformat(request.data["startTimestamp"]),datetime.datetime.fromisoformat(request.data["endTimestamp"])], EventNames, Authority)
            return Response(status=200, data=data)

        return Response(status=404)

class AdaptiveStimulation(RestViews.APIView):
    parser_classes = [RestParsers.MultiPartParser, RestParsers.FormParser]

    def get(self, request):
        if not request.user.is_authenticated:
            return redirect("signin")

        if not "patient_deidentified_id" in request.session:
            return redirect("patients")

        if request.session["patient_deidentified_id"] == "":
            return redirect("patients")

        Authority = {}
        Authority["Level"] = database.verifyAccess(request.user, request.session["patient_deidentified_id"])
        if Authority["Level"] == 0:
            return redirect("patients")

        context = dict()
        context["User"] = database.extractUserInfo(request.user)
        context["PageView"] = {"ExpandProcessed": True}
        context["Patient"] = database.extractPatientInfo(request.user, request.session["patient_deidentified_id"])
        context["PatientID"] = request.session["patient_deidentified_id"]
        return render(request, "pipeline_adaptivestimulation.html", context=context)

    def post(self, request):
        if not request.user.is_authenticated:
            return redirect("signin")

        if "requestData" in request.data and "timezoneOffset" in request.data:
            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.data["requestData"])
            if Authority["Level"] == 0:
                return Response(status=404)

            elif Authority["Level"] == 1:
                Authority["Permission"] = database.verifyPermission(request.user, request.data["requestData"], Authority, "ChronicLFPs")
                PatientID = request.data["requestData"]

            elif Authority["Level"] == 2:
                PatientInfo = database.extractAccess(request.user, request.data["requestData"])
                deidentification = database.extractPatientInfo(request.user, PatientInfo.authorized_patient_id)
                Authority["Permission"] = database.verifyPermission(request.user, PatientInfo.authorized_patient_id, Authority, "ChronicLFPs")
                PatientID = PatientInfo.authorized_patient_id

            data = dict()
            TherapyHistory = database.queryTherapyHistory(request.user, PatientID, Authority)
            data["ChronicData"] = database.queryChronicLFPs(request.user, PatientID, TherapyHistory, Authority)

            Events = list()
            EventColor = dict()
            for i in range(len(data["ChronicData"])):
                if "EventName" in data["ChronicData"][i].keys():
                    for k in range(len(data["ChronicData"][i]["EventName"])):
                        if not type(data["ChronicData"][i]["EventName"][k]) == list:
                            Events.extend(data["ChronicData"][i]["EventName"][k].tolist())
            EventNames = uniqueList(Events)

            data["ChronicData"] = database.processChronicLFPs(data["ChronicData"], EventNames, int(request.data["timezoneOffset"]))
            #data["EventPSDs"] = database.processEventPSDs(data["EventPSDs"], EventNames)
            data["EventMarker"] = database.processEventMarkers(data["ChronicData"], EventNames)
            #data["CustomEvent"] = database.queryEventPSDs(request.user, request.data["requestData"], TherapyHistory)

            return Response(status=200, data=data)

        return Response(status=404)

# Upload JSON File
class SessionUpload(RestViews.APIView):
    parser_classes = [RestParsers.MultiPartParser, RestParsers.FormParser]

    def post(self, request):
        if not request.user.is_authenticated:
            return Response(status=404)

        if not "file" in request.data:
            return Response(status=404)

        rawBytes = request.data["file"].read()
        if request.user.is_clinician:
            result, patient, JSON = database.processPerceptJSON(request.user, request.data["file"].name, rawBytes)
            if "SessionView" in request.data and JSON:
                request.session["SessionOverview"] = database.processSessionFile(JSON)
                request.session["patient_deidentified_id"] = str(JSON["PatientID"])
                request.session.modified = True

            if result == "Success":
                data = dict()
                if not patient == None:
                    data["newPatient"] = database.extractPatientTableRow(request.user, patient)
                return Response(status=200, data=data)
            else:
                print(result)
        else:
            if "SessionView" in request.data:
                _, _, JSON = database.processPerceptJSON(request.user, request.data["file"].name, rawBytes, process=False)
                if JSON:
                    request.session["SessionOverview"] = database.processSessionFile(JSON)
                    request.session["patient_deidentified_id"] = "TemporarySession"
                    request.session.modified = True
                    return Response(status=200, data=dict())
                else:
                    return Response(status=500)

            elif "deviceID" in request.data:
                Authority = {}
                Authority["Level"] = database.verifyAccess(request.user, request.session["patient_deidentified_id"])
                print(Authority)
                if Authority["Level"] == 0 or Authority["Level"] == 2:
                    return Response(status=403)

                if Authority["Level"] == 1:
                    deidentification = database.extractPatientInfo(request.user, request.session["patient_deidentified_id"])
                    DeviceIDs = [deidentification["Devices"][i]["ID"] for i in range(len(deidentification["Devices"]))]
                    if not request.data["deviceID"] in DeviceIDs:
                        return Response(status=403)

                result, patient, _ = database.processPerceptJSON(request.user, request.data["file"].name, rawBytes, request.data["deviceID"])
                if result == "Success":
                    return Response(status=200)
                else:
                    print(result)

        return Response(status=404)

class UpdateSessionInfo(RestViews.APIView):
    parser_classes = [RestParsers.MultiPartParser, RestParsers.FormParser]

    def post(self, request):
        if not request.user.is_authenticated:
            return redirect("signin")

        if "patientID" in request.data:
            if database.verifyAccess(request.user, request.data["patientID"]) == 0:
                return Response(status=400)

            request.session["patient_deidentified_id"] = request.data["patientID"]
            request.session.modified = True
            return Response(status=200)

        if "processingConfiguration" in request.data and "processingGroup" in request.data:
            if not request.data["processingConfiguration"] in request.session["ProcessingSettings"][request.data["processingGroup"]].keys():
                return Response(status=400)

            request.session["ProcessingSettings"][request.data["processingGroup"]][request.data["processingConfiguration"]] = request.data[request.data["processingConfiguration"]]
            request.session.modified = True
            return Response(status=200)

        return Response(status=404)

class RequestAPIEndpoints(RestViews.APIView):
    parser_classes = [RestParsers.MultiPartParser, RestParsers.FormParser]

    def post(self, request):
        if not request.user.is_authenticated:
            return Response(status=400)

        if not "RequestType" in request.data:
            return Response(status=400)

        if request.data["RequestType"] == "queryPatientList":
            data = database.extractPatientList(request.user)
            return Response(status=200, data=data)

        elif request.data["RequestType"] == "requestPatientInfo":
            data = database.extractPatientInfo(request.user, request.data["PatientID"])
            return Response(status=200, data=data)

        elif request.data["RequestType"] == "queryImpedanceMeasurement":
            data = database.queryImpedanceMeasurement(request.user, request.data["PatientID"])
            return Response(status=200, data=data)

        elif request.data["RequestType"] == "queryBrainSenseSurveys":
            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.data["PatientID"])
            if Authority["Level"] == 0:
                return Response(status=403)
            if Authority["Level"] == 1:
                Authority["Permission"] = database.verifyPermission(request.user, request.data["PatientID"], Authority, "BrainSenseSurvey")
                data = database.querySurveyResults(request.user, request.data["PatientID"], Authority)
                return Response(status=200, data=data)
            elif Authority["Level"] == 2:
                PatientInfo = database.extractAccess(request.user, request.data["PatientID"])
                Authority["Permission"] = database.verifyPermission(request.user, PatientInfo.authorized_patient_id, Authority, "BrainSenseSurvey")
                data = database.querySurveyResults(request.user, PatientInfo.authorized_patient_id, Authority)
                return Response(status=200, data=data)
            return Response(status=403)

        elif request.data["RequestType"] == "queryBrainSenseStreams":
            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.data["PatientID"])
            if Authority["Level"] == 1:
                Authority["Permission"] = database.verifyPermission(request.user, request.data["PatientID"], Authority, "BrainSenseStream")
                data = database.queryRealtimeStreamOverview(request.user, request.data["PatientID"], Authority)
                return Response(status=200, data=data)
            elif Authority["Level"] == 2:
                PatientInfo = database.extractAccess(request.user, request.data["PatientID"])
                Authority["Permission"] = database.verifyPermission(request.user, PatientInfo.authorized_patient_id, Authority, "BrainSenseStream")
                data = database.queryRealtimeStreamOverview(request.user, PatientInfo.authorized_patient_id, Authority)
                return Response(status=200, data=data)
            return Response(status=403)

        elif request.data["RequestType"] == "retrieveTherapyConfigurations" and "RequestData" in request.data:
            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.data["RequestData"])
            if Authority["Level"] == 0:
                return Response(status=403, data=data)
            if Authority["Level"] == 1:
                Authority["Permission"] = database.verifyPermission(request.user, request.data["RequestData"], Authority, "TherapyHistory")
                data["TherapyChangeLogs"] = database.queryTherapyHistory(request.user, request.data["RequestData"], Authority)
                TherapyConfigurations = database.queryTherapyConfigurations(request.user, request.data["RequestData"], Authority)
                data["TherapyConfigurations"] = database.processTherapyDetails(TherapyConfigurations, TherapyChangeLog=data["TherapyChangeLogs"])
            elif Authority["Level"] == 2:
                PatientInfo = database.extractAccess(request.user, request.data["RequestData"])
                Authority["Permission"] = database.verifyPermission(request.user, PatientInfo.authorized_patient_id, Authority, "TherapyHistory")
                data["TherapyChangeLogs"] = database.queryTherapyHistory(request.user, PatientInfo.authorized_patient_id, Authority)
                TherapyConfigurations = database.queryTherapyConfigurations(request.user, PatientInfo.authorized_patient_id, Authority)
                data["TherapyConfigurations"] = database.processTherapyDetails(TherapyConfigurations, TherapyChangeLog=data["TherapyChangeLogs"])
            return Response(status=200, data=data)

        elif request.data["RequestType"] == "retrieveChronicLFPs" and "RequestData" in request.data:
            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.data["RequestData"])
            if Authority["Level"] == 0:
                return Response(status=404)
            elif Authority["Level"] == 1:
                Authority["Permission"] = database.verifyPermission(request.user, request.data["RequestData"], Authority, "ChronicLFPs")
                PatientID = request.data["RequestData"]
            elif Authority["Level"] == 2:
                PatientInfo = database.extractAccess(request.user, request.data["RequestData"])
                deidentification = database.extractPatientInfo(request.user, PatientInfo.authorized_patient_id)
                Authority["Permission"] = database.verifyPermission(request.user, PatientInfo.authorized_patient_id, Authority, "ChronicLFPs")
                PatientID = PatientInfo.authorized_patient_id
            TherapyHistory = database.queryTherapyHistory(request.user, PatientID, Authority)
            data = database.queryChronicLFPs(request.user, PatientID, TherapyHistory, Authority)
            return Response(status=200, data=data)

        elif request.data["RequestType"] == "retrieveRealtimeStream" and "RequestData" in request.data:
            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.data["RequestData"])
            if Authority["Level"] == 0:
                return Response(status=403)

            if Authority["Level"] == 1:
                Authority["Permission"] = database.verifyPermission(request.user, request.data["RequestData"], Authority, "BrainSenseStream")
                BrainSenseData, _ = database.queryRealtimeStreamData(request.user, request.data["RequestData"], int(request.data["requestTimestamp"]), Authority, refresh=False)
                if BrainSenseData == None:
                    return Response(status=400)
                data = database.processRealtimeStreamRenderingData(BrainSenseData, request.session["ProcessingSettings"]["RealtimeStream"])
                return Response(status=200, data=data)

            elif Authority["Level"] == 2:
                PatientInfo = database.extractAccess(request.user, request.data["RequestData"])
                deidentification = database.extractPatientInfo(request.user, PatientInfo.authorized_patient_id)
                DeviceIDs = [deidentification["Devices"][i]["ID"] for i in range(len(deidentification["Devices"]))]
                if not request.data["requestData"] in DeviceIDs:
                    return Response(status=403)
                Authority["Permission"] = database.verifyPermission(request.user, PatientInfo.authorized_patient_id, Authority, "BrainSenseStream")
                BrainSenseData, _ = database.queryRealtimeStreamData(request.user, request.data["RequestData"], int(request.data["RequestTimestamp"]), Authority, refresh=False)
                if BrainSenseData == None:
                    return Response(status=400)
                data = database.processRealtimeStreamRenderingData(BrainSenseData, request.session["ProcessingSettings"]["RealtimeStream"])
                return Response(status=200, data=data)

            return Response(status=403)

        elif request.data["RequestType"] == "retrieveIndefiniteStream" and "DeviceID" in request.data and "RequestData" in request.data:
            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.data["RequestData"])
            if Authority["Level"] == 0:
                return Response(status=404)

            PatientID = request.session["patient_deidentified_id"]
            if Authority["Level"] == 2:
                PatientInfo = database.extractAccess(request.user, request.data["RequestData"])
                deidentification = database.extractPatientInfo(request.user, PatientInfo.authorized_patient_id)
                DeviceIDs = [deidentification["Devices"][i]["ID"] for i in range(len(deidentification["Devices"]))]
                if not request.data["DeviceID"] in DeviceIDs:
                    return Response(status=403)
                PatientID = PatientInfo.authorized_patient_id

            Authority["Permission"] = database.verifyPermission(request.user, PatientID, Authority, "IndefiniteStream")
            data = database.queryMontageData(request.user, [request.data["DeviceID"]], [int(request.data["RequestTimestamp"])], Authority)
            return Response(status=200, data=data)

        elif request.data["RequestType"] == "retrieveRealtimeStreamStimulationEpochs" and "RequestData" in request.data:
            Authority = {}
            Authority["Level"] = database.verifyAccess(request.user, request.data["RequestData"])
            if Authority["Level"] == 0:
                return Response(status=403)

            if Authority["Level"] == 1:
                Authority["Permission"] = database.verifyPermission(request.user, request.data["RequestData"], Authority, "BrainSenseStream")
                BrainSenseData, _ = database.queryRealtimeStreamData(request.user, request.data["RequestData"], int(request.data["RequestTimestamp"]), Authority, refresh=False)
                if BrainSenseData == None:
                    return Response(status=400)
                BrainSenseData["Stimulation"] = database.processRealtimeStreamStimulationAmplitude(BrainSenseData)
                StimPSD = database.processRealtimeStreamStimulationPSD(BrainSenseData, request.data["ChannelID"], method="Spectrogram")
                return Response(status=200, data=StimPSD)

            elif Authority["Level"] == 2:
                PatientInfo = database.extractAccess(request.user, request.data["RequestData"])
                deidentification = database.extractPatientInfo(request.user, PatientInfo.authorized_patient_id)
                DeviceIDs = [deidentification["Devices"][i]["ID"] for i in range(len(deidentification["Devices"]))]
                if not request.data["requestData"] in DeviceIDs:
                    return Response(status=403)
                Authority["Permission"] = database.verifyPermission(request.user, PatientInfo.authorized_patient_id, Authority, "BrainSenseStream")
                BrainSenseData, _ = database.queryRealtimeStreamData(request.user, request.data["RequestData"], int(request.data["RequestTimestamp"]), Authority, refresh=False)
                if BrainSenseData == None:
                    return Response(status=400)
                BrainSenseData["Stimulation"] = database.processRealtimeStreamStimulationAmplitude(BrainSenseData)
                StimPSD = database.processRealtimeStreamStimulationPSD(BrainSenseData, request.data["ChannelID"], method="Spectrogram")
                return Response(status=200, data=StimPSD)
                
        return Response(status=404)