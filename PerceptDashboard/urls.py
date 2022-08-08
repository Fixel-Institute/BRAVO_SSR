from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views, auth

urlpatterns = [
	path('', views.index),
	path('index', views.index, name="index"),
	path('signin', auth.userAuth.as_view(), name="signin"),
	path('signup', auth.userRegister.as_view(), name="signup"),
	path('signout', auth.userSignout, name="signout"),
	path('admin/authorize_access', views.ResearchAccessView.as_view(), name="Research Access"),

	path('patients', views.patientList, name="patients"),
	path('patients/new', views.PatientInformationUpdate.as_view(), name="new patients"),
	path('patientOverview', views.patientOverview, name="patient overview"),
	path('patients/upload', views.SessionUpload.as_view(), name="session upload data"),
	path('patientOverview/update', views.PatientInformationUpdate.as_view(), name="patient information update"),

	path('report/therapyHistory', views.TherapyHistoryView.as_view(), name="Therapy History Views"),
	path('report/therapyHistory/resolveConflicts', views.ResolveTherapyHistoryConflicts.as_view(), name="Therapy History Views"),
	path('report/brainsenseSurvey', views.BrainSenseSurveyView.as_view(), name="BrainSense Survey Views"),
	path('report/indefiniteStreams', views.IndefiniteStreamView.as_view(), name="Indefinite Stream Views"),
	path('report/brainsenseStreams', views.RealtimeStreamView.as_view(), name="BrainSense Stream Views"),
	path('report/chronicLFPs', views.ChronicLFPView.as_view(), name="Chronic LFP Views"),
	path('report/sessionsManagement', views.PatientSessionFiles.as_view(), name="Recording Sessions Management"),
	path('report/sessionReport', views.PatientSessionReport.as_view(), name="Clinical Session Report"),

	path('updateSessionInfo', views.UpdateSessionInfo.as_view(), name="session cookie update"),

	path('APIEndpoints', views.RequestAPIEndpoints.as_view(), name="API Endpoints")
]
