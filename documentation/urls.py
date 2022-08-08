from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
	path('', views.index),
	path('/overview', views.overview),
	path('/installation', views.installation),
	path('/api', views.api),
]
