from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseNotFound

# Create your views here.
def index(request):
    return render(request, "documentation_index.html")

def overview(request):
    return render(request, "documentation_overview.html")

def installation(request):
    return render(request, "documentation_installation.html")

def api(request):
    return render(request, "documentation_installation.html")
