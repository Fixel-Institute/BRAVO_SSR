from django.shortcuts import render, redirect
from django.http import HttpResponse

from django.contrib.auth.backends import BaseBackend
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout, get_user_model

import rest_framework.views as RestViews
import rest_framework.parsers as RestParsers
from rest_framework.response import Response

from . import models

import re
def validateEmail(email):
    return re.fullmatch(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', email)

class userRegister(RestViews.APIView):
    parser_classes = [RestParsers.MultiPartParser, RestParsers.FormParser]

    def get(self, request):
        return render(request, 'signup.html')

    def post(self, request):
        if "Email" in request.POST and "Password" in request.POST and "FirstName" in request.POST and "LastName" in request.POST and "Institute" in request.POST:
            if len(request.POST["Password"]) < 8:
                return Response(status=201, data={"message": "Minimum password length is 8-character."})

            if not validateEmail(request.POST["Email"]):
                return Response(status=201, data={"message": "Incorrect email format"})

            if len(request.POST["FirstName"]) == 0 or len(request.POST["LastName"]) == 0 or len(request.POST["Institute"]) == 0:
                return Response(status=201, data={"message": "Incomplete registration form."})

            User = get_user_model()
            try:
                match = User.objects.get(email=request.POST["Email"])
                return Response(status=201, data={"message": "Email already been used."})

            except User.DoesNotExist:
                user = User.objects.create_user(email=request.POST["Email"], first_name=request.POST["FirstName"], last_name=request.POST["LastName"], institute=request.POST["Institute"], password=request.POST["Password"])
                user.save()
                user = authenticate(request, username=request.POST["Email"], password=request.POST["Password"])
                login(request, user)
                return Response(status=200)

        return Response(status=404)

class userAuth(RestViews.APIView):
    parser_classes = [RestParsers.MultiPartParser, RestParsers.FormParser]

    def get(self, request):
        if request.user.is_authenticated:
            return redirect("index")

        return render(request, 'signin.html')

    def post(self, request):
        if "Email" in request.POST and "Password" in request.POST:
            user = authenticate(request, username=request.POST["Email"], password=request.POST["Password"])
        if user is not None:

            if not models.UserConfigurations.objects.filter(user_id=user.uniqueUserID).exists():
                models.UserConfigurations(user_id=user.uniqueUserID).save()

            login(request, user)
            request.session.set_expiry(3600)
            return redirect("index")
        else:
            return Response(status=201, data={"message": "Incorrect Credentials"})

        return Response(status=400)

def userSignout(request):
	if not request.user.is_authenticated:
		return redirect("index")

	logout(request)

	return redirect("index")
