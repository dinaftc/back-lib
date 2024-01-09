from django.shortcuts import render

# Create your views here.
# views.py

from rest_framework import generics, permissions
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.authtoken.models import Token
from django.contrib.auth import get_user_model
from .serializers import UtilisateurSerializer
from django.http import JsonResponse
Utilisateur = get_user_model()

class InscriptionUtilisateur(generics.CreateAPIView):
    queryset = Utilisateur.objects.all()
    serializer_class = UtilisateurSerializer
    permission_classes = [permissions.AllowAny]
from rest_framework.authtoken.models import Token
from django.shortcuts import get_object_or_404
from rest_framework.authtoken.models import Token
from django.shortcuts import get_object_or_404
from django.contrib.auth import get_user_model
from rest_framework.authtoken.models import Token
from django.shortcuts import get_object_or_404
from django.contrib.auth import get_user_model
from django.db import models

Utilisateur = get_user_model()

from rest_framework.authtoken.models import Token
from django.contrib.auth import get_user_model

Utilisateur = get_user_model()
Token.objects = models.Manager()

from rest_framework.response import Response
from rest_framework.views import APIView

from django.contrib.auth import authenticate, login
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework.views import APIView

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.permissions import AllowAny

from rest_framework import permissions
from rest_framework import views
from rest_framework.response import Response

from . import serializers
class LoginView(views.APIView):
    # This view should be accessible also for unauthenticated users.
    permission_classes = (permissions.AllowAny,)

    def post(self, request, format=None):
        serializer = serializers.LoginSerializer(data=self.request.data,
            context={ 'request': self.request })
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        login(request, user)
        return JsonResponse({'id':user.id_utilisateur})


