# urls.py

from django.urls import path
from .views import InscriptionUtilisateur,LoginView



urlpatterns = [
    # ...
    path('inscription/', InscriptionUtilisateur.as_view(), name='inscription'),
    #path('api-token-auth/', obtain_auth_token, name='api_token_auth'),
   
    path('login/', LoginView.as_view(), name='hi'),

]
