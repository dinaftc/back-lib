from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import Livre , Evaluation,UtilisateurProfile
# Register your models here.
admin.site.register(Livre)
admin.site.register(Evaluation)
admin.site.register(UtilisateurProfile)