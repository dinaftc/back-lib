from django.db import models

# Create your models here.
from django.db import models

# Create your models here.
# livres/models.py

from django.db import models
from django.contrib.auth.models import User
from django.conf import settings
from comptes.models import Utilisateur


class Livre(models.Model):
    id_l = models.AutoField(primary_key=True)
    isbn = models.CharField(max_length=13, unique=True)
    titre = models.CharField(max_length=255)
    auteur = models.CharField(max_length=255)
    image_url_small = models.URLField()
    image_url_medium = models.URLField()
    image_url_large = models.URLField()
    categorie_nom = models.CharField(max_length=255)
    pdf = models.FileField(upload_to='livres_pdfs/', blank=True, null=True)
    
class Evaluation(models.Model):
    id_e = models.AutoField(primary_key=True)
    utilisateur = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    livre = models.ForeignKey(Livre, on_delete=models.CASCADE)
    note = models.IntegerField()


class UtilisateurProfile(models.Model):
    id_p = models.AutoField(primary_key=True)
    utilisateur = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    historique_categories = models.JSONField(null=True, blank=True)
  
