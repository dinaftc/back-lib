from django.db import models

# Create your models here.
# models.py

from django.contrib.auth.models import AbstractUser,Group ,Permission
from django.db import models

class Utilisateur(AbstractUser):
    # Ajoutez des champs supplémentaires ici
    id_utilisateur = models.AutoField(primary_key=True)

    # Définissez d'autres champs personnalisés si nécessaire
    age = models.IntegerField(null=True, blank=True)
    location = models.CharField(max_length=255, blank=True)

    groups = models.ManyToManyField(Group, related_name='utilisateur_groups')
    user_permissions = models.ManyToManyField(Permission, related_name='utilisateur_permissions')

    # Vous pouvez également ajouter des méthodes ou d'autres personnalisations ici

    def __str__(self):
        return self.username
