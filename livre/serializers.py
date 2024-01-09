# livres/serializers.py

from rest_framework import serializers
from .models import Livre, Evaluation,UtilisateurProfile

class LivreSerializer(serializers.ModelSerializer):
    class Meta:
        model = Livre
        fields = '__all__'

class EvaluationSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = Evaluation
        fields = '__all__'
        
class profileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UtilisateurProfile
        fields = '__all__'        
        

