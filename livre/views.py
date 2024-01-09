from django.shortcuts import render

# Create your views here.
from django.shortcuts import render

# Create your views here.


from rest_framework import generics
from .models import Livre, Evaluation
from .serializers import LivreSerializer, EvaluationSerializer

from rest_framework import generics
from rest_framework.response import Response
from .models import Livre
from .serializers import LivreSerializer

class RechercheLivreView(generics.ListAPIView):
    serializer_class = LivreSerializer

    def get_queryset(self):
        query = self.request.query_params.get('q', '')
        return Livre.objects.filter(titre__icontains=query)
from rest_framework import status
from .models import Evaluation
from .serializers import EvaluationSerializer
from rest_framework.permissions import IsAuthenticated

class EvaluationCreate(generics.CreateAPIView):
    queryset = Evaluation.objects.all()
    serializer_class = EvaluationSerializer
    permission_classes = [IsAuthenticated]
    
    def perform_create(self, serializer):
        # Permettre aux utilisateurs d'évaluer plusieurs fois
        #utilisateur = self.request.user
        serializer.save()

    def create(self, request, *args, **kwargs):
        
        note = int(request.data.get('note', 0))
        if 0 <= note <= 10:
            return super().create(request, *args, **kwargs)
        else:
            return Response(
                {'error': 'La note doit être entre 0 et 10.'},
                status=status.HTTP_400_BAD_REQUEST
            )
from django.http import HttpResponse  
class TelechargementLivreView(generics.RetrieveAPIView):
    queryset = Livre.objects.all()
    #permission_classes = [IsAuthenticated]

    def retrieve(self, request, *args, **kwargs):
        livre = self.get_object()
        file_path = livre.pdf.path
        response = HttpResponse(open(file_path, 'rb').read())
        response['Content-Type'] = 'application/pdf'
        response['Content-Disposition'] = f'attachment; filename="{livre.titre}.pdf"'
        return response    
    
class ListeLivreParCategorieView(generics.ListAPIView):
    serializer_class = LivreSerializer
    #permission_classes = [IsAuthenticated]

    def get_queryset(self):
        categorie = self.kwargs['categorie']
        return Livre.objects.filter(categorie_nom__iexact=categorie)    
    
    
# views.py
from rest_framework import views
from rest_framework.response import Response
from rest_framework import status
import pickle
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

from rest_framework import views
from rest_framework.response import Response
from rest_framework import status
import pickle
from .models import Livre   # Assurez-vous d'importer votre modèle de livre
import pandas as pd
from mlxtend.frequent_patterns import apriori


from rest_framework import views
from rest_framework.response import Response
from rest_framework import status
import pickle


from mlxtend.frequent_patterns import apriori






from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


from .models import Livre, Evaluation
from django.http import JsonResponse
from django.views import View
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors




from django.db.models import ExpressionWrapper, F
from django.db.models import  Value


from django.db import models


 
from joblib import load





from django.http import JsonResponse
from django.views import View
from django.http import JsonResponse
from django.views import View
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

#knn_model = load(r'C:\Users\DELL\Downloads\modele_knn.joblib')
from django.http import JsonResponse
from django.views import View
from sklearn.cluster import KMeans
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import joblib
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
# Importer les modules nécessaires
import json
import joblib
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from .models import Livre, Evaluation, UtilisateurProfile
from comptes.models import Utilisateur
import numpy as np
import logging

# Charger le modèle KNN à l'intérieur de la vue
#knn_model = joblib.load(r'C:\Users\DELL\Downloads\knn_model.joblib')
#vectorizer=joblib.load( r'C:\Users\DELL\Downloads\vectorizer.joblib')
#svd=joblib.load( r'C:\Users\DELL\Downloads\svd.joblib')

'''



def get_recommendations(request):
    data = json.loads(request.body.decode('utf-8'))
    user_id = request.user.id
    book_title = data.get('titre')

    query_result = Livre.objects.filter(titre__iexact=book_title)
    
    if not query_result.exists():
        return JsonResponse({'error': f'Aucune correspondance trouvée pour le titre du livre : {book_title}'})

    query_book = query_result.first()
    # Combine features
    #Livres = Livre.objects.all()
    Livre_with_features = [ query_book.titre + ' ' +  query_book.auteur + ' ' +  query_book.categorie_nom]
    logging.info(f"Livre_with_features : {Livre_with_features}")



# Text vectorization using CountVectorizer
   
    Xx = vectorizer.transform( Livre_with_features)

# Dimensionality reduction using TruncatedSVD

    X = svd.transform(Xx)

    # Utiliser les valeurs spécifiques du modèle pour construire le vecteur de caractéristiques
    

    distances, indices = knn_model.kneighbors(X)

    neighbor_indices = indices[0]
    neighbors = Livre.objects.filter(id__in=neighbor_indices + 1).values('isbn', 'titre', 'auteur', 'categorie_nom')

    
    recommendations = list(neighbors)

    return JsonResponse({'recommendations': recommendations, "Livre_with_features":Livre_with_features})


'''



# Combine features

Livres = Livre.objects.all()
Livres_with_features = []

for livre in Livres:
    combined_features = livre.titre + ' ' + livre.auteur + ' ' + livre.categorie_nom
    Livres_with_features.append(combined_features)

# Text vectorization using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([livre for livre in Livres_with_features])

# Dimensionality reduction using TruncatedSVD
svd = TruncatedSVD(n_components=13)
X = svd.fit_transform(X)

# Initialize Nearest Neighbors model
k_neighbors = 4
knn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
knn_model.fit(X)

@csrf_exempt
# Function to find k-nearest neighbors for a given book title
def find_neighbors(request, exclude_rated=True):
    data = json.loads(request.body.decode('utf-8'))
    user_id = request.user
    book_title = data.get('titre')
    query_result = Livre.objects.filter(titre__iexact=book_title)

    if not query_result.exists():
        print(f"No match found for book title: {book_title}")
        return pd.DataFrame()  # Return an empty DataFrame or handle it as needed

    query_book_index = query_result.first().id_l

    # Use NumPy array indexing to access the feature vector
    query_book_features = X[query_book_index].reshape(1, -1)
    distances, indices = knn_model.kneighbors(query_book_features)

    # Convertir les indices en objets Livre
    neighbors = Livre.objects.filter(id_l__in=indices[0]).values('isbn', 'titre', 'auteur', 'categorie_nom')
   

    # Filtrer les livres déjà évalués par l'utilisateur
    if exclude_rated:
        user_rated_books = Evaluation.objects.filter(utilisateur_id=user_id).values_list('livre__isbn', flat=True)
        
        neighbors = [livre for livre in neighbors if livre['isbn'] not in user_rated_books]

    #return JsonResponse({'neighbors': neighbors, 'user_rated_books':list(user_rated_books)})
    # Ajouter les distances et les indices à la réponse JSON
    response_data = {
        'neighbors': neighbors,
        'user_rated_books': list(user_rated_books),
        'distances': distances.tolist(),  # Convertir les distances en une liste JSON
        'indices': indices.tolist(),      # Convertir les indices en une liste JSON
    }

    return JsonResponse(response_data)



from django.shortcuts import get_object_or_404
from django.contrib.auth.decorators import login_required

@csrf_exempt


def ajouter_evaluation(request, livre_id, note):
    # Obtenez l'utilisateur connecté
    utilisateur = request.user

    # Obtenez le livre en fonction de l'ISBN (supposons que le livre soit identifié par l'ISBN)
    livre = get_object_or_404(Livre,id_l=livre_id)

    # Ajoutez l'évaluation à la base de données
    Evaluation.objects.create(utilisateur=utilisateur, livre=livre, note=note)

    # Mettez à jour le profil de l'utilisateur avec l'historique des catégories
    historique_categories = Evaluation.objects.filter(utilisateur=utilisateur).values_list('livre__categorie_nom', flat=True).distinct()
    utilisateur_profile, created = UtilisateurProfile.objects.get_or_create(utilisateur=utilisateur)
    utilisateur_profile.historique_categories = list(historique_categories)
    utilisateur_profile.save()

    return JsonResponse({'message': 'Evaluation ajoutée avec succès.'})



#knn de users
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
    
    
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

from comptes.models import Utilisateur


# Combine features for users
users = UtilisateurProfile.objects.all()
users_with_features = []

for user_profile in users:
    combined_features = user_profile.utilisateur.location + ' ' + str(user_profile.utilisateur.age) + ' ' + ' '.join(user_profile.historique_categories)

    users_with_features.append(combined_features)

# Text vectorization using TF-IDF
vectorizer_users = TfidfVectorizer()
X_users = vectorizer_users.fit_transform(users_with_features)

# Dimensionality reduction using TruncatedSVD (optional)

# Initialize Nearest Neighbors model for users
k_neighbors_users = 4
knn_model_users = NearestNeighbors(n_neighbors=k_neighbors_users, metric='cosine')
knn_model_users.fit(X_users)


@csrf_exempt
# Function to find k-nearest neighbors for a given user
def find_neighbors_users(request):
    
    user_id = request.user
    try:
        user_profile = UtilisateurProfile.objects.get(utilisateur=user_id)
        user = Utilisateur.objects.get(id_utilisateur=user_id.id_utilisateur)
    except UtilisateurProfile.DoesNotExist:
        return JsonResponse({'error': 'l utilisateur n\' pas d historique.'})
    except Utilisateur.DoesNotExist:
        return JsonResponse({'error': 'L\'utilisateur n\'existe pas.'})
    combined_features_user = user.location + ' ' + str(user.age) + ' ' + ' '.join(user_profile.historique_categories)
    query_user_features = vectorizer_users.transform([combined_features_user])
        
    distances, indices = knn_model_users.kneighbors(query_user_features)

    if len(indices[0]) == 0:
        print("No neighbors found.")
        return JsonResponse({'error': 'No neighbors found.'})

    neighbors = Utilisateur.objects.filter(id_utilisateur__in=indices[0]).values('id_utilisateur','username')
    neighbor_ids = [neighbor['id_utilisateur'] for neighbor in neighbors]
 # Extraire les livres évalués par l'utilisateur
    evaluated_books_user = Evaluation.objects.filter(utilisateur=user_id)
    evaluated_books_user_ids = [evaluation.livre.id_l for evaluation in evaluated_books_user]

    # Extraire les livres évalués par les utilisateurs similaires
    evaluated_books_similar_users = Evaluation.objects.filter(utilisateur__in= neighbor_ids)
    evaluated_books_similar_users_ids = [evaluation.livre.id_l for evaluation in evaluated_books_similar_users]
    # Vérifier si l'utilisateur a évalué des livres
    if not evaluated_books_user.exists():
        return JsonResponse({'error': 'L\'utilisateur n\'a évalué aucun livre.'})
    baskets = []
    for user in neighbor_ids:
        user_evaluated_books = Evaluation.objects.filter(utilisateur=user)
        user_basket = [evaluation.livre.titre for evaluation in user_evaluated_books]
        baskets.append(user_basket)
        
    te_basket = TransactionEncoder()
    te_ary_user = te_basket.fit_transform(baskets)
    df_baskets = pd.DataFrame(te_ary_user, columns=te_basket.columns_)

# Appliquer l'algorithme Apriori
    min_support = 0.01  # Ajustez la valeur selon vos besoins
    frequent_itemsets_user = apriori(df_baskets, min_support=min_support, use_colnames=True)
    # Générer les règles d'association
    rules_user = association_rules(frequent_itemsets_user, metric="confidence", min_threshold=1)
    # ...
# Votre code pour obtenir les règles (rules_user)

# Extraire uniquement les antécédents et les conséquents
    # ...
# Votre code pour obtenir les règles (rules_user)

# Convertir les frozensets en listes pour permettre la sérialisation JSON
    rules_user['antecedents'] = rules_user['antecedents'].apply(list)
    rules_user['consequents'] = rules_user['consequents'].apply(list)

    # Extraire uniquement les antécédents et les conséquents
    antecedents = rules_user['antecedents'].tolist()
    consequents = rules_user['consequents'].tolist()

    # Créer une liste de règles avec antécédents et conséquents
    formatted_rules = [{'antecedents': ant, 'consequents': con} for ant, con in zip(antecedents, consequents)]

    # Retourner la réponse JSON
    return JsonResponse({
        'formatted_rules': formatted_rules,
        'neighbors': list(neighbors),
        'distances': distances.tolist(),
        'indices': indices.tolist(),
        'evaluated_books': {
            'user': evaluated_books_user_ids,
            'similar_users': evaluated_books_similar_users_ids
        }
    }, safe=False)


    #return JsonResponse({'rules_user': rules_user.to_json()}, safe=False)


