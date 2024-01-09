# livres/urls.py

from django.urls import path
from .views import ListeTousLivresView,DetailLivreView,RechercheLivreView,EvaluationCreate, TelechargementLivreView,ListeLivreParCategorieView,ajouter_evaluation
from .views import find_neighbors,find_neighbors_users
urlpatterns = [
     path('recherche/', RechercheLivreView.as_view(), name='recherche-livre'),
    path('par-categorie/<str:categorie>/', ListeLivreParCategorieView.as_view(), name='livres-par-categorie'),
   path('<int:pk>/telecharger/', TelechargementLivreView.as_view(), name='telecharger-livre'),
    path('evaluations/', EvaluationCreate.as_view(), name='evaluation-create'),
    path('aj_eval/<int:livre_id>/<int:note>/',ajouter_evaluation, name='evaluation') ,
   
  path('recommandation/',find_neighbors, name='recommandation'),
  path('recommandation_u/',find_neighbors_users, name='recommandation_user'),
    path('livres/', ListeTousLivresView.as_view(), name='liste_tous_livres'),
   path('livres/<int:id_l>/', DetailLivreView.as_view(), name='livre-detail'),

    
]
   

