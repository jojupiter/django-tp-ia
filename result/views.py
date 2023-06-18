from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
 
from result.models import Result
from message.models import Message
from algo.models import Algo
from result.serializers import ResultSerializer
import joblib 
from sklearn.feature_extraction.text import CountVectorizer


class ResultAPIView(APIView):
 
    def post(self, request):
        text = request.data.get('text')
        print("text",text)
        result = self.detectspam(text)
        serializer = ResultSerializer(result)
        return Response(serializer.data)
    

    def detectspam(self, text):
        model1 = joblib.load('nom_du_fichier.pkl')   # charger le modele
        X_test= self.preprocess(text)                    # pretraiter le mail
        predictions = model1.predict(X_test)
        print ('predictions', predictions)
        message = Message.objects.create(message_text=text)
        algo = Algo.objects.create(name="FORET")
        result = Result.objects.create(
            metrique1=0.5,
            metrique2=0.8,
            metrique3=0.3,
            metrique4=0.2,
            metrique5=0.6,
            metrique6=0.9,
            metrique7=0.4,
            metrique8=0.7,
            spam=True,
            message=message,  # Remplacez "message_instance" par l'instance appropriée de la classe Message
            algo=algo  # Remplacez "algo_instance" par l'instance appropriée de la classe Algo
        )
        return result
    
    def preprocess(self, text):
        vectorizer = CountVectorizer()
        # Importer le vocabulaire du vectoriseur
        vocab = joblib.load('vectorizer_vocab.pkl')
        # Charger les paramètres du vectoriseur
        vectorizer_params = joblib.load('vectorizer_params.pkl')
        # Effectuer la vectorisation 
        vectorizer.set_params(**vectorizer_params)
        vectorizer.vocabulary_=vocab
        X_test = vectorizer.transform([text])
        print("X shape",X_test.shape)
        print("X test",X_test)
        return X_test