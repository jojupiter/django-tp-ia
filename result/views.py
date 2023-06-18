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
        results = self.detectspam(text)
        serializer1 = ResultSerializer(results[0])
        serializer2 = ResultSerializer(results[1])
        serializer3 = ResultSerializer(results[2])
        return Response([text,serializer1.data,serializer2.data,serializer3.data])
    

    def detectspam(self, text):
        model1 = joblib.load('nom_du_fichier.pkl')   # charger le modele
        X_test= self.preprocess(text)                    # pretraiter le mail
        predictions = model1.predict(X_test)
        print ('predictions', predictions)
        message = Message.objects.create(message_text=text)
        algo = Algo.objects.create(name="FORET")
        result1 = Result.objects.create(
            accuracy=0.5,
            precision=0.8,
            recall=0.3,
            f1=0.2,
            spam=True,
            message=message,  # Remplacez "message_instance" par l'instance appropriée de la classe Message
            algo=algo  # Remplacez "algo_instance" par l'instance appropriée de la classe Algo
        )
        result2 = Result.objects.create(
            accuracy=0.5,
            precision=0.8,
            recall=0.3,
            f1=0.2,
            spam=True,
            message=message,  # Remplacez "message_instance" par l'instance appropriée de la classe Message
            algo=algo  # Remplacez "algo_instance" par l'instance appropriée de la classe Algo
        )
        result3 = Result.objects.create(
            accuracy=0.5,
            precision=0.8,
            recall=0.3,
            f1=0.2,
            spam=True,
            message=message,  # Remplacez "message_instance" par l'instance appropriée de la classe Message
            algo=algo  # Remplacez "algo_instance" par l'instance appropriée de la classe Algo
        )
        results =[result1,result2,result3]
        return results
    
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