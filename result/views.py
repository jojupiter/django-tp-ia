from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from google.cloud import translate_v2 as translate
from result.models import Result
from message.models import Message
from algo.models import Algo
from result.serializers import ResultSerializer
import joblib 
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


class ResultAPIView(APIView):
 
    def post(self, request):
        text = request.data.get('text')
        text= self.translate_text(text)
        text = self.clean_text(text)
        print("text",text)
        results = self.detectspam(text)
        serializer1 = ResultSerializer(results[0])
        serializer2 = ResultSerializer(results[1])
        serializer3 = ResultSerializer(results[2])
        return Response([text,serializer1.data,serializer2.data,serializer3.data])
    

    def detectspam(self, text):
        model1 = joblib.load('modele1.pkl')   # charger le modele
        model2 = joblib.load('modele2.pkl')   # charger le modele
        model3 = joblib.load('modele3.pkl')   # charger le modele
        X_test= self.preprocess(text)                    # pretraiter le mail
        prediction1 = model1.predict(X_test)
        prediction2 = model2.predict(X_test)
        prediction3 = model3.predict(X_test)
        print ('prediction1', prediction1)
        print ('prediction2', prediction2)
        print ('prediction3', prediction3)
        message = Message.objects.create(message_text=text)
        algo1 = Algo.objects.create(name="Regression Logistique")
        algo2 = Algo.objects.create(name="Forets Aleatoire")
        algo3 = Algo.objects.create(name="SVM(Support Vector Machine)")
        result1 = Result.objects.create(
            accuracy=0,
            precision=0,
            recall=0,
            f1=0,
            spam=self.getSpam(prediction1[0]),
            message=message,  # Remplacez "message_instance" par l'instance appropriée de la classe Message
            algo=algo1  # Remplacez "algo_instance" par l'instance appropriée de la classe Algo
        )
        result2 = Result.objects.create(
            accuracy=0,
            precision=0,
            recall=0,
            f1=0,
            spam=self.getSpam(prediction2[0]),
            message=message,  # Remplacez "message_instance" par l'instance appropriée de la classe Message
            algo=algo2  # Remplacez "algo_instance" par l'instance appropriée de la classe Algo
        )
        result3 = Result.objects.create(
            accuracy=0,
            precision=0,
            recall=0,
            f1=0,
            spam=self.getSpam(prediction3[0]),
            message=message,  # Remplacez "message_instance" par l'instance appropriée de la classe Message
            algo=algo3  # Remplacez "algo_instance" par l'instance appropriée de la classe Algo
        )
        results =[result1,result2,result3]
        return results
    
    def preprocess(self, text):
        vectorizer = CountVectorizer()
        # Importer le vocabulaire du vectoriseur
        vocab = joblib.load('vectorizer_vocab.pkl')
        # Charger les paramètres du vectoriseur
        vectorizer_params = joblib.load('vectorizer_params.pkl')
        # Charger le modele Feature Selection
        svd = joblib.load('selection_features_model.pkl')
        # Effectuer la vectorisation 
        vectorizer.set_params(**vectorizer_params)
        vectorizer.vocabulary_=vocab
        X_test = vectorizer.transform([text])
        print("X shape",X_test.shape)
        print("X test",X_test)
        # Feature Selection
        X_test_reduced=svd.transform(X_test)
        return X_test_reduced

    def clean_text(self,text):
        ps = PorterStemmer()
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        text = text.split()
        text = [ps.stem(word) for word in text if not word in stop_words]  # use stop_words instead of stopwords
        text = ' '.join(text)
        return text

    def getSpam(self, prediction):
        if(prediction==0) :
            return False
        else :
            return True

    def translate_text(target: str, text: str) -> dict:
        translate_client = translate.Client()

        if isinstance(text, bytes):
            text = text.decode("utf-8")

        # Text can also be a sequence of strings, in which case this method
        # will return a sequence of results for each text.
        result = translate_client.translate(text, target_language=target)

        print("Text: {}".format(result["input"]))
        print("Translation: {}".format(result["translatedText"]))
        print("Detected source language: {}".format(result["detectedSourceLanguage"]))

        return result