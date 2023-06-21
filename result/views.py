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
        #text= self.translate_text('en_US',text)
        print("text",text)
        results = self.detectspam(text)
        serializer1 = ResultSerializer(results[0])
        serializer2 = ResultSerializer(results[1])
        serializer3 = ResultSerializer(results[2])
        serializer4 = ResultSerializer(results[3])
        serializer5 = ResultSerializer(results[4])
        serializer6 = ResultSerializer(results[5])
        serializer7 = ResultSerializer(results[6])

        return Response([text,serializer1.data,serializer2.data,serializer3.data,serializer4.data,serializer5.data,serializer6.data,serializer7.data])
    

    def detectspam(self, text):
        model1 = joblib.load('modele1.pkl')   # charger le modele
        model2 = joblib.load('modele2.pkl')   # charger le modele
        model3 = joblib.load('modele3.pkl')   # charger le modele
        rf_model = joblib.load('rf_model.pkl')   # charger le modele
        nb_model = joblib.load('nb_model.pkl')   # charger le modele
        logreg_model = joblib.load('logreg_model.pkl')   # charger le modele
        svm_model = joblib.load('svm_model.pkl')   # charger le modele
        # pretraiter le mail
        X_test= self.preprocess(text)
        prediction1 = model1.predict(X_test)
        prediction2 = model2.predict(X_test)
        prediction3 = model3.predict(X_test)
        # pretraiter le mail avec des modeles sans reductions de dimensions
        X_test_no_dim = self.preprocess_no_dim(text)
        prediction4 = rf_model.predict(X_test_no_dim)
        prediction5 = nb_model.predict(X_test_no_dim)
        prediction6 = logreg_model.predict(X_test_no_dim)
        prediction7 = svm_model.predict(X_test_no_dim)

        print ('prediction1', prediction1)
        print ('prediction2', prediction2)
        print ('prediction3', prediction3)
        #tester les mails avec les modeles sans reductions de dimention
        print('prediction4', prediction4)
        print('prediction5', prediction5)
        print('prediction6', prediction6)
        print('prediction7', prediction7)

        message = Message.objects.create(message_text=text)
        algo1 = Algo.objects.create(name="Regression Logistique")
        algo2 = Algo.objects.create(name="Forets Aleatoire")
        algo3 = Algo.objects.create(name="SVM(Support Vector Machine)")
        algo4 = Algo.objects.create(name="Forests Aleatoires")
        algo5 = Algo.objects.create(name="Naifs Bayes")
        algo6 = Algo.objects.create(name="regression logistique")
        algo7 = Algo.objects.create(name="SVM(Support Vector Machine)")
        result1 = Result.objects.create(
            accuracy=0.97757847,
            precision=0.9489051,
            recall= 0.872483,
            f1=0.9090909,
            spam=self.getSpam(prediction1[0]),
            message=message,  # Remplacez "message_instance" par l'instance appropriée de la classe Message
            algo=algo1  # Remplacez "algo_instance" par l'instance appropriée de la classe Algo
        )
        result2 = Result.objects.create(
            accuracy= 0.9578475336,
            precision=1.0,
            recall=0.684563758,
            f1=0.812749003,
            spam=self.getSpam(prediction2[0]),
            message=message,  # Remplacez "message_instance" par l'instance appropriée de la classe Message
            algo=algo2  # Remplacez "algo_instance" par l'instance appropriée de la classe Algo
        )
        result3 = Result.objects.create(
            accuracy=0.977578475,
            precision=0.992063492,
            recall=0.8389261744,
            f1=0.9090909,
            spam=self.getSpam(prediction3[0]),
            message=message,  # Remplacez "message_instance" par l'instance appropriée de la classe Message
            algo=algo3  # Remplacez "algo_instance" par l'instance appropriée de la classe Algo
        )
        result4 = Result.objects.create(
            accuracy=0.977578475,
            precision=0.992063492,
            recall=0.8389261744,
            f1=0.9090909,
            spam=self.getSpam(prediction4[0]),
            message=message,  # Remplacez "message_instance" par l'instance appropriée de la classe Message
            algo=algo4  # Remplacez "algo_instance" par l'instance appropriée de la classe Algo
        )
        result5 = Result.objects.create(
            accuracy=0.977578475,
            precision=0.992063492,
            recall=0.8389261744,
            f1=0.9090909,
            spam=self.getSpam(prediction5[0]),
            message=message,  # Remplacez "message_instance" par l'instance appropriée de la classe Message
            algo=algo5  # Remplacez "algo_instance" par l'instance appropriée de la classe Algo
        )
        result6 = Result.objects.create(
            accuracy=0.977578475,
            precision=0.992063492,
            recall=0.8389261744,
            f1=0.9090909,
            spam=self.getSpam(prediction6[0]),
            message=message,  # Remplacez "message_instance" par l'instance appropriée de la classe Message
            algo=algo6  # Remplacez "algo_instance" par l'instance appropriée de la classe Algo
        )
        result7 = Result.objects.create(
            accuracy=0.977578475,
            precision=0.992063492,
            recall=0.8389261744,
            f1=0.9090909,
            spam=self.getSpam(prediction7[0]),
            message=message,  # Remplacez "message_instance" par l'instance appropriée de la classe Message
            algo=algo7  # Remplacez "algo_instance" par l'instance appropriée de la classe Algo
        )
        results =[result1,result2,result3,result4,result5,result6,result7]
        return results
    
    def preprocess(self, text):
        text = self.clean_text(text)
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

    def translate_text(self,target: str, text: str) -> dict:
        translate_client = translate.Client()

        if isinstance(text, bytes):
            text = text.decode("utf-8")

        # Text can also be a sequence of strings, in which case this method
        # will return a sequence of results for each text.
        result = translate_client.translate(text, target_language=target)

        print("Text: {}".format(result["input"]))
        print("Translation: {}".format(result["translatedText"]))
        print("Detected source language: {}".format(result["detectedSourceLanguage"]))

        return result["translatedText"]

    def preprocess_no_dim(self,text):
        vectorizer = CountVectorizer()
        # Importer le vocabulaire du vectoriseur
        vocab = joblib.load('vectorizer_vocab_no_dim.pkl')
        # Charger les paramètres du vectoriseur
        vectorizer_params = joblib.load('vectorizer_params_no_dim.pkl')
        vectorizer.set_params(**vectorizer_params)
        vectorizer.vocabulary_ = vocab
        X_test = vectorizer.transform([text])
        print("X shape no dim", X_test.shape)
        print("X test no dim", X_test)
        return  X_test