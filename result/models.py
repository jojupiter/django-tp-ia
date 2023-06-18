from django.db import models

# Create your models here.
from message.models import Message
from algo.models import Algo

class Result(models.Model):
    
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1 = models.FloatField()
    spam = models.BooleanField()
    message = models.ForeignKey(Message, on_delete=models.CASCADE)
    algo = models.ForeignKey(Algo, on_delete=models.CASCADE)

    created_on = models.DateTimeField(auto_now_add=True)
