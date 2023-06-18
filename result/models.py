from django.db import models

# Create your models here.
from message.models import Message
from algo.models import Algo

class Result(models.Model):
    
    metrique1 = models.FloatField()
    metrique2 = models.FloatField()
    metrique3 = models.FloatField()
    metrique4 = models.FloatField()
    metrique5 = models.FloatField()
    metrique6 = models.FloatField()
    metrique7 = models.FloatField()
    metrique8 = models.FloatField()
    spam = models.BooleanField()
    message = models.ForeignKey(Message, on_delete=models.CASCADE)
    algo = models.ForeignKey(Algo, on_delete=models.CASCADE)

    created_on = models.DateTimeField(auto_now_add=True)
