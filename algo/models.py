from django.db import models

# Create your models here.


class Algo(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField()
    created_on = models.DateTimeField(auto_now_add=True)
