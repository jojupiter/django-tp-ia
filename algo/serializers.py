from rest_framework import serializers
from .models import Algo


class AlgoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Algo
        fields = '__all__' 
