from rest_framework import serializers
from .models import Result
from message.serializers import MessageSerializer
from algo.serializers import AlgoSerializer

class ResultSerializer(serializers.ModelSerializer):
    message = MessageSerializer()
    algo = AlgoSerializer()
    custom_field = serializers.SerializerMethodField()
    class Meta:
        model = Result
        fields = ['custom_field','algo', 'message']

    def get_custom_field(self, obj):
        # Retourner les attributs souhaités dans le champ personnalisé
        return {'accuracy':obj.accuracy,'precision': obj.precision,'recall':obj.recall,'f1':obj.f1}
