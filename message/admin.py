from django.contrib import admin

# Register your models here.
from message import models

admin.site.register(models.Message)