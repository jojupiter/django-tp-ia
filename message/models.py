from django.db import models



class Message(models.Model):
    message_text = models.TextField()
    created_on = models.DateTimeField(auto_now_add=True)

    def __init__(self, message_text=None, *args, **kwargs):
        super(Message, self).__init__(*args, **kwargs)
        if message_text is not None:
            self.message_text = message_text
