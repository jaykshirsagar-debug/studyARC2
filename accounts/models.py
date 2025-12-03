from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """
    Custom user model for StudyARC.
    Extends Django's AbstractUser so we can add extra fields.
    """

    # optional extra fields for now – keep it light
    display_name = models.CharField(max_length=100, blank=True)
    bio = models.TextField(blank=True)
    # you can add more later (e.g. year_level, university, etc.)

    def __str__(self):
        # what shows up in admin etc.
        return self.display_name or self.username
