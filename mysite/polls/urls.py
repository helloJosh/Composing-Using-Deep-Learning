#-*-coding:utf-8
from django.conf.urls import url
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name="index"),
    path('happy', views.create_happy, name="happy"),
    path('urgency', views.create_urgency, name="urgency"),
    path('calm', views.create_calm, name="calm"),
    path('3', views.down, name="down")
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)