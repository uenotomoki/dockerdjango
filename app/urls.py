from django.urls import path
from . import views
from .views import HelloView

urlpatterns = [
    path('', HelloView.as_view(),name='index'),
    #path('form', views.form,name='form'),
]