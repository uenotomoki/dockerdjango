from django import forms

class HelloForm(forms.Form):
    questionnaire = forms.CharField(label='questionnaire')