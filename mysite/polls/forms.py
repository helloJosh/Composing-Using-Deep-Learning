from django import forms

class UserForm(forms.Form):
    mood = forms.CharField(label='mood')