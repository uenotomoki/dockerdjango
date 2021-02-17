from django.shortcuts import render
from django.http import HttpResponse
from .forms import HelloForm
from django.views.generic import TemplateView
from .questionnaire import questionnaire_result

class HelloView(TemplateView):
    def __init__(self):
        self.params = {
            'title':'Hello',
            'msg':'これはサンプルページです',
            'form':HelloForm(),
        }

    def get(self,request):
        return render(request,'app/index.html',self.params)

    def post(self,request):
        #self.params['msg'] = request.POST['questionnaire']
        self.params['msg'] = questionnaire_result(request.POST['questionnaire'])
        self.params['form'] = HelloForm(request.POST)
        return render(request,'app/index.html',self.params)

    #def form(request):
    #    msg = request.POST['msg']
    #    params = {
    #        'title':'form',
    #        'msg':'こんにちは' + msg + 'さん'
    #    }
    #    return render(request,'app/index.html',params)