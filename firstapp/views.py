from distutils.command.upload import upload
from django.shortcuts import render
from django.http import HttpResponse
#from .forms import UploadFileForm

def index(request):
    #output = "<h2>Прогнозирование свежести фрукта по фотографии</h2>"
    #return HttpResponse(output)
    #if request.method == "POST":
        #form = UploadFileForm(request.POST, request.FILES)
    return render(request, "index.html")

def predict(request):
    output = "<h2>Предсказать</h2>"
    return HttpResponse(output)


#https://learndjango.com/tutorials/django-file-and-image-uploads-tutorial