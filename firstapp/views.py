from distutils.command.upload import upload
from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import default_storage
import cv2

def index(request):
    # print(request.FILES)
    # print(request.FILES['photo'])
    if request.method == 'POST':
        # print(request.FILES['photo'])
        # print(request.FILES['photo'].read())
        # print(dir(request.FILES['photo']))
        with open("apple_saved.jpg", "wb") as f:
            f.write(request.FILES['photo'].read())
            f.close()
        # with default_storage.open("tmp/apple_saved.jpg", 'wb') as destination:
            # for chunk in request.FILES['photo'].chunks():
                # destination.write(chunk)
        path = "apple_saved.jpg"
        cv2.imread(path)



    return render(request, "index.html")


def result(request):
    output = "<h2>Предсказать</h2>"
    return HttpResponse(output)


# https://learndjango.com/tutorials/django-file-and-image-uploads-tutorial
# https://www.learningaboutelectronics.com/Articles/How-to-display-an-image-in-Django.php
