from distutils.command.upload import upload
from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import default_storage
# import cv2
supported_formats = [".png", ".jpg", "jpeg"]

def index(request):
    # print(request.FILES)
    # print(request.FILES['photo'])
    if request.method == 'POST':
        # print(request.FILES['photo'])
        # print(request.FILES['photo'].read())
        #print(dir(request.FILES['photo']))
        if bool(request.FILES) == False:
            data = {"image": "False"}
            return render(request, "index.html", context=data)

        image_format = list(request.FILES['photo'].name)
        image_format = image_format[-4] + image_format[-3] + image_format[-2] + image_format[-1]
        #print(image_format)
        #image_format = "jpg"

        if image_format not in supported_formats:
            data = {"image": "False"}
            return render(request, "index.html", context=data)
        
            
        with open(r"static/images/apple_saved.jpg", "wb") as f:
            f.write(request.FILES['photo'].read())
            f.close()
        # with default_storage.open("tmp/apple_saved.jpg", 'wb') as destination:
            # for chunk in request.FILES['photo'].chunks():
                # destination.write(chunk)
        # path = "apple_saved.jpg"
        # cv2.imread(path)
        data = {"image": "images/apple_saved.jpg"}
    
    else:
        data = {"image": "None"}



    return render(request, "index.html", context=data)


def result(request):
    output = "<h2>Предсказать</h2>"
    return HttpResponse(output)


# https://learndjango.com/tutorials/django-file-and-image-uploads-tutorial
# https://www.learningaboutelectronics.com/Articles/How-to-display-an-image-in-Django.php
