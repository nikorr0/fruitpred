from django.shortcuts import render
from django.http import HttpResponse
import cv2
import os
from fruitpred.settings import BASE_DIR
from . import predict as pr
from django.views.decorators.csrf import csrf_protect
import numpy as np

supported_formats = ["png", "jpg"]
pr.loadmodels(0)

@csrf_protect
def index(request):

    if request.method == 'POST':

        if bool(request.FILES) == False:
            data = {"image": "False"}
            return render(request, "index.html", context=data)

        image_format = list(request.FILES['photo'].name)
        image_format = (image_format[-3] + image_format[-2] + image_format[-1]).lower()


        if image_format not in supported_formats:
            data = {"image": "False"}
            return render(request, "index.html", context=data)
        
        
        IMAGES_DIR = os.path.join(BASE_DIR, "static\images")
        if image_format == "png":
            with open(os.path.join(IMAGES_DIR, "image_saved.png"), "wb") as f:
                f.write(request.FILES['photo'].read())
                f.close()

            cv2.imwrite(os.path.join(IMAGES_DIR, "image_saved.jpg"), cv2.imread(os.path.join(IMAGES_DIR, f"image_saved.{image_format}")))

        else:
            with open(os.path.join(IMAGES_DIR, "image_saved.jpg"), "wb") as f:
                f.write(request.FILES['photo'].read())
                f.close()
            cv2.imwrite(os.path.join(IMAGES_DIR, "image_saved.jpg"), cv2.imread(os.path.join(IMAGES_DIR, f"image_saved.{image_format}")))

        image = pr.Predfruitfreshness(os.path.join(IMAGES_DIR, "image_saved.jpg"))

        image_with_detection, crop_image = image.odpred()

        if (image_with_detection == "None") and (crop_image == "None"):
            data = {"image": "NotFound"}
            return render(request, "index.html", context=data)

        IMAGE_W_DETECT_DIR = os.path.join(IMAGES_DIR, "image_w_detect_saved.jpg")
        cv2.imwrite(IMAGE_W_DETECT_DIR, image_with_detection)

        IMAGE_CROP_DIR = os.path.join(IMAGES_DIR, "image_crop_saved.jpg")
        cv2.imwrite(IMAGE_CROP_DIR, crop_image[0])

        fruit_predictions = pr.Predfruitfreshness.predfruit(crop_image)
        freshness_predictions = pr.Predfruitfreshness.predfreshness(crop_image)
            
        height, width, color = image_with_detection.shape #reverse?
        #print(width)
        #print(height)
        k = round(width / height, 4)
        print(k)
        #height = image_with_detection.shape()[1]

        if k > 1.15:
            width_edit, height_edit = 640, 360

        if k < 0.9:
            width_edit, height_edit = 256, 410
        
        if (k >= 0.9) and (k <= 1.15):
            width_edit, height_edit = 400, 400


        data = {"image": os.path.join(IMAGES_DIR, "image_saved.jpg"), 
        "image_with_detection": IMAGE_W_DETECT_DIR, 
        "crop_image": IMAGE_CROP_DIR, 
        "fruit_predictions": fruit_predictions.items(), 
        "freshness_predictions": freshness_predictions.items(),
        "width_edit": width_edit,
        "height_edit": height_edit}


    
    else:
        data = {"image": "None"}
        return render(request, "index.html", context=data)



    return render(request, "index.html", context=data)


def result(request):
    output = "<h2>Предсказать</h2>"
    return HttpResponse(output)


# https://learndjango.com/tutorials/django-file-and-image-uploads-tutorial
# https://www.learningaboutelectronics.com/Articles/How-to-display-an-image-in-Django.php
