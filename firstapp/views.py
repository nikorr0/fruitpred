<<<<<<< Updated upstream
from distutils.command.upload import upload
=======
from tkinter import N
>>>>>>> Stashed changes
from django.shortcuts import render
from django.http import HttpResponse
#from .forms import UploadFileForm

<<<<<<< Updated upstream
=======
supported_formats = ["png", "jpg"]
pr.loadmodels()

@csrf_protect
>>>>>>> Stashed changes
def index(request):
    #output = "<h2>Прогнозирование свежести фрукта по фотографии</h2>"
    #return HttpResponse(output)
    #if request.method == "POST":
        #form = UploadFileForm(request.POST, request.FILES)
    return render(request, "index.html")

<<<<<<< Updated upstream
def predict(request):
=======
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

        image_with_detections, crop_images = image.odpred()

        if (image_with_detections == "None") and (crop_images == "None"):
            data = {"image": "NotFound"}
            return render(request, "index.html", context=data)
        
        n = 0
        IMAGE_W_DETECT_DIRs = list()
        for image_w_detection in image_with_detections:
            n += 1
            IMAGE_W_DETECT_DIR = os.path.join(IMAGES_DIR, f"image_w_detect_saved{n}.jpg")
            IMAGE_W_DETECT_DIRs.append(IMAGE_W_DETECT_DIR)
            cv2.imwrite(IMAGE_W_DETECT_DIR, image_w_detection)

        n = 0
        IMAGE_CROP_DIRs = list()
        for crop_image in crop_images:
            n += 1
            IMAGE_CROP_DIR = os.path.join(IMAGES_DIR, f"image_crop_saved{n}.jpg")
            IMAGE_CROP_DIRs.append(IMAGE_CROP_DIR)
            cv2.imwrite(IMAGE_CROP_DIR, crop_image[0])

        fruit_predictions = list()
        freshness_predictions = list()
        for crop_image in crop_images:
            fruit_prediction = pr.Predfruitfreshness.predfruit(crop_image)
            fruit_predictions.append(max(fruit_prediction.items()))

            freshness_prediction = pr.Predfruitfreshness.predfreshness(crop_image)
            freshness_predictions.append(max(freshness_prediction.items()))


            
        height, width, color = image_with_detections[0].shape #reverse?
        #print(width)
        #print(height)
        k = round(width / height, 4)
        #height = image_with_detection.shape()[1]

        if k > 1.15:
            width_edit, height_edit = 640, 360

        if k < 0.9:
            width_edit, height_edit = 256, 410
        
        if (k >= 0.9) and (k <= 1.15):
            width_edit, height_edit = 400, 400

        images_detections_and_crops = zip(IMAGE_W_DETECT_DIRs, IMAGE_CROP_DIRs)

        predictions = zip(fruit_predictions, freshness_predictions)

        data = {"image": os.path.join(IMAGES_DIR, "image_saved.jpg"), 
        #"image_with_detections": IMAGE_W_DETECT_DIRs, 
        #"crop_images": IMAGE_CROP_DIRs, 
        #"fruit_predictions": fruit_predictions, 
        #"freshness_predictions": freshness_predictions,
        "images_detections_and_crops": images_detections_and_crops,
        "predictions": predictions,
        "width_edit": width_edit,
        "height_edit": height_edit}


    
    else:
        data = {"image": "None"}
        return render(request, "index.html", context=data)



    return render(request, "index.html", context=data)


def result(request):
>>>>>>> Stashed changes
    output = "<h2>Предсказать</h2>"
    return HttpResponse(output)


#https://learndjango.com/tutorials/django-file-and-image-uploads-tutorial