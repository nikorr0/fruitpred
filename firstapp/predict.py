# from tensorflow.keras.models import Model # , Sequential
# from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
# from tensorflow.keras.utils import plot_model, to_categorical
# from tensorflow.keras.callbacks import LambdaCallback
# from tensorflow.keras import utils
from PIL import Image, ImageDraw #, ImageFont
# from xml.etree import ElementTree as et
from tensorflow import keras

# import tensorflow.keras.backend as K
# import math
# import pandas as pd
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import os
import time
import cv2
# import sys
from pathlib import Path

# sys.path.insert(1, '/models/modelOD/models/research/object_detection/utils')
import sys



# import file
#from tensorflow_models.modelOD.models.research.object_detection.utils import label_map_util
#from tensorflow_models.modelOD.models.research.object_detection.utils import visualization_utils as viz_utils


# from fruitpred.settings import MODELS_DIR
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = os.path.join(BASE_DIR, "firstapp")#\tensorflow_models")
MODELS_DIR = os.path.join(MODELS_DIR, "tensorflow_models")

sys.path.append(os.path.join(MODELS_DIR, "modelOD\models\research\object_detection"))
#print(MODELS_DIR)

#firstapp/tensorflow_models/modelOD/models/research/object_detection

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils



def loadmodels(func_lever=1):
  global lever
  if func_lever == 1:
    lever = 1
    start_time_full = time.time()

    #OD
    print()
    print('Loading OD model...')
    start_time = time.time()

    global detect_fn
    # url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz'
    # PATH_TO_MODEL_DIR = tf.keras.utils.get_file(fname='ssd_resnet101_v1_fpn_640x640_coco17_tpu-8', origin=url, untar=True)
    # PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
    PATH_TO_SAVED_MODEL = os.path.join(MODELS_DIR, "modelOD\ssd_resnet101_v1_fpn_640x640_coco17_tpu-8\saved_model")
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Done! OD model loaded in {round(elapsed_time, 1)} seconds.')

    #3classes
    print('Loading 3classes model...')
    start_time = time.time()

    global modelConcat
    MODELCONCAT_DIR = os.path.join(MODELS_DIR, "modelConcat")
    modelConcat = keras.models.load_model(os.path.join(MODELCONCAT_DIR, "model"))
    modelConcat.load_weights(os.path.join(MODELCONCAT_DIR, "weights\model_concat"))
    

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Done! 3classes model loaded in {round(elapsed_time, 1)} seconds.')
    #2classes
    print('Loading 2classes model...')
    start_time = time.time()
    
    global modelLayers
    MODELLAYERS_DIR = os.path.join(MODELS_DIR, "modelLayers")
    modelLayers = keras.models.load_model(os.path.join(MODELLAYERS_DIR, "model"))
    modelLayers.load_weights(os.path.join(MODELLAYERS_DIR, "weights\model_layers"))
    

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Done! 2classes model loaded in {round(elapsed_time, 1)} seconds.')


    end_time = time.time()
    elapsed_time = end_time - start_time_full
    print(f'Done! all models loaded in {round(elapsed_time, 1)} seconds.') 
    print()

  if func_lever == 0:
    lever = 0
    print()
    print("Skipping all models...")
    print()




class Predfruitfreshness:
  path_labelmap = os.path.join(MODELS_DIR, "modelOD\label_map.pbtxt")
  #lever_class = lever


  def __init__(self, img_path):
    self.img_path = img_path
  
  def test(self):
    print(self.img_path)
    print(Predfruitfreshness.path_labelmap)

  def odpred(self):


    category_index = label_map_util.create_category_index_from_labelmap(Predfruitfreshness.path_labelmap, use_display_name=True)

    path_to_img = self.img_path

    if lever == 0:
      img = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)
      width = img.shape[0]
      height = img.shape[1]
      image_w_detect = img
      crop_img = np.array([cv2.resize(img[0:300, 0:300], (256, 256))])

      return image_w_detect, crop_img





    def load_image_into_numpy_array(path_to_img):
      return np.array(Image.open(path_to_img))
    
    def detect_objects(path_to_img):

    #print('Running inference for {}... '.format(self.image_path), end='')

      image_np = load_image_into_numpy_array(path_to_img)

      input_tensor = tf.convert_to_tensor(image_np)

      input_tensor = input_tensor[tf.newaxis, ...]
      
      detections = detect_fn(input_tensor)

      num_detections = int(detections.pop('num_detections'))
      detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
      detections['num_detections'] = num_detections

    # detection_classes should be ints.
      detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # show classes
    # unique_classes = set(detections['detection_classes'])
    # print("Classes found:")
    # for c in unique_classes:
    #     print(category_index[c]['name'])
      image_np_with_detections = image_np.copy()

      viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_detections, detections['detection_boxes'], detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=10,#200,
          min_score_thresh=.50,
          agnostic_mode=False)
      
      image_np_with_detections = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)

      return detections, image_np_with_detections
      #plt.figure(figsize=(15, 10))
      #plt.imshow(image_np_with_detections)
      #print('Done')
      #plt.show()
      
    detections, image_w_detect = detect_objects(self.img_path)
  
    img = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)
    width = img.shape[0]
    height = img.shape[1]

    # num_obj = list()
    num_obj = list(filter(lambda x: (x[1] > 0.5) and ((x[0] == 53) or (x[0] == 52) or (x[0] == 55)),
                          list(zip(detections['detection_classes'],
                                   detections['detection_scores'],
                                   detections['detection_boxes'], [i for i in range(100)]))))
    print("NUM_OBJ", len(num_obj))
    # for i in range(10):
    #   n = detections['detection_classes'][i]
    #   if (n == 53) or (n == 52) or (n == 55):
    #     num_obj.append(i)

    if len(num_obj) == 0:
      num_obj = "None"
      return num_obj, num_obj

    # i = num_obj
    crop_imgs = list()
    image_w_detects = list()
    for i in range(len(num_obj)):

      xmin = num_obj[i][2][0] * width # detections['detection_boxes'][num_obj[0]][0] * width
      ymin = num_obj[i][2][1] * height # detections['detection_boxes'][num_obj[0]][1] * height
      wid = (num_obj[i][2][2] * width) - (num_obj[i][2][0] * width) # (detections['detection_boxes'][num_obj[0]][2] * width) - (detections['detection_boxes'][num_obj[0]][0] * width)
      heig = (num_obj[i][2][3] * height) - (num_obj[i][2][1] * height) # (detections['detection_boxes'][num_obj[0]][3] * height) - (detections['detection_boxes'][num_obj[0]][1] * height)

      x = int(round(xmin, 0)) #xmin
      y = int(round(ymin, 0)) #ymin
      w = int(round(wid, 0)) #xmax - xmin
      h = abs(int(round(heig, 0))) #ymax - ymin

      xmax = num_obj[i][2][2] * width
      ymax = num_obj[i][2][3] * height

      crop_img = cv2.cvtColor(img[x:x+w, y:y+h], cv2.COLOR_BGR2RGB)
      crop_img = cv2.resize(crop_img, (256, 256)) #/ 255
      crop_img = np.array([crop_img])
      crop_imgs.append(crop_img)

      image_w_one_detect = Image.open(self.img_path)
      image_w_one_detect_copy = ImageDraw.Draw(image_w_one_detect)
      for n in range(10):
        image_w_one_detect_copy.rectangle([xmin+n, ymin+n, xmax-n, ymax-n], outline = 'black') #неправильно рисует, добавляется только одно изображение с детекцией (может из-за неправильно рисует)
        
      image_w_detects.append(np.array(image_w_one_detect))


    return image_w_detects, crop_imgs

  language = 'ru'
  global fruits_lang
  global freshness_lang
  if language == 'ru':
    fruits_lang = ['яблоко', 'апельсин', 'банан']
    freshness_lang = ['свежий', 'несвежий']
  if language == 'en':
    fruits_lang = ['apple', 'orange', 'banana']
    freshness_lang = ['fresh', 'stale']

  def predfruit(crop_img):
    prediction_fruit = dict()

    if lever == 0:
      prediction_fruit["apple"] = "0 %"
      prediction_fruit["orange"] = "0 %"
      prediction_fruit["banana"] = "0 %"
      return prediction_fruit

    prediction_fruit[f"{fruits_lang[0]}"] = f"{round(modelConcat.predict(crop_img)[0][0]*100, 1)} %"
    prediction_fruit[f"{fruits_lang[1]}"] = f"{round(modelConcat.predict(crop_img)[0][1]*100, 1)} %"
    prediction_fruit[f"{fruits_lang[2]}"] = f"{round(modelConcat.predict(crop_img)[0][2]*100, 1)} %"
    return prediction_fruit

  def predfreshness(crop_img):
    prediction_freshness = dict()

    if lever == 0:
      prediction_freshness["fresh"] = "0 %"
      prediction_freshness["stale"] = "0 %"
      return prediction_freshness

    prediction_freshness[f"{freshness_lang[0]}"] = f"{round(modelLayers.predict(crop_img)[0][0]*100, 1)} %"
    prediction_freshness[f"{freshness_lang[1]}"] = f"{round(modelLayers.predict(crop_img)[0][1]*100, 1)} %"
    return prediction_freshness
