import io

from django.shortcuts import render
import keras
from django.http import HttpResponseRedirect
from django.shortcuts import render


from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import FileSystemStorage
# Create your views here.
import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import cv2
from tensorflow import Graph
import  numpy as np
img_size = 224
model = None
modelDensenet201 = tf.keras.models.load_model('./models/BreastCancer224.h5')
modelDensenet201.summary()
modelXception = tf.keras.models.load_model('./models/BreastCancer224xception.h5')
modelXception.summary()
modelDensenet121 = tf.keras.models.load_model('./models/BreastCancer224densenet121.h5')
modelDensenet121.summary()
def index(request):
    context = {'a': 1}
    return render(request, 'index.html', context)

gpuoptions = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpuoptions))




# Code mới !
# Create your views here.
@api_view(['GET'])
def index_page(request):
    return_data = {
        "error" : "0",
        "message" : "Successful",
    }
    return Response(return_data)


def create_opencv_image_from_stringio(img_stream, cv2_img_flag=0):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)


@api_view(["POST"])
def predictImageDensenet201(request):
    try:
        img = request.FILES['filePath'].read()
        if img is not None:
            img_size = 224
            img = cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (img_size, img_size), 3)
            img = np.array(img) / 255
            img = np.array(img)
            img = img.reshape(-1, img_size, img_size, 3)
            print(img)
            # Generate predictions for samples
            prediction = modelDensenet201.predict(img)
            r = prediction[0]
            classes = np.argmax(prediction, axis=1)
            percent = np.max(prediction) * 100
            for i in classes:
                if (i == 0):
                    a = 'Benign'
                elif (i == 1):
                    a = 'InSitu'
                elif (i == 2):
                    a = 'Invasive'
                else:
                    a = 'Normal'

            predictions = {
                'error': '0',
                'model': 'Densenet201',
                'message': 'Successfull',
                'Benign' : r[0]*100,
                'InSitu' : r[1]*100,
                'Invasive': r[2]*100,
                'Normal' : r[3]*100,
                'result': a,
                'accuracy': percent,
            }
        else:
            predictions = {
                'error': '1',
                'message': 'Invalid Parameters'
            }
    except Exception as e:
        predictions = {
            'error': '2',
            "message": str(e)
        }


    return Response(predictions)

@api_view(["POST"])
def predictImageXception(request):
    try:
        img = request.FILES['filePath'].read()
        if img is not None:
            img_size = 224
            img = cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (img_size, img_size), 3)
            img = np.array(img) / 255
            img = np.array(img)
            img = img.reshape(-1, img_size, img_size, 3)
            print(img)
            # Generate predictions for samples
            prediction = modelXception.predict(img)
            r = prediction[0]
            classes = np.argmax(prediction, axis=1)
            percent = np.max(prediction) * 100
            for i in classes:
                if (i == 0):
                    a = 'Benign'
                elif (i == 1):
                    a = 'InSitu'
                elif (i == 2):
                    a = 'Invasive'
                else:
                    a = 'Normal'

            predictions = {
                'error': '0',
                'model': 'Xception',
                'message': 'Successfull',
                'Benign' : r[0]*100,
                'InSitu' : r[1]*100,
                'Invasive': r[2]*100,
                'Normal' : r[3]*100,
                'result': a,
                'accuracy': percent,
            }
        else:
            predictions = {
                'error': '1',
                'message': 'Invalid Parameters'
            }
    except Exception as e:
        predictions = {
            'error': '2',
            "message": str(e)
        }


    return Response(predictions)

@api_view(["POST"])
def predictImageDensenet121(request):
    try:
        img = request.FILES['filePath'].read()
        if img is not None:
            img_size = 224
            img = cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (img_size, img_size), 3)
            img = np.array(img) / 255
            img = np.array(img)
            img = img.reshape(-1, img_size, img_size, 3)
            print(img)
            # Generate predictions for samples
            prediction = modelDensenet121.predict(img)
            r = prediction[0]
            classes = np.argmax(prediction, axis=1)
            percent = np.max(prediction) * 100
            for i in classes:
                if (i == 0):
                    a = 'Benign'
                elif (i == 1):
                    a = 'InSitu'
                elif (i == 2):
                    a = 'Invasive'
                else:
                    a = 'Normal'

            predictions = {
                'error': '0',
                'model': 'Densenet121',
                'message': 'Successfull',
                'Benign' : r[0]*100,
                'InSitu' : r[1]*100,
                'Invasive': r[2]*100,
                'Normal' : r[3]*100,
                'result': a,
                'accuracy': percent,
            }
        else:
            predictions = {
                'error': '1',
                'message': 'Invalid Parameters'
            }
    except Exception as e:
        predictions = {
            'error': '2',
            "message": str(e)
        }


    return Response(predictions)