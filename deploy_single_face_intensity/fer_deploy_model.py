#!/usr/bin/env python
# coding: utf-8

# In[30]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf


# In[31]:


from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS, cross_origin
import base64
import io


# In[32]:


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# In[33]:


Expressions={0:"Angry",1:"Disgust",2:"Fear",3:"Happy",4:"Sad",5:"Surprise",6:"Neutral"}


# In[34]:


from tensorflow.keras.backend import sigmoid


# In[35]:


def swish(x, beta = 1):
  return (x * sigmoid(beta * x))


# In[36]:


from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation


# In[37]:


get_custom_objects().update({'swish': Activation(swish)})


# In[38]:


def get_model():
    global model
    model=tf.keras.models.load_model("C:\\Users\\pushkar\\Desktop\\beproject_updated\\models\\model_fer_swish.h5")
    return model


# In[39]:


def detect_face(image):
    cascPath = 'C:\\Users\\pushkar\\Desktop\\beproject_updated\\extra_files\\haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)
  #BGR to gray conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  #Cascade multiscale classifier
    detected_faces = faceCascade.detectMultiScale(image,minNeighbors=3,
                                                minSize=(10,10))
    sub = gray

    for x, y, w, h in detected_faces:
        sub = gray[y:y+h,x:x+w]
    return sub,detected_faces


# In[40]:


print('Loading model')
get_model()


# In[41]:


@app.route('/static')
@app.route('/predict',methods=['POST'])
@cross_origin()
def make_prediction():
    message=request.get_json(force=True)
    encoded=message['image']
    encoded_data = encoded.split(',')[1]
    #print(encoded_data)
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    x=image.shape[0]
    y=image.shape[1]
    y1=int(y/(x/300))
    print(y1)
    img2 = cv2.resize(image,(y1,300),interpolation = cv2.INTER_AREA)
    sub_face,detected_faces = detect_face(img2)
    cv2.imwrite('sub_face.jpg',sub_face)
    img=cv2.resize(sub_face,(48,48), interpolation = cv2.INTER_AREA)
    img = img/255.0
    img=np.reshape(img,(1,48,48,1))
    predict_array=model.predict(img)
    predict=np.argmax(predict_array)
    intensity_label=""
    predict_array_oned=np.squeeze(predict_array)
    intensity_value=predict_array_oned[predict]
    if(intensity_value>=0.65):
            intensity_label="Intensity: Extreme"
    elif(intensity_value>=0.4 and intensity_value<0.65):
            intensity_label="Intensity: Moderate"
    elif(intensity_value<0.4):
            intensity_label="Intensity: Low"
    predicted_expression=Expressions[predict]+", "+intensity_label
    response ={
        'prediction':{
            'emotion':predicted_expression
        }
    }
    return jsonify(response)


# In[34]:


# global MEDIA_FOLDER
# MEDIA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("C:\\Users\\pushkar\\Desktop\\flask_apps")))), 'sub_face')


# In[35]:


# @app.route('/')
# @cross_origin
# def download_file(filename):
#     return send_from_directory(MEDIA_FOLDER,filename, as_attachment=True)


# In[ ]:




