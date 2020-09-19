#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf


# In[2]:


from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS, cross_origin
import base64
import io


# In[3]:


from PIL import Image


# In[4]:


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# In[5]:


Expressions={0:"Angry",1:"Disgust",2:"Fear",3:"Happy",4:"Sad",5:"Surprise",6:"Neutral"}


# In[6]:


from tensorflow.keras.backend import sigmoid


# In[10]:


def swish(x, beta = 1):
  return (x * sigmoid(beta * x))


# In[11]:


from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation


# In[12]:


get_custom_objects().update({'swish': Activation(swish)})


# In[13]:


def get_model():
    global model
    model=tf.keras.models.load_model("C:\\Users\\pushkar\\Desktop\\fer2013\\fer2013_finalmodels\\model_fer_swish.h5")
    return model


# In[14]:


def detect_face(image):
    cascPath = 'C:\\Users\\pushkar\\Desktop\\beproject_updated\\haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)
  #BGR to gray conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  #Cascade multiscale classifier
    detected_faces = faceCascade.detectMultiScale(image,minNeighbors=4,
                                                minSize=(10,10))
    sub = []

    for x, y, w, h in detected_faces:
        sub.append(gray[y:y+h,x:x+w])
    return sub,detected_faces,len(detected_faces)


# In[15]:


print('Loading model')
get_model()


# In[16]:


@app.route('/predict1',methods=['POST'])
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
    sub_face,detected_faces,numFaces = detect_face(img2) 
    predicted_expression = "["
    predict=[]
    for i in range(numFaces):
        img=cv2.resize(sub_face[i],(48,48), interpolation = cv2.INTER_AREA)
        img = img/255.0
        img=np.reshape(img,(1,48,48,1))
        predict_array=model.predict(img)
        predict.append(np.argmax(predict_array))
        predicted_expression=predicted_expression + " "+ Expressions[np.argmax(predict_array)]
        
    for face,sub,pred in zip(detected_faces,sub_face,predict):
        for _ in face:
            cv2.rectangle(img2,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,0,0),2)
            font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img2,str(Expressions[pred]),(face[0],face[1]-5),font,0.6,(255,0,0),2,cv2.LINE_AA)
    
    
    predicted_expression = predicted_expression + " ]"
    
    
    img_byte_arr = io.BytesIO()
    img = Image.fromarray(img2,mode='RGB')
    img.save(img_byte_arr, format='PNG')
    encode_img =  base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    
    
    
    response ={
        'prediction':{
            'emotion':predicted_expression
        },
        'image_faces_detected':{
            'image':encode_img
        }
    }
    return jsonify(response)


# In[ ]:




