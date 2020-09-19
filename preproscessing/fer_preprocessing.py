import pandas as pd
import numpy as np
import os
import dlib
import cv2

dataset = pd.read_csv('/content/drive/My Drive/Datasets/fer2013.csv')

def get_images(usage):
  df = dataset[dataset['Usage'] == usage]
  y = tf.keras.utils.to_categorical(df['emotion'])
  X = np.array([values.split() for values in df['pixels']])
  X = X.astype(np.float).reshape(len(df), 48, 48, 1)
  return (X, y)


X_train, y_train = get_images('Training')
X_test, y_test =  get_images('PublicTest')
X_validate, y_validate = get_images('PrivateTest') 


print('X_train:', X_train.shape)
print('y_train:', y_train.shape)

print('X_test:', X_test.shape)
print('y_test:', y_test.shape)

print('X_validate:', X_validate.shape)
print('y_validate:', y_validate.shape)


def get_label(arg):
  labels = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}
  return (labels.get(arg,"Invalid Emotion"))


plt.figure(figsize=(12,6))
plt.hist(dataset['emotion'], bins=30)
plt.title("Number of images per emotion")
plt.show()


train_data = X_train.astype('float32')
test_data = X_test.astype('float32')
val_data = X_validate.astype('float32')

train_data /= 255
test_data /= 255
val_data /= 255


def detect_face(frame):
  #Cascade classifier pretrained model
  cascPath = '/content/drive/My Drive/BE Project/sarvesh/haarcascades/haarcascade_frontalface_default.xml'
  faceCascade = cv2.CascadeClassifier(cascPath)
  #BGR to gray conversion
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  #Cascade multiscale classifier
  detected_faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                                minSize=(48,48),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
  coord=[]
  for x, y, w, h in detected_faces:
    if w>100:
      sub_img=frame[y:y+h,x:x+w]
      coord.append([x,y,w,h])
  return gray, detected_faces, coord


def extract_face_features(faces, offset_coeffs=(0.075,0.05)):
  gray=faces[0]
  detected_face=faces[1]
  new_face=[]
  for det in detected_face:
    x,y,w,h=det
    #offset-coeff, np.floor takes lowest int (delete border of image)
    hor_offset=np.int(np.floor(offset_coeffs[0]*w))
    ver_offset=np.int(np.floor(offset_coeffs[1]*h))
    extracted_face=gray[y+ver_offset:y+h, x+hor_offset:x-hor_offset+w]
    #zoom
    new_extracted_face=scipy.ndimage.zoom(extracted_face,(48/extracted_face.shape[0],48/extracted_face.shape[1]))
    new_extracted_face=new_extracted_face.astype(np.float32)
    #scale
    new_extracted_face/=float(new_extracted_face.max())
    new_face.append(new_extracted_face)
  return new_face   