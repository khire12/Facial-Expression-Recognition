

# Commented out IPython magic to ensure Python compatibility.
from google.colab.patches import cv2_imshow
import os
import pathlib
import cv2
# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np
import pandas as pd

def front_dataset(path):
  count=0
  for file in sorted(os.listdir(path)):
    if file[6]=='S':
      substr=file[4:6]
      if substr=='AF':
        img=cv2.imread(path+"/"+file,-1)
        cv2.imwrite("/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front/fear/"+file,img)
        count+=1
      elif substr=='AN':
        img=cv2.imread(path+"/"+file,-1)
        cv2.imwrite("/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front/anger/"+file,img)
        count+=1
      elif substr=='DI':
        img=cv2.imread(path+"/"+file,-1)
        cv2.imwrite("/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front/disgust/"+file,img)
        count+=1
      elif substr=='HA':
        img=cv2.imread(path+"/"+file,-1)
        cv2.imwrite("/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front/happiness/"+file,img)
        count+=1
      elif substr=='NE':
        img=cv2.imread(path+"/"+file,-1)
        cv2.imwrite("/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front/neutral/"+file,img)
        count+=1
      elif substr=='SA':
        img=cv2.imread(path+"/"+file,-1)
        cv2.imwrite("/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front/sadness/"+file,img)
        count+=1
      elif substr=='SU':
        img=cv2.imread(path+"/"+file,-1)
        cv2.imwrite("/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front/surprise/"+file,img)
        count+=1
  return count

def front_half_dataset(path):
  count=0
  for file in sorted(os.listdir(path)):
    if file[6]=='S' or file[6]=='H':
      substr=file[4:6]
      if substr=='AF':
        img=cv2.imread(path+"/"+file,-1)
        cv2.imwrite("/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half/fear/"+file,img)
        count+=1
      elif substr=='AN':
        img=cv2.imread(path+"/"+file,-1)
        cv2.imwrite("/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half/anger/"+file,img)
        count+=1
      elif substr=='DI':
        img=cv2.imread(path+"/"+file,-1)
        cv2.imwrite("/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half/disgust/"+file,img)
        count+=1
      elif substr=='HA':
        img=cv2.imread(path+"/"+file,-1)
        cv2.imwrite("/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half/happiness/"+file,img)
        count+=1
      elif substr=='NE':
        img=cv2.imread(path+"/"+file,-1)
        cv2.imwrite("/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half/neutral/"+file,img)
        count+=1
      elif substr=='SA':
        img=cv2.imread(path+"/"+file,-1)
        cv2.imwrite("/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half/sadness/"+file,img)
        count+=1
      elif substr=='SU':
        img=cv2.imread(path+"/"+file,-1)
        cv2.imwrite("/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half/surprise/"+file,img)
        count+=1
  return count

def front_half_full_dataset(path):
  count=0
  for file in sorted(os.listdir(path)):
    substr=file[4:6]
    if substr=='AF':
      img=cv2.imread(path+"/"+file,-1)
      cv2.imwrite("/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half_full/fear/"+file,img)
      count+=1
    elif substr=='AN':
      img=cv2.imread(path+"/"+file,-1)
      cv2.imwrite("/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half_full/anger/"+file,img)
      count+=1
    elif substr=='DI':
      img=cv2.imread(path+"/"+file,-1)
      cv2.imwrite("/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half_full/disgust/"+file,img)
      count+=1
    elif substr=='HA':
      img=cv2.imread(path+"/"+file,-1)
      cv2.imwrite("/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half_full/happiness/"+file,img)
      count+=1
    elif substr=='NE':
      img=cv2.imread(path+"/"+file,-1)
      cv2.imwrite("/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half_full/neutral/"+file,img)
      count+=1
    elif substr=='SA':
      img=cv2.imread(path+"/"+file,-1)
      cv2.imwrite("/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half_full/sadness/"+file,img)
      count+=1
    elif substr=='SU':
      img=cv2.imread(path+"/"+file,-1)
      cv2.imwrite("/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half_full/surprise/"+file,img)
      count+=1
  return count

path='/content/drive/My Drive/Datasets/kdef_akdef/kdef'
f_c=0
f_h_c=0
f_h_f_c=0
for file in sorted(os.listdir(path)):
  c1=front_dataset(path+'/'+file)
  c2=front_half_dataset(path+'/'+file)
  c3=front_half_full_dataset(path+'/'+file)
  f_c+=c1
  f_h_c+=c2
  f_h_f_c+=c3

print("front: ",f_c)
print("front+half: ",f_h_c)
print("front+half+full: ",f_h_f_c)

!pip install split-folders

import split_folders

split_folders.ratio('/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front', output='/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front(with train-test-val split)', seed=1337, ratio=(.7, .15, .15))
split_folders.ratio('/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half', output='/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half(with train-test-val split)', seed=1337, ratio=(.7, .15, .15))
split_folders.ratio('/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half_full', output='/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half_full(with train-test-val split)', seed=1337, ratio=(.7, .15, .15))