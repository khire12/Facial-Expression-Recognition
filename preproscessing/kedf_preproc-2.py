import os
import pathlib
import cv2
import split_folders

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



split_folders.ratio('/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front', output='/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front(with train-test-val split)', seed=1337, ratio=(.7, .15, .15))
split_folders.ratio('/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half', output='/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half(with train-test-val split)', seed=1337, ratio=(.7, .15, .15))
split_folders.ratio('/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half_full', output='/content/drive/My Drive/Datasets/kdef_akdef/KDEF_front_half_full(with train-test-val split)', seed=1337, ratio=(.7, .15, .15))