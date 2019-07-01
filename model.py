import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import glob
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.preprocessing.image import random_rotation, random_shift, random_zoom
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import load_model


#モデルの読み込み
model = load_model('weights.05-0.38-0.82-0.36-0.88.h5')
character=['青信号','赤信号']

img=glob.glob('信号赤/*')
img=img_to_array(load_img(img[0],target_size=(30,30)))
X = np.asarray(img)
X = X.astype('float32')
X = X / 255.0  
X = np.array(X)
#予測
#4次元にする
img_last=np.expand_dims(X,axis=0)#axis=0は0番目に1を追加(1,28,28,3)次元を増やす
#予測
img_last = model.predict(img_last)
#最も大きな値
img_last = np.argmax(img_last)
print('これは'+str(character[img_last]))


