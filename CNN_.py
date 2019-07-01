import glob
from keras.preprocessing.image import img_to_array,load_img
import numpy as np
import cv2
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

X = []
Y = []

files=glob.glob('信号青/*')#「信号青フォルダ」から写真のファイル名一覧を取得
for i in range(len(files)):
    picture = files[i]
    picture2 = load_img(picture, target_size=(30, 30))#PIL形式に変換　サイズ縮小
    picture3 = img_to_array(picture2)
    X.append(picture3)
    Y.append(0)#ラベル
    

files=glob.glob('信号赤/*')#「信号赤フォルダ」から写真のファイル名一覧を取得
for i in range(len(files)):
    picture = files[i]
    picture_red2 = load_img(picture, target_size=(30, 30))
    picture_red3 = img_to_array(picture_red2)
    X.append(picture_red3)
    Y.append(1)#ラベル
X = np.asarray(X) # numpyのarrayに変換
Y = np_utils.to_categorical(Y,2)#One-Hotエンコーディング 

X = X / 255.0#正規化
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)#「トレーニング用」と「テスト用」のデータに分割7:3 



model = Sequential()#ニューラルネットワークモデル構築する
model.add(Conv2D(32,(3,3),padding='same',input_shape=(30, 30, 3),activation='relu'))#畳み込み⇨特徴マップ
model.add(MaxPooling2D(pool_size=(2,2)))#プーリング処理
model.add(Dropout(0.3))#過学習
model.add(Conv2D(32, (3, 3))) #畳み込み
model.add(Activation('relu'))#活性化関数：ReLU 
model.add(Dropout(0.25))#過学習

model.add(Flatten())#　入力層
model.add(Dense(units=16,activation='relu'))#中間層　
model.add(Dense(units=2,activation ='softmax'))#出力層

model.summary()#モデルの表示
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam' ,metrics =['accuracy'] )#コンパイル
#損失関数


#モデル保存
cp = ModelCheckpoint('学習済みモデル/3/weights.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.h5', monitor="val_acc", verbose=1,mode='max',save_best_only=True, save_weights_only=False)
#save_weights_only: False=モデル全体を保存　True=モデルの重みが保存。

#学習
history = model.fit(X_train, y_train,batch_size=10,verbose=1,epochs=5,callbacks=[cp],validation_data=(X_test,y_test))

#評価
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])#損失関数の平均化
print('Test accuracy:', score[1])#正解率


