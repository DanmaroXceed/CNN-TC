#AppCNN - Tumor cerebral
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
import tensorflow as tf
import random
import sklearn.metrics as metrics
import glob
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

os.system("cls")

x_train = []
y_train = []
x_test = []
y_test = []
dataTr = []
t = 150;

for filename in glob.glob(os.path.join('tumores/appcnn/train/yes', '*.jpg')):
	dataTr.append([1,cv2.resize(cv2.imread(filename), dsize=(t,t), interpolation=cv2.INTER_CUBIC)])
for filename in glob.glob(os.path.join('tumores/appcnn/train/no', '*.jpg')):
	dataTr.append([0,cv2.resize(cv2.imread(filename), dsize=(t,t), interpolation=cv2.INTER_CUBIC)])
print("-----Extraccion de imagenes listo-----")

from random import shuffle
shuffle(dataTr)

for i, j in dataTr:
	y_train.append(i)#Columna de etiquetas 0 y 1
	x_train.append(j)#Columna de imagenes

x_train = np.array(x_train)/255
y_train = np.array(y_train)

##############################

for filename in glob.glob(os.path.join('tumores/appcnn/test/yes', '*.jpg')):
	x_test.append(cv2.resize(cv2.imread(filename), dsize=(t,t), interpolation=cv2.INTER_CUBIC))
	y_test.append(1)
for filename in glob.glob(os.path.join('tumores/appcnn/test/no', '*.jpg')):
	x_test.append(cv2.resize(cv2.imread(filename), dsize=(t,t), interpolation=cv2.INTER_CUBIC))
	y_test.append(0)

x_test = np.array(x_test)/255
y_test = np.array(y_test)

##############################
#######CREAR MODELO CNN#######
##############################

modelo = Sequential()
#1er Capa
modelo.add(Convolution2D(32,(3,3), input_shape=(t,t,3), activation='relu'))
modelo.add(MaxPooling2D(pool_size=((2,2))))
#2da Capa
modelo.add(Convolution2D(64,(3,3), activation='relu'))
modelo.add(MaxPooling2D(pool_size=((2,2))))
#3er Capa
modelo.add(Convolution2D(128,(3,3), activation='relu'))
modelo.add(MaxPooling2D(pool_size=((2,2))))
#4er Capa
modelo.add(Convolution2D(128,(3,3), activation='relu'))
modelo.add(MaxPooling2D(pool_size=((2,2))))
#Matriz a vector
modelo.add(Flatten())
#Capas densas
modelo.add(Dense(128, activation='relu'))
modelo.add(Dropout(0.5))#Desactiva el 50% de neuronas en cada epoca
modelo.add(Dense(64, activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(32, activation='relu'))
modelo.add(Dense(1, activation='sigmoid'))#Solido binario
#Optimizador ADAM
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
modelo.summary()
print("Configuracion de red: LISTO")
#Entrenamiento
epocas=10
rev=modelo.fit(x_train,y_train, batch_size=64, epochs=epocas, validation_data=(x_test, y_test))
#Prediccion
ruta1='tumores/appcnn/val/no/no1350.jpg'
ruta2='tumores/appcnn/val/yes/y1350.jpg'

I = cv2.imread(ruta1)
I_resz = cv2.resize(I,dsize=(t,t), interpolation=cv2.INTER_CUBIC)
I_resz = I_resz.astype('float')

I2 = cv2.imread(ruta2)
I2_resz = cv2.resize(I2,dsize=(t,t), interpolation=cv2.INTER_CUBIC)
I2_resz = I2_resz.astype('float')

if round(modelo.predict(np.array([I_resz]))[0][0])==1:
	print("Tumor")
	I = cv2.imread(ruta1)
	cv2.putText(I, 'Tumor cerebral', (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0),3)
	cv2.imshow('Tumor cerebral', I)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
else:
	print("Sin tumor")
	I = cv2.imread(ruta1)
	cv2.putText(I, 'Cerebro sin tumor', (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0),3)
	cv2.imshow('Cerebro sin tumor', I)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if round(modelo.predict(np.array([I2_resz]))[0][0])==1:
	print("Tumor")
	I2 = cv2.imread(ruta2)
	cv2.putText(I2, 'Tumor cerebral', (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0),3)
	cv2.imshow('Tumor cerebral', I2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
else:
	print("Sin tumor")
	I2 = cv2.imread(ruta2)
	cv2.putText(I2, 'Cerebro sin tumor', (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0),3)
	cv2.imshow('Cerebro sin tumor', I2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

snn=rev
print(snn.history)

#Graficar accuracy

plt.figure(0)
plt.plot(snn.history['acc'], 'r')
plt.plot(snn.history['val_acc'], 'g')
plt.xticks(np.arange(0, epocas, 2.0))
plt.rcParams['figure.figsize'] = (8,6)
plt.xlabel('Num of Epochs')
plt.ylabel('Accuracy')
plt.title('Training accuracy vs Validation accuracy')
plt.legend(['train', 'validacion'])

#Grafico de perdida

plt.plot(snn.history['loss'], 'r')
plt.plot(snn.history['val_loss'], 'g')
plt.xticks(np.arange(0, epocas, 2.0))
plt.rcParams['figure.figsize'] = (8,6)
plt.xlabel('Num of Epochs')
plt.ylabel('Loss')
plt.title('Training loss vs Validation loss')
plt.legend(['train', 'validacion'])

#Matriz de confusion

snn_pred=modelo.predict(x_test.astype('float'))
snn_pred=np.rint(snn_pred).astype('int')
snn_predicted=snn_pred.ravel()
snn_cm=confusion_matrix(y_test, snn_predicted)

snn_de_cm=pd.DataFrame(snn_cm)
plt.figure(figsize=(15,15))
sbn.set(font_scale=1.4)
sbn.heatmap(snn_de_cm, annot=True, annot_kws={"size":22})
plt.ylabel('Valor real')
plt.xlabel('Predicciones')
plt.show()

cnn_report= classification_report(y_test, snn_predicted)
print('Accuracy = {:.2f}'.format(accuracy_score(y_test,snn_predicted)))
print(cnn_report)

#Curva ROC

fpr, tpr, threshold = metrics.roc_curve(y_test, snn_predicted)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Curva ROC de la red neuronal convolucional')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1.01])
plt.ylim([0,1.01])
plt.xlabel('Sensibilidad')
plt.ylabel('1- Especificacion')
plt.show()