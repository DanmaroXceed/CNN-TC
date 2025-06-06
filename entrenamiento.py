#AppCNN - Tumor cerebral
import os
import cv2 
import numpy as np 
import glob
import matplotlib.pyplot as plt 
import seaborn as sbn 
import pandas as pd 
import sklearn.metrics as metrics 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
from keras import Sequential, Input 
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization # type: ignore
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore

t = 150

x_train = []
y_train = []
x_test = []
y_test = []
dataTr = []

# ------------------ Carga de imágenes ------------------

for filename in glob.glob(os.path.join('tumores/appcnn/train/yes', '*.jpg')):
	dataTr.append([1,cv2.resize(cv2.imread(filename), dsize=(t,t), interpolation=cv2.INTER_CUBIC)])
for filename in glob.glob(os.path.join('tumores/appcnn/train/no', '*.jpg')):
	dataTr.append([0,cv2.resize(cv2.imread(filename), dsize=(t,t), interpolation=cv2.INTER_CUBIC)])
print("-----Extraccion de imagenes listo-----")

# Mezclar y separar características y etiquetas
from random import shuffle
shuffle(dataTr)

for i, j in dataTr:
	y_train.append(i)#Columna de etiquetas 0 y 1
	x_train.append(j)#Columna de imagenes

x_train = np.array(x_train)/255
y_train = np.array(y_train)

for filename in glob.glob(os.path.join('tumores/appcnn/test/yes', '*.jpg')):
	x_test.append(cv2.resize(cv2.imread(filename), dsize=(t,t), interpolation=cv2.INTER_CUBIC))
	y_test.append(1)
for filename in glob.glob(os.path.join('tumores/appcnn/test/no', '*.jpg')):
	x_test.append(cv2.resize(cv2.imread(filename), dsize=(t,t), interpolation=cv2.INTER_CUBIC))
	y_test.append(0)

x_test = np.array(x_test)/255
y_test = np.array(y_test)

modelo = Sequential()
# 1era Capa
modelo.add(Input(shape=(t, t, 3)))
modelo.add(Conv2D(32, (3,3), activation='relu', padding='same'))
modelo.add(BatchNormalization())
modelo.add(MaxPooling2D(pool_size=(2,2)))

# 2da Capa
modelo.add(Conv2D(64, (3,3), activation='relu', padding='same'))
modelo.add(BatchNormalization())
modelo.add(MaxPooling2D(pool_size=(2,2)))

# 3era Capa
modelo.add(Conv2D(128, (3,3), activation='relu', padding='same'))
modelo.add(BatchNormalization())
modelo.add(MaxPooling2D(pool_size=(2,2)))

# 4ta Capa
modelo.add(Conv2D(128, (3,3), activation='relu', padding='same'))
modelo.add(BatchNormalization())
modelo.add(MaxPooling2D(pool_size=(2,2)))

# Aplanado
modelo.add(Flatten())

# Capas densas
modelo.add(Dense(128, activation='relu'))
modelo.add(Dropout(0.4))  
modelo.add(Dense(64, activation='relu'))
modelo.add(Dropout(0.3))  
modelo.add(Dense(32, activation='relu'))
modelo.add(Dense(1, activation='sigmoid'))

# Compilación
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Resumen
modelo.summary()
print("Configuracion de red: LISTO")

epocas = 30

# Funcion para detener el entrenamiento
class DetenerPorValPrecision(Callback):
    def __init__(self, precision_objetivo=0.99):
        super().__init__()
        self.precision_objetivo = precision_objetivo

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get('val_accuracy')
        if val_acc is not None and val_acc >= self.precision_objetivo:
            print(f"\n✔️ Se alcanzó val_accuracy = {val_acc:.4f}. Entrenamiento detenido.")
            self.model.stop_training = True

detener_callback = DetenerPorValPrecision(precision_objetivo=0.99)

# Callback para guardar el mejor modelo según val_accuracy
guardar_mejor_modelo = ModelCheckpoint(
    filepath='mejor_modelo_tumores.keras', # Guardar el modelo
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

# Callback EarlyStopping estándar
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    mode='max',
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

entrenamiento = modelo.fit(
	x_train, y_train,
	batch_size=32,
	epochs=epocas,
	validation_data=(x_test, y_test),
	callbacks=[detener_callback, early_stopping, reduce_lr, guardar_mejor_modelo]
)

print("Modelo entrenado exitosamente.")

print(entrenamiento.history)

# ------------------ Evaluación y Gráficas ------------------
#Graficar accuracy

plt.figure(0)
plt.plot(entrenamiento.history['accuracy'], 'r')
plt.plot(entrenamiento.history['val_accuracy'], 'b')
plt.xticks(np.arange(0, epocas, 2.0))
plt.rcParams['figure.figsize'] = (8,6)
plt.xlabel('Num of Epochs')
plt.ylabel('Accuracy')
plt.title('Training accuracy vs Validation accuracy')
plt.legend(['train', 'validacion'])
plt.show()

#Grafico de perdida

plt.figure(1)
plt.plot(entrenamiento.history['loss'], 'r')
plt.plot(entrenamiento.history['val_loss'], 'g')
plt.xticks(np.arange(0, epocas, 2.0))
plt.rcParams['figure.figsize'] = (6,6)
plt.xlabel('Num of Epochs')
plt.ylabel('Loss')
plt.title('Training loss vs Validation loss')
plt.legend(['train', 'validacion'])
plt.show()

#Matriz de confusion

entrenamiento_pred=modelo.predict(x_test.astype('float'))
entrenamiento_pred=np.rint(entrenamiento_pred).astype('int')
entrenamiento_predicted=entrenamiento_pred.ravel()
entrenamiento_cm=confusion_matrix(y_test, entrenamiento_predicted)

entrenamiento_de_cm=pd.DataFrame(entrenamiento_cm)
plt.figure(figsize=(15,15))
sbn.set_theme(font_scale=1.4)
sbn.heatmap(entrenamiento_de_cm, annot=True, annot_kws={"size":22})
plt.ylabel('Valor real')
plt.xlabel('Predicciones')
plt.title('Matriz de Confusión')
plt.show()

# Reporte
cnn_report= classification_report(y_test, entrenamiento_predicted)
print('Accuracy = {:.2f}'.format(accuracy_score(y_test,entrenamiento_predicted)))
print(cnn_report)

#Curva ROC

fpr, tpr, threshold = metrics.roc_curve(y_test, entrenamiento_predicted)
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