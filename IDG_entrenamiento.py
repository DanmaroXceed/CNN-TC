# ------------------ AppCNN - Tumor cerebral ------------------

# === IMPORTACIONES ===
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from keras import Sequential, Input
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D  # type: ignore
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau  # type: ignore
from random import shuffle
from tensorflow.keras.regularizers import l2  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# === PARÁMETROS ===
t = 150
epocas = 30
x_train, y_train = [], []
x_test, y_test = [], []
dataTr = []

# ------------------ CARGA Y PREPROCESAMIENTO DE DATOS ------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)  # sin aumentos para test

train_generator = train_datagen.flow_from_directory(
    'tumores/appcnn/train/',
    target_size=(t, t),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    'tumores/appcnn/test/',
    target_size=(t, t),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# ------------------ DEFINICIÓN DEL MODELO ------------------

modelo = Sequential()

# Entrada
modelo.add(Input(shape=(t, t, 3)))

# Capas convolucionales
modelo.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
modelo.add(BatchNormalization())
modelo.add(MaxPooling2D(pool_size=(2, 2)))

modelo.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
modelo.add(BatchNormalization())
modelo.add(MaxPooling2D(pool_size=(2, 2)))

modelo.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
modelo.add(BatchNormalization())
modelo.add(MaxPooling2D(pool_size=(2, 2)))

# Aplanamiento y capas densas
modelo.add(GlobalAveragePooling2D())
modelo.add(Dense(64, activation='relu'))
modelo.add(Dropout(0.5))
modelo.add(Dense(1, activation='sigmoid'))

# Compilación
modelo.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
modelo.summary()
print("Configuracion de red: LISTO")

# ------------------ CALLBACKS ------------------

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

guardar_mejor_modelo = ModelCheckpoint(
    filepath='IDG_mejor_modelo_tumores.keras',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=12,
    mode='max',
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# ------------------ ENTRENAMIENTO ------------------

entrenamiento = modelo.fit(
    train_generator,
    validation_data=test_generator,
    batch_size=32,
    epochs=epocas,
    callbacks=[detener_callback, early_stopping, reduce_lr, guardar_mejor_modelo]
)

print("Modelo entrenado exitosamente.")
print(entrenamiento.history)

# ------------------ EVALUACIÓN Y VISUALIZACIÓN ------------------

# Accuracy
plt.figure(0)
plt.plot(entrenamiento.history['accuracy'], 'r')
plt.plot(entrenamiento.history['val_accuracy'], 'b')
plt.xticks(np.arange(0, epocas, 2.0))
plt.rcParams['figure.figsize'] = (8, 6)
plt.xlabel('Num of Epochs')
plt.ylabel('Accuracy')
plt.title('Training accuracy vs Validation accuracy')
plt.legend(['train', 'validacion'])
plt.show()

# Pérdida
plt.figure(1)
plt.plot(entrenamiento.history['loss'], 'r')
plt.plot(entrenamiento.history['val_loss'], 'g')
plt.xticks(np.arange(0, epocas, 2.0))
plt.rcParams['figure.figsize'] = (6, 6)
plt.xlabel('Num of Epochs')
plt.ylabel('Loss')
plt.title('Training loss vs Validation loss')
plt.legend(['train', 'validacion'])
plt.show()

# Predicciones y matriz de confusión
entrenamiento_pred = modelo.predict(test_generator)
entrenamiento_pred = np.rint(entrenamiento_pred).astype('int').ravel()
y_test = test_generator.classes

entrenamiento_cm = confusion_matrix(y_test, entrenamiento_pred)
entrenamiento_df_cm = pd.DataFrame(entrenamiento_cm)

plt.figure(figsize=(15, 15))
sbn.set_theme(font_scale=1.4)
sbn.heatmap(entrenamiento_df_cm, annot=True, annot_kws={"size": 22}, cmap="Blues")
plt.ylabel('Valor real')
plt.xlabel('Predicciones')
plt.title('Matriz de Confusión')
plt.show()

# Reporte de clasificación
y_test = test_generator.classes

# Reporte y accuracy
cnn_report = classification_report(y_test, entrenamiento_pred)
print('Accuracy = {:.2f}'.format(accuracy_score(y_test, entrenamiento_pred)))
print(cnn_report)

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, entrenamiento_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.title('Curva ROC de la red neuronal convolucional')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1.01])
plt.ylim([0, 1.01])
plt.xlabel('Tasa de falsos positivos (1 - Especificidad)')
plt.ylabel('Tasa de verdaderos positivos (Sensibilidad)')
plt.grid(True)
plt.show()